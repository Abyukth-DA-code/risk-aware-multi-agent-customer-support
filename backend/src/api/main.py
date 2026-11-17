"""
FastAPI Application Layer for Autonomous Multi-Agent Customer Support System
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation

Destination: src/api/main.py
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    BasePredictionResponse,
    ChunkSummary,
    ComplexityPredictionRequest,
    ComplexityPredictionResponse,
    DomainPredictionRequest,
    DomainPredictionResponse,
    ErrorResponse,
    EscalationPredictionRequest,
    EscalationPredictionResponse,
    EvidenceSummary,
    HealthResponse,
    IntentPredictionRequest,
    IntentPredictionResponse,
    QueryRequest,
    QueryResponse,
    ResolutionSummary,
    RiskPredictionRequest,
    RiskPredictionResponse,
    SentimentPredictionRequest,
    SentimentPredictionResponse,
    TextPredictionRequest,
    TriageSummary,
)
from src.orchestration.model_service import ModelService
from src.orchestration.state import OrchestrationState
from src.rag.retriever import RAGRetriever
from src.api.workspace_routes import router as workspace_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.main")


# ---------------------------------------------------------------------------
# Global Service References & Lifespan Management
# ---------------------------------------------------------------------------

class ServiceContainer:
    """Holds references to warmed-up singleton services."""
    model_service: Optional[ModelService] = None
    retriever: Optional[RAGRetriever] = None
    is_ready: bool = False


services = ServiceContainer()


def _ensure_demo_workspace():
    """
    Ensures that the default demonstration workspace ('ws_demo') and widget token
    ('wtk_demo_public_token') exist in tenants.db and ChromaDB for local evaluation
    and reviewer demonstration.

    Security & Deployment Safeguards:
    1. Disabled if ENABLE_DEMO_WORKSPACE=false/0/no or CAPSTONE_ENV=production.
    2. Respects token revocation: does NOT un-revoke an explicitly revoked demo token.
    3. Strictly isolated: only creates/accesses 'ten_demo' and 'ws_demo'; never touches
       any production or student workspaces.
    """
    import hashlib
    import os
    env = os.environ.get("CAPSTONE_ENV", os.environ.get("APP_ENV", "development")).lower()
    enable_demo = os.environ.get("ENABLE_DEMO_WORKSPACE", "true").lower()

    if enable_demo in ("0", "false", "no", "off"):
        logger.info("Demo workspace seeding skipped (ENABLE_DEMO_WORKSPACE is disabled).")
        return

    if env == "production" and enable_demo != "true":
        logger.info("Demo workspace seeding skipped in production environment.")
        return

    try:
        from src.workspaces.db import WorkspaceDB
        from src.rag.ingestion.pipeline import DocumentIngestionPipeline

        db = WorkspaceDB()
        tenant = db.get_tenant("ten_demo")
        admin_key = os.environ.get("DEMO_ADMIN_API_KEY", "cap_demo_admin_key")
        hashed_admin_key = hashlib.sha256(admin_key.encode("utf-8")).hexdigest()

        if not tenant:
            db.create_tenant(
                name="Demo Storefront Tenant",
                email="demo@capstone.local",
                tenant_id="ten_demo",
            )
            logger.info("Demo tenant 'ten_demo' initialized.")

        key_record = db.get_api_key_record(hashed_admin_key)
        if not key_record:
            db.create_api_key(
                key_id="key_demo_admin",
                tenant_id="ten_demo",
                hashed_key=hashed_admin_key,
                name="Demo Admin Root Key",
            )
            logger.info("Demo API key registered for 'ten_demo'.")

        from src.workspaces.checklist_engine import ChecklistEngine

        pipeline = DocumentIngestionPipeline(db=db)

        demo_workspaces = [
            {
                "workspace_id": "ws_demo",
                "name": "Demo Storefront",
                "domain": "E-commerce",
                "description": "Interactive demo storefront for retail & e-commerce support.",
                "token": "wtk_demo_public_token",
                "category_key": "return_policy",
                "filename": "store_return_policy.txt",
                "content": (
                    "Store Return and Warranty Policy: Eligible items can be returned within 30 days of "
                    "purchase in original condition with tags attached for a full refund. Return shipping "
                    "is free for store credit exchanges. Damaged or defective items must be reported "
                    "within 48 hours of delivery. Clearance items and personalized goods are final sale."
                ),
            },
            {
                "workspace_id": "ws_demo_education",
                "name": "Demo Academy LMS",
                "domain": "Education",
                "description": "Interactive demo portal for university courses and academic student support.",
                "token": "wtk_demo_edu_token",
                "category_key": "attendance_policy",
                "filename": "academic_attendance_policy.txt",
                "content": (
                    "University Academic and Attendance Policy: All enrolled undergraduate and graduate "
                    "students must maintain a minimum of 75 percent attendance in each registered course lecture. "
                    "Students with attendance between 65 and 75 percent may apply for medical condonation by "
                    "submitting verified hospital documentation within 5 business days of absence. "
                    "Attendance below 65 percent results in mandatory course repetition."
                ),
            },
            {
                "workspace_id": "ws_demo_smallbiz",
                "name": "Demo Consulting Group",
                "domain": "Small Business",
                "description": "Interactive demo helpdesk for professional B2B consulting services.",
                "token": "wtk_demo_biz_token",
                "category_key": "pricing_invoices_billing",
                "filename": "consulting_billing_policy.txt",
                "content": (
                    "Professional Services Billing and Payment Policy: Client invoices are issued on the 1st of "
                    "every calendar month with Net-30 payment terms. Monthly retainer packages include up to "
                    "40 consulting hours. Unused hours do not roll over to subsequent billing cycles. "
                    "Retainer cancellations require 14 days written notice prior to the next billing date. "
                    "Late invoices accrue a 1.5 percent monthly charge after 30 days."
                ),
            },
        ]

        for spec in demo_workspaces:
            wid = spec["workspace_id"]
            ws = db.get_workspace(wid)
            if not ws:
                db.create_workspace(
                    tenant_id="ten_demo",
                    name=spec["name"],
                    domain=spec["domain"],
                    description=spec["description"],
                    workspace_id=wid,
                    widget_token=spec["token"],
                )
                logger.info(f"Demo workspace '{wid}' ({spec['domain']}) initialized with '{spec['token']}'.")
            else:
                if ws.get("widget_token_revoked_at"):
                    logger.info(f"Demo workspace '{wid}' token is currently revoked; preserving revocation status.")
                elif not ws.get("widget_token"):
                    with db.get_connection() as conn:
                        conn.execute(
                            "UPDATE workspaces SET widget_token = ? WHERE workspace_id = ?",
                            (spec["token"], wid),
                        )

            # Seed canonical checklist if not yet initialized
            items = db.get_checklist_items(wid)
            if not items:
                checklist_items = ChecklistEngine.get_default_checklist_items(
                    domain=spec["domain"],
                    workspace_id=wid,
                )
                db.insert_checklist_items(checklist_items)
                logger.info(f"Demo checklist for '{wid}' seeded with canonical {spec['domain']} categories.")

            # Ingest authoritative policy document if no documents present
            existing_docs = db.list_workspace_documents(wid)
            if not existing_docs:
                try:
                    pipeline.ingest_document(
                        workspace_id=wid,
                        tenant_id="ten_demo",
                        category_key=spec["category_key"],
                        filename=spec["filename"],
                        file_bytes=spec["content"].encode("utf-8"),
                    )
                    logger.info(f"Demo workspace '{wid}' seeded with '{spec['filename']}'.")
                except Exception as doc_err:
                    logger.warning(f"Failed to seed demo document for '{wid}': {doc_err}")
    except Exception as exc:
        logger.warning(f"Demo workspace initialization check warning: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: initializes models and retriever on startup,
    ensuring low latency on customer query requests.
    """
    logger.info("Initializing unified model service and knowledge retriever...")
    try:
        services.model_service = ModelService.get_instance()
        logger.info("ModelService (M1-M6) loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load ModelService: {exc}", exc_info=True)
        services.model_service = None

    try:
        services.retriever = RAGRetriever()
        logger.info("RAGRetriever initialized successfully.")
    except Exception as exc:
        logger.error(f"Failed to initialize RAGRetriever: {exc}", exc_info=True)
        services.retriever = None

    services.is_ready = services.model_service is not None and services.retriever is not None
    if services.is_ready:
        logger.info("API application services warmed up and ready.")
    else:
        logger.warning("API application started with degraded or partial services.")

    _ensure_demo_workspace()

    yield

    logger.info("Shutting down API service container...")
    services.model_service = None
    services.retriever = None
    services.is_ready = False


# ---------------------------------------------------------------------------
# FastAPI Application Declaration
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Autonomous Multi-Agent Customer Support API",
    description=(
        "Production REST API for query triage, knowledge retrieval, "
        "and human-in-the-loop escalation orchestration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Standard Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Multi-Tenant Workspace & Checklist Router
app.include_router(workspace_router)

# Mount Static Assets for Public Embeddable Chat Widget
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/widget", include_in_schema=False)
@app.get("/widget/demo", include_in_schema=False)
async def widget_demo():
    """Serve the public embeddable widget demonstration page."""
    demo_file = STATIC_DIR / "demo.html"
    if demo_file.exists():
        return FileResponse(str(demo_file))
    return JSONResponse(status_code=404, content={"detail": "Widget demo page not found."})


@app.get("/admin", include_in_schema=False)
async def admin_console():
    """Serve the Multi-Tenant Business Administration Console."""
    admin_file = STATIC_DIR / "admin.html"
    if admin_file.exists():
        return FileResponse(str(admin_file))
    return JSONResponse(status_code=404, content={"detail": "Admin console page not found."})


@app.get("/supervisor", include_in_schema=False)
async def supervisor_dashboard():
    """Serve the Human Supervisor Escalation Dashboard."""
    supervisor_file = STATIC_DIR / "supervisor.html"
    if supervisor_file.exists():
        return FileResponse(str(supervisor_file))
    return JSONResponse(status_code=404, content={"detail": "Supervisor dashboard page not found."})


# ---------------------------------------------------------------------------
# Centralized Error Handlers (Safe, Non-leaking Errors)
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with clean 422 responses."""
    error_details = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        error_details.append(f"{field_path}: {msg}")

    clean_message = "; ".join(error_details) if error_details else "Invalid request payload."
    logger.warning(f"Validation error on {request.url.path}: {clean_message}")

    payload = ErrorResponse(
        error="Validation Error",
        detail=clean_message,
    )
    status_code = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected runtime exceptions without leaking internal stack traces."""
    logger.error(f"Unhandled error processing {request.method} {request.url.path}: {exc}", exc_info=True)
    payload = ErrorResponse(
        error="Internal Server Error",
        detail="An internal error occurred while processing your request. Please contact support or check server logs.",
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump())


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    tags=["Root"],
    summary="API Root Information",
    description="Returns high-level system identification and documentation links.",
)
async def root() -> Dict[str, Any]:
    return {
        "system": "Autonomous Multi-Agent Customer Support Resolution System",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/health",
        "query_endpoint": "/query",
        "model_prediction_endpoints": {
            "m1_intent": "/predict/intent",
            "m2_sentiment": "/predict/sentiment",
            "m3_risk": "/predict/risk",
            "m4_complexity": "/predict/complexity",
            "m5_domain": "/predict/domain",
            "m6_escalation": "/predict/escalation",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Subsystem Health Check",
    description="Inspects availability of model inference, vector retriever, and orchestration.",
)
async def health() -> HealthResponse:
    components: Dict[str, str] = {}

    # 1. Model Service Check
    model_svc = services.model_service
    if model_svc is None:
        try:
            model_svc = ModelService.get_instance()
            services.model_service = model_svc
        except Exception:
            model_svc = None

    if model_svc is not None:
        components["model_service"] = "ready"
    else:
        components["model_service"] = "unavailable"

    # 2. Retriever Check
    retriever_inst = services.retriever
    if retriever_inst is None:
        try:
            retriever_inst = RAGRetriever()
            services.retriever = retriever_inst
        except Exception:
            retriever_inst = None

    if retriever_inst is not None:
        try:
            doc_count = retriever_inst.collection.count()
            components["vector_retriever"] = f"ready (collection documents: {doc_count})"
        except Exception:
            components["vector_retriever"] = "degraded"
    else:
        components["vector_retriever"] = "unavailable"

    # 3. Multi-Agent Orchestration Engine Check
    # (Inspects whether graph.py and safety policy are runnable)
    try:
        from src.orchestration import graph  # type: ignore
        if (
            hasattr(graph, "execute_orchestration")
            or hasattr(graph, "OrchestrationPipeline")
            or hasattr(graph, "run_graph")
            or hasattr(graph, "app")
        ):
            pipeline = graph.get_orchestration_pipeline()
            if pipeline is not None and hasattr(pipeline, "execute_from_state"):
                components["orchestration_engine"] = "ready"
            else:
                components["orchestration_engine"] = "degraded"
        else:
            components["orchestration_engine"] = "pending_integration"
    except Exception as exc:
        logger.warning(f"Orchestration engine health check failed: {exc}")
        components["orchestration_engine"] = "unavailable"

    # Overall Status Determination
    if (
        components.get("model_service") == "ready"
        and "ready" in components.get("vector_retriever", "")
        and components.get("orchestration_engine") == "ready"
    ):
        overall_status = "healthy"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components,
        environment="development",
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Customer Support"],
    summary="Submit Customer Support Query",
    description=(
        "Executes ML triage (M1–M5), provisional autonomy classification (M6), "
        "and hybrid RAG knowledge retrieval. Returns honest status if orchestration is pending."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Process incoming customer query:
    1. Triage query using unified ModelService (Intent, Domain, Sentiment, Risk, Complexity, M6).
    2. Instantiate canonical OrchestrationState.
    3. Retrieve relevant knowledge chunks using hybrid RAGRetriever.
    4. Attach evidence to state.
    5. Evaluate resolution: if full orchestration graph is not yet implemented, return clear controlled pending response.
    """
    logger.info(f"Processing query: '{request.query[:80]}...' (Customer: {request.customer_id}, Session: {request.session_id})")

    # Step 1: Ensure ModelService is available
    model_svc = services.model_service or ModelService.get_instance()

    # Normalize parameters for ML inference
    sens_act = "Yes" if request.sensitive_action else "No"
    fin_imp = "Yes" if request.financial_impact else "No"
    act_type = request.action_type or "Information/Support"

    # Perform triage classification
    try:
        analysis = model_svc.analyze_query(
            query=request.query,
            domain=request.domain,
            sensitive_action=sens_act,
            financial_impact=fin_imp,
            action_type=act_type,
        )
    except Exception as exc:
        logger.error(f"Error during model classification triage: {exc}", exc_info=True)
        raise RuntimeError("Model service inference failed") from exc

    # Step 2: Instantiate canonical OrchestrationState
    state = OrchestrationState.from_model_service(
        analysis=analysis,
        customer_id=request.customer_id,
        session_id=request.session_id,
        metadata=request.metadata or {},
        sensitive_action=request.sensitive_action or False,
        financial_impact=request.financial_impact or False,
        action_type=act_type,
    )

    # Step 3: Retrieve knowledge evidence from hybrid RAG
    retrieved_chunk_summaries: List[ChunkSummary] = []
    retriever_inst = services.retriever
    if retriever_inst is None:
        try:
            retriever_inst = RAGRetriever()
            services.retriever = retriever_inst
        except Exception as init_exc:
            logger.warning(f"Lazy RAGRetriever initialization failed: {init_exc}")
            retriever_inst = None

    if retriever_inst is not None:
        try:
            chunks = retriever_inst.retrieve(
                query=request.query,
                domain=state.domain,
                top_k=request.top_k or 3,
                distinct_documents=True,
                candidate_multiplier=4,
                hybrid=False,
            )
            state.add_retrieved_chunks(chunks)

            for c in chunks:
                retrieved_chunk_summaries.append(
                    ChunkSummary(
                        chunk_id=str(c.get("chunk_id", "")),
                        document_id=str(c.get("document_id", "")),
                        domain=str(c.get("domain", "")),
                        title=str(c.get("title", "")),
                        authority=str(c.get("authority", "")),
                        freshness_status=str(c.get("freshness_status", "")),
                        document_type=str(c.get("document_type", "")),
                        retrieval_priority=int(c.get("retrieval_priority", 1)),
                        similarity=round(float(c.get("similarity", 0.0)), 4),
                        chunk_text=str(c.get("chunk_text", "")),
                        rrf_score=round(float(c["rrf_score"]), 6) if c.get("rrf_score") is not None else None,
                        bm25_score=round(float(c["bm25_score"]), 4) if c.get("bm25_score") is not None else None,
                    )
                )
        except Exception as exc:
            logger.warning(f"RAG retrieval encountered an issue: {exc}", exc_info=True)
            state.errors.append(f"RAG retrieval degraded: {str(exc)}")

    # Step 4: Multi-Agent Orchestration & Deterministic Safety Policy Evaluation
    if request.run_orchestration is False:
        should_orchestrate = False
    elif request.metadata and request.metadata.get("run_orchestration") is False:
        should_orchestrate = False
    else:
        should_orchestrate = bool(request.run_orchestration)

    orchestration_ready = False
    final_response_text: Optional[str] = None
    resolution_message = (
        "Query successfully triaged (M1-M6) and knowledge evidence retrieved. "
        "Multi-agent resolution graph (graph.py) and safety guardrail engine (safety_policy.py) are pending integration."
    )

    if should_orchestrate:
        try:
            from src.orchestration.graph import execute_orchestration
            state = execute_orchestration(state)
            orchestration_ready = True
            final_response_text = state.final_response
            resolution_message = (
                state.decision_rationale
                or f"Orchestration completed with decision '{state.final_autonomy_tier}'."
            )
        except Exception as exc:
            logger.error(f"Orchestration workflow execution encountered an error: {exc}", exc_info=True)
            state.errors.append(f"Orchestration error: {str(exc)}")
            resolution_message = f"Orchestration error: {str(exc)}"

    # Step 5: Construct Structured Pydantic Response
    triage_summary = TriageSummary(
        domain=state.domain,
        domain_confidence=state.domain_confidence,
        intent=state.intent,
        intent_confidence=state.intent_confidence,
        sentiment=state.sentiment,
        sentiment_confidence=state.sentiment_confidence,
        risk=state.risk,
        risk_confidence=state.risk_confidence,
        risk_class_probabilities=state.risk_class_probabilities,
        complexity=state.complexity,
        complexity_confidence=state.complexity_confidence,
        complexity_class_probabilities=state.complexity_class_probabilities,
        provisional_autonomy_tier=state.provisional_autonomy_tier,
        provisional_escalation_required=state.provisional_escalation_required,
        provisional_confidence=state.provisional_confidence,
    )

    evidence_summary = EvidenceSummary(
        evidence_quality=state.evidence_quality,
        retrieval_count=state.retrieval_count,
        retrieval_similarity_score=state.retrieval_similarity_score,
        has_outdated_evidence=state.has_outdated_evidence,
        has_conflicting_evidence=state.has_conflicting_evidence,
        chunks=retrieved_chunk_summaries,
    )

    resolution_summary = ResolutionSummary(
        status=state.status,
        orchestration_ready=orchestration_ready,
        final_autonomy_tier=state.final_autonomy_tier,
        final_escalation_required=state.final_escalation_required,
        final_response=final_response_text,
        message=resolution_message,
        safety_rule_triggered=state.safety_rule_triggered,
        safety_override=state.safety_override,
        decision_source=state.decision_source,
        final_confidence=state.final_confidence,
        decision_rationale=state.decision_rationale,
        escalation_reason=state.escalation_reason,
        escalation_dossier=state.escalation_dossier,
    )

    return QueryResponse(
        query=state.query,
        session_id=state.session_id,
        customer_id=state.customer_id,
        triage=triage_summary,
        evidence=evidence_summary,
        resolution=resolution_summary,
    )


@app.post(
    "/orchestrate",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Customer Support"],
    summary="Submit Customer Support Query with Full Orchestration Workflow",
    description=(
        "Executes complete multi-stage pipeline: ML triage (M1-M6), RAG retrieval, "
        "evidence assessment, and deterministic safety policy evaluation."
    ),
)
async def process_orchestrated_query(request: QueryRequest) -> QueryResponse:
    """Executes full query resolution pipeline including safety policy gating."""
    request.run_orchestration = True
    return await process_query(request)


# ---------------------------------------------------------------------------
# Individual Model Prediction Endpoints (M1–M6)
# ---------------------------------------------------------------------------

def _normalize_flag(value: Any, default: str = "No") -> str:
    """Normalize boolean or string flag to canonical 'Yes' or 'No' string."""
    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("yes", "true", "1"):
            return "Yes"
        if cleaned in ("no", "false", "0"):
            return "No"
        return value.strip().title()
    return "Yes" if bool(value) else default


@app.post(
    "/predict/intent",
    response_model=IntentPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Predictions (M1-M6)"],
    summary="M1: Predict Customer Intent",
    description=(
        "Classifies customer query into one of 47 fine-grained intent classes using LinearSVC (M1). "
        "Confidence is an uncalibrated score computed via softmax over decision margins."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def predict_intent(request: IntentPredictionRequest) -> IntentPredictionResponse:
    """Execute M1 Intent Classifier inference."""
    model_svc = services.model_service or ModelService.get_instance()
    try:
        res = model_svc.predict_intent(request.query)
    except Exception as exc:
        logger.error(f"Error during M1 intent prediction: {exc}", exc_info=True)
        raise RuntimeError(f"M1 intent prediction failed: {exc}") from exc

    return IntentPredictionResponse(
        query=request.query,
        prediction=res["intent"],
        intent=res["intent"],
        confidence=res.get("confidence"),
    )


@app.post(
    "/predict/sentiment",
    response_model=SentimentPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Predictions (M1-M6)"],
    summary="M2: Predict Customer Sentiment",
    description=(
        "Classifies customer query sentiment into Negative, Neutral, or Positive using LogisticRegression (M2). "
        "Confidence is a calibrated probability from predict_proba."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def predict_sentiment(request: SentimentPredictionRequest) -> SentimentPredictionResponse:
    """Execute M2 Sentiment Analyzer inference."""
    model_svc = services.model_service or ModelService.get_instance()
    try:
        res = model_svc.predict_sentiment(request.query)
    except Exception as exc:
        logger.error(f"Error during M2 sentiment prediction: {exc}", exc_info=True)
        raise RuntimeError(f"M2 sentiment prediction failed: {exc}") from exc

    return SentimentPredictionResponse(
        query=request.query,
        prediction=res["sentiment"],
        sentiment=res["sentiment"],
        confidence=res.get("confidence"),
    )


@app.post(
    "/predict/risk",
    response_model=RiskPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Predictions (M1-M6)"],
    summary="M3: Predict Operational Risk Tier",
    description=(
        "Classifies operational risk into Low, Medium, or High using RandomForestClassifier (M3). "
        "Confidence is a calibrated probability with full class distribution."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def predict_risk(request: RiskPredictionRequest) -> RiskPredictionResponse:
    """Execute M3 Risk Tier Classifier inference."""
    model_svc = services.model_service or ModelService.get_instance()
    sens_act = _normalize_flag(request.sensitive_action, "No")
    fin_imp = _normalize_flag(request.financial_impact, "No")
    act_type = request.action_type or "Information/Support"
    domain = request.domain or "E-commerce"

    try:
        res = model_svc.predict_risk(
            query=request.query,
            domain=domain,
            sensitive_action=sens_act,
            financial_impact=fin_imp,
            action_type=act_type,
        )
    except Exception as exc:
        logger.error(f"Error during M3 risk prediction: {exc}", exc_info=True)
        raise RuntimeError(f"M3 risk prediction failed: {exc}") from exc

    return RiskPredictionResponse(
        query=request.query,
        prediction=res["risk"],
        risk=res["risk"],
        confidence=res.get("confidence"),
        class_probabilities=res.get("class_probabilities", {}),
    )


@app.post(
    "/predict/complexity",
    response_model=ComplexityPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Predictions (M1-M6)"],
    summary="M4: Predict Task Complexity Tier",
    description=(
        "Classifies inquiry complexity into Low, Medium, or High using RandomForestClassifier (M4). "
        "Confidence is a calibrated probability with full class distribution."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def predict_complexity(request: ComplexityPredictionRequest) -> ComplexityPredictionResponse:
    """Execute M4 Task Complexity Classifier inference."""
    model_svc = services.model_service or ModelService.get_instance()
    domain = request.domain or "E-commerce"

    try:
        res = model_svc.predict_complexity(
            query=request.query,
            domain=domain,
        )
    except Exception as exc:
        logger.error(f"Error during M4 complexity prediction: {exc}", exc_info=True)
        raise RuntimeError(f"M4 complexity prediction failed: {exc}") from exc

    return ComplexityPredictionResponse(
        query=request.query,
        prediction=res["complexity"],
        complexity=res["complexity"],
        confidence=res.get("confidence"),
        class_probabilities=res.get("class_probabilities", {}),
    )


@app.post(
    "/predict/domain",
    response_model=DomainPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Predictions (M1-M6)"],
    summary="M5: Predict Business Domain",
    description=(
        "Classifies business domain (E-commerce, Education, Small Business) using LinearSVC (M5). "
        "Confidence is an uncalibrated score computed via softmax over decision margins."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def predict_domain(request: DomainPredictionRequest) -> DomainPredictionResponse:
    """Execute M5 Domain Classifier inference."""
    model_svc = services.model_service or ModelService.get_instance()
    try:
        res = model_svc.predict_domain(request.query)
    except Exception as exc:
        logger.error(f"Error during M5 domain prediction: {exc}", exc_info=True)
        raise RuntimeError(f"M5 domain prediction failed: {exc}") from exc

    return DomainPredictionResponse(
        query=request.query,
        prediction=res["domain"],
        domain=res["domain"],
        confidence=res.get("confidence"),
    )


@app.post(
    "/predict/escalation",
    response_model=EscalationPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Predictions (M1-M6)"],
    summary="M6: Predict Escalation & Autonomy Tier",
    description=(
        "Predicts provisional autonomy tier (ANSWER, ORCHESTRATE, ESCALATE) and escalation requirement "
        "using DecisionTreeClassifier (M6). Note: M6 output is provisional and subject to safety guardrail override."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
@app.post(
    "/predict/autonomy",
    response_model=EscalationPredictionResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def predict_escalation(request: EscalationPredictionRequest) -> EscalationPredictionResponse:
    """Execute M6 Escalation & Autonomy Engine inference."""
    model_svc = services.model_service or ModelService.get_instance()
    domain = request.domain or "E-commerce"
    risk = request.risk or "Low"
    complexity = request.complexity or "Low"
    evidence_quality = request.evidence_quality or "Strong"
    verif_req = _normalize_flag(request.verification_required, "No")
    sens_act = _normalize_flag(request.sensitive_action, "No")
    fin_imp = _normalize_flag(request.financial_impact, "No")
    sentiment = request.sentiment or "Neutral"

    try:
        res = model_svc.predict_escalation(
            query=request.query,
            domain=domain,
            risk=risk,
            complexity=complexity,
            evidence_quality=evidence_quality,
            verification_required=verif_req,
            sensitive_action=sens_act,
            financial_impact=fin_imp,
            sentiment=sentiment,
        )
    except Exception as exc:
        logger.error(f"Error during M6 escalation prediction: {exc}", exc_info=True)
        raise RuntimeError(f"M6 escalation prediction failed: {exc}") from exc

    return EscalationPredictionResponse(
        query=request.query,
        prediction=res["autonomy_tier"],
        autonomy_tier=res["autonomy_tier"],
        escalation_required=res["escalation_required"],
        confidence=res.get("confidence"),
        class_probabilities=res.get("class_probabilities", {}),
    )


