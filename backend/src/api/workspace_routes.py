"""
FastAPI Router for Multi-Tenant Workspaces & Dynamic Checklists
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
"""

from collections import defaultdict
import re
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.schemas import (
    ChunkSummary,
    EvidenceSummary,
    ResolutionSummary,
    TriageSummary,
)
from src.rag.ingestion.extractors import TextExtractionError
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.workspaces.integration_kit import generate_integration_kit_zip
from src.workspaces.models import (
    ChecklistItemResponse,
    CustomCategoryRequest,
    CustomerMemoryCreateRequest,
    CustomerMemoryDeleteResponse,
    CustomerMemoryListResponse,
    CustomerMemoryResponse,
    EscalationListItem,
    EscalationResponse,
    EscalationReviewRequest,
    EscalationStatus,
    ReadinessReport,
    TenantRegisterRequest,
    TenantResponse,
    WidgetChatRequest,
    WidgetChatResponse,
    WidgetConfigResponse,
    WidgetTokenResponse,
    WorkspaceCreateRequest,
    WorkspaceQueryRequest,
    WorkspaceQueryResponse,
    WorkspaceResponse,
)
from src.workspaces.service import WorkspaceService

router = APIRouter(prefix="/api/v1", tags=["Workspaces & Multi-Tenancy"])

# Bearer token security scheme (auto_error=False to allow clean custom error handling)
security = HTTPBearer(auto_error=False)
_workspace_service = WorkspaceService()
_ingestion_pipeline = DocumentIngestionPipeline()


class WidgetRateLimiter:
    """Sliding-window in-memory rate limiter per client key."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        self.requests[key] = [t for t in self.requests[key] if t > cutoff]
        if len(self.requests[key]) >= self.max_requests:
            return False
        self.requests[key].append(now)
        return True

    def is_allowed(self, key: str) -> bool:
        """Alias for check() to support standard rate limiter interface."""
        return self.check(key)


_widget_rate_limiter = WidgetRateLimiter()


def get_workspace_service() -> WorkspaceService:
    return _workspace_service


def get_ingestion_pipeline() -> DocumentIngestionPipeline:
    return _ingestion_pipeline


def get_current_tenant(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_widget_token: Optional[str] = Header(None, alias="X-Widget-Token"),
    service: WorkspaceService = Depends(get_workspace_service),
) -> Dict[str, Any]:
    """
    Strict Tenant Authentication Dependency.
    Extracts API key from Authorization header or X-API-Key.
    Never trusts client-supplied tenant_id parameter.
    """
    raw_key = None
    if credentials and credentials.credentials:
        raw_key = credentials.credentials
    elif x_api_key:
        raw_key = x_api_key
    elif x_widget_token:
        raw_key = x_widget_token

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Security Invariant: Widget tokens cannot authenticate administrative endpoints
    if raw_key.startswith("wtk_"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Widget tokens cannot be used to authenticate administrative endpoints. Use an admin API key.",
        )

    tenant_auth = service.authenticate_api_key(raw_key)
    if not tenant_auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return tenant_auth


def get_widget_auth(
    workspace_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_widget_token: Optional[str] = Header(None, alias="X-Widget-Token"),
    service: WorkspaceService = Depends(get_workspace_service),
) -> Dict[str, Any]:
    """
    Public Scoped Widget Authentication Dependency.
    Validates workspace-scoped wtk_... token for customer chat.
    Rejects administrative keys (sk_...) and mismatched or revoked tokens.
    """
    raw_token = None
    if credentials and credentials.credentials:
        raw_token = credentials.credentials
    elif x_widget_token:
        raw_token = x_widget_token

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Widget authentication required. Supply X-Widget-Token or Bearer wtk_... token.",
        )

    # Protect against client misuse: admin keys must never be accepted on widget endpoint
    if raw_token.startswith("sk_") or raw_token.startswith("cap_"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API keys (sk_... or cap_...) cannot be used on public widget chat. Please use a scoped widget token.",
        )

    try:
        ws = service.verify_widget_token(workspace_id=workspace_id, widget_token=raw_token)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    return ws


# ---------------------------------------------------------------------------
# Tenant Onboarding
# ---------------------------------------------------------------------------

@router.post(
    "/tenants/register",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new business tenant and receive root API key",
)
def register_tenant(
    req: TenantRegisterRequest,
    service: WorkspaceService = Depends(get_workspace_service),
):
    try:
        return service.register_tenant(req)
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A tenant with email '{req.email}' is already registered.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Workspace Management
# ---------------------------------------------------------------------------

@router.post(
    "/workspaces",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new business workspace with dynamic checklist",
)
def create_workspace(
    req: WorkspaceCreateRequest,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    # Tenant ID is securely extracted from authenticated credential, never client request body!
    tenant_id = current_tenant["tenant_id"]
    return service.create_workspace(authenticated_tenant_id=tenant_id, req=req)


@router.get(
    "/workspaces",
    response_model=List[WorkspaceResponse],
    summary="List all workspaces owned by the authenticated tenant",
)
def list_workspaces(
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    return service.list_workspaces(authenticated_tenant_id=tenant_id)


@router.get(
    "/workspaces/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Get workspace details (enforcing tenant authorization)",
)
def get_workspace(
    workspace_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.get_workspace(authenticated_tenant_id=tenant_id, workspace_id=workspace_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


# ---------------------------------------------------------------------------
# Dynamic Checklist & Readiness
# ---------------------------------------------------------------------------

@router.get(
    "/workspaces/{workspace_id}/checklist",
    response_model=List[ChecklistItemResponse],
    summary="Get domain-specific dynamic knowledge-base checklist",
)
def get_checklist(
    workspace_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.get_checklist(authenticated_tenant_id=tenant_id, workspace_id=workspace_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.post(
    "/workspaces/{workspace_id}/checklist/custom",
    response_model=ChecklistItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a custom category to the checklist beyond the domain default",
)
def add_custom_category(
    workspace_id: str,
    req: CustomCategoryRequest,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.add_custom_category(
            authenticated_tenant_id=tenant_id,
            workspace_id=workspace_id,
            req=req,
        )
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.get(
    "/workspaces/{workspace_id}/readiness",
    response_model=ReadinessReport,
    summary="Get true backend operational readiness report for chatbot",
)
def get_readiness_report(
    workspace_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.get_readiness_report(
            authenticated_tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


# ---------------------------------------------------------------------------
# Multi-Format Document Ingestion & Management
# ---------------------------------------------------------------------------

@router.post(
    "/workspaces/{workspace_id}/documents/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload, extract, chunk, embed, and index a document into workspace knowledge base",
)
async def upload_document(
    workspace_id: str,
    category_key: str = Form(..., description="Target checklist category key"),
    replace_document_id: Optional[str] = Form(None, description="Optional document ID to replace"),
    file: UploadFile = File(..., description="Document file (.pdf, .docx, .txt, .csv)"),
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        service.verify_workspace_access(tenant_id, workspace_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    try:
        content = await file.read()
        result = pipeline.ingest_document(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            category_key=category_key,
            filename=file.filename or "unknown",
            file_bytes=content,
            replace_document_id=replace_document_id,
        )
        return result
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        if "Duplicate document detected" in str(exc):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except TextExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(exc)}",
        )


@router.get(
    "/workspaces/{workspace_id}/documents",
    summary="List all active ingested documents for the workspace",
)
def list_documents(
    workspace_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        service.verify_workspace_access(tenant_id, workspace_id)
        return service.db.list_workspace_documents(workspace_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.delete(
    "/workspaces/{workspace_id}/documents/{document_id}",
    summary="Delete document, purge vector chunks, and reset checklist category",
)
def delete_document(
    workspace_id: str,
    document_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        service.verify_workspace_access(tenant_id, workspace_id)
        pipeline.delete_document(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            document_id=document_id,
        )
        return {
            "message": f"Document '{document_id}' successfully deleted and vector chunks purged.",
            "document_id": document_id,
            "workspace_id": workspace_id,
        }
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


# ---------------------------------------------------------------------------
# Tenant-Aware Isolated RAG Query & Orchestration
# ---------------------------------------------------------------------------

@router.post(
    "/workspaces/{workspace_id}/query",
    response_model=WorkspaceQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit query to workspace-isolated knowledge base with HITL orchestration",
    description=(
        "Executes multi-agent triage, strictly isolated dense retrieval from workspace collection, "
        "deterministic safety policy evaluation, and grounded citation answer synthesis."
    ),
)
def query_workspace(
    workspace_id: str,
    request: WorkspaceQueryRequest,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        ws = service.verify_workspace_access(tenant_id, workspace_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    from src.orchestration.graph import get_orchestration_pipeline
    pipeline = get_orchestration_pipeline()

    meta = dict(request.metadata)
    meta["tenant_id"] = tenant_id
    meta["db"] = service.db
    meta["workspace_id"] = workspace_id
    meta["generate_grounded_response"] = True

    # Retrieve chronological session history strictly bounded to this workspace and session
    history = service.get_session_history(workspace_id, request.session_id, limit=6) if request.session_id else []
    cust_memories = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in service.list_customer_memories(tenant_id, workspace_id, request.customer_id)] if request.customer_id else []

    # Execute isolated pipeline bounded strictly to this workspace
    state = pipeline.run(
        query=request.query,
        customer_id=request.customer_id,
        session_id=request.session_id,
        domain=ws["domain"],
        sensitive_action=request.sensitive_action,
        financial_impact=request.financial_impact,
        action_type=request.action_type,
        top_k=request.top_k,
        metadata=meta,
        workspace_id=workspace_id,
        generate_grounded_response=True,
        conversation_history=history,
        customer_memory=cust_memories,
    )

    # Persist turn in short-term session memory if session_id is provided
    if request.session_id:
        user_res = state.final_response
        if state.final_escalation_required and isinstance(state.escalation_dossier, dict):
            user_res = state.escalation_dossier.get("escalation_notice")
        service.save_session_turn(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            session_id=request.session_id,
            user_query=request.query,
            resolved_query=state.resolved_query,
            bot_response=user_res,
            final_autonomy_tier=state.final_autonomy_tier,
            citations=state.retrieved_chunks,
        )

    # Format chunks
    retrieved_chunk_summaries = []
    for c in state.retrieved_chunks:
        if isinstance(c, dict):
            retrieved_chunk_summaries.append(
                ChunkSummary(
                    chunk_id=str(c.get("chunk_id", "")),
                    document_id=str(c.get("document_id", "")),
                    domain=str(c.get("domain", "")),
                    title=str(c.get("title", "")),
                    authority=str(c.get("authority", "High")),
                    freshness_status=str(c.get("freshness_status", "Current")),
                    document_type=str(c.get("document_type", "Policy")),
                    retrieval_priority=int(c.get("retrieval_priority", 1)),
                    similarity=round(float(c.get("similarity", 0.0)), 4),
                    chunk_text=str(c.get("chunk_text", "")),
                    rrf_score=round(float(c["rrf_score"]), 6) if c.get("rrf_score") is not None else None,
                    bm25_score=round(float(c["bm25_score"]), 4) if c.get("bm25_score") is not None else None,
                )
            )

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

    resolution_message = (
        state.decision_rationale
        or f"Orchestration completed with decision '{state.final_autonomy_tier}'."
    )

    resolution_summary = ResolutionSummary(
        status=state.status,
        orchestration_ready=True,
        final_autonomy_tier=state.final_autonomy_tier,
        final_escalation_required=state.final_escalation_required,
        final_response=state.final_response,
        message=resolution_message,
        safety_rule_triggered=state.safety_rule_triggered,
        safety_override=state.safety_override,
        decision_source=state.decision_source,
        final_confidence=state.final_confidence,
        decision_rationale=state.decision_rationale,
        escalation_reason=state.escalation_reason,
        escalation_dossier=state.escalation_dossier,
    )

    return WorkspaceQueryResponse(
        workspace_id=workspace_id,
        query=state.query,
        session_id=state.session_id,
        customer_id=state.customer_id,
        grounded_response=(
            state.final_response
            if not state.final_escalation_required
            else None
        ),
        triage=triage_summary.model_dump(),
        evidence=evidence_summary.model_dump(),
        resolution=resolution_summary.model_dump(),
    )


# ---------------------------------------------------------------------------
# Human-in-the-Loop (HITL) Escalation Queue Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/workspaces/{workspace_id}/escalations",
    response_model=List[EscalationListItem],
    summary="List escalation queue items for authorized workspace",
)
def list_escalations(
    workspace_id: str,
    status_filter: Optional[EscalationStatus] = Query(None, alias="status"),
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        status_val = status_filter.value if status_filter else None
        return service.list_escalations(
            authenticated_tenant_id=tenant_id,
            workspace_id=workspace_id,
            status=status_val,
        )
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.get(
    "/workspaces/{workspace_id}/escalations/{escalation_id}",
    response_model=EscalationResponse,
    summary="Retrieve full escalation dossier with evidence and audit history",
)
def get_escalation(
    workspace_id: str,
    escalation_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.get_escalation(
            authenticated_tenant_id=tenant_id,
            workspace_id=workspace_id,
            escalation_id=escalation_id,
        )
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Escalation dossier '{escalation_id}' not found in workspace '{workspace_id}'.",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.post(
    "/workspaces/{workspace_id}/escalations/{escalation_id}/review",
    response_model=EscalationResponse,
    summary="Submit human supervisor review decision with audit trail",
)
def review_escalation(
    workspace_id: str,
    escalation_id: str,
    req: EscalationReviewRequest,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.review_escalation(
            authenticated_tenant_id=tenant_id,
            workspace_id=workspace_id,
            escalation_id=escalation_id,
            req=req,
        )
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Escalation dossier '{escalation_id}' not found in workspace '{workspace_id}'.",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get(
    "/workspaces/{workspace_id}/sessions/{session_id}/turns",
    summary="Retrieve chronological conversation history turns for a session dossier",
)
def get_session_turns(
    workspace_id: str,
    session_id: str,
    limit: int = Query(20, ge=1, le=100),
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        service.verify_workspace_access(tenant_id, workspace_id)
        return service.get_session_history(
            workspace_id=workspace_id,
            session_id=session_id,
            limit=limit,
        )
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found.",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


# ---------------------------------------------------------------------------
# Public Embeddable Widget Endpoints (Scoped Token & Chat)
# ---------------------------------------------------------------------------

@router.post(
    "/workspaces/{workspace_id}/widget/chat",
    response_model=WidgetChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit query from embeddable JavaScript chat widget with multi-turn session memory",
)
def widget_chat(
    workspace_id: str,
    request: WidgetChatRequest,
    req_obj: Request,
    ws: Dict[str, Any] = Depends(get_widget_auth),
    service: WorkspaceService = Depends(get_workspace_service),
):
    # 1. Sliding-Window Rate Limiting per Client IP
    client_ip = req_obj.client.host if req_obj.client else "unknown"
    client_key = f"{client_ip}:{workspace_id}"
    if not _widget_rate_limiter.check(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a few seconds before submitting another message.",
        )

    # 2. Identifier Sanitization & Validation (Reject Script Tags / Suspicious Chars)
    session_id = request.session_id
    if session_id:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", session_id) or len(session_id) > 64:
            session_id = f"sess_{uuid.uuid4().hex[:16]}"
    else:
        session_id = f"sess_{uuid.uuid4().hex[:16]}"

    customer_id = request.customer_id
    if customer_id and (not re.match(r"^[a-zA-Z0-9_\-]+$", customer_id) or len(customer_id) > 64):
        customer_id = None

    tenant_id = ws["tenant_id"]

    # 3. Retrieve Chronological Session History (Strictly Bounded to this Session)
    history = service.get_session_history(workspace_id, session_id, limit=6)

    # 4. Retrieve Long-Term Customer Context (Non-Authoritative)
    cust_memories = []
    if customer_id:
        cust_memories = service.db.list_customer_memories(workspace_id, customer_id)

    from src.orchestration.graph import get_orchestration_pipeline
    pipeline = get_orchestration_pipeline()

    meta = dict(request.metadata)
    meta["tenant_id"] = tenant_id
    meta["db"] = service.db
    meta["workspace_id"] = workspace_id
    meta["generate_grounded_response"] = True

    # 5. Execute Multi-Agent Orchestration with Reference Resolution
    state = pipeline.run(
        query=request.query,
        customer_id=customer_id,
        session_id=session_id,
        domain=ws["domain"],
        sensitive_action=False,
        financial_impact=False,
        action_type="Information/Support",
        top_k=3,
        metadata=meta,
        workspace_id=workspace_id,
        generate_grounded_response=True,
        conversation_history=history,
        customer_memory=cust_memories,
    )

    # Format Citation Metadata for UI
    citations = []
    for c in state.retrieved_chunks:
        if isinstance(c, dict):
            citations.append({
                "chunk_id": str(c.get("chunk_id", "")),
                "document_id": str(c.get("document_id", "")),
                "document_name": str(c.get("title", "Policy Document")),
                "title": str(c.get("title", "Policy Document")),
                "category_key": str(c.get("category_key") or c.get("document_type", "")),
                "authority": str(c.get("authority", "High")),
                "freshness_status": str(c.get("freshness_status", "Current")),
                "similarity": round(float(c.get("similarity", 0.0)), 4),
                "chunk_text": str(c.get("chunk_text", ""))[:200],
            })

    # Determine Structured User-Facing Response
    escalation_id = None
    clarification_prompt = None
    user_response = state.final_response

    if state.final_escalation_required:
        if isinstance(state.escalation_dossier, dict):
            escalation_id = state.escalation_dossier.get("escalation_id")
            user_response = state.escalation_dossier.get("escalation_notice")
        if not user_response:
            user_response = state.decision_rationale or "Your inquiry has been routed to a human support specialist."
    elif state.final_autonomy_tier == "CLARIFY":
        clarification_prompt = state.final_response

    # 6. Save Verified Turn into Session Memory
    service.save_session_turn(
        workspace_id=workspace_id,
        tenant_id=tenant_id,
        session_id=session_id,
        user_query=request.query,
        resolved_query=state.resolved_query,
        bot_response=user_response,
        final_autonomy_tier=state.final_autonomy_tier,
        citations=citations,
    )

    return WidgetChatResponse(
        workspace_id=workspace_id,
        session_id=session_id,
        query=state.query,
        resolved_query=state.resolved_query,
        response=user_response,
        status=state.status,
        autonomy_tier=state.final_autonomy_tier,
        escalation_required=bool(state.final_escalation_required),
        escalation_id=escalation_id,
        clarification_prompt=clarification_prompt,
        citations=citations,
    )


# ---------------------------------------------------------------------------
# Public Widget Management & Configuration Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/workspaces/{workspace_id}/widget/token",
    response_model=WidgetTokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Get active public widget token for website embedding",
)
def get_widget_token(
    workspace_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.get_widget_token(tenant_id, workspace_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.post(
    "/workspaces/{workspace_id}/widget/token/rotate",
    response_model=WidgetTokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Rotate public widget token (invalidates previous embed snippet)",
)
def rotate_widget_token(
    workspace_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.rotate_widget_token(tenant_id, workspace_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.post(
    "/workspaces/{workspace_id}/widget/token/revoke",
    response_model=WidgetTokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Revoke public widget token (immediately disables widget queries)",
)
def revoke_widget_token(
    workspace_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.revoke_widget_token(tenant_id, workspace_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.get(
    "/workspaces/{workspace_id}/widget/config",
    response_model=WidgetConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Get public workspace display configuration for embeddable widget",
)
def get_widget_config(
    workspace_id: str,
    x_widget_token: Optional[str] = Header(None, alias="X-Widget-Token"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    service: WorkspaceService = Depends(get_workspace_service),
):
    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    elif x_widget_token:
        token = x_widget_token

    if token and token.startswith("wtk_"):
        try:
            ws = service.verify_widget_token(workspace_id, token)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    else:
        ws = service.db.get_workspace(workspace_id)
        if not ws:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")

    return WidgetConfigResponse(
        workspace_id=ws["workspace_id"],
        name=ws["name"],
        domain=ws["domain"],
        description=ws.get("description"),
        is_active=ws["is_active"],
    )


# ---------------------------------------------------------------------------
# Integration Kit Download Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/integration/download-kit",
    summary="Download default public framework integration kit ZIP",
    description=(
        "Downloads the public integration kit ZIP containing widget.js, embed_example.html, "
        "INTEGRATION_GUIDE.md, api_client_example.py, and api_client_example.js. "
        "Safe for public download with zero sensitive credential leakage."
    ),
    response_class=Response,
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": "Downloadable ZIP integration kit containing client widget and integration samples.",
        }
    },
)
def download_default_integration_kit(
    request: Request,
):
    api_base = str(request.base_url).rstrip("/")
    if not api_base or api_base == "http://testserver":
        api_base = "http://localhost:8000"

    zip_bytes = generate_integration_kit_zip(api_base=api_base)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="capstone-integration-kit.zip"',
            "Content-Type": "application/zip",
        },
    )


@router.get(
    "/workspaces/{workspace_id}/integration/download-kit",
    summary="Download workspace-specific integration kit ZIP",
    description=(
        "Downloads the integration kit ZIP customized for the authorized workspace with pre-populated "
        "scoped public widget token (wtk_...) and workspace ID. Requires tenant authentication."
    ),
    response_class=Response,
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": "Downloadable ZIP integration kit customized for this workspace.",
        },
        401: {"description": "Authentication required"},
        403: {"description": "Forbidden (tenant does not own workspace or used widget token)"},
        404: {"description": "Workspace not found"},
    },
)
def download_workspace_integration_kit(
    workspace_id: str,
    request: Request,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        service.verify_workspace_access(tenant_id, workspace_id)
        ws = service.db.get_workspace(workspace_id)
        if not ws:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")

        token_resp = service.get_widget_token(tenant_id, workspace_id)
        widget_token = token_resp.widget_token

        api_base = str(request.base_url).rstrip("/")
        if not api_base or api_base == "http://testserver":
            api_base = "http://localhost:8000"

        zip_bytes = generate_integration_kit_zip(
            workspace_id=ws["workspace_id"],
            widget_token=widget_token,
            workspace_name=ws["name"],
            domain=ws["domain"],
            api_base=api_base,
        )

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="integration-kit-{workspace_id}.zip"',
                "Content-Type": "application/zip",
            },
        )
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


# ---------------------------------------------------------------------------
# Long-Term Customer Memory Endpoints (Admin Authenticated)
# ---------------------------------------------------------------------------

@router.post(
    "/workspaces/{workspace_id}/customers/{customer_id}/memory",
    response_model=CustomerMemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create customer memory with configurable TTL retention",
)
def create_customer_memory(
    workspace_id: str,
    customer_id: str,
    req: CustomerMemoryCreateRequest,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        return service.create_customer_memory(
            authenticated_tenant_id=tenant_id,
            workspace_id=workspace_id,
            customer_id=customer_id,
            category=req.category,
            content=req.content,
            confidence=req.confidence,
            ttl_days=req.ttl_days,
        )
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.get(
    "/workspaces/{workspace_id}/customers/{customer_id}/memory",
    response_model=CustomerMemoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="List active customer memories with automatic read-time TTL expiration filtering",
)
def list_customer_memories(
    workspace_id: str,
    customer_id: str,
    include_expired: bool = Query(default=False),
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        memories = service.list_customer_memories(
            authenticated_tenant_id=tenant_id,
            workspace_id=workspace_id,
            customer_id=customer_id,
            include_expired=include_expired,
        )
        return CustomerMemoryListResponse(
            workspace_id=workspace_id,
            customer_id=customer_id,
            memories=memories,
            total_count=len(memories),
        )
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.delete(
    "/workspaces/{workspace_id}/customers/{customer_id}/memory/{memory_id}",
    response_model=CustomerMemoryDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete specific customer memory entry",
)
def delete_customer_memory(
    workspace_id: str,
    customer_id: str,
    memory_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        deleted = service.delete_customer_memory(
            authenticated_tenant_id=tenant_id,
            workspace_id=workspace_id,
            memory_id=memory_id,
        )
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory '{memory_id}' not found.")
        return CustomerMemoryDeleteResponse(
            workspace_id=workspace_id,
            customer_id=customer_id,
            memory_id=memory_id,
            deleted_count=1,
            message="Customer memory record permanently deleted.",
        )
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.delete(
    "/workspaces/{workspace_id}/customers/{customer_id}/memory",
    response_model=CustomerMemoryDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Technical Right-to-be-Forgotten: Purge all customer memory records for this customer",
)
def delete_all_customer_memories(
    workspace_id: str,
    customer_id: str,
    current_tenant: Dict[str, Any] = Depends(get_current_tenant),
    service: WorkspaceService = Depends(get_workspace_service),
):
    tenant_id = current_tenant["tenant_id"]
    try:
        count = service.delete_all_customer_memories(
            authenticated_tenant_id=tenant_id,
            workspace_id=workspace_id,
            customer_id=customer_id,
        )
        return CustomerMemoryDeleteResponse(
            workspace_id=workspace_id,
            customer_id=customer_id,
            deleted_count=count,
            message=f"Purged {count} memory records for customer '{customer_id}' under right-to-be-forgotten.",
        )
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


