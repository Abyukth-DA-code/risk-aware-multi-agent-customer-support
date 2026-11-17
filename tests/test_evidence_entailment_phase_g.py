"""
Comprehensive Test Suite for Phase G: Evidence Entailment, RAG Safety & Selective Answering
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
Research Direction: Risk-Aware Adaptive Multi-Agent Orchestration

Covers 15 required verification scenarios:
1. Fully supported answer with citation
2. Payment-method query vs invoice policy E009 (INSUFFICIENT_EVIDENCE -> ESCALATE)
3. Subscription cancellation vs order policy E028 (INSUFFICIENT_EVIDENCE -> ESCALATE)
4. Partially supported answer routing to clarification / abstention
5. Contradictory evidence detection (CONTRADICTED -> ESCALATE via EVID_CONF_03)
6. Empty or whitespace-only retrieval (INSUFFICIENT_EVIDENCE -> score 0.0 -> ESCALATE)
7. High similarity (>0.50) but insufficient entailment gate
8. Low similarity (<0.45) boundary
9. Verification error/timeout fails closed (VERIFICATION_ERROR -> ESCALATE)
10. Mandatory CRIT_01 overrides supported answer
11. Mandatory SENS_FIN_06 overrides supported financial query
12. Escalation dossier and customer notice integrity
13. Workspace evidence isolation across tenants
14. Public demo query backward compatibility
15. Legitimate answerable queries remain answerable with citations
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.agents.grounded_generator import GroundedAnswerGenerator
from src.agents.policy_agent import PolicyAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.verification_agent import EvidenceEntailmentVerifier, VerificationAgent
from src.orchestration.graph import OrchestrationPipeline
from src.orchestration.safety_policy import SafetyPolicyEngine
from src.orchestration.state import OrchestrationState
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.rag.retriever import RAGRetriever
from src.workspaces.db import WorkspaceDB
from src.workspaces.service import WorkspaceService


@pytest.fixture
def test_env(tmp_path):
    """Provides isolated test database, Chroma store, and multi-agent pipeline."""
    import chromadb

    db_file = tmp_path / "test_tenants_phase_g.db"
    db = WorkspaceDB(db_path=db_file)
    chroma_dir = tmp_path / "chroma_phase_g"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    service = WorkspaceService(db=db)
    ingestion = DocumentIngestionPipeline(db=db, chroma_client=chroma_client)
    retriever = RAGRetriever(client=chroma_client)
    pipeline = OrchestrationPipeline(retriever=retriever, db=db)

    return {
        "db": db,
        "chroma_client": chroma_client,
        "service": service,
        "ingestion": ingestion,
        "retriever": retriever,
        "pipeline": pipeline,
    }


@pytest.fixture
def test_client(test_env, monkeypatch):
    """Wires FastAPI application to isolated test components."""
    from src.api.main import app
    from src.api import workspace_routes
    from src.orchestration import graph

    monkeypatch.setattr(workspace_routes, "_workspace_service", test_env["service"])
    monkeypatch.setattr(workspace_routes, "_ingestion_pipeline", test_env["ingestion"])
    monkeypatch.setattr(graph, "_pipeline_instance", test_env["pipeline"])

    client = TestClient(app)
    return client


# ===========================================================================
# Scenario 1: Fully Supported Answer with Verifiable Citation
# ===========================================================================

def test_scenario_1_fully_supported_answer_with_citation(test_env):
    """Grounded query with authentic supporting evidence receives SUPPORTED status and ANSWER tier."""
    pipeline = test_env["pipeline"]
    policy_agent = pipeline.policy_agent
    supervisor_agent = pipeline.supervisor_agent

    state = OrchestrationState(
        query="What is your product return window?",
        intent="return_request",
        risk="Low",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        provisional_confidence=0.88,
    )
    state.retrieved_chunks = [
        {
            "chunk_id": "chk_ret_01",
            "document_id": "E002",
            "title": "Return and Exchange Policy",
            "chunk_text": "Customers may return unused items within 30 days of purchase for a full refund.",
            "similarity": 0.78,
            "authority": "Official",
            "freshness_status": "Current",
        }
    ]
    state.retrieval_similarity_score = 0.78

    # Step 1: Policy Agent
    state = policy_agent.evaluate_evidence(state)
    assert state.entailment_status == "SUPPORTED"
    assert state.entailment_score >= 0.45
    assert state.evidence_quality == "Strong"

    # Step 2: Safety Policy Evaluation
    state = policy_agent.safety_engine.evaluate(state)
    assert state.final_autonomy_tier == "ANSWER"
    assert not state.final_escalation_required

    # Step 3: Supervisor Agent
    state = supervisor_agent.execute(state, generate_grounded_response=True)
    assert state.final_response is not None
    assert "[E002]" in state.final_response
    assert "Return and Exchange Policy" in state.final_response
    assert state.status == "RESOLVED"


# ===========================================================================
# Scenario 2: Payment-Method Gap Query vs Invoice Policy E009
# ===========================================================================

def test_scenario_2_payment_method_gap_rejected(test_env):
    """
    Payment methods questions retrieving invoice policy E009 must be rejected
    by the evidence entailment gate (INSUFFICIENT_EVIDENCE -> ESCALATE).
    """
    pipeline = test_env["pipeline"]
    policy_agent = pipeline.policy_agent
    supervisor_agent = pipeline.supervisor_agent

    query = "What payment methods are accepted for online orders?"
    state = OrchestrationState(
        query=query,
        intent="check_payment_methods",
        risk="Low",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        provisional_confidence=0.82,
    )
    # E009 describes invoices and receipts, NOT accepted payment methods
    state.retrieved_chunks = [
        {
            "chunk_id": "chk_inv_01",
            "document_id": "E009",
            "title": "Invoice and Receipt Policy",
            "chunk_text": (
                "Official invoices and digital receipts are issued automatically upon order confirmation. "
                "Customers can download VAT receipts from their account order history within 12 months."
            ),
            "similarity": 0.51,  # High lexical similarity due to commerce/order context
            "authority": "Official",
            "freshness_status": "Current",
        }
    ]
    state.retrieval_similarity_score = 0.51

    state = policy_agent.evaluate_evidence(state)
    assert state.entailment_status == "INSUFFICIENT_EVIDENCE"
    assert state.evidence_quality == "Insufficient"
    assert any("payment" in f.lower() for f in state.verification_findings)

    state = policy_agent.safety_engine.evaluate(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True

    state = supervisor_agent.execute(state, generate_grounded_response=True)
    # Grounded answer must be refused
    assert state.final_response is None
    assert state.status == "ESCALATED"
    assert state.escalation_dossier is not None
    assert state.escalation_dossier.get("escalation_id") is not None


# ===========================================================================
# Scenario 3: Subscription Cancellation Gap Query vs Order Policy E028
# ===========================================================================

def test_scenario_3_subscription_cancellation_gap_rejected(test_env):
    """
    Subscription cancellation query matched to physical order cancellation policy E028
    must be blocked by predicate demand verification.
    """
    pipeline = test_env["pipeline"]
    policy_agent = pipeline.policy_agent
    supervisor_agent = pipeline.supervisor_agent

    query = "Can I cancel the business subscription plan?"
    state = OrchestrationState(
        query=query,
        intent="cancel_order",
        risk="Low",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        provisional_confidence=0.79,
    )
    # E028 covers physical order shipments, not software subscriptions
    state.retrieved_chunks = [
        {
            "chunk_id": "chk_ord_01",
            "document_id": "E028",
            "title": "Order Cancellation Policy",
            "chunk_text": (
                "Physical orders may be cancelled within 60 minutes of placing the order before warehouse dispatch. "
                "Once the parcel has shipped, cancellation is unavailable and the standard return process applies."
            ),
            "similarity": 0.59,  # High similarity due to cancellation verb
            "authority": "Official",
            "freshness_status": "Current",
        }
    ]
    state.retrieval_similarity_score = 0.59

    state = policy_agent.evaluate_evidence(state)
    assert state.entailment_status == "INSUFFICIENT_EVIDENCE"
    assert state.evidence_quality == "Insufficient"
    assert any("subscription" in f.lower() for f in state.verification_findings)

    state = policy_agent.safety_engine.evaluate(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True

    state = supervisor_agent.execute(state, generate_grounded_response=True)
    assert state.final_response is None
    assert state.status == "ESCALATED"


# ===========================================================================
# Scenario 4: Partially Supported Answer Routing (Clarification vs Abstention)
# ===========================================================================

def test_scenario_4_partially_supported_routing_clarify_and_abstain(test_env):
    """
    Partially supported evidence:
    - In low-risk inquiries: routes safely to CLARIFY prompt.
    - In high-risk inquiries: routes to ESCALATE.
    """
    pipeline = test_env["pipeline"]
    policy_agent = pipeline.policy_agent
    supervisor_agent = pipeline.supervisor_agent

    # Case A: Low-risk inquiry
    state_low = OrchestrationState(
        query="I want to cancel something right now",
        intent="cancel_order",
        risk="Low",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        provisional_confidence=0.78,
    )
    state_low.retrieved_chunks = [
        {
            "chunk_id": "chk_gen_01",
            "document_id": "E028",
            "title": "Order Cancellation Policy",
            "chunk_text": "Orders can be cancelled before warehouse processing begins.",
            "similarity": 0.42,  # Marginal similarity triggering PARTIALLY_SUPPORTED
            "authority": "Official",
            "freshness_status": "Current",
        }
    ]
    state_low.retrieval_similarity_score = 0.42

    # Manually configure verifier to return PARTIALLY_SUPPORTED
    class MockPartialVerifier(EvidenceEntailmentVerifier):
        def verify(self, *args, **kwargs):
            return {
                "status": "PARTIALLY_SUPPORTED",
                "score": 0.40,
                "findings": ["Partial evidence match across claims."],
                "supported_claims": ["Orders can be cancelled"],
                "unsupported_claims": ["which item to cancel"],
                "evidence_references": ["E028"],
                "latency_ms": 1.2,
            }

    custom_policy = PolicyAgent(safety_engine=pipeline.safety_engine, verifier=MockPartialVerifier())
    state_low = custom_policy.evaluate_evidence(state_low)
    assert state_low.entailment_status == "PARTIALLY_SUPPORTED"
    assert state_low.metadata.get("is_ambiguous") is True

    state_low = custom_policy.safety_engine.evaluate(state_low)
    state_low = supervisor_agent.execute(state_low, generate_grounded_response=True)
    assert state_low.final_autonomy_tier == "CLARIFY"
    assert "clarify" in state_low.final_response.lower()

    # Case B: High-risk inquiry with partial evidence
    state_high = OrchestrationState(
        query="Dispute unauthorized charges and reverse payment",
        intent="dispute_charge",
        risk="High",
        complexity="Medium",
        provisional_autonomy_tier="ESCALATE",
        provisional_confidence=0.60,
    )
    state_high.retrieved_chunks = state_low.retrieved_chunks
    state_high.retrieval_similarity_score = 0.42

    state_high = custom_policy.evaluate_evidence(state_high)
    assert state_high.entailment_status == "PARTIALLY_SUPPORTED"
    assert state_high.evidence_quality == "Insufficient"

    state_high = custom_policy.safety_engine.evaluate(state_high)
    assert state_high.final_autonomy_tier == "ESCALATE"
    assert state_high.final_escalation_required is True


# ===========================================================================
# Scenario 5: Contradictory Evidence Detection
# ===========================================================================

def test_scenario_5_contradictory_evidence_conflict_escalation(test_env):
    """Conflicting chunks or explicit conflict flags trigger CONTRADICTED and EVID_CONF_03."""
    pipeline = test_env["pipeline"]
    policy_agent = pipeline.policy_agent

    state = OrchestrationState(
        query="What is the return window for electronics?",
        intent="return_request",
        risk="Medium",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        provisional_confidence=0.85,
    )
    state.retrieved_chunks = [
        {
            "chunk_id": "c1",
            "document_id": "DOC_A",
            "title": "Standard Return Policy",
            "category_key": "returns",
            "chunk_text": "Returns are strictly accepted within 14 days of delivery.",
            "similarity": 0.65,
            "freshness_status": "Current",
        },
        {
            "chunk_id": "c2",
            "document_id": "DOC_B",
            "title": "Extended Holiday Policy",
            "category_key": "returns",
            "chunk_text": "Returns are accepted within 60 days of delivery during promotional seasons.",
            "similarity": 0.64,
            "freshness_status": "Current",
        },
    ]
    state.retrieval_similarity_score = 0.65

    state = policy_agent.evaluate_evidence(state)
    assert state.entailment_status == "CONTRADICTED"
    assert state.evidence_quality == "Conflicting"
    assert state.has_conflicting_evidence is True

    state = policy_agent.safety_engine.evaluate(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True
    assert state.safety_rule_triggered == "EVID_CONF_03"


# ===========================================================================
# Scenario 6: Empty or Whitespace-Only Retrieval
# ===========================================================================

def test_scenario_6_empty_and_whitespace_retrieval(test_env):
    """Empty chunk lists and whitespace-only chunks fail closed with score 0.0 and INSUFFICIENT_EVIDENCE."""
    verifier = EvidenceEntailmentVerifier()

    # Empty list
    res_empty = verifier.verify("How do I return?", [])
    assert res_empty["status"] == "INSUFFICIENT_EVIDENCE"
    assert res_empty["score"] == 0.0

    # Whitespace-only chunk
    res_ws = verifier.verify("How do I return?", [{"chunk_id": "c1", "chunk_text": "   \n\t  "}])
    assert res_ws["status"] == "INSUFFICIENT_EVIDENCE"
    assert res_ws["score"] == 0.0


# ===========================================================================
# Scenario 7: High Similarity But Insufficient Entailment
# ===========================================================================

def test_scenario_7_high_similarity_insufficient_entailment(test_env):
    """
    Passage with high keyword overlap (similarity > 0.50) but missing the factual
    predicate must not qualify for autonomous answer.
    """
    verifier = EvidenceEntailmentVerifier()

    query = "What is the placement schedule and campus drive timeline for engineering?"
    # Passage discusses course registration during engineering semesters, high lexical overlap
    retrieved = [
        {
            "chunk_id": "c_acad_01",
            "document_id": "E101",
            "title": "Engineering Academic Calendar",
            "chunk_text": (
                "Engineering undergraduate semester registrations take place in August and January. "
                "Students must register for all laboratory and elective courses prior to the academic deadline."
            ),
            "similarity": 0.54,
        }
    ]

    res = verifier.verify(query, retrieved)
    assert res["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("placement" in f.lower() for f in res["findings"])


# ===========================================================================
# Scenario 8: Low Similarity Boundary (<0.45)
# ===========================================================================

def test_scenario_8_low_similarity_boundary(test_env):
    """Retrieval similarity below threshold (<0.45) results in INSUFFICIENT_EVIDENCE."""
    pipeline = test_env["pipeline"]
    policy_agent = pipeline.policy_agent

    state = OrchestrationState(
        query="Can I use international currencies?",
        intent="currency_inquiry",
        risk="Low",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        provisional_confidence=0.75,
    )
    state.retrieved_chunks = [
        {
            "chunk_id": "chk_c01",
            "document_id": "E009",
            "title": "Invoice Policy",
            "chunk_text": "Invoices are generated upon checkout in the local currency.",
            "similarity": 0.38,  # Below default 0.45 threshold
            "freshness_status": "Current",
        }
    ]
    state.retrieval_similarity_score = 0.38

    state = policy_agent.evaluate_evidence(state)
    assert state.evidence_quality == "Insufficient"

    state = policy_agent.safety_engine.evaluate(state)
    assert state.final_autonomy_tier == "ESCALATE"


# ===========================================================================
# Scenario 9: Verification Error Fails Closed
# ===========================================================================

def test_scenario_9_verification_error_fails_closed(test_env):
    """Any exception during verification gracefully fails closed to VERIFICATION_ERROR and ESCALATE."""
    pipeline = test_env["pipeline"]

    class BrokenVerifier(EvidenceEntailmentVerifier):
        def verify(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory / simulated crash")

    policy_agent = PolicyAgent(safety_engine=pipeline.safety_engine, verifier=BrokenVerifier())

    state = OrchestrationState(
        query="What is your return policy?",
        intent="return_request",
        risk="Low",
        complexity="Low",
    )
    state.retrieved_chunks = [
        {"chunk_id": "c1", "chunk_text": "Valid return text", "similarity": 0.85}
    ]
    state.retrieval_similarity_score = 0.85

    # PolicyAgent catches verifier error and fails closed
    state = policy_agent.evaluate_evidence(state)
    assert state.entailment_status in ("VERIFICATION_ERROR", "INSUFFICIENT_EVIDENCE")
    assert state.evidence_quality == "Insufficient"

    state = policy_agent.safety_engine.evaluate(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True


# ===========================================================================
# Scenario 10: Mandatory CRIT_01 Overrides Supported Evidence
# ===========================================================================

def test_scenario_10_mandatory_crit_01_overrides_supported(test_env):
    """Even with 100% verified evidence entailment, risk == 'Critical' forces CRIT_01 escalation."""
    pipeline = test_env["pipeline"]
    policy_agent = pipeline.policy_agent

    state = OrchestrationState(
        query="Threatening legal action and reporting server breach",
        intent="security_incident",
        risk="Critical",
        complexity="High",
        provisional_autonomy_tier="ANSWER",
        provisional_confidence=0.95,
    )
    state.retrieved_chunks = [
        {
            "chunk_id": "c_sec_01",
            "document_id": "SEC01",
            "title": "Incident Escalation Protocol",
            "chunk_text": "All reported security breaches are handled by the incident response team.",
            "similarity": 0.90,
            "freshness_status": "Current",
        }
    ]
    state.retrieval_similarity_score = 0.90

    state = policy_agent.execute(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True
    assert state.safety_rule_triggered == "CRIT_01"
    assert state.safety_override is True


# ===========================================================================
# Scenario 11: Mandatory SENS_FIN_06 Overrides Supported Financial Query
# ===========================================================================

def test_scenario_11_mandatory_sens_fin_06_overrides_supported(test_env):
    """Sensitive action + Financial impact + High risk mandates SENS_FIN_06 human escalation."""
    pipeline = test_env["pipeline"]
    policy_agent = pipeline.policy_agent

    state = OrchestrationState(
        query="Execute wire refund of $50,000 for disputed inventory",
        intent="refund_request",
        risk="High",
        complexity="High",
        sensitive_action=True,
        financial_impact=True,
        provisional_autonomy_tier="ANSWER",
        provisional_confidence=0.92,
    )
    state.retrieved_chunks = [
        {
            "chunk_id": "c_fin_01",
            "document_id": "FIN01",
            "title": "High-Value Refund Procedures",
            "chunk_text": "Refunds above $10,000 must undergo executive verification and approval.",
            "similarity": 0.88,
            "freshness_status": "Current",
        }
    ]
    state.retrieval_similarity_score = 0.88

    state = policy_agent.execute(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True
    assert state.safety_rule_triggered in ("SENS_FIN_06", "RISK_VERIF_02")


# ===========================================================================
# Scenario 12: Escalation Dossier and Customer Notice Integrity
# ===========================================================================

def test_scenario_12_escalation_dossier_and_notice_integrity(test_client):
    """Escalation generated produces valid escalation_id, customer notice, and dossier in SQLite."""
    # Register tenant and workspace
    t = test_client.post(
        "/api/v1/tenants/register", json={"name": "Dossier Corp", "email": "dossier@corp.com"}
    ).json()
    w = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Dossier Desk", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    ).json()
    ws_id = w["workspace_id"]
    headers = {"Authorization": f"Bearer {t['api_key']}"}

    # Query without knowledge triggers LOW_CONF_05 / EVID_INSUF
    res = test_client.post(
        f"/api/v1/workspaces/{ws_id}/query",
        json={"query": "Wire transfer $100,000 to offshore account"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["resolution"]["final_autonomy_tier"] == "ESCALATE"
    dossier = data["resolution"]["escalation_dossier"]
    assert dossier is not None
    esc_id = dossier.get("escalation_id")
    assert esc_id is not None
    assert esc_id.startswith("esc_")
    assert "routed to a human" in dossier.get("escalation_notice", "")

    # Verify dossier persisted in SQLite queue
    list_res = test_client.get(f"/api/v1/workspaces/{ws_id}/escalations", headers=headers)
    assert list_res.status_code == 200
    dossiers = list_res.json()
    assert isinstance(dossiers, list)
    assert any(d["escalation_id"] == esc_id for d in dossiers)


# ===========================================================================
# Scenario 13: Workspace Evidence Isolation Across Tenants
# ===========================================================================

def test_scenario_13_workspace_evidence_isolation(test_client):
    """Tenant A's ingested policy cannot be retrieved or verified by Tenant B."""
    # Tenant A
    t_a = test_client.post(
        "/api/v1/tenants/register", json={"name": "Corp A", "email": "a@corp.com"}
    ).json()
    w_a = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Desk A", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t_a['api_key']}"},
    ).json()
    doc_a = "Secret Product Code 9988: Internal launch scheduled for December 2026.".encode("utf-8")
    up_res = test_client.post(
        f"/api/v1/workspaces/{w_a['workspace_id']}/documents/upload",
        data={"category_key": "confidential_launch"},
        files={"file": ("launch.txt", doc_a, "text/plain")},
        headers={"Authorization": f"Bearer {t_a['api_key']}"},
    )
    assert up_res.status_code in (200, 201)

    # Tenant B
    t_b = test_client.post(
        "/api/v1/tenants/register", json={"name": "Corp B", "email": "b@corp.com"}
    ).json()
    w_b = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Desk B", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t_b['api_key']}"},
    ).json()

    # Query from Tenant B must NOT retrieve or verify against Tenant A's document
    res_b = test_client.post(
        f"/api/v1/workspaces/{w_b['workspace_id']}/query",
        json={"query": "What is Secret Product Code 9988 launch date?"},
        headers={"Authorization": f"Bearer {t_b['api_key']}"},
    )
    assert res_b.status_code == 200
    data_b = res_b.json()
    # Tenant B has empty workspace, so retrieval yields 0 chunks and escalates
    assert len(data_b["evidence"]["chunks"]) == 0
    assert data_b["resolution"]["final_autonomy_tier"] == "ESCALATE"


# ===========================================================================
# Scenario 14: Public Demo Query Backward Compatibility
# ===========================================================================

def test_scenario_14_public_demo_query_backward_compatibility(test_client):
    """Demo query interface (/query) returns valid triage, evidence, and resolution."""
    res = test_client.post("/query", json={"query": "How do I return a damaged product?"})
    assert res.status_code == 200
    data = res.json()
    assert "triage" in data
    assert "evidence" in data
    assert "resolution" in data
    assert "final_autonomy_tier" in data["resolution"]


# ===========================================================================
# Scenario 15: Legitimate Answerable Queries Remain Answerable with Citations
# ===========================================================================

def test_scenario_15_legitimate_answerable_queries_remain_answerable(test_client):
    """Authentic policy questions uploaded to a workspace are answered autonomously with citations."""
    t = test_client.post(
        "/api/v1/tenants/register", json={"name": "Answerable Corp", "email": "ans@corp.com"}
    ).json()
    w = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Answer Desk", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    ).json()
    ws_id = w["workspace_id"]
    headers = {"Authorization": f"Bearer {t['api_key']}"}

    policy_text = (
        "Warranty Registration Guidelines:\n"
        "Customers must register their hardware devices within 14 days of purchase at warranty.example.com. "
        "The standard limited warranty covers manufacturing defects for a duration of 24 months from the invoice date."
    ).encode("utf-8")

    upload_res = test_client.post(
        f"/api/v1/workspaces/{ws_id}/documents/upload",
        data={"category_key": "warranty_registration_policy"},
        files={"file": ("warranty.txt", policy_text, "text/plain")},
        headers=headers,
    )
    assert upload_res.status_code in (200, 201)

    query_res = test_client.post(
        f"/api/v1/workspaces/{ws_id}/query",
        json={"query": "How long is the limited hardware warranty and when must I register?"},
        headers=headers,
    )
    assert query_res.status_code == 200
    data = query_res.json()

    assert data["resolution"]["final_autonomy_tier"] == "ANSWER"
    assert data["evidence"]["evidence_quality"] == "Strong"
    assert data["grounded_response"] is not None
    assert "24 months" in data["grounded_response"] or "14 days" in data["grounded_response"]
    assert "Source Citation:" in data["grounded_response"]
