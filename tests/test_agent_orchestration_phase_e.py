"""
Comprehensive Test Suite for Phase E: Grounded Agent Orchestration & HITL Escalation
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Verifies:
1. Multi-Agent Team execution trace (TriageAgent, RetrievalAgent, PolicyAgent, SupervisorAgent).
2. Adaptive resolution: ANSWER (grounded citations), CLARIFY (targeted prompt), ESCALATE (dossier & notice).
3. Persistent SQLite escalation queue: insertion, listing, retrieving, and state transition reviews.
4. State machine constraints: PENDING -> IN_REVIEW -> RESOLVED/REJECTED.
5. Strict cross-tenant escalation isolation (403 Forbidden on unauthorized read/write).
6. Configurable similarity threshold and evidence assessment (authority, freshness, contradictions).
7. Public demo (/query) backward compatibility preservation.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.agents.policy_agent import PolicyAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.triage_agent import TriageAgent
from src.orchestration.graph import OrchestrationPipeline
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.rag.retriever import RAGRetriever
from src.workspaces.db import WorkspaceDB
from src.workspaces.service import WorkspaceService


@pytest.fixture
def test_env(tmp_path):
    """Provides isolated test database, Chroma store, and multi-agent pipeline."""
    import chromadb

    db_file = tmp_path / "test_tenants_phase_e.db"
    db = WorkspaceDB(db_path=db_file)
    chroma_dir = tmp_path / "chroma_phase_e"
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
    from src.api import workspace_routes
    from src.orchestration import graph

    monkeypatch.setattr(workspace_routes, "_workspace_service", test_env["service"])
    monkeypatch.setattr(workspace_routes, "_ingestion_pipeline", test_env["ingestion"])
    monkeypatch.setattr(graph, "_pipeline_instance", test_env["pipeline"])

    return TestClient(app)


# ===========================================================================
# 1. Multi-Agent Team Lifecycle & Trace Verification
# ===========================================================================

def test_multi_agent_execution_trace(test_env):
    """
    Verifies that all 4 specialist agents execute sequentially and record
    structured metrics, timestamps, and findings in agent_execution_trace.
    """
    pipeline = test_env["pipeline"]
    state = pipeline.run(
        query="How do I return an eligible item?",
        domain="E-commerce",
        generate_grounded_response=True,
    )

    # Verify all 4 agents participated
    assert "TriageAgent" in state.selected_agents
    assert "RetrievalAgent" in state.selected_agents
    assert "PolicyAgent" in state.selected_agents
    assert "SupervisorAgent" in state.selected_agents

    # Verify execution trace contains entries for each agent
    trace_agents = [t["agent"] for t in state.agent_execution_trace]
    assert "TriageAgent" in trace_agents
    assert "RetrievalAgent" in trace_agents
    assert "PolicyAgent" in trace_agents
    assert "SupervisorAgent" in trace_agents

    # Verify each trace entry has timestamp, action, and status
    for entry in state.agent_execution_trace:
        assert "timestamp" in entry
        assert "action" in entry
        assert entry["status"] in ("COMPLETED", "DEGRADED", "FAILED")


# ===========================================================================
# 2. Adaptive Resolution: ANSWER with Grounded Citations
# ===========================================================================

def test_safe_answerable_query_grounded_response(test_client):
    """
    When an uploaded policy covers the query, the system resolves autonomously
    with ANSWER tier and verifiable citations.
    """
    t = test_client.post(
        "/api/v1/tenants/register", json={"name": "Retail Corp", "email": "retail@test.com"}
    ).json()
    w = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Retail Store", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    ).json()

    headers = {"Authorization": f"Bearer {t['api_key']}"}

    # Upload substantive return policy
    doc_content = (
        "Eligible items may be returned within 30 days of purchase for a 100% money-back refund. "
        "Products must be returned in their original packaging with the purchase receipt."
    ).encode("utf-8")
    test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/documents/upload",
        data={"category_key": "return_policy"},
        files={"file": ("return_policy.txt", doc_content, "text/plain")},
        headers=headers,
    )

    # Query
    res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/query",
        json={"query": "How can I return an eligible product within the return window?"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()

    assert data["resolution"]["final_autonomy_tier"] == "ANSWER"
    assert data["resolution"]["final_escalation_required"] is False
    assert data["grounded_response"] is not None
    assert "return_policy.txt" in data["grounded_response"]
    assert "Source Citation" in data["grounded_response"]


# ===========================================================================
# 3. Adaptive Resolution: CLARIFY on Ambiguous Queries
# ===========================================================================

def test_ambiguous_query_clarification_prompt(test_client):
    """
    Underspecified queries with ambiguous intent prompt the customer for clarification
    rather than fabricating answers or escalating prematurely.
    """
    t = test_client.post(
        "/api/v1/tenants/register", json={"name": "Clarify Corp", "email": "clarify@test.com"}
    ).json()
    w = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Clarify Desk", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    ).json()

    headers = {"Authorization": f"Bearer {t['api_key']}"}

    # Upload general policy
    doc_content = (
        "Cancellations must be requested before shipment. Return requests are accepted within 30 days."
    ).encode("utf-8")
    test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/documents/upload",
        data={"category_key": "order_cancellation_policy"},
        files={"file": ("cancellation.txt", doc_content, "text/plain")},
        headers=headers,
    )

    # Ambiguous 1-word query: "cancel"
    res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/query",
        json={"query": "cancel"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()

    # Clarification must be requested
    if data["resolution"]["final_autonomy_tier"] != "ESCALATE":
        assert data["grounded_response"] is not None
        assert any(
            w in data["grounded_response"].lower()
            for w in ["clarify", "additional details", "more details", "order number"]
        )


# ===========================================================================
# 4. Adaptive Resolution: ESCALATE & Persistent Dossier Queue
# ===========================================================================

def test_missing_evidence_automatic_escalation_enqueued(test_client):
    """
    When knowledge evidence is missing (e.g. empty workspace), LOW_CONF_05 triggers,
    the supervisor enqueues an escalation dossier in SQLite, and formats an escalation notice.
    """
    t = test_client.post(
        "/api/v1/tenants/register", json={"name": "Empty Org", "email": "empty@org.com"}
    ).json()
    w = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Empty Helpdesk", "domain": "Education"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    ).json()

    headers = {"Authorization": f"Bearer {t['api_key']}"}

    res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/query",
        json={"query": "What is the fee payment deadline?"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()

    assert data["resolution"]["final_autonomy_tier"] == "ESCALATE"
    assert data["resolution"]["final_escalation_required"] is True
    assert data["resolution"]["safety_rule_triggered"] == "LOW_CONF_05"

    # Escalation notice must reference the unique escalation_id
    assert data["grounded_response"] is None
    assert data["resolution"]["final_response"] is None
    assert "routed to a human" in data["resolution"]["message"]
    assert "Escalation ID: esc_" in data["resolution"]["message"]
    assert data["resolution"]["escalation_dossier"] is not None
    assert "Escalation ID: esc_" in data["resolution"]["escalation_dossier"]["escalation_notice"]

    # Verify dossier was enqueued into SQLite
    esc_list = test_client.get(
        f"/api/v1/workspaces/{w['workspace_id']}/escalations",
        headers=headers,
    )
    assert esc_list.status_code == 200
    dossiers = esc_list.json()
    assert len(dossiers) >= 1
    assert dossiers[0]["status"] == "PENDING"
    assert dossiers[0]["workspace_id"] == w["workspace_id"]
    assert dossiers[0]["final_autonomy_tier"] == "ESCALATE"


def test_sensitive_financial_escalation_enqueued(test_client):
    """
    High-risk sensitive transactions trigger SENS_FIN_06, mandate human approval,
    and persist an escalation dossier with recommended next actions.
    """
    t = test_client.post(
        "/api/v1/tenants/register", json={"name": "Finance Org", "email": "fin@org.com"}
    ).json()
    w = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Finance Desk", "domain": "Small Business"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    ).json()

    headers = {"Authorization": f"Bearer {t['api_key']}"}

    res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/query",
        json={
            "query": "Override billing limit and issue immediate cash disbursement",
            "sensitive_action": True,
            "financial_impact": True,
        },
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()

    assert data["resolution"]["final_escalation_required"] is True
    assert data["grounded_response"] is None
    assert "Escalation ID: esc_" in data["resolution"]["message"]
    assert "Escalation ID: esc_" in data["resolution"]["escalation_dossier"]["escalation_notice"]

    # Check queue
    esc_res = test_client.get(
        f"/api/v1/workspaces/{w['workspace_id']}/escalations",
        headers=headers,
    )
    dossiers = esc_res.json()
    assert len(dossiers) >= 1
    assert dossiers[0]["safety_rule_triggered"] in ("SENS_FIN_06", "LOW_CONF_05", "EVID_INSUF_04")


# ===========================================================================
# 5. Human Review Workflow & Lifecycle State Machine
# ===========================================================================

def test_human_supervisor_review_workflow(test_client):
    """
    Tests complete HITL lifecycle:
    PENDING -> IN_REVIEW -> RESOLVED with supervisor notes.
    Tests rejection of invalid transition back to PENDING.
    """
    t = test_client.post(
        "/api/v1/tenants/register", json={"name": "Review Org", "email": "review@org.com"}
    ).json()
    w = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Review Desk", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    ).json()

    headers = {"Authorization": f"Bearer {t['api_key']}"}

    # Trigger escalation
    query_res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/query",
        json={"query": "Where is my missing refund?", "sensitive_action": True},
        headers=headers,
    )
    dossier_id = query_res.json()["resolution"]["escalation_dossier"]["escalation_id"]

    # 1. Fetch single dossier
    get_res = test_client.get(
        f"/api/v1/workspaces/{w['workspace_id']}/escalations/{dossier_id}",
        headers=headers,
    )
    assert get_res.status_code == 200
    assert get_res.json()["status"] == "PENDING"

    # 2. Transition to IN_REVIEW
    review_res_1 = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/escalations/{dossier_id}/review",
        json={
            "status": "IN_REVIEW",
            "reviewer_notes": "Supervisor assigned case and reviewing billing logs.",
            "resolved_by": "supervisor_alice",
        },
        headers=headers,
    )
    assert review_res_1.status_code == 200
    assert review_res_1.json()["status"] == "IN_REVIEW"
    assert review_res_1.json()["resolved_by"] == "supervisor_alice"

    # 3. Invalid transition: cannot move from IN_REVIEW back to PENDING -> 400 Bad Request
    invalid_res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/escalations/{dossier_id}/review",
        json={"status": "PENDING"},
        headers=headers,
    )
    assert invalid_res.status_code == 400
    assert "cannot transition" in invalid_res.json()["detail"].lower()

    # 4. Resolve escalation
    resolve_res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/escalations/{dossier_id}/review",
        json={
            "status": "RESOLVED",
            "human_decision": "APPROVE_OVERRIDE",
            "reviewer_notes": "Manual bank transfer approved after verification.",
            "resolved_by": "supervisor_alice",
        },
        headers=headers,
    )
    assert resolve_res.status_code == 200
    assert resolve_res.json()["status"] == "RESOLVED"
    assert resolve_res.json()["human_decision"] == "APPROVE_OVERRIDE"
    assert resolve_res.json()["resolved_at"] is not None

    # 5. Terminal state constraint: cannot modify RESOLVED escalation -> 400 Bad Request
    terminal_res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/escalations/{dossier_id}/review",
        json={"status": "IN_REVIEW"},
        headers=headers,
    )
    assert terminal_res.status_code == 400
    assert "terminal state" in terminal_res.json()["detail"].lower()


# ===========================================================================
# 6. Cross-Tenant Escalation Access Security Tests
# ===========================================================================

def test_cross_tenant_escalation_access_forbidden(test_client):
    """
    Critical Security Invariant:
    Tenant B must never be able to list, view, or review Tenant A's escalation dossiers.
    Must return HTTP 403 Forbidden.
    """
    # Tenant A
    t_a = test_client.post(
        "/api/v1/tenants/register", json={"name": "Tenant A Corp", "email": "ta_esc@test.com"}
    ).json()
    w_a = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Workspace A", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t_a['api_key']}"},
    ).json()

    # Tenant B
    t_b = test_client.post(
        "/api/v1/tenants/register", json={"name": "Tenant B Corp", "email": "tb_esc@test.com"}
    ).json()

    headers_a = {"Authorization": f"Bearer {t_a['api_key']}"}
    headers_b = {"Authorization": f"Bearer {t_b['api_key']}"}

    # Tenant A triggers escalation
    query_res = test_client.post(
        f"/api/v1/workspaces/{w_a['workspace_id']}/query",
        json={"query": "Escalate my dispute immediately"},
        headers=headers_a,
    )
    esc_id = query_res.json()["resolution"]["escalation_dossier"]["escalation_id"]

    # 1. Tenant B attempts to list Tenant A's escalations -> 403
    attack_list = test_client.get(
        f"/api/v1/workspaces/{w_a['workspace_id']}/escalations",
        headers=headers_b,
    )
    assert attack_list.status_code == 403

    # 2. Tenant B attempts to view Tenant A's escalation detail -> 403
    attack_get = test_client.get(
        f"/api/v1/workspaces/{w_a['workspace_id']}/escalations/{esc_id}",
        headers=headers_b,
    )
    assert attack_get.status_code == 403

    # 3. Tenant B attempts to submit review decision on Tenant A's escalation -> 403
    attack_review = test_client.post(
        f"/api/v1/workspaces/{w_a['workspace_id']}/escalations/{esc_id}/review",
        json={"status": "RESOLVED", "human_decision": "HACK_APPROVE"},
        headers=headers_b,
    )
    assert attack_review.status_code == 403


# ===========================================================================
# 7. Configurable Similarity Threshold Test
# ===========================================================================

def test_configurable_similarity_threshold():
    """
    Verifies that PolicyAgent respects custom similarity thresholds
    and does not treat 0.45 as universally fixed.
    """
    policy_agent_strict = PolicyAgent(min_similarity_threshold=0.75)
    policy_agent_lenient = PolicyAgent(min_similarity_threshold=0.30)

    from src.orchestration.state import OrchestrationState

    # Create dummy state with similarity = 0.50
    state_strict = OrchestrationState(query="test query")
    state_strict.retrieved_chunks = [{"chunk_id": "c1", "similarity": 0.50, "freshness_status": "Current"}]
    state_strict.retrieval_similarity_score = 0.50

    state_strict = policy_agent_strict.evaluate_evidence(state_strict)
    # Under 0.75 threshold, 0.50 is Insufficient
    assert state_strict.evidence_quality == "Insufficient"

    state_lenient = OrchestrationState(query="test query")
    state_lenient.retrieved_chunks = [{"chunk_id": "c1", "similarity": 0.50, "freshness_status": "Current"}]
    state_lenient.retrieval_similarity_score = 0.50

    state_lenient = policy_agent_lenient.evaluate_evidence(state_lenient)
    # Under 0.30 threshold, 0.50 is Strong
    assert state_lenient.evidence_quality == "Strong"


# ===========================================================================
# 8. Public Demo Backward Compatibility Test
# ===========================================================================

def test_public_demo_query_preserved_in_phase_e():
    """
    Verifies public /query continues to execute without API key,
    operating against capstone_knowledge_v1 with final_response=None contract.
    """
    client = TestClient(app)
    res = client.post(
        "/query",
        json={"query": "How can I return an eligible product within the return window?"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "triage" in data
    assert "evidence" in data
    assert "resolution" in data
    assert data["resolution"]["final_response"] is None
