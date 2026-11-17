"""
Comprehensive Test Suite for Phase D: Tenant-Aware Isolated RAG Retrieval
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Verifies:
1. Public demo query (/query) backward compatibility is 100% preserved.
2. Authenticated tenant query requires Bearer API key.
3. Cross-tenant access attempts are strictly rejected with HTTP 403 Forbidden.
4. Physical Chroma isolation across tenant workspaces (zero cross-tenant leakage).
5. Safe failure: empty or missing workspace collections NEVER silently fall back to capstone_knowledge_v1.
6. Grounded extractive answer generation with source citations for ANSWER tier.
7. Refusal of generation and escalation dossier generation when safety invariants trigger (ESCALATE).
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.orchestration.graph import OrchestrationPipeline, get_orchestration_pipeline
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.rag.retriever import RAGRetriever
from src.workspaces.db import WorkspaceDB
from src.workspaces.models import BusinessDomain
from src.workspaces.service import WorkspaceService


@pytest.fixture
def test_env(tmp_path):
    """
    Creates isolated test database, Chroma client, and pipeline.
    Uses dedicated temporary directory to guarantee zero test interference.
    """
    import chromadb

    db_file = tmp_path / "test_tenants_phase_d.db"
    db = WorkspaceDB(db_path=db_file)
    chroma_dir = tmp_path / "chroma_phase_d"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    service = WorkspaceService(db=db)
    ingestion = DocumentIngestionPipeline(db=db, chroma_client=chroma_client)
    retriever = RAGRetriever(client=chroma_client)
    pipeline = OrchestrationPipeline(retriever=retriever)

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
# 1. Public Demo Query Backward Compatibility Test
# ===========================================================================

def test_public_demo_query_unaffected():
    """
    Requirement 1: Public demo query (/query) must remain working without
    requiring an API key, retaining original contract.
    """
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"query": "How can I return an eligible product within the return window?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "triage" in data
    assert "evidence" in data
    assert "resolution" in data
    # Demo endpoint preserves final_response as None
    assert data["resolution"]["final_response"] is None


# ===========================================================================
# 2. Authentication & Authorization Security Tests
# ===========================================================================

def test_workspace_query_requires_auth(test_client):
    """Unauthenticated query attempts must receive HTTP 401 Unauthorized."""
    res = test_client.post(
        "/api/v1/workspaces/ws_sample123/query",
        json={"query": "What is the return policy?"},
    )
    assert res.status_code == 401
    assert "Authentication required" in res.json()["detail"]


def test_workspace_query_invalid_api_key(test_client):
    """Invalid credentials must receive HTTP 401 Unauthorized."""
    res = test_client.post(
        "/api/v1/workspaces/ws_sample123/query",
        json={"query": "What is the return policy?"},
        headers={"Authorization": "Bearer invalid_secret_token_123"},
    )
    assert res.status_code == 401
    assert "Invalid, expired, or revoked API key" in res.json()["detail"]


def test_workspace_query_nonexistent_workspace(test_client):
    """Authenticated query for non-existent workspace receives HTTP 404."""
    reg = test_client.post(
        "/api/v1/tenants/register",
        json={"name": "Test Org", "email": "org@test.com"},
    ).json()
    api_key = reg["api_key"]

    res = test_client.post(
        "/api/v1/workspaces/ws_nonexistent_9999/query",
        json={"query": "What is the return policy?"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_cross_tenant_workspace_query_forbidden(test_client):
    """
    Critical Security Invariant:
    Tenant B must never be able to query Tenant A's workspace, even with valid Tenant B credentials.
    Must return HTTP 403 Forbidden.
    """
    # 1. Register Tenant A and create Workspace A
    t_a = test_client.post(
        "/api/v1/tenants/register",
        json={"name": "Tenant A Corp", "email": "tenant_a@test.com"},
    ).json()
    w_a = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Storefront A", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t_a['api_key']}"},
    ).json()

    # 2. Register Tenant B and create Workspace B
    t_b = test_client.post(
        "/api/v1/tenants/register",
        json={"name": "Tenant B University", "email": "tenant_b@test.com"},
    ).json()

    # 3. Cross-Tenant Query Attack: Tenant B attempts to query Workspace A
    attack_res = test_client.post(
        f"/api/v1/workspaces/{w_a['workspace_id']}/query",
        json={"query": "Tell me your return policy"},
        headers={"Authorization": f"Bearer {t_b['api_key']}"},
    )
    assert attack_res.status_code == 403
    assert "Unauthorized" in attack_res.json()["detail"]


# ===========================================================================
# 3. Multi-Tenant Document & Retrieval Isolation Tests
# ===========================================================================

def test_multi_tenant_rag_isolation(test_client):
    """
    Verifies strict physical vector retrieval isolation:
    - Tenant A (E-commerce) uploads a return policy.
    - Tenant B (Education) uploads an attendance policy.
    - Querying Workspace A retrieves strictly Tenant A's return document.
    - Querying Workspace B retrieves strictly Tenant B's attendance document.
    - Zero cross-tenant chunk leakage between collections.
    """
    # Setup Tenant A (E-commerce)
    t_a = test_client.post(
        "/api/v1/tenants/register",
        json={"name": "E-Shop Global", "email": "contact@eshop.com"},
    ).json()
    w_a = test_client.post(
        "/api/v1/workspaces",
        json={"name": "E-Shop Portal", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t_a['api_key']}"},
    ).json()

    # Setup Tenant B (Education)
    t_b = test_client.post(
        "/api/v1/tenants/register",
        json={"name": "Apex University", "email": "admin@apex.edu"},
    ).json()
    w_b = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Student Affairs", "domain": "Education"},
        headers={"Authorization": f"Bearer {t_b['api_key']}"},
    ).json()

    headers_a = {"Authorization": f"Bearer {t_a['api_key']}"}
    headers_b = {"Authorization": f"Bearer {t_b['api_key']}"}

    # Upload Return Policy to Workspace A
    policy_a = (
        "Eligible items may be returned within 30 days of purchase with original receipt for a full refund. "
        "Products must be in their original undamaged condition and original packaging."
    ).encode("utf-8")
    up_a = test_client.post(
        f"/api/v1/workspaces/{w_a['workspace_id']}/documents/upload",
        data={"category_key": "return_policy"},
        files={"file": ("ecommerce_returns.txt", policy_a, "text/plain")},
        headers=headers_a,
    )
    assert up_a.status_code == 201
    doc_id_a = up_a.json()["document_id"]

    # Upload Attendance Policy to Workspace B
    policy_b = (
        "Students must maintain at least 75 percent class attendance to be eligible for final semester examinations. "
        "Absences exceeding 25 percent mandate an official medical or bereavement certificate."
    ).encode("utf-8")
    up_b = test_client.post(
        f"/api/v1/workspaces/{w_b['workspace_id']}/documents/upload",
        data={"category_key": "academic_regulations"},
        files={"file": ("university_attendance.txt", policy_b, "text/plain")},
        headers=headers_b,
    )
    assert up_b.status_code == 201
    doc_id_b = up_b.json()["document_id"]

    # 1. Query Workspace A for return policy
    res_a = test_client.post(
        f"/api/v1/workspaces/{w_a['workspace_id']}/query",
        json={"query": "How many days do I have to return an eligible item for a refund?"},
        headers=headers_a,
    )
    assert res_a.status_code == 200
    data_a = res_a.json()

    # Verify Workspace A retrieval
    assert data_a["workspace_id"] == w_a["workspace_id"]
    assert data_a["evidence"]["retrieval_count"] >= 1
    assert data_a["evidence"]["evidence_quality"] == "Strong"
    # Document IDs retrieved must match Tenant A's document, NEVER Tenant B's
    retrieved_doc_ids_a = [c["document_id"] for c in data_a["evidence"]["chunks"]]
    assert doc_id_a in retrieved_doc_ids_a
    assert doc_id_b not in retrieved_doc_ids_a

    # Verify Grounded Answer synthesis
    assert data_a["grounded_response"] is not None
    assert "ecommerce_returns.txt" in data_a["grounded_response"]
    assert "30 days" in data_a["grounded_response"]
    assert "Source Citation" in data_a["grounded_response"]

    # 2. Query Workspace B for attendance policy
    res_b = test_client.post(
        f"/api/v1/workspaces/{w_b['workspace_id']}/query",
        json={"query": "What is the minimum attendance required for final examinations?"},
        headers=headers_b,
    )
    assert res_b.status_code == 200
    data_b = res_b.json()

    # Verify Workspace B retrieval
    assert data_b["workspace_id"] == w_b["workspace_id"]
    assert data_b["evidence"]["retrieval_count"] >= 1
    assert data_b["evidence"]["evidence_quality"] == "Strong"
    retrieved_doc_ids_b = [c["document_id"] for c in data_b["evidence"]["chunks"]]
    assert doc_id_b in retrieved_doc_ids_b
    assert doc_id_a not in retrieved_doc_ids_b

    # Verify Grounded Answer synthesis for Tenant B
    assert data_b["grounded_response"] is not None
    assert "university_attendance.txt" in data_b["grounded_response"]
    assert "75 percent" in data_b["grounded_response"]


# ===========================================================================
# 4. Safe Failure & Zero Fallback Invariant Tests
# ===========================================================================

def test_empty_workspace_safe_failure_no_demo_fallback(test_client):
    """
    Critical Invariant:
    When a workspace has no uploaded documents, retrieval must return empty results.
    It MUST NEVER silently fall back to capstone_knowledge_v1.
    Low evidence quality must trigger safety rule LOW_CONF_05 and ESCALATE tier.
    Zero hallucinated response must be generated (grounded_response is None).
    """
    t = test_client.post(
        "/api/v1/tenants/register",
        json={"name": "Empty Store", "email": "empty@store.com"},
    ).json()
    w = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Fresh Storefront", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    ).json()

    # Query with text that matches capstone_knowledge_v1 in the public demo
    res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/query",
        json={"query": "How can I return an eligible product within the return window?"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    )
    assert res.status_code == 200
    data = res.json()

    # Evidence must be 0 chunks, Insufficient quality
    assert data["evidence"]["retrieval_count"] == 0
    assert data["evidence"]["evidence_quality"] == "Insufficient"
    assert len(data["evidence"]["chunks"]) == 0

    # Decision must escalate due to missing evidence
    assert data["resolution"]["final_autonomy_tier"] == "ESCALATE"
    assert data["resolution"]["final_escalation_required"] is True
    assert data["resolution"]["safety_rule_triggered"] == "LOW_CONF_05"

    # Escalation dossier must be structured
    assert data["resolution"]["escalation_dossier"] is not None
    assert data["resolution"]["escalation_reason"] == "Low Confidence Review"

    # Zero hallucinated text: grounded_response must be None
    assert data["grounded_response"] is None
    assert data["resolution"]["final_response"] is None


# ===========================================================================
# 5. Grounded Answer Refusal on Safety Gate Escalation
# ===========================================================================

def test_grounded_answer_refusal_when_safety_rule_triggers(test_client):
    """
    When safety invariants mandate escalation (e.g., sensitive/financial transaction),
    the system must refuse autonomous answer generation even if matching documents exist.
    """
    t = test_client.post(
        "/api/v1/tenants/register",
        json={"name": "Secure Finance", "email": "sec@fin.com"},
    ).json()
    w = test_client.post(
        "/api/v1/workspaces",
        json={"name": "Secured Workspace", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t['api_key']}"},
    ).json()

    headers = {"Authorization": f"Bearer {t['api_key']}"}

    # Upload document
    policy = (
        "Refund processing requires authorization. Bank transfers take 3-5 business days."
    ).encode("utf-8")
    test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/documents/upload",
        data={"category_key": "refund_policy"},
        files={"file": ("refunds.txt", policy, "text/plain")},
        headers=headers,
    )

    # Submit query with sensitive and financial impact flags
    res = test_client.post(
        f"/api/v1/workspaces/{w['workspace_id']}/query",
        json={
            "query": "Initiate an immediate bank refund transfer for my account.",
            "sensitive_action": True,
            "financial_impact": True,
        },
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()

    # Must mandate escalation
    assert data["resolution"]["final_escalation_required"] is True
    # Grounded response must be strictly refused
    assert data["grounded_response"] is None
    assert data["resolution"]["final_response"] is None
    assert data["resolution"]["escalation_dossier"] is not None
