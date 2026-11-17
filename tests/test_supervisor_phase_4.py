"""
Comprehensive Test Suite for Phase 4: Human Supervisor Escalation Dashboard
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
Research Direction: Risk-Aware Adaptive Multi-Agent Orchestration

Verifies:
1. GET /supervisor serves the complete supervisor dashboard HTML.
2. Escalation queue retrieval for authorized workspaces with risk and status filters.
3. Detailed case dossier retrieval with inference diagnostics, evidence, and safety rules.
4. Session-turn history endpoint providing chronological conversation context for dossiers.
5. Human review lifecycle transitions (PENDING -> IN_REVIEW -> RESOLVED) with reviewer audit trail.
6. Terminal-state immutability (modifications to RESOLVED/REJECTED return HTTP 400).
7. Strict cross-tenant isolation (Tenant B forbidden from Tenant A escalations).
8. Public widget tokens forbidden from administrative supervisor endpoints.
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.workspaces.db import WorkspaceDB
from src.workspaces.models import EscalationStatus
from src.workspaces.service import WorkspaceService


@pytest.fixture
def temp_db(tmp_path):
    """Provides isolated, temporary SQLite database for tests."""
    db_file = tmp_path / "test_supervisor_phase4.db"
    return WorkspaceDB(db_path=db_file)


@pytest.fixture
def client(temp_db, monkeypatch):
    """TestClient wired to isolated test database."""
    test_service = WorkspaceService(db=temp_db)
    test_pipeline = DocumentIngestionPipeline(db=temp_db)
    from src.api import workspace_routes

    monkeypatch.setattr(workspace_routes, "_workspace_service", test_service)
    monkeypatch.setattr(workspace_routes, "_ingestion_pipeline", test_pipeline)
    return TestClient(app)


# Helper: register a tenant, create a workspace, and insert a real escalation record
def _setup_tenant_and_escalation(client, temp_db, tenant_name="Supervisor Tenant", domain="E-commerce"):
    reg_resp = client.post(
        "/api/v1/tenants/register",
        json={"name": tenant_name, "email": f"{tenant_name.lower().replace(' ', '_')}@test.local"},
    )
    assert reg_resp.status_code == 201
    auth_data = reg_resp.json()
    api_key = auth_data["api_key"]
    tenant_id = auth_data["tenant_id"]
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    ws_resp = client.post(
        "/api/v1/workspaces",
        headers=auth_headers,
        json={"name": f"{tenant_name} Workspace", "domain": domain},
    )
    assert ws_resp.status_code == 201
    ws_id = ws_resp.json()["workspace_id"]
    token = temp_db.get_or_create_widget_token(ws_id)

    # Insert traceable escalation record into DB
    import uuid
    unique_esc_id = f"esc_{uuid.uuid4().hex[:10]}"
    esc_record = {
        "escalation_id": unique_esc_id,
        "workspace_id": ws_id,
        "tenant_id": tenant_id,
        "session_id": "sess_audit_401",
        "customer_id": "cust_vip_99",
        "query": "I demand an immediate unapproved charge reversal of $4,500!",
        "domain": domain,
        "intent": "chargeback_dispute",
        "intent_confidence": 0.88,
        "risk": "High",
        "risk_confidence": 0.92,
        "complexity": "High",
        "provisional_autonomy_tier": "ANSWER",
        "final_autonomy_tier": "ESCALATE",
        "evidence_quality": "Insufficient",
        "retrieval_similarity_score": 0.54,
        "retrieved_chunks_json": json.dumps([
            {
                "chunk_id": "chk_pol_01",
                "document_id": "doc_pol_01",
                "title": "Dispute Resolution Policy",
                "document_type": "dispute_policy",
                "similarity": 0.54,
                "chunk_text": "Chargebacks and transaction disputes exceeding $1,000 require manual supervisor review."
            }
        ]),
        "safety_rule_triggered": "FINANCIAL_RISK_02",
        "escalation_reason": "High Financial Risk Transaction Threshold",
        "recommended_action": "Verify transaction audit logs with fraud division",
        "status": "PENDING",
    }
    temp_db.create_escalation(esc_record)

    return {
        "api_key": api_key,
        "tenant_id": tenant_id,
        "auth_headers": auth_headers,
        "workspace_id": ws_id,
        "widget_token": token,
        "escalation_id": esc_record["escalation_id"],
        "session_id": esc_record["session_id"],
    }


# ===========================================================================
# 1. Supervisor Dashboard Route Verification
# ===========================================================================

def test_supervisor_route_serves_html(client):
    """GET /supervisor returns HTTP 200 and serves the complete supervisor interface."""
    resp = client.get("/supervisor")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    content = resp.text
    assert "Supervisor Desk" in content
    assert "Supervisor Authentication" in content
    assert "cap_demo_admin_key" in content
    assert "queueTableBody" in content
    assert "dossierModal" in content


# ===========================================================================
# 2. Escalation Queue Listing with Filters
# ===========================================================================

def test_escalation_queue_listing(client, temp_db):
    """Lists real escalation records strictly for the authorized workspace with filtering."""
    env = _setup_tenant_and_escalation(client, temp_db)
    ws_id = env["workspace_id"]
    headers = env["auth_headers"]

    # 1. List without filter
    resp = client.get(f"/api/v1/workspaces/{ws_id}/escalations", headers=headers)
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    item = items[0]
    assert item["escalation_id"] == env["escalation_id"]
    assert item["risk"] == "High"
    assert item["status"] == "PENDING"
    assert item["workspace_id"] == ws_id
    assert "charge reversal" in item["query"]

    # 2. List with matching status filter
    resp_pending = client.get(f"/api/v1/workspaces/{ws_id}/escalations?status=PENDING", headers=headers)
    assert resp_pending.status_code == 200
    assert len(resp_pending.json()) == 1

    # 3. List with non-matching status filter
    resp_resolved = client.get(f"/api/v1/workspaces/{ws_id}/escalations?status=RESOLVED", headers=headers)
    assert resp_resolved.status_code == 200
    assert len(resp_resolved.json()) == 0


# ===========================================================================
# 3. Detailed Case Dossier Retrieval
# ===========================================================================

def test_escalation_dossier_retrieval(client, temp_db):
    """Retrieves full escalation dossier containing diagnostics, evidence, and safety policy."""
    env = _setup_tenant_and_escalation(client, temp_db)
    ws_id = env["workspace_id"]
    esc_id = env["escalation_id"]
    headers = env["auth_headers"]

    resp = client.get(f"/api/v1/workspaces/{ws_id}/escalations/{esc_id}", headers=headers)
    assert resp.status_code == 200
    dossier = resp.json()

    assert dossier["escalation_id"] == esc_id
    assert dossier["intent"] == "chargeback_dispute"
    assert dossier["intent_confidence"] == pytest.approx(0.88, abs=0.01)
    assert dossier["risk"] == "High"
    assert dossier["risk_confidence"] == pytest.approx(0.92, abs=0.01)
    assert dossier["safety_rule_triggered"] == "FINANCIAL_RISK_02"
    assert dossier["final_autonomy_tier"] == "ESCALATE"
    assert dossier["status"] == "PENDING"

    # Verify retrieved chunks JSON parses accurately
    chunks = json.loads(dossier["retrieved_chunks_json"])
    assert len(chunks) == 1
    assert chunks[0]["title"] == "Dispute Resolution Policy"
    assert "Dispute Resolution Policy" in chunks[0]["title"]


# ===========================================================================
# 4. Session Turns Context Endpoint
# ===========================================================================

def test_session_turns_context_endpoint(client, temp_db):
    """GET /workspaces/{id}/sessions/{id}/turns returns chronological turn history for dossier context."""
    env = _setup_tenant_and_escalation(client, temp_db)
    ws_id = env["workspace_id"]
    sess_id = env["session_id"]
    headers = env["auth_headers"]

    # Save turn in session memory
    temp_db.save_session_turn(
        workspace_id=ws_id,
        tenant_id=env["tenant_id"],
        session_id=sess_id,
        user_query="Hello, I have an issue with transaction #441",
        resolved_query="Hello, I have an issue with transaction #441",
        bot_response="I can help you review transaction #441. What occurred?",
        final_autonomy_tier="ANSWER",
        citations=[{"document_name": "FAQ", "similarity": 0.85}],
    )

    resp = client.get(f"/api/v1/workspaces/{ws_id}/sessions/{sess_id}/turns", headers=headers)
    assert resp.status_code == 200
    turns = resp.json()
    assert len(turns) == 1
    assert turns[0]["turn_index"] == 1
    assert turns[0]["user_query"] == "Hello, I have an issue with transaction #441"
    assert turns[0]["bot_response"] == "I can help you review transaction #441. What occurred?"
    assert turns[0]["final_autonomy_tier"] == "ANSWER"


# ===========================================================================
# 5. Human Review Lifecycle Transitions (State Machine)
# ===========================================================================

def test_human_review_lifecycle_transitions(client, temp_db):
    """
    Verifies state machine transitions:
    Step 1: PENDING -> IN_REVIEW (claimed by supervisor)
    Step 2: IN_REVIEW -> RESOLVED (with official notes and decision code)
    """
    env = _setup_tenant_and_escalation(client, temp_db)
    ws_id = env["workspace_id"]
    esc_id = env["escalation_id"]
    headers = env["auth_headers"]

    # Step 1: Claim case (Set IN_REVIEW)
    review_resp_1 = client.post(
        f"/api/v1/workspaces/{ws_id}/escalations/{esc_id}/review",
        headers=headers,
        json={
            "status": "IN_REVIEW",
            "human_decision": "REQUEST_MORE_INFO",
            "reviewer_notes": "Supervisor Sarah investigating cardholder transaction logs.",
            "resolved_by": "supervisor_sarah",
        },
    )
    assert review_resp_1.status_code == 200
    d1 = review_resp_1.json()
    assert d1["status"] == "IN_REVIEW"
    assert d1["resolved_by"] == "supervisor_sarah"
    assert d1["resolved_at"] is None  # Not terminal yet

    # Step 2: Final Resolution (Set RESOLVED)
    review_resp_2 = client.post(
        f"/api/v1/workspaces/{ws_id}/escalations/{esc_id}/review",
        headers=headers,
        json={
            "status": "RESOLVED",
            "human_decision": "APPROVE_OVERRIDE",
            "reviewer_notes": "Customer provided verified police report; manual credit granted.",
            "resolved_by": "supervisor_sarah",
        },
    )
    assert review_resp_2.status_code == 200
    d2 = review_resp_2.json()
    assert d2["status"] == "RESOLVED"
    assert d2["human_decision"] == "APPROVE_OVERRIDE"
    assert d2["resolved_at"] is not None  # Terminal timestamp populated


# ===========================================================================
# 6. Terminal State Immutability Enforcement
# ===========================================================================

def test_terminal_state_immutability(client, temp_db):
    """Once an escalation is RESOLVED or REJECTED, subsequent modifications return HTTP 400."""
    env = _setup_tenant_and_escalation(client, temp_db)
    ws_id = env["workspace_id"]
    esc_id = env["escalation_id"]
    headers = env["auth_headers"]

    # Resolve escalation
    res = client.post(
        f"/api/v1/workspaces/{ws_id}/escalations/{esc_id}/review",
        headers=headers,
        json={
            "status": "RESOLVED",
            "human_decision": "CONFIRM_POLICY_LIMITATION",
            "reviewer_notes": "Resolution completed.",
            "resolved_by": "lead_supervisor",
        },
    )
    assert res.status_code == 200

    # Attempt to reopen or modify resolved escalation
    res_reopen = client.post(
        f"/api/v1/workspaces/{ws_id}/escalations/{esc_id}/review",
        headers=headers,
        json={
            "status": "IN_REVIEW",
            "human_decision": "APPROVE_OVERRIDE",
            "reviewer_notes": "Attempting illegal state modification.",
            "resolved_by": "hacker_tamper",
        },
    )
    assert res_reopen.status_code == 400
    assert "terminal state" in res_reopen.json()["detail"].lower()


# ===========================================================================
# 7. Strict Cross-Tenant Authorization Isolation
# ===========================================================================

def test_supervisor_cross_tenant_isolation(client, temp_db):
    """Tenant B is strictly rejected (HTTP 403) from listing, viewing, or reviewing Tenant A's escalations."""
    env_a = _setup_tenant_and_escalation(client, temp_db, tenant_name="Tenant Alpha")
    env_b = _setup_tenant_and_escalation(client, temp_db, tenant_name="Tenant Beta")

    ws_a = env_a["workspace_id"]
    esc_a = env_a["escalation_id"]
    headers_b = env_b["auth_headers"]

    # 1. Tenant B tries to list Tenant A's escalation queue
    list_attempt = client.get(f"/api/v1/workspaces/{ws_a}/escalations", headers=headers_b)
    assert list_attempt.status_code == 403

    # 2. Tenant B tries to view Tenant A's case dossier
    get_attempt = client.get(f"/api/v1/workspaces/{ws_a}/escalations/{esc_a}", headers=headers_b)
    assert get_attempt.status_code == 403

    # 3. Tenant B tries to submit a review decision on Tenant A's case
    review_attempt = client.post(
        f"/api/v1/workspaces/{ws_a}/escalations/{esc_a}/review",
        headers=headers_b,
        json={
            "status": "RESOLVED",
            "human_decision": "APPROVE_OVERRIDE",
            "reviewer_notes": "Unauthorized review attempt",
            "resolved_by": "tenant_b_spy",
        },
    )
    assert review_attempt.status_code == 403

    # 4. Tenant B tries to inspect Tenant A's session dialogue turns
    turns_attempt = client.get(f"/api/v1/workspaces/{ws_a}/sessions/{env_a['session_id']}/turns", headers=headers_b)
    assert turns_attempt.status_code == 403


# ===========================================================================
# 8. Public Widget Token Forbidden from Supervisor Endpoints
# ===========================================================================

def test_widget_token_forbidden_from_supervisor(client, temp_db):
    """Public widget tokens (wtk_...) cannot access or review supervisor escalation endpoints."""
    env = _setup_tenant_and_escalation(client, temp_db)
    ws_id = env["workspace_id"]
    esc_id = env["escalation_id"]
    widget_token = env["widget_token"]

    widget_headers = {"X-Widget-Token": widget_token}

    # Queue listing without master Bearer token
    resp_list = client.get(f"/api/v1/workspaces/{ws_id}/escalations", headers=widget_headers)
    assert resp_list.status_code in (401, 403)

    # Dossier viewing with widget token
    resp_dossier = client.get(f"/api/v1/workspaces/{ws_id}/escalations/{esc_id}", headers=widget_headers)
    assert resp_dossier.status_code in (401, 403)

    # Review submission with widget token
    resp_review = client.post(
        f"/api/v1/workspaces/{ws_id}/escalations/{esc_id}/review",
        headers=widget_headers,
        json={"status": "RESOLVED", "resolved_by": "unauthorized_widget"},
    )
    assert resp_review.status_code in (401, 403)
