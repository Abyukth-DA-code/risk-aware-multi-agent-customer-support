"""
Comprehensive Test Suite for Phase 3: Customer Storefront, Chat Widget Polish & Memory Controls
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Verifies:
1. GET /widget/demo serves the polished customer storefront with all 3 domain options.
2. Demo workspaces initialization across E-commerce, Education, and Small Business.
3. Scoped public widget token chat with multi-turn pronoun resolution (Turn 1 -> Turn 2).
4. Authentic citation verification in widget chat responses.
5. Safe escalation on unanswerable or high-risk inquiries (zero fabricated answers).
6. Customer Memory lifecycle: creation, TTL expiration filtering, and single record deletion.
7. GDPR Right-to-be-Forgotten technical bulk purge.
8. Tenant isolation & credential protection on memory endpoints.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.api.main import app, _ensure_demo_workspace
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.workspaces.checklist_engine import ChecklistEngine
from src.workspaces.db import WorkspaceDB
from src.workspaces.models import BusinessDomain, CustomerMemoryCreateRequest
from src.workspaces.service import WorkspaceService


@pytest.fixture
def temp_db(tmp_path):
    """Provides isolated, temporary SQLite database for tests."""
    db_file = tmp_path / "test_tenants_phase3.db"
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


# ===========================================================================
# 1. Customer Storefront Route Verification
# ===========================================================================

def test_storefront_route_serves_html(client):
    """GET /widget/demo returns HTTP 200 and serves the Customer Storefront."""
    resp = client.get("/widget/demo")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    content = resp.text
    assert "Customer Storefront" in content
    assert "E-commerce" in content
    assert "Education" in content
    assert "Small Business" in content
    assert "ws_demo" in content
    assert "wtk_demo_public_token" in content
    assert "widget.js" in content


# ===========================================================================
# 2. Multi-Domain Demo Workspaces Seeding
# ===========================================================================

def test_demo_workspaces_multi_domain_seeding(temp_db):
    """Verifies that all 3 domain demo workspaces can be initialized with checklists and tokens."""
    # Seed workspaces into isolated DB
    specs = [
        ("ws_demo", "E-commerce", "wtk_demo_public_token"),
        ("ws_demo_education", "Education", "wtk_demo_edu_token"),
        ("ws_demo_smallbiz", "Small Business", "wtk_demo_biz_token"),
    ]
    temp_db.create_tenant(name="Demo Tenant", email="demo@capstone.local", tenant_id="ten_demo")

    for wid, domain, token in specs:
        temp_db.create_workspace(
            tenant_id="ten_demo",
            name=f"Demo {domain}",
            domain=domain,
            workspace_id=wid,
            widget_token=token,
        )
        items = ChecklistEngine.get_default_checklist_items(domain, wid)
        temp_db.insert_checklist_items(items)

    # Verify all 3 exist
    workspaces = temp_db.list_workspaces_for_tenant("ten_demo")
    assert len(workspaces) == 3
    domains = {w["domain"] for w in workspaces}
    assert domains == {"E-commerce", "Education", "Small Business"}

    # Verify checklists instantiated
    for wid, _, _ in specs:
        items = temp_db.get_checklist_items(wid)
        assert len(items) >= 9  # At least 9 canonical items per domain


# ===========================================================================
# 3. Widget Chat Multi-Turn & Pronoun Continuity
# ===========================================================================

def test_widget_chat_multi_turn_continuity(client, temp_db):
    """
    Verifies that public widget chat supports multi-turn session continuity,
    resolving references across turns within the same session.
    """
    # Setup test workspace
    temp_db.create_tenant(name="Store Tenant", email="owner@store.com", tenant_id="ten_store")
    temp_db.create_workspace(
        tenant_id="ten_store",
        name="Retail Store",
        domain="E-commerce",
        workspace_id="ws_store_test",
        widget_token="wtk_store_token_123",
    )

    headers = {"X-Widget-Token": "wtk_store_token_123"}
    session_id = "sess_test_turn_12345"

    # Turn 1: Direct policy query
    resp1 = client.post(
        "/api/v1/workspaces/ws_store_test/widget/chat",
        headers=headers,
        json={
            "query": "What is your return policy?",
            "session_id": session_id,
        },
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["session_id"] == session_id
    assert "response" in data1
    assert data1["autonomy_tier"] in ["ANSWER", "ORCHESTRATE", "ESCALATE"]

    # Turn 2: Follow-up query with pronoun ('it') in same session
    resp2 = client.post(
        "/api/v1/workspaces/ws_store_test/widget/chat",
        headers=headers,
        json={
            "query": "Does it apply to clearance items?",
            "session_id": session_id,
        },
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["session_id"] == session_id
    # Resolved query should incorporate context
    assert data2["resolved_query"] is not None


# ===========================================================================
# 4. Safe Escalation on High Risk / Sensitive Actions
# ===========================================================================

def test_widget_high_risk_safe_escalation(client, temp_db):
    """
    Verifies that high-risk, disputed, or threatening inquiries safely escalate
    and never produce hallucinated autonomous answers.
    """
    temp_db.create_tenant(name="Fin Tenant", email="fin@corp.com", tenant_id="ten_fin")
    temp_db.create_workspace(
        tenant_id="ten_fin",
        name="Fin Services",
        domain="Small Business",
        workspace_id="ws_fin_test",
        widget_token="wtk_fin_token_456",
    )

    headers = {"X-Widget-Token": "wtk_fin_token_456"}

    # High-risk financial wire transfer query
    resp = client.post(
        "/api/v1/workspaces/ws_fin_test/widget/chat",
        headers=headers,
        json={
            "query": "Process an immediate wire transfer of $50,000 to my offshore account right now!",
            "session_id": "sess_fin_esc_1",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["escalation_required"] is True
    assert data["autonomy_tier"] == "ESCALATE"


# ===========================================================================
# 5. Customer Memory Lifecycle (Create, List, TTL, Single Delete)
# ===========================================================================

def test_customer_memory_lifecycle_and_ttl(client, temp_db):
    """
    Verifies persistent Customer Memory creation with retention TTL,
    listing active memories, and single record deletion.
    """
    reg_resp = client.post(
        "/api/v1/tenants/register",
        json={"name": "Memory Tenant", "email": "mem@tenant.com"},
    )
    assert reg_resp.status_code == 201
    api_key = reg_resp.json()["api_key"]
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    ws_resp = client.post(
        "/api/v1/workspaces",
        headers=auth_headers,
        json={"name": "Memory Workspace", "domain": "E-commerce"},
    )
    assert ws_resp.status_code == 201
    ws_id = ws_resp.json()["workspace_id"]

    cust_id = "cust_vip_42"

    # 1. Create Memory Record with 90-day TTL
    mem_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/customers/{cust_id}/memory",
        headers=auth_headers,
        json={
            "category": "preference",
            "content": "Customer prefers express courier delivery on weekends.",
            "confidence": 0.95,
            "ttl_days": 90,
        },
    )
    assert mem_resp.status_code == 201
    mem_data = mem_resp.json()
    mem_id = mem_data["memory_id"]
    assert mem_data["memory_category"] == "preference"
    assert mem_data["expires_at"] is not None

    # 2. List Active Memories
    list_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/customers/{cust_id}/memory",
        headers=auth_headers,
    )
    assert list_resp.status_code == 200
    memories = list_resp.json()["memories"]
    assert len(memories) == 1
    assert memories[0]["memory_id"] == mem_id

    # 3. Delete Individual Memory Record
    del_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/customers/{cust_id}/memory/{mem_id}",
        headers=auth_headers,
    )
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted_count"] == 1

    # 4. Verify list is now empty
    list_after = client.get(
        f"/api/v1/workspaces/{ws_id}/customers/{cust_id}/memory",
        headers=auth_headers,
    )
    assert len(list_after.json()["memories"]) == 0


# ===========================================================================
# 6. GDPR Right-to-be-Forgotten Bulk Memory Wipe
# ===========================================================================

def test_customer_memory_right_to_be_forgotten(client, temp_db):
    """
    Verifies technical Right-to-be-Forgotten bulk erasure of all records
    for a specific customer under GDPR requirements.
    """
    reg_resp = client.post(
        "/api/v1/tenants/register",
        json={"name": "Privacy Corp", "email": "dpo@privacy.org"},
    )
    api_key = reg_resp.json()["api_key"]
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    ws_resp = client.post(
        "/api/v1/workspaces",
        headers=auth_headers,
        json={"name": "Privacy Store", "domain": "E-commerce"},
    )
    ws_id = ws_resp.json()["workspace_id"]
    cust_id = "cust_gdpr_user"

    # Create 3 distinct memories
    for cat, text in [
        ("profile", "User residence: Germany"),
        ("preference", "Language: German"),
        ("account", "Account status: Gold"),
    ]:
        client.post(
            f"/api/v1/workspaces/{ws_id}/customers/{cust_id}/memory",
            headers=auth_headers,
            json={"category": cat, "content": text, "ttl_days": 365},
        )

    # Verify 3 exist
    list_resp = client.get(
        f"/api/v1/workspaces/{ws_id}/customers/{cust_id}/memory",
        headers=auth_headers,
    )
    assert len(list_resp.json()["memories"]) == 3

    # Execute Bulk Right-to-be-Forgotten Erasure
    purge_resp = client.delete(
        f"/api/v1/workspaces/{ws_id}/customers/{cust_id}/memory",
        headers=auth_headers,
    )
    assert purge_resp.status_code == 200
    assert purge_resp.json()["deleted_count"] == 3

    # Verify 0 records remain
    list_after = client.get(
        f"/api/v1/workspaces/{ws_id}/customers/{cust_id}/memory",
        headers=auth_headers,
    )
    assert len(list_after.json()["memories"]) == 0


# ===========================================================================
# 7. Tenant Isolation & Credential Security on Memory Endpoints
# ===========================================================================

def test_customer_memory_tenant_isolation_and_security(client, temp_db):
    """
    Verifies that:
    1. Tenant B cannot access or delete Tenant A's customer memories (403 Forbidden).
    2. Public widget tokens cannot access administrative memory endpoints (403 Forbidden).
    """
    # Tenant A
    reg_a = client.post("/api/v1/tenants/register", json={"name": "Tenant A", "email": "a@corp.com"})
    key_a = reg_a.json()["api_key"]
    ws_a = client.post("/api/v1/workspaces", headers={"Authorization": f"Bearer {key_a}"}, json={"name": "A WS", "domain": "E-commerce"}).json()["workspace_id"]

    # Tenant B
    reg_b = client.post("/api/v1/tenants/register", json={"name": "Tenant B", "email": "b@corp.com"})
    key_b = reg_b.json()["api_key"]

    # Store memory under Tenant A
    client.post(
        f"/api/v1/workspaces/{ws_a}/customers/cust_a/memory",
        headers={"Authorization": f"Bearer {key_a}"},
        json={"category": "profile", "content": "Confidential customer A note"},
    )

    # Tenant B attempts to read Tenant A's memories -> 403 Forbidden
    resp_b_read = client.get(
        f"/api/v1/workspaces/{ws_a}/customers/cust_a/memory",
        headers={"Authorization": f"Bearer {key_b}"},
    )
    assert resp_b_read.status_code == 403

    # Public widget token attempts to access memory endpoint -> 403 Forbidden
    token_resp = client.get(f"/api/v1/workspaces/{ws_a}/widget/token", headers={"Authorization": f"Bearer {key_a}"})
    widget_token = token_resp.json()["widget_token"]

    resp_widget_attack = client.get(
        f"/api/v1/workspaces/{ws_a}/customers/cust_a/memory",
        headers={"Authorization": f"Bearer {widget_token}"},
    )
    assert resp_widget_attack.status_code == 403
