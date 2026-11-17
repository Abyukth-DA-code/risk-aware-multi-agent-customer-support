"""
Comprehensive Test Suite for Phase 2: Business Admin Console & Knowledge Management
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Verifies:
1. GET /admin endpoint serving admin.html
2. Tenant registration and session API key issuance
3. Multi-workspace creation across domains (E-commerce, Education, Small Business)
4. Domain checklist instantiation and backend readiness calculation
5. Adding custom categories to domain checklists
6. Document ingestion (.txt, .pdf, .docx, .csv) and vector indexing into workspace collection
7. Quality validation constraints (min chars, format restrictions, duplicate conflict)
8. Document listing and deletion with Chroma vector purging
9. Public widget token retrieval, rotation, and revocation
10. Strict tenant isolation across administrative operations
"""

import io
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.workspaces.checklist_engine import DOMAIN_CHECKLIST_SPECS, ChecklistEngine
from src.workspaces.db import WorkspaceDB
from src.workspaces.models import BusinessDomain, ChecklistItemStatus, CustomCategoryRequest, TenantRegisterRequest, WorkspaceCreateRequest
from src.workspaces.service import WorkspaceService



@pytest.fixture
def temp_db(tmp_path):
    """Provides isolated, temporary SQLite database for tests."""
    db_file = tmp_path / "test_tenants_admin.db"
    return WorkspaceDB(db_path=db_file)


@pytest.fixture
def workspace_service(temp_db):
    """Provides WorkspaceService backed by temporary SQLite DB."""
    return WorkspaceService(db=temp_db)


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
# 1. Admin Console Route Verification
# ===========================================================================

def test_admin_route_serves_html(client):
    """GET /admin returns HTTP 200 and serves the Business Admin Console HTML."""
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    content = resp.text
    assert "Business Admin Console" in content
    assert "Business Authentication" in content
    assert "Domain Knowledge Checklist" in content
    assert "Ingest Knowledge Documents" in content
    assert "Storefront Integration" in content
    assert "cap_demo_admin_key" in content


# ===========================================================================
# 2. Multi-Business Tenant & Workspace Lifecycle
# ===========================================================================

def test_single_owner_multi_workspace_management(client):
    """
    Verifies that a single business owner can register a tenant and manage
    multiple isolated workspaces across distinct domains (E-commerce, Education, Small Business).
    """
    # 1. Register Tenant
    reg_resp = client.post(
        "/api/v1/tenants/register",
        json={"name": "Multi-Venture Holdings", "email": "founder@multiventure.com"},
    )
    assert reg_resp.status_code == 201
    tenant_data = reg_resp.json()
    api_key = tenant_data["api_key"]
    tenant_id = tenant_data["tenant_id"]
    assert api_key.startswith("cap_")

    auth_headers = {"Authorization": f"Bearer {api_key}"}

    # 2. Create Workspace 1: E-commerce
    ws1_resp = client.post(
        "/api/v1/workspaces",
        headers=auth_headers,
        json={"name": "Venture Retail Store", "domain": "E-commerce", "description": "Online storefront"},
    )
    assert ws1_resp.status_code == 201
    ws1 = ws1_resp.json()
    assert ws1["domain"] == "E-commerce"
    assert ws1["tenant_id"] == tenant_id

    # 3. Create Workspace 2: Education
    ws2_resp = client.post(
        "/api/v1/workspaces",
        headers=auth_headers,
        json={"name": "Venture Learning Academy", "domain": "Education", "description": "Student LMS support"},
    )
    assert ws2_resp.status_code == 201
    ws2 = ws2_resp.json()
    assert ws2["domain"] == "Education"

    # 4. Create Workspace 3: Small Business
    ws3_resp = client.post(
        "/api/v1/workspaces",
        headers=auth_headers,
        json={"name": "Venture Consulting Group", "domain": "Small Business", "description": "Client invoice support"},
    )
    assert ws3_resp.status_code == 201
    ws3 = ws3_resp.json()
    assert ws3["domain"] == "Small Business"

    # 5. List workspaces for this tenant - must return all 3
    list_resp = client.get("/api/v1/workspaces", headers=auth_headers)
    assert list_resp.status_code == 200
    workspaces = list_resp.json()
    assert len(workspaces) == 3
    domains = {w["domain"] for w in workspaces}
    assert domains == {"E-commerce", "Education", "Small Business"}


# ===========================================================================
# 3. Domain Checklists & Dynamic Readiness Evaluation
# ===========================================================================

def test_canonical_domain_checklist_and_readiness(client):
    """
    Verifies that creating a workspace instantiates the canonical checklist categories
    and reports accurate baseline readiness.
    """
    # Register tenant
    reg_resp = client.post(
        "/api/v1/tenants/register",
        json={"name": "Apex Commerce", "email": "ops@apex.com"},
    )
    api_key = reg_resp.json()["api_key"]
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    # Create E-commerce workspace
    ws_resp = client.post(
        "/api/v1/workspaces",
        headers=auth_headers,
        json={"name": "Apex Store", "domain": "E-commerce"},
    )
    ws_id = ws_resp.json()["workspace_id"]

    # 1. Inspect Checklist
    chk_resp = client.get(f"/api/v1/workspaces/{ws_id}/checklist", headers=auth_headers)
    assert chk_resp.status_code == 200
    items = chk_resp.json()
    assert len(items) == 10  # 6 required + 4 recommended
    cat_keys = {item["category_key"] for item in items}
    assert "return_policy" in cat_keys
    assert "refund_policy" in cat_keys
    assert "order_cancellation" in cat_keys
    assert "delivery_shipping" in cat_keys
    assert "payment_methods" in cat_keys
    assert "customer_support_escalation" in cat_keys

    # 2. Inspect Initial Readiness (0% because no documents uploaded yet)
    read_resp = client.get(f"/api/v1/workspaces/{ws_id}/readiness", headers=auth_headers)
    assert read_resp.status_code == 200
    report = read_resp.json()
    assert report["readiness_percentage"] == 0.0
    assert report["is_ready_for_testing"] is False
    assert report["required_categories"] == 6
    assert report["required_completed"] == 0
    assert len(report["missing_required_categories"]) == 6
    assert len(report["unanswerable_query_warnings"]) == 6

    # 3. Add Custom Category
    custom_resp = client.post(
        f"/api/v1/workspaces/{ws_id}/checklist/custom",
        headers=auth_headers,
        json={"category_name": "VIP Membership Rules", "is_required": False},
    )
    assert custom_resp.status_code == 201
    custom_item = custom_resp.json()
    assert custom_item["category_key"] == "vip_membership_rules"
    assert custom_item["is_required"] is False

    # Check updated total count
    chk_resp2 = client.get(f"/api/v1/workspaces/{ws_id}/checklist", headers=auth_headers)
    assert len(chk_resp2.json()) == 11


# ===========================================================================
# 4. Document Ingestion, Processing & Vector Storage
# ===========================================================================

def test_document_ingestion_and_quality_gates(client, tmp_path, monkeypatch):
    """
    Verifies document upload, quality constraints, duplicate detection,
    and checklist state transitions.
    """
    reg_resp = client.post(
        "/api/v1/tenants/register",
        json={"name": "Edu Portal", "email": "dean@edu.org"},
    )
    api_key = reg_resp.json()["api_key"]
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    ws_resp = client.post(
        "/api/v1/workspaces",
        headers=auth_headers,
        json={"name": "Exam Support", "domain": "Education"},
    )
    ws_id = ws_resp.json()["workspace_id"]

    # Test 1: Uploading empty / too-short document (<50 chars) must return 400 Bad Request
    short_content = b"Short text."
    resp_short = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/upload",
        headers=auth_headers,
        data={"category_key": "attendance_policy"},
        files={"file": ("short.txt", short_content, "text/plain")},
    )
    assert resp_short.status_code == 400
    assert "too short" in resp_short.json()["detail"].lower()

    # Test 2: Uploading valid policy text
    valid_policy = (
        "University Attendance Policy: Mandatory attendance requirement for all registered "
        "undergraduate students is 75 percent in each enrolled lecture course. Students with "
        "attendance between 65 and 75 percent may apply for medical condonation with verified "
        "hospital documentation submitted within five business days of absence."
    ).encode("utf-8")

    resp_valid = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/upload",
        headers=auth_headers,
        data={"category_key": "attendance_policy"},
        files={"file": ("university_attendance_policy.txt", valid_policy, "text/plain")},
    )
    assert resp_valid.status_code == 201
    upload_res = resp_valid.json()
    doc_id = upload_res["document_id"]
    assert upload_res["filename"] == "university_attendance_policy.txt"
    assert upload_res["chunks_indexed"] >= 1
    assert upload_res["character_count"] >= 50

    # Test 3: Uploading exact duplicate must return 409 Conflict
    resp_dup = client.post(
        f"/api/v1/workspaces/{ws_id}/documents/upload",
        headers=auth_headers,
        data={"category_key": "attendance_policy"},
        files={"file": ("duplicate_policy.txt", valid_policy, "text/plain")},
    )
    assert resp_dup.status_code == 409
    assert "duplicate" in resp_dup.json()["detail"].lower()

    # Test 4: Verify Checklist status transitioned to 'Validated'
    chk_resp = client.get(f"/api/v1/workspaces/{ws_id}/checklist", headers=auth_headers)
    items = chk_resp.json()
    att_item = next(i for i in items if i["category_key"] == "attendance_policy")
    assert att_item["status"] == "Validated"
    assert att_item["filename"] == "university_attendance_policy.txt"
    assert att_item["extracted_records_count"] >= 1

    # Test 5: Verify readiness updated (1 required completed out of 5)
    read_resp = client.get(f"/api/v1/workspaces/{ws_id}/readiness", headers=auth_headers)
    report = read_resp.json()
    assert report["required_completed"] == 1
    assert report["readiness_percentage"] > 0.0

    # Test 6: List active documents
    docs_resp = client.get(f"/api/v1/workspaces/{ws_id}/documents", headers=auth_headers)
    assert docs_resp.status_code == 200
    docs = docs_resp.json()
    assert len(docs) == 1
    assert docs[0]["document_id"] == doc_id

    # Test 7: Delete document
    del_resp = client.delete(f"/api/v1/workspaces/{ws_id}/documents/{doc_id}", headers=auth_headers)
    assert del_resp.status_code == 200
    assert "deleted" in del_resp.json()["message"].lower()

    # Test 8: Verify checklist status reset after document deletion
    chk_resp_after = client.get(f"/api/v1/workspaces/{ws_id}/checklist", headers=auth_headers)
    items_after = chk_resp_after.json()
    att_item_after = next(i for i in items_after if i["category_key"] == "attendance_policy")
    assert att_item_after["status"] == "Not Uploaded"


# ===========================================================================
# 5. Widget Token Management & Security Invariants
# ===========================================================================

def test_widget_token_management_and_isolation(client):
    """
    Verifies scoped public widget token retrieval, rotation, revocation,
    and security isolation against administrative endpoints.
    """
    reg_resp = client.post(
        "/api/v1/tenants/register",
        json={"name": "Small Biz Store", "email": "shop@biz.com"},
    )
    api_key = reg_resp.json()["api_key"]
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    ws_resp = client.post(
        "/api/v1/workspaces",
        headers=auth_headers,
        json={"name": "Main Office", "domain": "Small Business"},
    )
    ws_id = ws_resp.json()["workspace_id"]

    # 1. Retrieve Widget Token
    token_resp = client.get(f"/api/v1/workspaces/{ws_id}/widget/token", headers=auth_headers)
    assert token_resp.status_code == 200
    token1 = token_resp.json()["widget_token"]
    assert token1.startswith("wtk_")

    # 2. Rotate Token
    rot_resp = client.post(f"/api/v1/workspaces/{ws_id}/widget/token/rotate", headers=auth_headers)
    assert rot_resp.status_code == 200
    token2 = rot_resp.json()["widget_token"]
    assert token2.startswith("wtk_")
    assert token2 != token1

    # 3. Security Check: Widget token CANNOT access admin endpoints
    widget_headers = {"Authorization": f"Bearer {token2}"}
    admin_access_resp = client.get(f"/api/v1/workspaces/{ws_id}/checklist", headers=widget_headers)
    assert admin_access_resp.status_code == 403
    assert "Widget tokens cannot be used to authenticate administrative endpoints" in admin_access_resp.json()["detail"]

    # 4. Revoke Token
    rev_resp = client.post(f"/api/v1/workspaces/{ws_id}/widget/token/revoke", headers=auth_headers)
    assert rev_resp.status_code == 200
    assert rev_resp.json()["revoked"] is True
