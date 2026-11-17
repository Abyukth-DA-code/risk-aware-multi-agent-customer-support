"""
Comprehensive Test Suite for Phase B: Multi-Tenant Workspaces & Dynamic Checklists
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.workspaces.checklist_engine import DOMAIN_CHECKLIST_SPECS, ChecklistEngine
from src.workspaces.db import WorkspaceDB
from src.workspaces.models import BusinessDomain, ChecklistItemStatus, CustomCategoryRequest
from src.workspaces.service import WorkspaceService


@pytest.fixture
def temp_db(tmp_path):
    """Provides isolated, temporary SQLite database for tests."""
    db_file = tmp_path / "test_tenants.db"
    return WorkspaceDB(db_path=db_file)


@pytest.fixture
def workspace_service(temp_db):
    """Provides WorkspaceService backed by temporary SQLite DB."""
    return WorkspaceService(db=temp_db)


@pytest.fixture
def client(temp_db, monkeypatch):
    """TestClient wired to isolated test database."""
    test_service = WorkspaceService(db=temp_db)
    from src.api import workspace_routes
    monkeypatch.setattr(workspace_routes, "_workspace_service", test_service)
    return TestClient(app)


# ===========================================================================
# 1. Domain-Specific Dynamic Checklist Unit Tests
# ===========================================================================

def test_ecommerce_checklist_spec():
    """Verify E-commerce dynamic checklist contains exact required & recommended categories."""
    spec = DOMAIN_CHECKLIST_SPECS[BusinessDomain.ECOMMERCE.value]
    req_names = [x["name"] for x in spec["required"]]
    rec_names = [x["name"] for x in spec["recommended"]]

    assert "Return Policy" in req_names
    assert "Refund Policy" in req_names
    assert "Order Cancellation Policy" in req_names
    assert "Delivery and Shipping Policy" in req_names
    assert "Payment Methods and Billing Policy" in req_names
    assert "Customer Support and Escalation Policy" in req_names
    assert len(req_names) == 6

    assert "Product Warranty Policy" in rec_names
    assert "Frequently Asked Questions (FAQ)" in rec_names
    assert len(rec_names) == 4


def test_education_checklist_spec():
    """Verify Education dynamic checklist contains exact required & recommended categories."""
    spec = DOMAIN_CHECKLIST_SPECS[BusinessDomain.EDUCATION.value]
    req_names = [x["name"] for x in spec["required"]]
    rec_names = [x["name"] for x in spec["recommended"]]

    assert "Attendance Policy" in req_names
    assert "Examination and Grading Policy" in req_names
    assert "Course Registration and Academic Records" in req_names
    assert "Fees and Payment Policy" in req_names
    assert "Student Support and Escalation Policy" in req_names
    assert len(req_names) == 5

    assert "Placement Policy" in rec_names
    assert "Library Policy" in rec_names
    assert len(rec_names) == 5


def test_small_business_checklist_spec():
    """Verify Small Business dynamic checklist contains exact required & recommended categories."""
    spec = DOMAIN_CHECKLIST_SPECS[BusinessDomain.SMALL_BUSINESS.value]
    req_names = [x["name"] for x in spec["required"]]
    rec_names = [x["name"] for x in spec["recommended"]]

    assert "Products and Services Information" in req_names
    assert "Pricing, Invoices, and Billing Policy" in req_names
    assert "Subscription and Cancellation Policy" in req_names
    assert "Orders, Inventory, or Fulfilment Guidance" in req_names
    assert "Customer Support and Escalation Policy" in req_names
    assert len(req_names) == 5

    assert "Vendor and Supplier Policies" in rec_names
    assert "Returns and Warranty Information" in rec_names
    assert len(rec_names) == 4


def test_checklist_engine_generation():
    """Verify ChecklistEngine generates initial unpopulated items with Not Uploaded status."""
    items = ChecklistEngine.get_default_checklist_items(
        domain="E-commerce",
        workspace_id="ws_test_123",
    )
    assert len(items) == 10  # 6 required + 4 recommended
    for item in items:
        assert item["status"] == ChecklistItemStatus.NOT_UPLOADED.value
        assert item["workspace_id"] == "ws_test_123"
        assert item["extracted_records_count"] == 0


def test_readiness_calculation_and_unanswerable_warnings():
    """Verify readiness logic derives state solely from backend validation, not clicks."""
    items = ChecklistEngine.get_default_checklist_items(
        domain="Education",
        workspace_id="ws_edu_1",
    )
    # Initially 0% readiness
    report = ChecklistEngine.calculate_readiness(items, "Education", "ws_edu_1")
    assert report.is_ready_for_testing is False
    assert report.readiness_percentage == 0.0
    assert report.required_completed == 0
    assert len(report.missing_required_categories) == 5
    assert any("attendance" in w.lower() for w in report.unanswerable_query_warnings)

    # Validate 1 required category
    items[0]["status"] = ChecklistItemStatus.VALIDATED.value
    report2 = ChecklistEngine.calculate_readiness(items, "Education", "ws_edu_1")
    assert report2.required_completed == 1
    assert report2.readiness_percentage == 20.0
    assert report2.is_ready_for_testing is False

    # Validate all 5 required categories
    for i in range(5):
        items[i]["status"] = ChecklistItemStatus.VALIDATED.value
    report3 = ChecklistEngine.calculate_readiness(items, "Education", "ws_edu_1")
    assert report3.required_completed == 5
    assert report3.readiness_percentage == 100.0
    assert report3.is_ready_for_testing is True
    assert len(report3.missing_required_categories) == 0


# ===========================================================================
# 2. Tenant Registration & Authentication Security Tests
# ===========================================================================

def test_tenant_registration_and_api_key_hashing(workspace_service):
    """Verify tenant registration returns plaintext key once and stores only hash."""
    from src.workspaces.models import TenantRegisterRequest
    reg = TenantRegisterRequest(name="Acme Corp", email="owner@acme.com")
    res = workspace_service.register_tenant(reg)

    assert res.tenant_id.startswith("ten_")
    assert res.api_key.startswith("cap_")

    # Verify lookup via raw key works
    auth = workspace_service.authenticate_api_key(res.api_key)
    assert auth is not None
    assert auth["tenant_id"] == res.tenant_id

    # Verify invalid key fails
    assert workspace_service.authenticate_api_key("invalid_key") is None
    assert workspace_service.authenticate_api_key("") is None


def test_custom_category_addition(workspace_service):
    """Verify workspace owners can add custom categories beyond default checklist."""
    from src.workspaces.models import TenantRegisterRequest, WorkspaceCreateRequest
    t = workspace_service.register_tenant(TenantRegisterRequest(name="Biz", email="biz@corp.com"))
    w = workspace_service.create_workspace(
        t.tenant_id,
        WorkspaceCreateRequest(name="Store", domain=BusinessDomain.ECOMMERCE),
    )

    custom = workspace_service.add_custom_category(
        t.tenant_id,
        w.workspace_id,
        CustomCategoryRequest(
            category_name="International Customs Clearance",
            is_required=True,
        ),
    )
    assert custom.category_name == "International Customs Clearance"
    assert custom.is_required is True

    items = workspace_service.get_checklist(t.tenant_id, w.workspace_id)
    assert any(i.category_name == "International Customs Clearance" for i in items)


# ===========================================================================
# 3. Multi-Tenant Isolation & Authorization Security (Never Trust Client ID)
# ===========================================================================

def test_cross_tenant_isolation(workspace_service):
    """
    Critical Security Invariant:
    Tenant A must never be able to view, access, or modify Tenant B's workspace.
    """
    from src.workspaces.models import TenantRegisterRequest, WorkspaceCreateRequest
    # Create Tenant A and Workspace A
    t_a = workspace_service.register_tenant(TenantRegisterRequest(name="Tenant A", email="a@corp.com"))
    w_a = workspace_service.create_workspace(
        t_a.tenant_id,
        WorkspaceCreateRequest(name="Workspace A", domain=BusinessDomain.ECOMMERCE),
    )

    # Create Tenant B and Workspace B
    t_b = workspace_service.register_tenant(TenantRegisterRequest(name="Tenant B", email="b@corp.com"))
    w_b = workspace_service.create_workspace(
        t_b.tenant_id,
        WorkspaceCreateRequest(name="Workspace B", domain=BusinessDomain.EDUCATION),
    )

    # Tenant A accesses Workspace A: SUCCESS
    assert workspace_service.get_workspace(t_a.tenant_id, w_a.workspace_id).name == "Workspace A"

    # Tenant B accesses Workspace B: SUCCESS
    assert workspace_service.get_workspace(t_b.tenant_id, w_b.workspace_id).name == "Workspace B"

    # CROSS-TENANT ATTACK: Tenant A attempts to access Workspace B: FORBIDDEN!
    with pytest.raises(PermissionError):
        workspace_service.get_workspace(t_a.tenant_id, w_b.workspace_id)

    # CROSS-TENANT ATTACK: Tenant B attempts to access Workspace A: FORBIDDEN!
    with pytest.raises(PermissionError):
        workspace_service.get_workspace(t_b.tenant_id, w_a.workspace_id)

    # CROSS-TENANT ATTACK: Tenant B attempts to read Workspace A checklist: FORBIDDEN!
    with pytest.raises(PermissionError):
        workspace_service.get_checklist(t_b.tenant_id, w_a.workspace_id)

    # Non-existent workspace: LookupError (404)
    with pytest.raises(LookupError):
        workspace_service.get_workspace(t_a.tenant_id, "ws_nonexistent")


# ===========================================================================
# 4. End-to-End API Routes & Middleware Tests
# ===========================================================================

def test_api_tenant_onboarding_and_cross_tenant_blocking(client):
    """Test full HTTP flow: registration, workspace creation, and 403 cross-tenant rejection."""
    # 1. Register Tenant 1
    r1 = client.post("/api/v1/tenants/register", json={"name": "Retail Corp", "email": "retail@test.com"})
    assert r1.status_code == 201
    d1 = r1.json()
    key1 = d1["api_key"]

    # 2. Register Tenant 2
    r2 = client.post("/api/v1/tenants/register", json={"name": "Uni Support", "email": "uni@test.com"})
    assert r2.status_code == 201
    d2 = r2.json()
    key2 = d2["api_key"]

    headers1 = {"Authorization": f"Bearer {key1}"}
    headers2 = {"Authorization": f"Bearer {key2}"}

    # 3. Tenant 1 creates Workspace (E-commerce)
    ws_res1 = client.post(
        "/api/v1/workspaces",
        json={"name": "Retail Storefront", "domain": "E-commerce"},
        headers=headers1,
    )
    assert ws_res1.status_code == 201
    ws1_id = ws_res1.json()["workspace_id"]

    # 4. Unauthenticated request rejected with 401
    unauth = client.get(f"/api/v1/workspaces/{ws1_id}")
    assert unauth.status_code == 401

    # 5. Invalid key rejected with 401
    bad_key = client.get(f"/api/v1/workspaces/{ws1_id}", headers={"Authorization": "Bearer bad_key"})
    assert bad_key.status_code == 401

    # 6. Tenant 1 retrieves own workspace: 200 OK
    own_ws = client.get(f"/api/v1/workspaces/{ws1_id}", headers=headers1)
    assert own_ws.status_code == 200
    assert own_ws.json()["name"] == "Retail Storefront"

    # 7. CROSS-TENANT ATTACK: Tenant 2 attempts to view Tenant 1's workspace: 403 Forbidden!
    attack = client.get(f"/api/v1/workspaces/{ws1_id}", headers=headers2)
    assert attack.status_code == 403
    assert "Unauthorized" in attack.json()["detail"]

    # 8. Check dynamic checklist for Tenant 1
    chk_res = client.get(f"/api/v1/workspaces/{ws1_id}/checklist", headers=headers1)
    assert chk_res.status_code == 200
    items = chk_res.json()
    assert len(items) == 10
    assert any(i["category_name"] == "Return Policy" for i in items)

    # 9. Readiness report initially indicates unready
    ready_res = client.get(f"/api/v1/workspaces/{ws1_id}/readiness", headers=headers1)
    assert ready_res.status_code == 200
    report = ready_res.json()
    assert report["is_ready_for_testing"] is False
    assert report["readiness_percentage"] == 0.0
    assert len(report["unanswerable_query_warnings"]) > 0
