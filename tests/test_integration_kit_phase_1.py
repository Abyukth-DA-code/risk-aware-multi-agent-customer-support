"""
Comprehensive Test Suite for Phase 1: Integration Kit Download & Credential Isolation
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Verifies:
1. GET /api/v1/integration/download-kit (Public Default Kit)
2. GET /api/v1/workspaces/{workspace_id}/integration/download-kit (Authorized Workspace Kit)
3. ZIP archive integrity and exactly 5 required files
4. Strict tenant isolation (Tenant B cannot download Tenant A's kit)
5. Credential isolation (Widget tokens rejected for admin download; Master API key never leaked in ZIP)
6. Clean execution of _ensure_demo_workspace and DocumentIngestionPipeline
"""

import io
from pathlib import Path
import zipfile

import pytest
from starlette.testclient import TestClient

from src.api.main import _ensure_demo_workspace, app
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.workspaces.db import WorkspaceDB
from src.workspaces.integration_kit import generate_integration_kit_zip
from src.workspaces.models import BusinessDomain, TenantRegisterRequest, WorkspaceCreateRequest
from src.workspaces.service import WorkspaceService


@pytest.fixture
def temp_db(tmp_path):
    """Provides isolated, temporary SQLite database for tests."""
    db_file = tmp_path / "test_tenants_kit.db"
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


@pytest.fixture
def test_setup(workspace_service):
    """Sets up two distinct tenants with workspaces to test tenant and credential isolation."""
    # Tenant 1: Apex Retail
    tenant_1 = workspace_service.register_tenant(
        TenantRegisterRequest(name="Apex Retail Corp", email="owner@apexretail.com")
    )
    ws_1 = workspace_service.create_workspace(
        authenticated_tenant_id=tenant_1.tenant_id,
        req=WorkspaceCreateRequest(
            name="Apex Flagship Store",
            domain=BusinessDomain.ECOMMERCE,
            description="High-volume online retail store",
        ),
    )

    # Tenant 2: Beta Academy
    tenant_2 = workspace_service.register_tenant(
        TenantRegisterRequest(name="Beta Academy Inc", email="admin@betaacademy.edu")
    )
    ws_2 = workspace_service.create_workspace(
        authenticated_tenant_id=tenant_2.tenant_id,
        req=WorkspaceCreateRequest(
            name="Online Learning Portal",
            domain=BusinessDomain.EDUCATION,
            description="Higher education student support",
        ),
    )

    return {
        "tenant_1": tenant_1,
        "ws_1": ws_1,
        "tenant_2": tenant_2,
        "ws_2": ws_2,
    }


# ===========================================================================
# 1. Unit Tests for ZIP Generation & Content Guarantees
# ===========================================================================

def test_generate_integration_kit_zip_default():
    """Verifies that the default integration kit ZIP builds with exactly the 5 required files and placeholders."""
    zip_bytes = generate_integration_kit_zip()
    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0

    bio = io.BytesIO(zip_bytes)
    assert zipfile.is_zipfile(bio)

    with zipfile.ZipFile(bio, "r") as zf:
        namelist = sorted(zf.namelist())
        expected_files = sorted([
            "widget.js",
            "embed_example.html",
            "INTEGRATION_GUIDE.md",
            "api_client_example.py",
            "api_client_example.js",
        ])
        assert namelist == expected_files

        # Check widget.js content
        widget_code = zf.read("widget.js").decode("utf-8")
        assert "AutonomousSupportWidget" in widget_code
        assert "SupportChatWidget" in widget_code

        # Check embed_example.html content
        html_code = zf.read("embed_example.html").decode("utf-8")
        assert "YOUR_WORKSPACE_ID" in html_code
        assert "YOUR_WIDGET_TOKEN" in html_code
        # Credential isolation: Master API key prefix must NEVER appear in HTML
        assert "cap_" not in html_code

        # Check INTEGRATION_GUIDE.md content
        guide_text = zf.read("INTEGRATION_GUIDE.md").decode("utf-8")
        assert "Autonomous Multi-Agent Customer Support" in guide_text
        assert "wtk_" in guide_text
        assert "React / Next.js" in guide_text

        # Check API client examples
        py_code = zf.read("api_client_example.py").decode("utf-8")
        assert "YOUR_WORKSPACE_ID" in py_code
        assert "YOUR_WIDGET_TOKEN" in py_code
        assert "cap_" not in py_code

        js_code = zf.read("api_client_example.js").decode("utf-8")
        assert "YOUR_WORKSPACE_ID" in js_code
        assert "YOUR_WIDGET_TOKEN" in js_code
        assert "cap_" not in js_code


def test_generate_integration_kit_zip_customized():
    """Verifies that the customized integration kit uses the scoped widget token and workspace ID."""
    ws_id = "ws_test_ecommerce_101"
    token = "wtk_test_scoped_token_987"
    name = "Apex Retail Flagship"
    domain = "E-commerce"

    zip_bytes = generate_integration_kit_zip(
        workspace_id=ws_id,
        widget_token=token,
        workspace_name=name,
        domain=domain,
        api_base="http://localhost:8000",
    )
    bio = io.BytesIO(zip_bytes)
    assert zipfile.is_zipfile(bio)

    with zipfile.ZipFile(bio, "r") as zf:
        html_code = zf.read("embed_example.html").decode("utf-8")
        assert ws_id in html_code
        assert token in html_code
        assert name in html_code
        assert "cap_" not in html_code  # Master key must NEVER leak

        py_code = zf.read("api_client_example.py").decode("utf-8")
        assert f'WORKSPACE_ID = "{ws_id}"' in py_code
        assert f'WIDGET_TOKEN = "{token}"' in py_code
        assert "cap_" not in py_code

        js_code = zf.read("api_client_example.js").decode("utf-8")
        assert f'const WORKSPACE_ID = "{ws_id}";' in js_code
        assert f'const WIDGET_TOKEN = "{token}";' in js_code
        assert "cap_" not in js_code


# ===========================================================================
# 2. Public Default Endpoint Tests
# ===========================================================================

def test_get_default_integration_kit_endpoint(client):
    """GET /api/v1/integration/download-kit returns 200 OK and valid downloadable ZIP."""
    resp = client.get("/api/v1/integration/download-kit")
    assert resp.status_code == 200
    assert "application/zip" in resp.headers.get("content-type", "")
    content_disp = resp.headers.get("content-disposition", "")
    assert "attachment" in content_disp
    assert "capstone-integration-kit.zip" in content_disp

    # Verify the downloaded payload is a valid ZIP
    bio = io.BytesIO(resp.content)
    assert zipfile.is_zipfile(bio)
    with zipfile.ZipFile(bio, "r") as zf:
        namelist = sorted(zf.namelist())
        assert len(namelist) == 5
        assert "widget.js" in namelist
        assert "embed_example.html" in namelist
        assert "INTEGRATION_GUIDE.md" in namelist
        assert "api_client_example.py" in namelist
        assert "api_client_example.js" in namelist

        # Verify zero credential leakage
        combined_text = (
            zf.read("embed_example.html").decode("utf-8")
            + zf.read("api_client_example.py").decode("utf-8")
            + zf.read("api_client_example.js").decode("utf-8")
        )
        assert "cap_" not in combined_text


# ===========================================================================
# 3. Workspace-Specific Endpoint Authentication & Authorization Tests
# ===========================================================================

def test_download_workspace_kit_unauthenticated(client, test_setup):
    """Unauthenticated request must be rejected with 401 Unauthorized."""
    ws_id = test_setup["ws_1"].workspace_id
    resp = client.get(f"/api/v1/workspaces/{ws_id}/integration/download-kit")
    assert resp.status_code == 401
    assert "Authentication required" in resp.json()["detail"]


def test_download_workspace_kit_invalid_key(client, test_setup):
    """Request with invalid API key must be rejected with 401 Unauthorized."""
    ws_id = test_setup["ws_1"].workspace_id
    headers = {"Authorization": "Bearer cap_invalid_fake_key_123456"}
    resp = client.get(f"/api/v1/workspaces/{ws_id}/integration/download-kit", headers=headers)
    assert resp.status_code == 401
    assert "Invalid, expired, or revoked API key" in resp.json()["detail"]


def test_download_workspace_kit_rejected_for_widget_token(client, test_setup):
    """
    Security Invariant: Public widget tokens (wtk_...) must NEVER be allowed
    to authenticate administrative/workspace endpoints.
    """
    ws_id = test_setup["ws_1"].workspace_id
    headers = {"Authorization": "Bearer wtk_public_client_token_123"}
    resp = client.get(f"/api/v1/workspaces/{ws_id}/integration/download-kit", headers=headers)
    assert resp.status_code == 403
    assert "Widget tokens cannot be used to authenticate administrative endpoints" in resp.json()["detail"]


def test_download_workspace_kit_tenant_isolation(client, test_setup):
    """
    Tenant Isolation Invariant: Tenant 2 cannot download the integration kit
    for a workspace owned by Tenant 1.
    """
    ws_1_id = test_setup["ws_1"].workspace_id
    tenant_2_key = test_setup["tenant_2"].api_key

    headers = {"Authorization": f"Bearer {tenant_2_key}"}
    resp = client.get(f"/api/v1/workspaces/{ws_1_id}/integration/download-kit", headers=headers)
    assert resp.status_code == 403
    assert "cannot access workspace" in resp.json()["detail"]


def test_download_workspace_kit_not_found(client, test_setup):
    """Non-existent workspace ID must return 404 Not Found."""
    tenant_1_key = test_setup["tenant_1"].api_key
    headers = {"Authorization": f"Bearer {tenant_1_key}"}
    resp = client.get("/api/v1/workspaces/ws_non_existent_9999/integration/download-kit", headers=headers)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_download_workspace_kit_authorized_success(client, test_setup):
    """
    Authorized business owner successfully downloads their workspace integration kit
    pre-configured with their scoped widget token, domain, and workspace ID.
    """
    ws_1 = test_setup["ws_1"]
    tenant_1 = test_setup["tenant_1"]

    headers = {"Authorization": f"Bearer {tenant_1.api_key}"}
    resp = client.get(f"/api/v1/workspaces/{ws_1.workspace_id}/integration/download-kit", headers=headers)

    assert resp.status_code == 200
    assert "application/zip" in resp.headers.get("content-type", "")
    content_disp = resp.headers.get("content-disposition", "")
    assert f"integration-kit-{ws_1.workspace_id}.zip" in content_disp

    bio = io.BytesIO(resp.content)
    assert zipfile.is_zipfile(bio)

    with zipfile.ZipFile(bio, "r") as zf:
        namelist = sorted(zf.namelist())
        assert len(namelist) == 5

        html_code = zf.read("embed_example.html").decode("utf-8")
        py_code = zf.read("api_client_example.py").decode("utf-8")
        js_code = zf.read("api_client_example.js").decode("utf-8")

        # Must contain the workspace ID
        assert ws_1.workspace_id in html_code
        assert ws_1.workspace_id in py_code
        assert ws_1.workspace_id in js_code

        # Must contain a valid wtk_... token
        assert "wtk_" in html_code
        assert "wtk_" in py_code
        assert "wtk_" in js_code

        # Must NOT contain the tenant's master API key
        assert tenant_1.api_key not in html_code
        assert tenant_1.api_key not in py_code
        assert tenant_1.api_key not in js_code
        assert "cap_" not in html_code


# ===========================================================================
# 4. Demo Workspace Seeding Bug Fix Verification
# ===========================================================================

def test_demo_workspace_seeding_without_import_error(monkeypatch, tmp_path):
    """
    Verifies that _ensure_demo_workspace executes cleanly with DocumentIngestionPipeline
    and does not trigger any ImportError.
    """
    test_db_path = tmp_path / "test_demo_tenants.db"
    monkeypatch.setenv("ENABLE_DEMO_WORKSPACE", "true")
    monkeypatch.setenv("CAPSTONE_ENV", "development")

    # Call _ensure_demo_workspace directly
    _ensure_demo_workspace()

    # Verify that WorkspaceDB can read ten_demo and ws_demo
    db = WorkspaceDB()
    demo_tenant = db.get_tenant("ten_demo")
    assert demo_tenant is not None
    assert demo_tenant["tenant_id"] == "ten_demo"

    demo_ws = db.get_workspace("ws_demo")
    assert demo_ws is not None
    assert demo_ws["workspace_id"] == "ws_demo"
    assert demo_ws["widget_token"] == "wtk_demo_public_token"
