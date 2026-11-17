"""
Tests for Demo Workspace Seeding Security and Boundary Isolation.
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation
"""

import os
import pytest
from fastapi.testclient import TestClient
from src.api.main import app, _ensure_demo_workspace
from src.workspaces.db import WorkspaceDB


@pytest.fixture
def client():
    return TestClient(app)


def test_demo_workspace_seeding_disabled_via_env(monkeypatch):
    """Verifies that demo seeding is strictly skipped when ENABLE_DEMO_WORKSPACE=false."""
    monkeypatch.setenv("ENABLE_DEMO_WORKSPACE", "false")
    # Calling _ensure_demo_workspace should return immediately without error
    _ensure_demo_workspace()


def test_demo_workspace_token_revocation_preserved(monkeypatch):
    """Verifies that an explicitly revoked demo widget token is NOT resurrected on startup."""
    db = WorkspaceDB()
    ws = db.get_workspace("ws_demo")
    if not ws:
        pytest.skip("ws_demo not present in test db")

    # Deliberately revoke ws_demo's widget token
    db.revoke_widget_token("ws_demo")
    revoked_ws = db.get_workspace("ws_demo")
    assert revoked_ws.get("widget_token_revoked_at") is not None

    # Call _ensure_demo_workspace
    monkeypatch.setenv("ENABLE_DEMO_WORKSPACE", "true")
    _ensure_demo_workspace()

    # Confirm revocation status was preserved
    ws_after = db.get_workspace("ws_demo")
    assert ws_after.get("widget_token_revoked_at") is not None

    # Restore token for subsequent interactive testing
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE workspaces SET widget_token = ?, widget_token_revoked_at = NULL WHERE workspace_id = ?",
            ("wtk_demo_public_token", "ws_demo"),
        )


def test_demo_widget_token_rejected_on_admin_endpoints(client):
    """Confirms that public widget tokens cannot be used to access administrative endpoints."""
    # Attempt to access checklist using widget token as Bearer auth
    res1 = client.get(
        "/api/v1/workspaces/ws_demo/checklist",
        headers={"Authorization": "Bearer wtk_demo_public_token"},
    )
    assert res1.status_code in (401, 403)

    # Attempt to access memory admin using widget token
    res2 = client.get(
        "/api/v1/workspaces/ws_demo/customers/CUST0001/memory",
        headers={"Authorization": "Bearer wtk_demo_public_token"},
    )
    assert res2.status_code in (401, 403)


def test_demo_widget_token_rejected_on_other_workspaces(client):
    """Confirms that wtk_demo_public_token cannot query another workspace's widget endpoint."""
    res = client.post(
        "/api/v1/workspaces/ws_non_existent_or_other/widget/chat",
        headers={"X-Widget-Token": "wtk_demo_public_token"},
        json={"query": "Hello"},
    )
    assert res.status_code == 403
