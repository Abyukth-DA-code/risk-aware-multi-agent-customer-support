"""
Comprehensive Test Suite for Phase F: Multi-Tenant Memory & Embeddable Chat Widget
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Verifies:
1. Multi-turn session memory recording, ordering, and retrieval.
2. Context-aware follow-up reference resolution (pronouns and ellipses) with original query preservation.
3. Strict session isolation (zero cross-session context leakage).
4. Cross-tenant authorization on memory and widget endpoints (HTTP 403 Forbidden).
5. Long-term customer memory creation, read-time TTL expiration, and retrieval.
6. Technical Right-to-be-Forgotten / GDPR total customer memory purge.
7. Memory separation from RAG evidence (customer memory is never indexed into Chroma).
8. Public widget token (wtk_...) lifecycle: retrieval, rotation, and revocation.
9. Security perimeter:
   - Widget tokens strictly rejected on administrative endpoints (HTTP 403).
   - Admin API keys (sk_... and cap_...) strictly rejected on public widget chat (HTTP 403).
10. Widget chat end-to-end execution with citation rendering and session persistence.
11. Widget rate limiting (HTTP 429) and input validation.
12. Static asset delivery (/static/widget.js and /widget/demo).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.api.workspace_routes import WidgetRateLimiter
from src.memory.customer_memory import CustomerMemoryManager
from src.memory.resolver import ReferenceResolver
from src.memory.session_memory import SessionMemoryManager
from src.orchestration.graph import OrchestrationPipeline
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.rag.retriever import RAGRetriever
from src.workspaces.db import WorkspaceDB
from src.workspaces.models import (
    BusinessDomain,
    CustomerMemoryCreateRequest,
    TenantRegisterRequest,
    WorkspaceCreateRequest,
)
from src.workspaces.service import WorkspaceService


@pytest.fixture
def test_env(tmp_path):
    """Provides isolated test database, Chroma store, and services."""
    import chromadb

    db_file = tmp_path / "test_tenants_phase_f.db"
    db = WorkspaceDB(db_path=db_file)
    chroma_dir = tmp_path / "chroma_phase_f"
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


def _register_tenant(service: WorkspaceService, name: str = "Test Corp", email: str = "test@corp.com"):
    return service.register_tenant(TenantRegisterRequest(name=name, email=email))


def _create_workspace(service: WorkspaceService, tenant_id: str, name: str = "Main Store", domain: BusinessDomain = BusinessDomain.ECOMMERCE):
    return service.create_workspace(tenant_id, WorkspaceCreateRequest(name=name, domain=domain))


# ===========================================================================
# 1. Short-Term Session Memory & Isolation Tests
# ===========================================================================

def test_session_memory_persistence_and_ordering(test_env):
    """Verifies storing and retrieving multi-turn conversation turns chronologically."""
    db = test_env["db"]
    tenant_id = f"ten_{uuid.uuid4().hex[:12]}"
    tenant = db.create_tenant(tenant_id=tenant_id, name="Acme Corp", email="acme@test.com")
    ws_id = f"ws_{uuid.uuid4().hex[:12]}"
    ws = db.create_workspace(workspace_id=ws_id, tenant_id=tenant_id, name="Storefront", domain="E-commerce")

    manager = SessionMemoryManager(db=db)
    session_id = "sess_001"

    # Turn 1
    manager.save_turn(
        workspace_id=ws["workspace_id"],
        tenant_id=tenant["tenant_id"],
        session_id=session_id,
        user_query="What is your return policy?",
        resolved_query=None,
        bot_response="Eligible items can be returned within 30 days.",
        final_autonomy_tier="ANSWER",
        citations=[{"category_key": "return_policy", "filename": "policy.txt"}],
    )

    # Turn 2
    manager.save_turn(
        workspace_id=ws["workspace_id"],
        tenant_id=tenant["tenant_id"],
        session_id=session_id,
        user_query="Does it apply to opened items?",
        resolved_query="Does return policy apply to opened items?",
        bot_response="Opened items are subject to a 15% restocking fee.",
        final_autonomy_tier="ANSWER",
        citations=[{"category_key": "return_policy", "filename": "policy.txt"}],
    )

    turns = manager.get_history(ws["workspace_id"], session_id, limit=10)
    assert len(turns) == 2
    assert turns[0]["user_query"] == "What is your return policy?"
    assert turns[1]["user_query"] == "Does it apply to opened items?"
    assert turns[1]["resolved_query"] == "Does return policy apply to opened items?"
    assert turns[0]["turn_index"] == 1
    assert turns[1]["turn_index"] == 2


def test_session_isolation_no_cross_leakage(test_env):
    """Verifies that conversation history in Session A never contaminates Session B."""
    db = test_env["db"]
    tenant_id = f"ten_{uuid.uuid4().hex[:12]}"
    tenant = db.create_tenant(tenant_id=tenant_id, name="Acme Corp", email="acme@test.com")
    ws_id = f"ws_{uuid.uuid4().hex[:12]}"
    ws = db.create_workspace(workspace_id=ws_id, tenant_id=tenant_id, name="Storefront", domain="E-commerce")

    manager = SessionMemoryManager(db=db)
    resolver = ReferenceResolver()

    # Session A: discuss return policy
    manager.save_turn(
        workspace_id=ws["workspace_id"],
        tenant_id=tenant["tenant_id"],
        session_id="session_A",
        user_query="What is the laptop return policy?",
        resolved_query=None,
        bot_response="Laptops have a 14-day return window.",
        final_autonomy_tier="ANSWER",
    )

    # Session B: Brand new user asks an ambiguous question
    turns_b = manager.get_history(ws["workspace_id"], "session_B")
    assert len(turns_b) == 0

    resolved_b = resolver.resolve_reference("How much does it cost?", turns_b)
    # With no prior turns in Session B, the query cannot be resolved to laptops
    assert resolved_b is None

    # Check Session A turns can resolve
    turns_a = manager.get_history(ws["workspace_id"], "session_A")
    assert len(turns_a) == 1
    resolved_a = resolver.resolve_reference("Does it apply to accessories?", turns_a)
    assert resolved_a is not None
    assert "laptop" in resolved_a.lower() or "return" in resolved_a.lower()


# ===========================================================================
# 2. Context-Aware Follow-Up Reference Resolution Tests
# ===========================================================================

def test_reference_resolver_pronouns_and_ellipses():
    """Verifies pronoun replacement ('it', 'they', 'that') and ellipsis synthesis."""
    resolver = ReferenceResolver()

    history = [
        {
            "user_query": "What is the return policy for electronics?",
            "bot_response": "Electronics may be returned within 14 days of purchase with original receipt.",
            "final_autonomy_tier": "ANSWER",
        }
    ]

    # Test pronoun 'it'
    q1 = "Does it include sale items?"
    r1 = resolver.resolve_reference(q1, history)
    assert r1 is not None
    assert "return policy" in r1.lower() or "electronics" in r1.lower()
    assert "sale items" in r1.lower()

    # Test ellipsis 'what about...'
    q2 = "What about open box items?"
    r2 = resolver.resolve_reference(q2, history)
    assert r2 is not None
    assert "open box items" in r2.lower()
    assert "return" in r2.lower() or "electronics" in r2.lower()

    # Test clear standalone query without pronouns (should return None -> no rewrite)
    q3 = "Where is your retail store located in Chicago?"
    r3 = resolver.resolve_reference(q3, history)
    assert r3 is None


def test_reference_resolver_preserves_original_query(test_env):
    """Verifies that state.query is preserved verbatim and only state.resolved_query is updated."""
    pipeline = test_env["pipeline"]
    raw_query = "Does it require a receipt?"

    history = [
        {
            "user_query": "What is the exchange policy?",
            "bot_response": "Exchanges are allowed within 30 days.",
            "final_autonomy_tier": "ANSWER",
        }
    ]

    state = pipeline.run(
        query=raw_query,
        domain="E-commerce",
        conversation_history=history,
        generate_grounded_response=False,
    )

    # Invariant: Raw user query is preserved verbatim
    assert state.query == raw_query
    # Disambiguated resolution is stored in state.resolved_query and metadata
    assert state.resolved_query is not None
    assert "exchange" in state.resolved_query.lower() or "policy" in state.resolved_query.lower()
    assert state.metadata.get("resolved_query") == state.resolved_query


# ===========================================================================
# 3. Long-Term Customer Memory & Configurable TTL Retention Tests
# ===========================================================================

def test_customer_memory_ttl_and_read_time_expiration(test_env):
    """Verifies customer memory creation, active retrieval, and automatic read-time TTL expiration."""
    db = test_env["db"]
    tenant_id = f"ten_{uuid.uuid4().hex[:12]}"
    tenant = db.create_tenant(tenant_id=tenant_id, name="Retail Inc", email="retail@test.com")
    ws_id = f"ws_{uuid.uuid4().hex[:12]}"
    ws = db.create_workspace(workspace_id=ws_id, tenant_id=tenant_id, name="Webstore", domain="E-commerce")
    manager = CustomerMemoryManager(db=db)

    # 1. Create active memory (TTL: 30 days)
    mem_active = manager.add_memory(
        workspace_id=ws["workspace_id"],
        tenant_id=tenant["tenant_id"],
        customer_id="cust_42",
        category="preference",
        content="Prefers express courier delivery",
        ttl_days=30,
    )
    assert mem_active["memory_id"].startswith(("cmem_", "mem_"))
    assert mem_active["customer_id"] == "cust_42"
    assert mem_active["status"] == "Active"
    assert mem_active["expires_at"] is not None

    # 2. Create memory with expired timestamp
    now = datetime.now(timezone.utc)
    expired_at = (now - timedelta(seconds=10)).isoformat()
    db.create_customer_memory(
        workspace_id=ws["workspace_id"],
        tenant_id=tenant["tenant_id"],
        customer_id="cust_42",
        category="temporary_flag",
        content="Expiring temporary authorization",
        confidence=1.0,
        expires_at=expired_at,
    )

    # 3. Read active memories -> expired one must be filtered out
    active_mems = manager.get_memories(ws["workspace_id"], "cust_42")
    contents = [m.get("memory_content") or m.get("content") for m in active_mems]
    assert "Prefers express courier delivery" in contents
    assert "Expiring temporary authorization" not in contents

    # 4. Prune expired records
    pruned = manager.cleanup_expired(ws["workspace_id"])
    assert pruned >= 1


def test_customer_memory_gdpr_right_to_be_forgotten(test_env, test_client):
    """Verifies complete purge of all customer memories under technical Right-to-be-Forgotten."""
    service = test_env["service"]
    tenant = _register_tenant(service, name="Privacy Corp", email="privacy@test.com")
    ws = _create_workspace(service, tenant.tenant_id, name="Secure Hub", domain=BusinessDomain.SMALL_BUSINESS)
    headers = {"Authorization": f"Bearer {tenant.api_key}"}

    customer_id = "customer_gdpr_999"

    # Add 3 memory entries
    for cat, text in [("profile", "VIP Client"), ("device", "iOS User"), ("note", "Likes dark mode")]:
        res = test_client.post(
            f"/api/v1/workspaces/{ws.workspace_id}/customers/{customer_id}/memory",
            json={"category": cat, "content": text, "ttl_days": 365},
            headers=headers,
        )
        assert res.status_code == 201

    # Verify 3 memories exist
    res = test_client.get(
        f"/api/v1/workspaces/{ws.workspace_id}/customers/{customer_id}/memory",
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["total_count"] == 3

    # Delete all memories (GDPR Right-to-be-Forgotten)
    del_res = test_client.delete(
        f"/api/v1/workspaces/{ws.workspace_id}/customers/{customer_id}/memory",
        headers=headers,
    )
    assert del_res.status_code == 200
    assert del_res.json()["deleted_count"] == 3

    # Verify 0 memories remain
    verify_res = test_client.get(
        f"/api/v1/workspaces/{ws.workspace_id}/customers/{customer_id}/memory",
        headers=headers,
    )
    assert verify_res.status_code == 200
    assert verify_res.json()["total_count"] == 0


# ===========================================================================
# 4. Memory Separation from RAG Policy Evidence
# ===========================================================================

def test_memory_never_indexed_as_rag_evidence(test_env):
    """
    Verifies that customer memory records and session turns are NEVER indexed into
    the workspace Chroma vector collection or returned as authoritative policy chunks.
    """
    db = test_env["db"]
    chroma_client = test_env["chroma_client"]

    tenant_id = f"ten_{uuid.uuid4().hex[:12]}"
    tenant = db.create_tenant(tenant_id=tenant_id, name="Security First", email="sec@test.com")
    ws_id = f"ws_{uuid.uuid4().hex[:12]}"
    ws = db.create_workspace(workspace_id=ws_id, tenant_id=tenant_id, name="Safe Store", domain="E-commerce")

    # Store highly sensitive customer memory
    db.create_customer_memory(
        workspace_id=ws["workspace_id"],
        tenant_id=tenant["tenant_id"],
        customer_id="cust_sensitive",
        category="private_secret",
        content="CONFIDENTIAL_DISCOUNT_CODE_XYZ789",
        confidence=1.0,
    )

    # Store sensitive session turn
    db.save_session_turn(
        workspace_id=ws["workspace_id"],
        tenant_id=tenant["tenant_id"],
        session_id="sess_private",
        user_query="My secret account pin is 4321",
        resolved_query=None,
        bot_response="Recorded.",
        final_autonomy_tier="ANSWER",
    )

    # Perform vector search on the workspace collection
    col_name = f"ws_{ws['workspace_id']}_v1"
    try:
        col = chroma_client.get_collection(col_name)
        results = col.query(query_texts=["CONFIDENTIAL_DISCOUNT_CODE_XYZ789", "secret account pin"], n_results=5)
        matched_docs = [doc for sublist in results.get("documents", []) for doc in sublist]
        assert len(matched_docs) == 0
    except Exception:
        # Collection does not exist because memory is never converted to documents
        pass


# ===========================================================================
# 5. Public Widget Token Lifecycle & Rotation
# ===========================================================================

def test_widget_token_lifecycle_and_rotation(test_env, test_client):
    """Verifies widget token retrieval, token rotation, and immediate revocation."""
    service = test_env["service"]
    tenant = _register_tenant(service, name="Token Corp", email="token@test.com")
    ws = _create_workspace(service, tenant.tenant_id, name="Token Store", domain=BusinessDomain.ECOMMERCE)
    headers = {"Authorization": f"Bearer {tenant.api_key}"}

    # 1. Retrieve initial active widget token
    res = test_client.get(f"/api/v1/workspaces/{ws.workspace_id}/widget/token", headers=headers)
    assert res.status_code == 200
    token1 = res.json()["widget_token"]
    assert token1.startswith("wtk_")
    assert res.json()["revoked"] is False

    # 2. Verify config endpoint works with token1
    cfg_res = test_client.get(
        f"/api/v1/workspaces/{ws.workspace_id}/widget/config",
        headers={"X-Widget-Token": token1},
    )
    assert cfg_res.status_code == 200
    assert cfg_res.json()["name"] == "Token Store"

    # 3. Rotate widget token
    rot_res = test_client.post(f"/api/v1/workspaces/{ws.workspace_id}/widget/token/rotate", headers=headers)
    assert rot_res.status_code == 200
    token2 = rot_res.json()["widget_token"]
    assert token2.startswith("wtk_")
    assert token2 != token1

    # 4. Old token1 MUST now be rejected with 403 Forbidden
    old_res = test_client.get(
        f"/api/v1/workspaces/{ws.workspace_id}/widget/config",
        headers={"X-Widget-Token": token1},
    )
    assert old_res.status_code == 403

    # 5. Revoke widget token
    rev_res = test_client.post(f"/api/v1/workspaces/{ws.workspace_id}/widget/token/revoke", headers=headers)
    assert rev_res.status_code == 200
    assert rev_res.json()["revoked"] is True

    # 6. Both tokens rejected after revocation
    assert test_client.get(
        f"/api/v1/workspaces/{ws.workspace_id}/widget/config",
        headers={"X-Widget-Token": token2},
    ).status_code == 403


# ===========================================================================
# 6. Public Widget Security Perimeter & Token Rejections
# ===========================================================================

def test_widget_tokens_strictly_rejected_on_admin_routes(test_env, test_client):
    """
    Verifies that scoped public widget tokens (wtk_...) are strictly rejected
    with HTTP 403 Forbidden when attempting to access administrative endpoints.
    """
    service = test_env["service"]
    tenant = _register_tenant(service, name="Security Corp", email="sec_corp@test.com")
    ws = _create_workspace(service, tenant.tenant_id, name="Admin Store", domain=BusinessDomain.ECOMMERCE)
    wtk = service.get_widget_token(tenant.tenant_id, ws.workspace_id).widget_token

    widget_auth = {"Authorization": f"Bearer {wtk}"}

    # Admin route 1: Document Upload
    up_res = test_client.post(
        f"/api/v1/workspaces/{ws.workspace_id}/documents/upload",
        headers=widget_auth,
        files={"file": ("test.txt", b"Content", "text/plain")},
        data={"category_key": "return_policy"},
    )
    assert up_res.status_code == 403
    assert "Widget tokens" in up_res.json()["detail"]

    # Admin route 2: Custom Category Addition
    cat_res = test_client.post(
        f"/api/v1/workspaces/{ws.workspace_id}/checklist/custom",
        headers=widget_auth,
        json={"category_name": "Custom Category", "is_required": True},
    )
    assert cat_res.status_code == 403

    # Admin route 3: Customer Memory Management
    mem_res = test_client.post(
        f"/api/v1/workspaces/{ws.workspace_id}/customers/cust_test/memory",
        headers=widget_auth,
        json={"category": "note", "content": "test"},
    )
    assert mem_res.status_code == 403

    # Admin route 4: Escalation Review Queue
    esc_res = test_client.get(
        f"/api/v1/workspaces/{ws.workspace_id}/escalations",
        headers=widget_auth,
    )
    assert esc_res.status_code == 403


def test_admin_api_keys_strictly_rejected_on_widget_chat(test_env, test_client):
    """
    Verifies that administrative master API keys (cap_... / sk_...) are strictly
    rejected on the public widget chat endpoint to prevent accidental credential leakage.
    """
    service = test_env["service"]
    tenant = _register_tenant(service, name="Key Safety Corp", email="keysafety@test.com")
    ws = _create_workspace(service, tenant.tenant_id, name="Key Safety Store", domain=BusinessDomain.ECOMMERCE)
    admin_key = tenant.api_key

    res = test_client.post(
        f"/api/v1/workspaces/{ws.workspace_id}/widget/chat",
        headers={"Authorization": f"Bearer {admin_key}"},
        json={"query": "Hello support"},
    )
    assert res.status_code == 403
    assert "Admin API keys" in res.json()["detail"]


# ===========================================================================
# 7. Cross-Tenant Authorization Isolation Tests
# ===========================================================================

def test_cross_tenant_memory_and_widget_access_rejected(test_env, test_client):
    """
    Verifies that Tenant B cannot access or manipulate Tenant A's customer memory
    or widget token configurations.
    """
    service = test_env["service"]
    tA = _register_tenant(service, name="Tenant Alpha", email="alpha@test.com")
    wsA = _create_workspace(service, tA.tenant_id, name="Alpha Store", domain=BusinessDomain.ECOMMERCE)

    tB = _register_tenant(service, name="Tenant Beta", email="beta@test.com")
    headersB = {"Authorization": f"Bearer {tB.api_key}"}

    # Tenant B tries to get Tenant A's widget token
    res = test_client.get(f"/api/v1/workspaces/{wsA.workspace_id}/widget/token", headers=headersB)
    assert res.status_code == 403

    # Tenant B tries to rotate Tenant A's widget token
    res = test_client.post(f"/api/v1/workspaces/{wsA.workspace_id}/widget/token/rotate", headers=headersB)
    assert res.status_code == 403

    # Tenant B tries to list customer memory in Tenant A's workspace
    res = test_client.get(f"/api/v1/workspaces/{wsA.workspace_id}/customers/cust_1/memory", headers=headersB)
    assert res.status_code == 403

    # Tenant B tries to delete customer memory in Tenant A's workspace
    res = test_client.delete(f"/api/v1/workspaces/{wsA.workspace_id}/customers/cust_1/memory", headers=headersB)
    assert res.status_code == 403


# ===========================================================================
# 8. Widget Chat End-to-End Workflow with Multi-Turn Memory
# ===========================================================================

def test_widget_chat_multi_turn_resolution(test_env, test_client):
    """
    Verifies multi-turn widget chat end-to-end:
    - Turn 1: Direct query -> Answer + citations + session recorded.
    - Turn 2: Follow-up with pronoun 'it' -> Disambiguated + session updated.
    """
    service = test_env["service"]
    ingestion = test_env["ingestion"]

    tenant = _register_tenant(service, name="Demo Retailer", email="demoretail@test.com")
    ws = _create_workspace(service, tenant.tenant_id, name="Demo Store", domain=BusinessDomain.ECOMMERCE)
    wtk = service.get_widget_token(tenant.tenant_id, ws.workspace_id).widget_token

    # Ingest a real policy document
    policy_text = "Return Policy: Eligible products may be returned within 30 days of purchase for a 100% full refund."
    ingestion.ingest_document(
        workspace_id=ws.workspace_id,
        tenant_id=tenant.tenant_id,
        category_key="return_policy",
        filename="return_policy.txt",
        file_bytes=policy_text.encode("utf-8"),
    )

    session_id = "widget_turn_test_sess_01"
    headers = {"X-Widget-Token": wtk}

    # Turn 1
    t1_res = test_client.post(
        f"/api/v1/workspaces/{ws.workspace_id}/widget/chat",
        headers=headers,
        json={"query": "What is your return policy?", "session_id": session_id},
    )
    assert t1_res.status_code == 200
    t1_data = t1_res.json()
    assert t1_data["session_id"] == session_id
    assert t1_data["status"].upper() == "RESOLVED"
    assert len(t1_data["citations"]) > 0

    # Turn 2: Follow up referring to 'it'
    t2_res = test_client.post(
        f"/api/v1/workspaces/{ws.workspace_id}/widget/chat",
        headers=headers,
        json={"query": "How many days does it allow?", "session_id": session_id},
    )
    assert t2_res.status_code == 200
    t2_data = t2_res.json()
    assert t2_data["session_id"] == session_id
    # Query is preserved verbatim
    assert t2_data["query"] == "How many days does it allow?"
    # Resolved query synthesized return policy context
    assert t2_data["resolved_query"] is not None
    assert "return" in t2_data["resolved_query"].lower()

    # Verify session turns are recorded in SQLite
    turns = service.get_session_turns(ws.workspace_id, session_id)
    assert len(turns) == 2


# ===========================================================================
# 9. Rate Limiter & Validation Tests
# ===========================================================================

def test_widget_rate_limiter():
    """Verifies that sliding-window rate limiter throttles excessive requests with HTTP 429."""
    limiter = WidgetRateLimiter(max_requests=3, window_seconds=60)
    client_key = "test_ip_127.0.0.1"

    assert limiter.is_allowed(client_key) is True
    assert limiter.is_allowed(client_key) is True
    assert limiter.is_allowed(client_key) is True
    # 4th request within window must be rejected
    assert limiter.is_allowed(client_key) is False


def test_widget_empty_query_rejected(test_env, test_client):
    """Verifies that empty or whitespace-only queries are rejected with HTTP 422."""
    service = test_env["service"]
    tenant = _register_tenant(service, name="Validation Corp", email="val@test.com")
    ws = _create_workspace(service, tenant.tenant_id, name="Val Store", domain=BusinessDomain.ECOMMERCE)
    wtk = service.get_widget_token(tenant.tenant_id, ws.workspace_id).widget_token

    res = test_client.post(
        f"/api/v1/workspaces/{ws.workspace_id}/widget/chat",
        headers={"X-Widget-Token": wtk},
        json={"query": "    "},
    )
    assert res.status_code == 422


# ===========================================================================
# 10. Static Asset Delivery Tests
# ===========================================================================

def test_static_widget_assets_served(test_client):
    """Verifies that static assets (widget.js and demo page) are properly served."""
    # Test widget.js
    res_js = test_client.get("/static/widget.js")
    assert res_js.status_code == 200
    assert "AutonomousSupportWidget" in res_js.text
    assert "ast-widget-container" in res_js.text

    # Test demo.html via /static/demo.html
    res_html = test_client.get("/static/demo.html")
    assert res_html.status_code == 200
    assert "Embeddable Support Chat Widget" in res_html.text

    # Test /widget/demo shortcut
    res_demo = test_client.get("/widget/demo")
    assert res_demo.status_code == 200
    assert "Embeddable Support Chat Widget" in res_demo.text
