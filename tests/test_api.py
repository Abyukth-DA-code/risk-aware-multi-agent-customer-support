"""
Automated Test Suite for FastAPI Layer
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation

Destination: tests/test_api.py
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify GET / returns service information and endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["documentation"] == "/docs"
    assert data["health_check"] == "/health"
    assert data["query_endpoint"] == "/query"


def test_health_endpoint():
    """Verify GET /health inspects subsystem readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert data["components"]["model_service"] == "ready"
    assert "ready" in data["components"]["vector_retriever"]
    assert data["components"]["orchestration_engine"] == "ready"


def test_query_endpoint_valid_request():
    """Verify POST /query performs ML triage, knowledge retrieval, and full orchestration."""
    payload = {
        "query": "Where is my order? It has been delayed for 3 days.",
        "customer_id": "CUST-9021",
        "session_id": "SESS-4412",
        "sensitive_action": False,
        "financial_impact": False,
        "top_k": 2,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Query & Customer Identity
    assert data["query"] == payload["query"]
    assert data["customer_id"] == "CUST-9021"
    assert data["session_id"] == "SESS-4412"

    # Triage Section (M1–M6)
    triage = data["triage"]
    assert triage["domain"] is not None
    assert triage["intent"] is not None
    assert triage["sentiment"] is not None
    assert triage["risk"] is not None
    assert triage["complexity"] is not None
    assert triage["provisional_autonomy_tier"] is not None

    # Evidence Section (RAG)
    evidence = data["evidence"]
    assert evidence["evidence_quality"] in ("Strong", "Adequate", "Insufficient", "Conflicting")
    assert evidence["retrieval_count"] >= 0
    assert isinstance(evidence["chunks"], list)

    # Resolution Section (Integrated Multi-Agent Orchestration)
    resolution = data["resolution"]
    assert resolution["orchestration_ready"] is True
    assert resolution["final_autonomy_tier"] in ("ANSWER", "ORCHESTRATE", "ESCALATE")
    assert isinstance(resolution["final_escalation_required"], bool)
    assert resolution["final_response"] is None
    assert resolution["message"] is not None


def test_query_endpoint_incomplete_workflow_when_orchestration_disabled():
    """Verify POST /query preserves pending resolution when run_orchestration=False."""
    payload = {
        "query": "Where is my order? It has been delayed for 3 days.",
        "customer_id": "CUST-9021",
        "session_id": "SESS-4412",
        "sensitive_action": False,
        "financial_impact": False,
        "top_k": 2,
        "run_orchestration": False,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Evidence is Pending before assessment
    assert data["evidence"]["evidence_quality"] == "Pending"

    # Resolution honestly reports pending integration
    resolution = data["resolution"]
    assert resolution["orchestration_ready"] is False
    assert resolution["final_response"] is None
    assert "pending integration" in resolution["message"].lower()
    assert resolution["final_autonomy_tier"] is None


def test_query_validation_empty_string():
    """Verify POST /query rejects empty string with HTTP 422."""
    payload = {"query": ""}
    response = client.post("/query", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


def test_query_validation_whitespace_only():
    """Verify POST /query rejects whitespace-only string with HTTP 422."""
    payload = {"query": "      "}
    response = client.post("/query", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert "whitespace" in data["detail"].lower()


def test_query_validation_missing_query():
    """Verify POST /query rejects missing query with HTTP 422."""
    payload = {"customer_id": "CUST-101"}
    response = client.post("/query", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


def test_query_validation_top_k_bounds():
    """Verify top_k must be between 1 and 10."""
    # top_k too large
    res_large = client.post("/query", json={"query": "test", "top_k": 50})
    assert res_large.status_code == 422

    # top_k too small
    res_small = client.post("/query", json={"query": "test", "top_k": 0})
    assert res_small.status_code == 422
