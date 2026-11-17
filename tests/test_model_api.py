"""
Automated Test Suite for Model-Specific Prediction API Endpoints (M1–M6)
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation

Destination: tests/test_model_api.py
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, services
from src.orchestration.model_service import ModelService

client = TestClient(app)


# ---------------------------------------------------------------------------
# Root and Discovery Tests
# ---------------------------------------------------------------------------

def test_root_lists_model_prediction_endpoints():
    """Verify GET / lists all model prediction endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "model_prediction_endpoints" in data
    endpoints = data["model_prediction_endpoints"]
    assert endpoints["m1_intent"] == "/predict/intent"
    assert endpoints["m2_sentiment"] == "/predict/sentiment"
    assert endpoints["m3_risk"] == "/predict/risk"
    assert endpoints["m4_complexity"] == "/predict/complexity"
    assert endpoints["m5_domain"] == "/predict/domain"
    assert endpoints["m6_escalation"] == "/predict/escalation"


# ---------------------------------------------------------------------------
# M1: Intent Classifier Tests
# ---------------------------------------------------------------------------

def test_predict_m1_intent_success():
    """Verify POST /predict/intent returns intent prediction and uncalibrated SVM score note."""
    payload = {"query": "Where is my package? The tracking number shows no movement."}
    response = client.post("/predict/intent", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "M1_Intent_Classifier"
    assert isinstance(data["prediction"], str) and len(data["prediction"]) > 0
    assert data["intent"] == data["prediction"]
    assert isinstance(data["confidence"], (int, float))
    assert data["confidence"] > 0
    assert data["confidence_type"] == "uncalibrated_score"
    assert "uncalibrated" in data["confidence_note"].lower()
    assert "linearsvc" in data["confidence_note"].lower()
    assert data["query"] == payload["query"]
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# M2: Sentiment Analyzer Tests
# ---------------------------------------------------------------------------

def test_predict_m2_sentiment_success():
    """Verify POST /predict/sentiment returns calibrated probability and valid sentiment class."""
    payload = {"query": "I am absolutely thrilled with the fast resolution and kind agent!"}
    response = client.post("/predict/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "M2_Sentiment_Analyzer"
    assert data["prediction"] in ["Positive", "Neutral", "Negative"]
    assert data["sentiment"] == data["prediction"]
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["confidence_type"] == "calibrated_probability"
    assert "logisticregression" in data["confidence_note"].lower()
    assert data["query"] == payload["query"]
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# M3: Risk Tier Classifier Tests
# ---------------------------------------------------------------------------

def test_predict_m3_risk_success():
    """Verify POST /predict/risk returns calibrated risk tier and full class distribution."""
    payload = {
        "query": "I request an immediate chargeback and refund of 850 dollars to my card.",
        "domain": "E-commerce",
        "sensitive_action": True,
        "financial_impact": True,
        "action_type": "Transaction",
    }
    response = client.post("/predict/risk", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "M3_Risk_Classifier"
    assert data["prediction"] in ["Low", "Medium", "High"]
    assert data["risk"] == data["prediction"]
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["confidence_type"] == "calibrated_probability"
    assert "randomforestclassifier" in data["confidence_note"].lower()

    # Verify probability distribution structure
    probas = data["class_probabilities"]
    assert isinstance(probas, dict)
    for c in ["Low", "Medium", "High"]:
        assert c in probas
        assert 0.0 <= probas[c] <= 1.0
    assert round(sum(probas.values()), 2) == 1.0
    assert data["query"] == payload["query"]


# ---------------------------------------------------------------------------
# M4: Complexity Classifier Tests
# ---------------------------------------------------------------------------

def test_predict_m4_complexity_success():
    """Verify POST /predict/complexity returns calibrated complexity tier and class distribution."""
    payload = {
        "query": "How do I configure OAuth2 multi-tenant authentication in my account portal?",
        "domain": "Education",
    }
    response = client.post("/predict/complexity", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "M4_Complexity_Classifier"
    assert data["prediction"] in ["Low", "Medium", "High"]
    assert data["complexity"] == data["prediction"]
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["confidence_type"] == "calibrated_probability"
    assert "randomforestclassifier" in data["confidence_note"].lower()

    probas = data["class_probabilities"]
    assert isinstance(probas, dict)
    for c in ["Low", "Medium", "High"]:
        assert c in probas
        assert 0.0 <= probas[c] <= 1.0
    assert round(sum(probas.values()), 2) == 1.0


# ---------------------------------------------------------------------------
# M5: Domain Classifier Tests
# ---------------------------------------------------------------------------

def test_predict_m5_domain_success():
    """Verify POST /predict/domain returns predicted domain and uncalibrated SVM score note."""
    payload = {"query": "How do I enroll my team in the advanced machine learning certificate course?"}
    response = client.post("/predict/domain", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "M5_Domain_Classifier"
    assert data["prediction"] in ["E-commerce", "Education", "Small Business"]
    assert data["domain"] == data["prediction"]
    assert isinstance(data["confidence"], (int, float))
    assert data["confidence"] > 0
    assert data["confidence_type"] == "uncalibrated_score"
    assert "uncalibrated" in data["confidence_note"].lower()
    assert "linearsvc" in data["confidence_note"].lower()
    assert data["query"] == payload["query"]


# ---------------------------------------------------------------------------
# M6: Escalation & Autonomy Engine Tests
# ---------------------------------------------------------------------------

def test_predict_m6_escalation_success():
    """Verify POST /predict/escalation returns provisional autonomy tier and escalation flag."""
    payload = {
        "query": "My billing has been disputed and I want an immediate executive escalation.",
        "domain": "E-commerce",
        "risk": "High",
        "complexity": "High",
        "evidence_quality": "Weak",
        "verification_required": "Yes",
        "sensitive_action": "Yes",
        "financial_impact": "Yes",
        "sentiment": "Negative",
    }
    response = client.post("/predict/escalation", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "M6_Escalation_Autonomy_Classifier"
    assert data["prediction"] in ["ANSWER", "ORCHESTRATE", "ESCALATE"]
    assert data["autonomy_tier"] == data["prediction"]
    assert data["escalation_required"] in ["Yes", "No"]
    assert isinstance(data["confidence"], (int, float))
    assert data["confidence_type"] == "calibrated_probability"
    assert "decisiontreeclassifier" in data["confidence_note"].lower()
    assert isinstance(data["class_probabilities"], dict)


def test_predict_m6_autonomy_alias_endpoint():
    """Verify POST /predict/autonomy alias routes correctly to M6."""
    payload = {"query": "What are your business hours?"}
    response = client.post("/predict/autonomy", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "M6_Escalation_Autonomy_Classifier"
    assert data["prediction"] in ["ANSWER", "ORCHESTRATE", "ESCALATE"]


# ---------------------------------------------------------------------------
# Input Validation Tests (HTTP 422) for all 6 endpoints
# ---------------------------------------------------------------------------

MODEL_ENDPOINTS = [
    "/predict/intent",
    "/predict/sentiment",
    "/predict/risk",
    "/predict/complexity",
    "/predict/domain",
    "/predict/escalation",
]


@pytest.mark.parametrize("endpoint", MODEL_ENDPOINTS)
def test_validation_empty_query_rejected(endpoint):
    """Verify all 6 model endpoints reject empty query string with HTTP 422."""
    response = client.post(endpoint, json={"query": ""})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


@pytest.mark.parametrize("endpoint", MODEL_ENDPOINTS)
def test_validation_whitespace_only_query_rejected(endpoint):
    """Verify all 6 model endpoints reject whitespace-only query with HTTP 422."""
    response = client.post(endpoint, json={"query": "    \t \n  "})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert "whitespace" in data["detail"].lower()


@pytest.mark.parametrize("endpoint", MODEL_ENDPOINTS)
def test_validation_missing_query_field_rejected(endpoint):
    """Verify all 6 model endpoints reject missing query field with HTTP 422."""
    response = client.post(endpoint, json={"domain": "E-commerce"})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


# ---------------------------------------------------------------------------
# Error Handling & Non-leaking Internal Errors (HTTP 500)
# ---------------------------------------------------------------------------

def test_model_error_handling_non_leaking(monkeypatch):
    """Verify model execution exceptions produce safe HTTP 500 without leaking stack traces."""
    # Obtain model service instance and monkeypatch predict_intent to simulate runtime error
    model_svc = services.model_service or ModelService.get_instance()

    def mock_broken_predict(*args, **kwargs):
        raise RuntimeError("Simulated pipeline math error or hardware fault")

    monkeypatch.setattr(model_svc, "predict_intent", mock_broken_predict)

    non_raising_client = TestClient(app, raise_server_exceptions=False)
    response = non_raising_client.post("/predict/intent", json={"query": "Normal customer query."})
    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "Internal Server Error"
    # Ensure safe message and no raw python traceback is leaked to client
    assert "Traceback" not in data["detail"]
    assert "hardware fault" not in data["detail"]
    assert "internal error" in data["detail"].lower()
