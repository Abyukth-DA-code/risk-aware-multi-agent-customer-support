"""
Pydantic Request and Response Schemas for Customer Support API
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation

Destination: src/api/schemas.py
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    """System health check and component readiness status."""
    status: str = Field(..., description="Overall service status (healthy, degraded, unhealthy)")
    version: str = Field(default="1.0.0", description="API Version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp",
    )
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Individual subsystem readiness indicators",
    )
    environment: str = Field(default="production", description="Operating environment")

    model_config = ConfigDict(extra="ignore")


class QueryRequest(BaseModel):
    """Customer query submission schema."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Customer support inquiry text (cannot be blank)",
    )
    customer_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional customer or account identifier",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional chat session or conversation identifier",
    )
    domain: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Optional explicit domain override (E-commerce, Education, Small Business)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional arbitrary session or customer metadata",
    )
    sensitive_action: Optional[bool] = Field(
        default=False,
        description="Whether this query involves sensitive account or credential actions",
    )
    financial_impact: Optional[bool] = Field(
        default=False,
        description="Whether this query involves monetary transactions, refunds, or fees",
    )
    action_type: Optional[str] = Field(
        default="Information/Support",
        description="Action category (Information/Support, Transaction, Administrative)",
    )
    top_k: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of RAG knowledge passages to retrieve (1 to 10)",
    )
    run_orchestration: Optional[bool] = Field(
        default=True,
        description="Whether to execute downstream evidence assessment and deterministic safety policy gating",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be blank or contain only whitespace.")
        return cleaned

    model_config = ConfigDict(extra="ignore")


class TriageSummary(BaseModel):
    """M1–M6 model triage classification outputs."""
    domain: Optional[str] = None
    domain_confidence: Optional[float] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    risk: Optional[str] = None
    risk_confidence: Optional[float] = None
    risk_class_probabilities: Dict[str, float] = Field(default_factory=dict)
    complexity: Optional[str] = None
    complexity_confidence: Optional[float] = None
    complexity_class_probabilities: Dict[str, float] = Field(default_factory=dict)
    provisional_autonomy_tier: Optional[str] = None
    provisional_escalation_required: Optional[str] = None
    provisional_confidence: Optional[float] = None

    model_config = ConfigDict(extra="ignore")


class ChunkSummary(BaseModel):
    """Summary of a retrieved knowledge base chunk."""
    chunk_id: str
    document_id: str
    domain: str
    title: str
    authority: str
    freshness_status: str
    document_type: str
    retrieval_priority: int
    similarity: float
    chunk_text: str
    rrf_score: Optional[float] = None
    bm25_score: Optional[float] = None

    model_config = ConfigDict(extra="ignore")


class EvidenceSummary(BaseModel):
    """RAG retrieval and knowledge evidence state."""
    evidence_quality: str = Field(default="Pending", description="Assessed evidence quality status")
    retrieval_count: int = Field(default=0, description="Total chunks retrieved")
    retrieval_similarity_score: Optional[float] = Field(
        default=None, description="Peak retrieval similarity score"
    )
    has_outdated_evidence: bool = Field(
        default=False, description="Whether any retrieved chunks are marked Outdated"
    )
    has_conflicting_evidence: bool = Field(
        default=False, description="Whether contradictory policy evidence was detected"
    )
    chunks: List[ChunkSummary] = Field(default_factory=list, description="Retrieved knowledge passages")

    model_config = ConfigDict(extra="ignore")


class ResolutionSummary(BaseModel):
    """Final resolution or orchestration status."""
    status: str = Field(..., description="Workflow execution status")
    orchestration_ready: bool = Field(
        default=False,
        description="Whether the full multi-agent resolution graph is integrated and executed",
    )
    final_autonomy_tier: Optional[str] = Field(
        default=None, description="Definitive autonomy gate decision"
    )
    final_escalation_required: Optional[bool] = Field(
        default=None, description="Whether human escalation was mandated"
    )
    final_response: Optional[str] = Field(
        default=None, description="Final generated answer (None if orchestration is pending)"
    )
    message: str = Field(..., description="Human-readable status or guidance message")
    safety_rule_triggered: Optional[str] = None
    safety_override: bool = False
    decision_source: str = "Pending"
    final_confidence: Optional[float] = None
    decision_rationale: Optional[str] = None
    escalation_reason: Optional[str] = None
    escalation_dossier: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


class QueryResponse(BaseModel):
    """Unified API response for POST /query."""
    query: str
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    triage: TriageSummary
    evidence: EvidenceSummary
    resolution: ResolutionSummary
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 response timestamp",
    )

    model_config = ConfigDict(extra="ignore")


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    error: str = Field(..., description="Error classification")
    detail: str = Field(..., description="Safe human-readable explanation")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 error timestamp",
    )

    model_config = ConfigDict(extra="ignore")


# ===========================================================================
# Model-Specific Request Schemas (M1–M6)
# ===========================================================================

class TextPredictionRequest(BaseModel):
    """Base request schema for text classification models (M1, M2, M5)."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Customer support inquiry text (cannot be blank)",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be blank or contain only whitespace.")
        return cleaned

    model_config = ConfigDict(extra="ignore")


class IntentPredictionRequest(TextPredictionRequest):
    """Request schema for M1 Intent Classifier."""
    pass


class SentimentPredictionRequest(TextPredictionRequest):
    """Request schema for M2 Sentiment Analyzer."""
    pass


class DomainPredictionRequest(TextPredictionRequest):
    """Request schema for M5 Domain Classifier."""
    pass


class RiskPredictionRequest(BaseModel):
    """Request schema for M3 Risk Tier Classifier."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Customer support inquiry text (cannot be blank)",
    )
    domain: Optional[str] = Field(
        default="E-commerce",
        max_length=50,
        description="Domain context (E-commerce, Education, Small Business)",
    )
    sensitive_action: Optional[Union[bool, str]] = Field(
        default=False,
        description="Whether query involves sensitive account or credential actions (bool or 'Yes'/'No')",
    )
    financial_impact: Optional[Union[bool, str]] = Field(
        default=False,
        description="Whether query involves monetary transactions, refunds, or fees (bool or 'Yes'/'No')",
    )
    action_type: Optional[str] = Field(
        default="Information/Support",
        max_length=50,
        description="Action category (Information/Support, Transaction, Administrative)",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be blank or contain only whitespace.")
        return cleaned

    model_config = ConfigDict(extra="ignore")


class ComplexityPredictionRequest(BaseModel):
    """Request schema for M4 Complexity Classifier."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Customer support inquiry text (cannot be blank)",
    )
    domain: Optional[str] = Field(
        default="E-commerce",
        max_length=50,
        description="Domain context (E-commerce, Education, Small Business)",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be blank or contain only whitespace.")
        return cleaned

    model_config = ConfigDict(extra="ignore")


class EscalationPredictionRequest(BaseModel):
    """Request schema for M6 Escalation & Autonomy Engine."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Customer support inquiry text (cannot be blank)",
    )
    domain: Optional[str] = Field(
        default="E-commerce",
        max_length=50,
        description="Domain context (E-commerce, Education, Small Business)",
    )
    risk: Optional[str] = Field(
        default="Low",
        max_length=20,
        description="Operational risk tier (Low, Medium, High)",
    )
    complexity: Optional[str] = Field(
        default="Low",
        max_length=20,
        description="Task complexity tier (Low, Medium, High)",
    )
    evidence_quality: Optional[str] = Field(
        default="Strong",
        max_length=30,
        description="Retrieved evidence quality assessment (Strong, Weak, Outdated, Conflicting)",
    )
    verification_required: Optional[Union[bool, str]] = Field(
        default="No",
        description="Whether explicit verification is required (bool or 'Yes'/'No')",
    )
    sensitive_action: Optional[Union[bool, str]] = Field(
        default="No",
        description="Whether query involves sensitive actions (bool or 'Yes'/'No')",
    )
    financial_impact: Optional[Union[bool, str]] = Field(
        default="No",
        description="Whether query involves financial transactions (bool or 'Yes'/'No')",
    )
    sentiment: Optional[str] = Field(
        default="Neutral",
        max_length=20,
        description="Customer sentiment (Negative, Neutral, Positive)",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be blank or contain only whitespace.")
        return cleaned

    model_config = ConfigDict(extra="ignore")


# ===========================================================================
# Model-Specific Response Schemas (M1–M6)
# ===========================================================================

class BasePredictionResponse(BaseModel):
    """Base response schema for individual model predictions."""
    model_name: str = Field(..., description="Machine learning model identifier")
    prediction: Any = Field(..., description="Primary prediction outcome")
    confidence: Optional[float] = Field(
        default=None, description="Prediction confidence score or probability"
    )
    confidence_type: str = Field(
        ...,
        description="Indicates whether confidence is 'calibrated_probability' or 'uncalibrated_score'",
    )
    confidence_note: str = Field(
        ...,
        description="Documentation explaining score calibration, limitations, or source",
    )
    query: str = Field(..., description="Query submitted for prediction")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 response timestamp",
    )

    model_config = ConfigDict(extra="ignore")


class IntentPredictionResponse(BasePredictionResponse):
    """M1 Intent Classifier prediction response."""
    intent: str = Field(..., description="Predicted customer intent across 47 classes")
    model_name: str = Field(default="M1_Intent_Classifier", description="Model identifier")
    confidence_type: str = Field(
        default="uncalibrated_score",
        description="Uncalibrated score derived from LinearSVC decision function",
    )
    confidence_note: str = Field(
        default=(
            "LinearSVC decision-margin softmax values are uncalibrated decision scores, "
            "not true calibrated probabilities. They reflect relative hyperplane distance "
            "and may appear low for multi-class classification (47 classes)."
        ),
        description="Score calibration note",
    )


class SentimentPredictionResponse(BasePredictionResponse):
    """M2 Sentiment Analyzer prediction response."""
    sentiment: str = Field(..., description="Predicted customer sentiment (Negative, Neutral, Positive)")
    model_name: str = Field(default="M2_Sentiment_Analyzer", description="Model identifier")
    confidence_type: str = Field(
        default="calibrated_probability",
        description="Calibrated probability from LogisticRegression predict_proba",
    )
    confidence_note: str = Field(
        default="Calibrated probability computed via LogisticRegression predict_proba.",
        description="Score calibration note",
    )


class RiskPredictionResponse(BasePredictionResponse):
    """M3 Risk Tier Classifier prediction response."""
    risk: str = Field(..., description="Predicted risk tier (Low, Medium, High)")
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Calibrated class probability distribution across all risk classes",
    )
    model_name: str = Field(default="M3_Risk_Classifier", description="Model identifier")
    confidence_type: str = Field(
        default="calibrated_probability",
        description="Calibrated probability from RandomForestClassifier predict_proba",
    )
    confidence_note: str = Field(
        default="Calibrated probability computed via RandomForestClassifier predict_proba.",
        description="Score calibration note",
    )


class ComplexityPredictionResponse(BasePredictionResponse):
    """M4 Task Complexity Classifier prediction response."""
    complexity: str = Field(..., description="Predicted complexity tier (Low, Medium, High)")
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Calibrated class probability distribution across complexity classes",
    )
    model_name: str = Field(default="M4_Complexity_Classifier", description="Model identifier")
    confidence_type: str = Field(
        default="calibrated_probability",
        description="Calibrated probability from RandomForestClassifier predict_proba",
    )
    confidence_note: str = Field(
        default="Calibrated probability computed via RandomForestClassifier predict_proba.",
        description="Score calibration note",
    )


class DomainPredictionResponse(BasePredictionResponse):
    """M5 Domain Classifier prediction response."""
    domain: str = Field(..., description="Predicted business domain (E-commerce, Education, Small Business)")
    model_name: str = Field(default="M5_Domain_Classifier", description="Model identifier")
    confidence_type: str = Field(
        default="uncalibrated_score",
        description="Uncalibrated score derived from LinearSVC decision function",
    )
    confidence_note: str = Field(
        default=(
            "LinearSVC decision-margin softmax values are uncalibrated decision scores, "
            "not true calibrated probabilities. They reflect relative hyperplane distance "
            "rather than empirical likelihood."
        ),
        description="Score calibration note",
    )


class EscalationPredictionResponse(BasePredictionResponse):
    """M6 Escalation & Autonomy Engine prediction response."""
    autonomy_tier: str = Field(..., description="Predicted autonomy tier (ANSWER, ORCHESTRATE, ESCALATE)")
    escalation_required: str = Field(..., description="Provisional escalation requirement ('Yes' or 'No')")
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Class probability distribution across autonomy classes",
    )
    model_name: str = Field(default="M6_Escalation_Autonomy_Classifier", description="Model identifier")
    confidence_type: str = Field(
        default="calibrated_probability",
        description="Probability distribution from DecisionTreeClassifier predict_proba",
    )
    confidence_note: str = Field(
        default="Class probability derived from DecisionTreeClassifier leaf sample distributions.",
        description="Score calibration note",
    )
