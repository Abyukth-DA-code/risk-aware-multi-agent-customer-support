"""
Pydantic Data Models for Multi-Tenant Workspace & Checklist Engine
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BusinessDomain(str, Enum):
    ECOMMERCE = "E-commerce"
    EDUCATION = "Education"
    SMALL_BUSINESS = "Small Business"


class ChecklistItemStatus(str, Enum):
    NOT_UPLOADED = "Not Uploaded"
    PROCESSING = "Processing"
    VALIDATED = "Validated"
    NEEDS_ATTENTION = "Needs Attention"
    OUTDATED = "Outdated"


class TenantRegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Business or organization name")
    email: str = Field(..., min_length=5, max_length=150, description="Owner contact email")


class TenantResponse(BaseModel):
    tenant_id: str
    name: str
    email: str
    created_at: str
    api_key: Optional[str] = Field(None, description="Plaintext API key returned only once upon creation")


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Workspace display name")
    domain: BusinessDomain = Field(..., description="Primary business domain")
    description: Optional[str] = Field(None, max_length=500, description="Optional business description")


class WorkspaceResponse(BaseModel):
    workspace_id: str
    tenant_id: str
    name: str
    domain: str
    description: Optional[str]
    is_active: bool
    created_at: str
    updated_at: str


class ChecklistItemResponse(BaseModel):
    item_id: str
    workspace_id: str
    domain: str
    category_key: str
    category_name: str
    is_required: bool
    status: ChecklistItemStatus
    document_id: Optional[str] = None
    filename: Optional[str] = None
    extracted_records_count: int = 0
    validation_errors: Optional[str] = None
    last_updated: str


class CustomCategoryRequest(BaseModel):
    category_name: str = Field(..., min_length=3, max_length=100, description="Custom document category name")
    description: Optional[str] = Field(None, max_length=300, description="Category purpose description")
    is_required: bool = Field(default=False, description="Whether this category is strictly required for readiness")


class ReadinessReport(BaseModel):
    workspace_id: str
    domain: str
    total_categories: int
    required_categories: int
    required_completed: int
    recommended_completed: int
    is_ready_for_testing: bool
    readiness_percentage: float
    missing_required_categories: List[str]
    unanswerable_query_warnings: List[str]
    status_summary: Dict[str, int]


class WorkspaceQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Customer inquiry text")
    customer_id: Optional[str] = Field(default=None, max_length=100)
    session_id: Optional[str] = Field(default=None, max_length=100)
    sensitive_action: bool = Field(default=False)
    financial_impact: bool = Field(default=False)
    action_type: str = Field(default="Information/Support")
    top_k: int = Field(default=3, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be blank or contain only whitespace.")
        return cleaned

    model_config = ConfigDict(extra="ignore")


class WorkspaceQueryResponse(BaseModel):
    workspace_id: str
    query: str
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    grounded_response: Optional[str] = None
    triage: Any = Field(default_factory=dict)
    evidence: Any = Field(default_factory=dict)
    resolution: Any = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    model_config = ConfigDict(extra="ignore")


class EscalationStatus(str, Enum):
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"


class EscalationReviewRequest(BaseModel):
    status: EscalationStatus = Field(..., description="Target lifecycle state (IN_REVIEW, RESOLVED, REJECTED)")
    human_decision: Optional[str] = Field(None, max_length=100, description="Decision code (e.g. APPROVE_OVERRIDE, REJECT_REQUEST, PROVIDE_CUSTOM_ANSWER)")
    reviewer_notes: Optional[str] = Field(None, max_length=1000, description="Human supervisor audit commentary")
    resolved_by: Optional[str] = Field(None, max_length=100, description="Reviewer identity or ID")

    model_config = ConfigDict(extra="ignore")


class EscalationListItem(BaseModel):
    escalation_id: str
    workspace_id: str
    tenant_id: str
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    query: str
    domain: str
    intent: Optional[str] = None
    risk: Optional[str] = None
    final_autonomy_tier: str
    safety_rule_triggered: Optional[str] = None
    escalation_reason: str
    status: EscalationStatus
    human_decision: Optional[str] = None
    created_at: str
    updated_at: str
    resolved_at: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class EscalationResponse(BaseModel):
    escalation_id: str
    workspace_id: str
    tenant_id: str
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    query: str
    domain: str
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    risk: Optional[str] = None
    risk_confidence: Optional[float] = None
    complexity: Optional[str] = None
    provisional_autonomy_tier: Optional[str] = None
    final_autonomy_tier: str
    evidence_quality: Optional[str] = None
    retrieval_similarity_score: Optional[float] = None
    retrieved_chunks_json: Optional[str] = None
    safety_rule_triggered: Optional[str] = None
    escalation_reason: str
    recommended_action: Optional[str] = None
    status: EscalationStatus
    human_decision: Optional[str] = None
    reviewer_notes: Optional[str] = None
    resolved_by: Optional[str] = None
    created_at: str
    updated_at: str
    resolved_at: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class CustomerMemoryCreateRequest(BaseModel):
    category: str = Field(..., min_length=1, max_length=100, description="Memory category (e.g. preference, order_history, account_note)")
    content: str = Field(..., min_length=1, max_length=2000, description="Descriptive customer context or historical record")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the memory observation")
    ttl_days: Optional[int] = Field(default=None, ge=1, le=3650, description="Configurable retention lifetime in days (None for indefinite)")

    @field_validator("category", "content")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank or contain only whitespace.")
        return cleaned

    model_config = ConfigDict(extra="ignore")


class CustomerMemoryResponse(BaseModel):
    memory_id: str
    workspace_id: str
    tenant_id: str
    customer_id: str
    memory_category: str
    memory_content: str
    confidence: float
    created_at: str
    expires_at: Optional[str] = None
    status: str

    model_config = ConfigDict(extra="ignore")


class CustomerMemoryListResponse(BaseModel):
    workspace_id: str
    customer_id: str
    memories: List[CustomerMemoryResponse]
    total_count: int

    model_config = ConfigDict(extra="ignore")


class CustomerMemoryDeleteResponse(BaseModel):
    workspace_id: str
    customer_id: Optional[str] = None
    memory_id: Optional[str] = None
    deleted_count: int
    message: str

    model_config = ConfigDict(extra="ignore")


class WidgetTokenResponse(BaseModel):
    workspace_id: str
    widget_token: str
    revoked: bool = False
    message: str

    model_config = ConfigDict(extra="ignore")


class WidgetConfigResponse(BaseModel):
    workspace_id: str
    name: str
    domain: str
    description: Optional[str] = None
    is_active: bool

    model_config = ConfigDict(extra="ignore")


class WidgetChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Customer message text")
    session_id: Optional[str] = Field(None, max_length=64, description="Client session identifier")
    customer_id: Optional[str] = Field(None, max_length=64, description="Optional customer identity")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be blank or contain only whitespace.")
        return cleaned

    model_config = ConfigDict(extra="ignore")


class WidgetChatResponse(BaseModel):
    workspace_id: str
    session_id: str
    query: str
    resolved_query: Optional[str] = None
    response: Optional[str] = None
    status: str
    autonomy_tier: str
    escalation_required: bool
    escalation_id: Optional[str] = None
    clarification_prompt: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    model_config = ConfigDict(extra="ignore")

