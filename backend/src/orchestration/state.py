"""
Canonical Orchestration State Contract
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation

Destination: src/orchestration/state.py

Defines the central typed state container passed across:
- Model Service triage (M1–M5 classifications + provisional M6 routing)
- Hybrid RAG retrieval (BM25 + Dense ChromaDB)
- Verification & Evidence Assessment
- Domain and Task Agents (E-commerce, Education, Small Business, Resolution)
- Deterministic Safety Guardrails (configs/autonomy_rules.yaml)
- Human-in-the-Loop Escalation Dossier
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Set, Union
import numpy as np


# ---------------------------------------------------------------------------
# Strict Literal Types (Aligned with Model Cards & autonomy_rules.yaml)
# ---------------------------------------------------------------------------

AutonomyTier = Literal["ANSWER", "ORCHESTRATE", "ESCALATE"]
EvidenceQuality = Literal["Pending", "Strong", "Insufficient", "Conflicting"]
VerificationStatus = Literal["Pending", "Passed", "Failed", "Not_Required"]
EntailmentStatus = Literal[
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
    "VERIFICATION_ERROR",
]
ExecutionStatus = Literal[
    "INITIALIZED",
    "CLASSIFIED",
    "RETRIEVED",
    "VERIFIED",
    "RESOLVED",
    "ESCALATED",
    "FAILED",
]
DecisionSource = Literal[
    "Pending",
    "ML_Model_Provisional",
    "Deterministic_Safety_Rule",
    "Model_Verified",
    "Human_Override",
]


# ---------------------------------------------------------------------------
# JSON-Safe Serialization Helper
# ---------------------------------------------------------------------------

def _make_json_safe(obj: Any) -> Any:
    """
    Recursively converts NumPy scalars/arrays, Path, datetime, sets,
    and dataclasses into native JSON-serializable Python primitives
    without discarding any fields or precision.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return str(obj)
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return str(val) if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_make_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return [_make_json_safe(x) for x in sorted(list(obj), key=str)]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {str(k): _make_json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Supporting Structured Sub-Containers
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """Represents a validated knowledge passage retrieved via RAGRetriever."""
    chunk_id: str
    document_id: str
    domain: str
    title: str
    authority: str
    freshness_status: str
    document_type: str
    retrieval_priority: int
    chunk_text: str
    cosine_distance: float
    similarity: float
    rrf_score: Optional[float] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievedChunk":
        known_keys = {
            "chunk_id", "document_id", "domain", "title", "authority",
            "freshness_status", "document_type", "retrieval_priority",
            "chunk_text", "cosine_distance", "similarity", "rrf_score",
            "bm25_score", "dense_rank", "lexical_rank",
        }
        core = {k: v for k, v in data.items() if k in known_keys}
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(**core, metadata=extra)


@dataclass
class EscalationDossier:
    """Traceable context packet handed over for Human-in-the-Loop review."""
    triggered_rule: str
    reason: str
    timestamp: str
    query: str
    domain: Optional[str] = None
    intent: Optional[str] = None
    risk: Optional[str] = None
    complexity: Optional[str] = None
    sentiment: Optional[str] = None
    evidence_quality: str = "Pending"
    verification_score: Optional[float] = None
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    agent_trace: List[Dict[str, Any]] = field(default_factory=list)
    recommended_action: Optional[str] = None
    context_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Master Orchestration State Contract
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationState:
    """
    Central state container coordinating the full query lifecycle:
    Triage -> Retrieval -> Verification -> Agent Execution -> Safety Gate -> Response
    """

    # Protected safety-gate fields that cannot be initialized by from_model_service()
    PROTECTED_FINAL_FIELDS: ClassVar[Set[str]] = {
        "final_autonomy_tier",
        "final_escalation_required",
        "final_confidence",
        "safety_rule_triggered",
        "safety_override",
        "decision_source",
        "decision_rationale",
    }

    # 1. Customer Query & Session Context
    query: str
    resolved_query: Optional[str] = None
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    customer_memory: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sensitive_action: bool = False
    financial_impact: bool = False
    action_type: str = "Information/Support"

    # 2. M1–M5 Classification Outputs
    domain: Optional[str] = None
    domain_confidence: Optional[float] = None

    intent: Optional[str] = None
    intent_confidence: Optional[float] = None

    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None

    risk: Optional[str] = None
    risk_confidence: Optional[float] = None
    risk_class_probabilities: Dict[str, float] = field(default_factory=dict)

    complexity: Optional[str] = None
    complexity_confidence: Optional[float] = None
    complexity_class_probabilities: Dict[str, float] = field(default_factory=dict)

    # 3. Provisional M6 Routing (Triage Hypothesis)
    provisional_autonomy_tier: Optional[AutonomyTier] = None
    provisional_escalation_required: Optional[str] = None  # Preserves explicit "Yes" / "No" from M6
    provisional_confidence: Optional[float] = None
    provisional_class_probabilities: Dict[str, float] = field(default_factory=dict)
    provisional_routing_reason: Optional[str] = None

    # 4. RAG Retrieval & Evidence Quality Assessment
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_count: int = 0
    evidence_quality: EvidenceQuality = "Pending"  # Invariant: NEVER Strong before assessment
    retrieval_similarity_score: Optional[float] = None
    has_conflicting_evidence: bool = False
    has_outdated_evidence: bool = False
    query_embedding: Optional[List[float]] = None

    # 5. Verification Signals & Safety Invariants
    verification_required: bool = False
    verification_score: Optional[float] = None
    verification_status: VerificationStatus = "Pending"
    verification_findings: List[str] = field(default_factory=list)
    verification_details: Dict[str, Any] = field(default_factory=dict)
    entailment_status: Optional[EntailmentStatus] = None
    entailment_score: Optional[float] = None
    entailment_details: Dict[str, Any] = field(default_factory=dict)
    verified_claims: List[Dict[str, Any]] = field(default_factory=list)

    # 6. Final Autonomy Decision (Deterministic Safety Gate)
    # Invariant: final_* fields remain None until evaluated by safety_policy.py
    final_autonomy_tier: Optional[AutonomyTier] = None
    final_escalation_required: Optional[bool] = None
    final_confidence: Optional[float] = None
    safety_rule_triggered: Optional[str] = None
    safety_override: bool = False
    decision_source: DecisionSource = "Pending"
    decision_rationale: Optional[str] = None

    # 7. Multi-Agent Coordination & Execution Trace
    selected_agents: List[str] = field(default_factory=list)
    agent_execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_response: Optional[str] = None
    response_citations: List[str] = field(default_factory=list)

    # 8. Execution Status, Errors & Escalation Tracking
    status: ExecutionStatus = "INITIALIZED"
    errors: List[str] = field(default_factory=list)
    escalation_reason: Optional[str] = None
    escalation_dossier: Optional[Dict[str, Any]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # -----------------------------------------------------------------------
    # Dual-Access Interoperability Helpers (Dict-like Subscription + Object)
    # -----------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(f"Key '{key}' not found in OrchestrationState") from exc

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(f"Cannot set invalid attribute '{key}' on OrchestrationState")
        setattr(self, key, value)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self, safe: bool = True) -> Dict[str, Any]:
        """
        Convert state into a pure dictionary.
        When safe=True (default), normalizes NumPy types, Path, and datetime
        for guaranteed standard JSON serializability.
        """
        raw = asdict(self)
        return _make_json_safe(raw) if safe else raw

    # -----------------------------------------------------------------------
    # Factory Methods for Seamless Subsystem Ingestion
    # -----------------------------------------------------------------------

    @classmethod
    def from_model_service(
        cls, analysis: Dict[str, Any], **kwargs: Any
    ) -> "OrchestrationState":
        """
        Factory to instantiate OrchestrationState from ModelService.analyze_query().
        Maps M1–M5 outputs, records M6 as provisional routing,
        and strictly rejects caller attempts to preset protected final decision fields.
        """
        forbidden = set(kwargs.keys()).intersection(cls.PROTECTED_FINAL_FIELDS)
        if forbidden:
            raise ValueError(
                f"Cannot initialize protected final safety-gate fields via from_model_service(): {sorted(forbidden)}. "
                "Final decision fields must only be evaluated and set post-retrieval by the deterministic safety policy engine."
            )

        extracted = dict(kwargs)

        q = extracted.pop("query", analysis.get("query", ""))
        dom_dict = analysis.get("domain") or {}
        intent_dict = analysis.get("intent") or {}
        sent_dict = analysis.get("sentiment") or {}
        risk_dict = analysis.get("risk") or {}
        comp_dict = analysis.get("complexity") or {}
        esc_dict = analysis.get("escalation") or {}

        resolved_domain = extracted.pop("domain", dom_dict.get("domain"))
        domain_conf = extracted.pop("domain_confidence", dom_dict.get("confidence"))
        resolved_intent = extracted.pop("intent", intent_dict.get("intent"))
        intent_conf = extracted.pop("intent_confidence", intent_dict.get("confidence"))
        resolved_sent = extracted.pop("sentiment", sent_dict.get("sentiment"))
        sent_conf = extracted.pop("sentiment_confidence", sent_dict.get("confidence"))
        resolved_risk = extracted.pop("risk", risk_dict.get("risk"))
        risk_conf = extracted.pop("risk_confidence", risk_dict.get("confidence"))
        risk_probs = extracted.pop("risk_class_probabilities", risk_dict.get("class_probabilities", {}))
        resolved_comp = extracted.pop("complexity", comp_dict.get("complexity"))
        comp_conf = extracted.pop("complexity_confidence", comp_dict.get("confidence"))
        comp_probs = extracted.pop("complexity_class_probabilities", comp_dict.get("class_probabilities", {}))

        prov_tier = extracted.pop("provisional_autonomy_tier", esc_dict.get("autonomy_tier"))
        raw_esc_req = esc_dict.get("escalation_required")  # "Yes" | "No" from M6
        prov_esc_req = extracted.pop("provisional_escalation_required", raw_esc_req)
        prov_conf = extracted.pop("provisional_confidence", esc_dict.get("confidence"))
        prov_probs = extracted.pop("provisional_class_probabilities", esc_dict.get("class_probabilities", {}))
        prov_reason = extracted.pop(
            "provisional_routing_reason",
            "Initial M6 classification under default pre-retrieval assumptions",
        )

        initial_status = extracted.pop("status", "CLASSIFIED")
        meta = extracted.pop("metadata", {})
        if not isinstance(meta, dict):
            meta = {"raw_metadata": meta}

        return cls(
            query=q,
            domain=resolved_domain,
            domain_confidence=domain_conf,
            intent=resolved_intent,
            intent_confidence=intent_conf,
            sentiment=resolved_sent,
            sentiment_confidence=sent_conf,
            risk=resolved_risk,
            risk_confidence=risk_conf,
            risk_class_probabilities=risk_probs,
            complexity=resolved_comp,
            complexity_confidence=comp_conf,
            complexity_class_probabilities=comp_probs,
            provisional_autonomy_tier=prov_tier,
            provisional_escalation_required=prov_esc_req,
            provisional_confidence=prov_conf,
            provisional_class_probabilities=prov_probs,
            provisional_routing_reason=prov_reason,
            metadata=meta,
            status=initial_status,
            **extracted,
        )

    def add_retrieved_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query_embedding: Optional[Union[List[float], np.ndarray]] = None,
    ) -> None:
        """
        Attaches retrieved RAG passages without assuming any fixed field set.
        Verified against RAGRetriever.retrieve() which returns List[Dict[str, Any]].
        Safely computes summary metrics if similarity or freshness exists.
        Leaves evidence_quality as 'Pending' (never promotes to 'Strong').
        Optionally preserves normalized query_embedding for downstream verifiers.
        """
        if not isinstance(chunks, list):
            chunks = list(chunks) if chunks is not None else []

        self.retrieved_chunks = chunks
        self.retrieval_count = len(chunks)

        if query_embedding is not None:
            try:
                if isinstance(query_embedding, np.ndarray):
                    self.query_embedding = [float(x) for x in query_embedding.flatten()]
                elif isinstance(query_embedding, (list, tuple)):
                    self.query_embedding = [float(x) for x in query_embedding]
            except Exception:
                self.query_embedding = None

        if chunks:
            valid_sims = []
            for c in chunks:
                if isinstance(c, dict):
                    sim = c.get("similarity")
                    if isinstance(sim, (int, float)) and not np.isnan(sim):
                        valid_sims.append(float(sim))
            self.retrieval_similarity_score = (
                round(max(valid_sims), 4) if valid_sims else None
            )
            self.has_outdated_evidence = any(
                isinstance(c, dict) and c.get("freshness_status") == "Outdated"
                for c in chunks
            )
        else:
            self.retrieval_similarity_score = None
            self.has_outdated_evidence = False

        self.status = "RETRIEVED"
        self.updated_at = datetime.now(timezone.utc).isoformat()
