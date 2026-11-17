"""
Policy & Governance Specialist Agent
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Role:
- Evaluates authentic evidence sufficiency, freshness, authority, and contradictions.
- Uses configurable similarity thresholds to assess evidence quality.
- Enforces deterministic safety invariants (CRIT_01 to SENS_FIN_06) via SafetyPolicyEngine.
- Establishes preliminary autonomy tier decision (ANSWER, ORCHESTRATE, ESCALATE).
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.verification_agent import EvidenceEntailmentVerifier
from src.orchestration.safety_policy import SafetyPolicyEngine
from src.orchestration.state import OrchestrationState

logger = logging.getLogger("agents.policy")

DEFAULT_SIMILARITY_THRESHOLD = 0.45


class PolicyAgent:
    """Specialist agent responsible for evidence assessment and safety gate enforcement."""

    def __init__(
        self,
        safety_engine: Optional[SafetyPolicyEngine] = None,
        verifier: Optional[EvidenceEntailmentVerifier] = None,
        min_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.safety_engine = safety_engine or SafetyPolicyEngine()
        self.verifier = verifier or EvidenceEntailmentVerifier()
        self.min_similarity_threshold = min_similarity_threshold

    def evaluate_evidence(self, state: OrchestrationState) -> OrchestrationState:
        """
        Evaluates authentic retrieved knowledge chunks using multi-criteria signals:
        relevance/similarity, evidence entailment verification (Phase G), freshness,
        authority, and contradictory policy detection.
        """
        threshold = float(
            state.metadata.get(
                "min_entailment_threshold",
                state.metadata.get("min_similarity_threshold", self.min_similarity_threshold),
            )
        )

        if not state.retrieved_chunks or state.retrieval_similarity_score is None:
            state.evidence_quality = "Insufficient"
            state.entailment_status = "INSUFFICIENT_EVIDENCE"
            state.entailment_score = 0.0
            state.verification_findings.append("No knowledge base passages retrieved.")
            return state

        sim = float(state.retrieval_similarity_score)

        # 1. Phase G: Evidence Entailment Verification
        query = state.resolved_query or state.query
        try:
            verif_res = self.verifier.verify(
                query=query,
                retrieved_chunks=state.retrieved_chunks,
                proposed_answer=state.final_response,
                threshold=threshold,
                retrieval_similarity=sim,
                query_embedding=state.query_embedding or state.metadata.get("query_embedding"),
            )
        except Exception as exc:
            logger.error(f"Evidence entailment verification failed: {exc}", exc_info=True)
            verif_res = {
                "status": "VERIFICATION_ERROR",
                "score": 0.0,
                "findings": [f"Evidence verification error: {str(exc)}"],
                "supported_claims": [],
                "unsupported_claims": [query] if query else [],
                "evidence_references": [],
                "latency_ms": 0.0,
            }

        state.entailment_status = verif_res["status"]
        state.entailment_score = verif_res["score"]
        state.entailment_details = verif_res
        state.verified_claims = verif_res.get("supported_claims", [])
        state.verification_findings.extend(verif_res.get("findings", []))

        # 2. Check for conflicting or contradictory evidence
        is_conflicting = (
            verif_res["status"] == "CONTRADICTED"
            or state.has_conflicting_evidence
            or state.metadata.get("has_conflicting_evidence", False)
            or any(
                isinstance(c, dict) and c.get("conflict_flag") is True
                for c in state.retrieved_chunks
            )
        )

        # 3. Check for outdated evidence
        has_outdated = (
            state.has_outdated_evidence
            or any(
                isinstance(c, dict) and str(c.get("freshness_status", "")).lower() == "outdated"
                for c in state.retrieved_chunks
            )
        )
        if has_outdated:
            state.has_outdated_evidence = True

        # 4. Categorize Evidence Quality & Route based on Entailment Status
        if is_conflicting:
            state.has_conflicting_evidence = True
            state.evidence_quality = "Conflicting"
            state.verification_findings.append(
                "Contradictory policy documentation detected across retrieved chunks."
            )
        elif has_outdated:
            state.evidence_quality = "Insufficient"
            state.verification_findings.append("Evidence quality insufficient due to outdated document.")
        elif verif_res["status"] in ("INSUFFICIENT_EVIDENCE", "VERIFICATION_ERROR"):
            state.evidence_quality = "Insufficient"
            state.verification_findings.append(
                f"Evidence quality insufficient due to entailment verification: {verif_res['status']}."
            )
        elif verif_res["status"] == "PARTIALLY_SUPPORTED":
            norm_risk = str(state.risk).strip().capitalize() if state.risk else "Low"
            if norm_risk == "Low" and not state.sensitive_action and not state.financial_impact:
                # Low-risk partial evidence routes safely to CLARIFY
                state.evidence_quality = "Strong"
                state.metadata["is_ambiguous"] = True
                state.verification_findings.append(
                    "Evidence partially supports query demand in low-risk context; routing to CLARIFY."
                )
            else:
                state.evidence_quality = "Insufficient"
                state.verification_findings.append(
                    "Evidence partially supports query demand in high-risk/sensitive context; insufficient for autonomous answer."
                )
        elif sim < threshold:
            state.evidence_quality = "Insufficient"
            state.verification_findings.append(
                f"Evidence quality insufficient due to low similarity ({sim:.4f} < {threshold:.2f})."
            )
        else:
            state.evidence_quality = "Strong"
            state.verification_findings.append(
                f"Strong evidence confirmed (similarity: {sim:.4f} >= {threshold:.2f}, fresh documents, verified entailment)."
            )

        # 5. Verification Requirements
        if state.metadata.get("verification_required") is not None:
            state.verification_required = bool(state.metadata["verification_required"])
        elif state.sensitive_action or state.financial_impact:
            state.verification_required = True

        if state.verification_required:
            raw_score = (
                state.verification_details.get("verification_score")
                or state.metadata.get("verification_score")
            )
            if raw_score is not None:
                try:
                    v_score = float(raw_score)
                    if not np.isnan(v_score) and not np.isinf(v_score):
                        state.verification_score = float(np.clip(v_score, 0.0, 1.0))
                        state.verification_status = "Passed" if state.verification_score >= 0.80 else "Failed"
                except (ValueError, TypeError):
                    state.verification_score = None
                    state.verification_status = "Pending"
            else:
                state.verification_score = None
                state.verification_status = "Pending"
        else:
            state.verification_status = "Not_Required"

        return state

    def execute(self, state: OrchestrationState) -> OrchestrationState:
        """
        Executes complete policy assessment:
        1. Evidence Evaluation & Entailment Verification
        2. Deterministic Safety Engine Invariants
        3. Trace Recording
        """
        start_time = datetime.now(timezone.utc).isoformat()
        if "PolicyAgent" not in state.selected_agents:
            state.selected_agents.append("PolicyAgent")

        try:
            # Step 1: Evaluate authentic evidence with entailment verification
            state = self.evaluate_evidence(state)
            state.metadata["latency_verification_ms"] = round(
                state.entailment_details.get("latency_ms", 0.0) if state.entailment_details else 0.0, 3
            )

            # Step 2: Enforce deterministic safety policy rules
            t_pol0 = time.perf_counter()
            state = self.safety_engine.evaluate(state)
            lat_pol = round((time.perf_counter() - t_pol0) * 1000, 3)
            state.metadata["latency_policy_ms"] = lat_pol

            state.agent_execution_trace.append({
                "agent": "PolicyAgent",
                "timestamp": start_time,
                "action": "POLICY_EVALUATION",
                "status": "COMPLETED",
                "latency_ms": round(state.metadata["latency_verification_ms"] + lat_pol, 3),
                "findings": {
                    "evidence_quality": state.evidence_quality,
                    "retrieval_similarity": state.retrieval_similarity_score,
                    "entailment_status": state.entailment_status,
                    "entailment_score": state.entailment_score,
                    "final_autonomy_tier": state.final_autonomy_tier,
                    "final_escalation_required": state.final_escalation_required,
                    "safety_rule_triggered": state.safety_rule_triggered,
                    "final_confidence": state.final_confidence,
                    "decision_rationale": state.decision_rationale,
                },
            })
            state.status = "VERIFIED"

        except Exception as exc:
            logger.error(f"PolicyAgent error: {exc}", exc_info=True)
            state.errors.append(f"PolicyAgent error: {str(exc)}")
            state.final_autonomy_tier = "ESCALATE"
            state.final_escalation_required = True
            state.escalation_reason = f"Policy evaluation error: {str(exc)}"
            state.agent_execution_trace.append({
                "agent": "PolicyAgent",
                "timestamp": start_time,
                "action": "POLICY_EVALUATION",
                "status": "FAILED",
                "error": str(exc),
            })

        state.updated_at = datetime.now(timezone.utc).isoformat()
        return state
