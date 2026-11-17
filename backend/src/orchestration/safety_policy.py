"""
Deterministic Safety Policy Engine
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation

Destination: src/orchestration/safety_policy.py

Enforces deterministic guardrails, observable confidence evaluation,
and safety-tier decision gating defined in configs/autonomy_rules.yaml.
Strictly ensures M6 triage outputs remain provisional and cannot bypass
mandatory safety invariants.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import numpy as np
import yaml

from src.orchestration.state import (
    AutonomyTier,
    DecisionSource,
    EscalationDossier,
    ExecutionStatus,
    OrchestrationState,
    _make_json_safe,
)


# Standard canonical rule IDs matching configs/autonomy_rules.yaml
RULE_CRIT_01 = "CRIT_01"
RULE_RISK_VERIF_02 = "RISK_VERIF_02"
RULE_EVID_CONF_03 = "EVID_CONF_03"
RULE_EVID_INSUF_04 = "EVID_INSUF_04"
RULE_LOW_CONF_05 = "LOW_CONF_05"
RULE_SENS_FIN_06 = "SENS_FIN_06"

# Authority string-to-score normalization mapping
AUTHORITY_SCORES: Dict[str, float] = {
    "OFFICIAL": 1.0,
    "HIGH": 1.0,
    "VERIFIED": 0.85,
    "MEDIUM": 0.70,
    "LOW": 0.40,
    "COMMUNITY": 0.30,
    "UNKNOWN": 0.30,
}


class SafetyPolicyEngine:
    """
    Deterministic Safety Policy Engine for Customer Support Multi-Agent Orchestration.

    Responsibilities:
    1. Loads canonical thresholds and rules from configs/autonomy_rules.yaml.
    2. Calculates observable composite confidence from authentic retrieval and evidence signals.
    3. Evaluates mandatory escalation rules (CRIT_01 through SENS_FIN_06) with strict priority.
    4. Evaluates tier qualification for direct ANSWER and dynamic ORCHESTRATE.
    5. Enforces fail-safe conservative fallback when neither tier qualifies.
    6. Constructs structured EscalationDossier for human review.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the safety policy engine with YAML configuration.
        """
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config(self.config_path)

        # Autonomy tier configurations
        tiers_cfg = self.config.get("autonomy_tiers", {})
        self.answer_conditions = tiers_cfg.get("ANSWER", {}).get("conditions", {})
        self.orchestrate_conditions = tiers_cfg.get("ORCHESTRATE", {}).get("conditions", {})

        escalate_cfg = tiers_cfg.get("ESCALATE", {})
        self.trigger_criteria: List[Dict[str, Any]] = escalate_cfg.get("trigger_criteria", [])

        # Observable confidence weights
        self.confidence_weights: Dict[str, float] = self.config.get(
            "observable_confidence_weights",
            {
                "retrieval_similarity_weight": 0.25,
                "reranker_score_weight": 0.25,
                "source_authority_weight": 0.15,
                "document_freshness_weight": 0.15,
                "evidence_agreement_weight": 0.10,
                "database_consistency_weight": 0.10,
            },
        )

    @staticmethod
    def _resolve_config_path(custom_path: Optional[Union[str, Path]]) -> Path:
        """Finds and resolves the autonomy_rules.yaml path safely."""
        if custom_path is not None:
            p = Path(custom_path).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Configuration file not found at: {p}")
            return p

        candidates = [
            Path("config/autonomy_rules.yaml"),
            Path("configs/autonomy_rules.yaml"),
            Path(__file__).resolve().parents[3] / "config" / "autonomy_rules.yaml",
            Path(__file__).resolve().parents[2] / "config" / "autonomy_rules.yaml",
            Path(__file__).resolve().parents[3] / "configs" / "autonomy_rules.yaml",
            Path(__file__).resolve().parents[2] / "configs" / "autonomy_rules.yaml",
            Path(__file__).resolve().parent / "autonomy_rules.yaml",
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()

        raise FileNotFoundError(
            "configs/autonomy_rules.yaml could not be found in current working directory "
            f"or relative to {__file__}."
        )

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        """Safely loads YAML configuration file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML configuration format in {path}")
        return data

    def compute_observable_confidence(self, state: OrchestrationState) -> float:
        """
        Computes the normalized observable confidence score [0.0, 1.0] from genuinely
        available signals without fabricating missing values.
        """
        breakdown = self.get_confidence_breakdown(state)
        return breakdown["composite_confidence"]

    def get_confidence_breakdown(self, state: OrchestrationState) -> Dict[str, Any]:
        """
        Inspects which signal values genuinely exist, re-normalizes weights over valid
        available signals, and produces a complete audit breakdown.
        """
        # 1. Retrieval similarity signal
        retrieval_similarity: Optional[float] = None
        if state.retrieval_similarity_score is not None:
            try:
                val = float(state.retrieval_similarity_score)
                if not np.isnan(val) and not np.isinf(val):
                    retrieval_similarity = float(np.clip(val, 0.0, 1.0))
            except (ValueError, TypeError):
                pass
        elif state.retrieved_chunks:
            valid_sims: List[float] = []
            for c in state.retrieved_chunks:
                if isinstance(c, dict):
                    sim = c.get("similarity")
                    if isinstance(sim, (int, float)) and not np.isnan(sim) and not np.isinf(sim):
                        valid_sims.append(float(sim))
            if valid_sims:
                retrieval_similarity = float(np.clip(max(valid_sims), 0.0, 1.0))
        elif "retrieval_similarity" in state.metadata:
            try:
                val = float(state.metadata["retrieval_similarity"])
                if not np.isnan(val) and not np.isinf(val):
                    retrieval_similarity = float(np.clip(val, 0.0, 1.0))
            except (ValueError, TypeError):
                pass

        # 2. Reranker score signal
        reranker_score: Optional[float] = None
        raw_rerank = (
            state.metadata.get("reranker_score")
            or state.intermediate_results.get("reranker_score")
        )
        if raw_rerank is not None:
            try:
                val = float(raw_rerank)
                if not np.isnan(val) and not np.isinf(val):
                    reranker_score = float(np.clip(val, 0.0, 1.0))
            except (ValueError, TypeError):
                pass
        elif state.retrieved_chunks:
            chunk_reranks = [
                float(c["reranker_score"])
                for c in state.retrieved_chunks
                if isinstance(c, dict)
                and c.get("reranker_score") is not None
                and not np.isnan(c["reranker_score"])
            ]
            if chunk_reranks:
                reranker_score = float(np.clip(max(chunk_reranks), 0.0, 1.0))

        # 3. Source authority signal
        source_authority: Optional[float] = None
        raw_auth = state.metadata.get("source_authority_score")
        if raw_auth is not None:
            try:
                val = float(raw_auth)
                if not np.isnan(val) and not np.isinf(val):
                    source_authority = float(np.clip(val, 0.0, 1.0))
            except (ValueError, TypeError):
                pass
        elif state.retrieved_chunks:
            auth_scores: List[float] = []
            for c in state.retrieved_chunks:
                if isinstance(c, dict):
                    auth_val = c.get("authority") or c.get("Authority")
                    if isinstance(auth_val, (int, float)) and not np.isnan(auth_val):
                        auth_scores.append(float(np.clip(auth_val, 0.0, 1.0)))
                    elif isinstance(auth_val, str) and auth_val.strip():
                        norm_key = auth_val.strip().upper()
                        score = AUTHORITY_SCORES.get(norm_key, 0.50)
                        auth_scores.append(score)
            if auth_scores:
                source_authority = float(np.clip(float(np.mean(auth_scores)), 0.0, 1.0))

        # 4. Document freshness signal
        document_freshness: Optional[float] = None
        raw_fresh = state.metadata.get("document_freshness_score")
        if raw_fresh is not None:
            try:
                val = float(raw_fresh)
                if not np.isnan(val) and not np.isinf(val):
                    document_freshness = float(np.clip(val, 0.0, 1.0))
            except (ValueError, TypeError):
                pass
        elif state.retrieved_chunks:
            if state.has_outdated_evidence or any(
                isinstance(c, dict) and c.get("freshness_status") == "Outdated"
                for c in state.retrieved_chunks
            ):
                document_freshness = 0.0
            else:
                document_freshness = 1.0

        # 5. Evidence agreement signal
        evidence_agreement: Optional[float] = None
        raw_agree = state.metadata.get("evidence_agreement_score")
        if raw_agree is not None:
            try:
                val = float(raw_agree)
                if not np.isnan(val) and not np.isinf(val):
                    evidence_agreement = float(np.clip(val, 0.0, 1.0))
            except (ValueError, TypeError):
                pass
        else:
            if state.has_conflicting_evidence or state.evidence_quality == "Conflicting":
                evidence_agreement = 0.0
            elif state.evidence_quality == "Strong":
                evidence_agreement = 1.0
            elif state.evidence_quality == "Insufficient":
                evidence_agreement = 0.5
            # If evidence_quality is Pending, agreement cannot be determined safely -> None

        # 6. Database consistency signal
        database_consistency: Optional[float] = None
        raw_db = (
            state.verification_details.get("database_consistency")
            or state.metadata.get("database_consistency_score")
            or state.intermediate_results.get("database_consistency")
        )
        if raw_db is not None:
            try:
                val = float(raw_db)
                if not np.isnan(val) and not np.isinf(val):
                    database_consistency = float(np.clip(val, 0.0, 1.0))
            except (ValueError, TypeError):
                pass

        # Mapping signals to configured weights
        signal_map: Dict[str, Tuple[Optional[float], float]] = {
            "retrieval_similarity": (
                retrieval_similarity,
                self.confidence_weights.get("retrieval_similarity_weight", 0.25),
            ),
            "reranker_score": (
                reranker_score,
                self.confidence_weights.get("reranker_score_weight", 0.25),
            ),
            "source_authority": (
                source_authority,
                self.confidence_weights.get("source_authority_weight", 0.15),
            ),
            "document_freshness": (
                document_freshness,
                self.confidence_weights.get("document_freshness_weight", 0.15),
            ),
            "evidence_agreement": (
                evidence_agreement,
                self.confidence_weights.get("evidence_agreement_weight", 0.10),
            ),
            "database_consistency": (
                database_consistency,
                self.confidence_weights.get("database_consistency_weight", 0.10),
            ),
        }

        # Filter to valid available signals
        available_signals = {
            sig_name: (val, weight)
            for sig_name, (val, weight) in signal_map.items()
            if val is not None
        }

        total_available_weight = sum(w for _, w in available_signals.values())

        if total_available_weight <= 0.0:
            # Conservative fail-safe: if no valid reliability signals exist, confidence is 0.0
            composite = 0.0
        else:
            weighted_sum = sum(v * w for v, w in available_signals.values())
            composite = float(np.clip(weighted_sum / total_available_weight, 0.0, 1.0))

        composite_rounded = round(composite, 4)

        return {
            "composite_confidence": composite_rounded,
            "total_available_weight": round(total_available_weight, 4),
            "available_signals": {
                k: {
                    "value": round(v, 4),
                    "base_weight": round(w, 4),
                    "normalized_weight": round(w / total_available_weight, 4),
                }
                for k, (v, w) in available_signals.items()
            },
            "unavailable_signals": [
                k for k, (v, _) in signal_map.items() if v is None
            ],
        }

    def _build_escalation_dossier(
        self,
        state: OrchestrationState,
        triggered_rule: str,
        reason: str,
        recommended_action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Constructs a JSON-safe EscalationDossier dict."""
        chunk_ids: List[str] = []
        for c in state.retrieved_chunks:
            if isinstance(c, dict) and c.get("chunk_id"):
                chunk_ids.append(str(c["chunk_id"]))

        if not recommended_action:
            if triggered_rule == RULE_CRIT_01:
                recommended_action = "Immediate transfer to Senior Security/Risk Specialist"
            elif triggered_rule == RULE_RISK_VERIF_02:
                recommended_action = "Manual identity and credentials verification before proceeding"
            elif triggered_rule in (RULE_EVID_CONF_03, RULE_EVID_INSUF_04):
                recommended_action = "Specialist review of policy documentation and customer records"
            elif triggered_rule == RULE_LOW_CONF_05:
                recommended_action = "Human agent review due to low retrieval/evidence confidence"
            elif triggered_rule == RULE_SENS_FIN_06:
                recommended_action = "Supervisor financial approval required for high-risk sensitive transaction"
            else:
                recommended_action = "Human customer support representative review and resolution"

        summary = (
            f"Query: '{state.query[:120]}...' | Domain: {state.domain or 'Unknown'} | "
            f"Risk: {state.risk or 'Unspecified'} | Complexity: {state.complexity or 'Unspecified'} | "
            f"Trigger: {triggered_rule} ({reason})"
        )

        dossier = EscalationDossier(
            triggered_rule=triggered_rule,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=state.query,
            domain=state.domain,
            intent=state.intent,
            risk=state.risk,
            complexity=state.complexity,
            sentiment=state.sentiment,
            evidence_quality=state.evidence_quality,
            verification_score=state.verification_score,
            retrieved_chunk_ids=chunk_ids,
            agent_trace=list(state.agent_execution_trace),
            recommended_action=recommended_action,
            context_summary=summary,
        )
        return _make_json_safe(asdict(dossier))

    def evaluate(self, state: OrchestrationState) -> OrchestrationState:
        """
        Evaluates the deterministic safety policy on an OrchestrationState.
        Modifies state in-place and returns it.
        """
        # 1. Compute observable confidence
        conf_breakdown = self.get_confidence_breakdown(state)
        confidence = conf_breakdown["composite_confidence"]
        state.final_confidence = confidence

        # Record confidence breakdown in intermediate_results for auditability
        state.intermediate_results["confidence_breakdown"] = conf_breakdown

        # 2. Evaluate Mandatory Escalation Rules (YAML priority order)
        escalation_result = self._check_mandatory_escalation_rules(state, confidence)
        if escalation_result is not None:
            rule_id, reason = escalation_result
            state.final_autonomy_tier = "ESCALATE"
            state.final_escalation_required = True
            state.safety_rule_triggered = rule_id
            state.safety_override = (state.provisional_autonomy_tier != "ESCALATE")
            state.decision_source = "Deterministic_Safety_Rule"
            state.decision_rationale = (
                f"Mandatory escalation rule triggered: [{rule_id}] {reason}. "
                f"Provisional tier '{state.provisional_autonomy_tier}' overridden for customer safety."
                if state.safety_override
                else f"Mandatory escalation rule triggered: [{rule_id}] {reason}."
            )
            state.escalation_reason = reason
            state.escalation_dossier = self._build_escalation_dossier(state, rule_id, reason)
            state.status = "ESCALATED"
            state.updated_at = datetime.now(timezone.utc).isoformat()
            return state

        # 3. Preserve Provisional Model Escalation if no specific rule fired
        if state.provisional_autonomy_tier == "ESCALATE":
            esc_reason = state.escalation_reason or "Provisional model triage recommended human escalation"
            state.final_autonomy_tier = "ESCALATE"
            state.final_escalation_required = True
            state.safety_rule_triggered = None
            state.safety_override = False
            state.decision_source = "Model_Verified"
            state.decision_rationale = (
                "Preserved provisional model escalation decision; no overriding safety rule triggered."
            )
            state.escalation_reason = esc_reason
            state.escalation_dossier = self._build_escalation_dossier(state, "NONE", esc_reason)
            state.status = "ESCALATED"
            state.updated_at = datetime.now(timezone.utc).isoformat()
            return state

        # 4. Evaluate Tier Qualification for ANSWER
        answer_qualified, answer_disqualify_reason = self._qualifies_for_answer(state, confidence)
        if answer_qualified:
            override = (state.provisional_autonomy_tier != "ANSWER")
            state.final_autonomy_tier = "ANSWER"
            state.final_escalation_required = False
            state.safety_rule_triggered = None
            state.safety_override = override
            state.decision_source = "Deterministic_Safety_Rule" if override else "Model_Verified"
            state.decision_rationale = (
                f"Qualified for direct autonomous ANSWER tier (overriding provisional '{state.provisional_autonomy_tier}')."
                if override
                else "Qualified for direct autonomous ANSWER tier (verified against deterministic safety policy)."
            )
            state.escalation_reason = None
            state.escalation_dossier = None
            state.status = "VERIFIED"
            state.updated_at = datetime.now(timezone.utc).isoformat()
            return state

        # 5. Evaluate Tier Qualification for ORCHESTRATE
        orchestrate_qualified, orchestrate_disqualify_reason = self._qualifies_for_orchestrate(state, confidence)
        if orchestrate_qualified:
            override = (state.provisional_autonomy_tier != "ORCHESTRATE")
            state.final_autonomy_tier = "ORCHESTRATE"
            state.final_escalation_required = False
            state.safety_rule_triggered = None
            state.safety_override = override
            state.decision_source = "Deterministic_Safety_Rule" if override else "Model_Verified"
            state.decision_rationale = (
                f"Qualified for multi-agent ORCHESTRATE tier (overriding provisional '{state.provisional_autonomy_tier}')."
                if override
                else "Qualified for multi-agent ORCHESTRATE tier (verified against deterministic safety policy)."
            )
            state.escalation_reason = None
            state.escalation_dossier = None
            state.status = "VERIFIED"
            state.updated_at = datetime.now(timezone.utc).isoformat()
            return state

        # 6. Conservative Fallback Escalation (neither ANSWER nor ORCHESTRATE qualified)
        fallback_reason = "Disqualified from autonomous execution tiers (conservative escalation)"
        rationale_detail = (
            f"Query disqualified from ANSWER ({answer_disqualify_reason}) and "
            f"disqualified from ORCHESTRATE ({orchestrate_disqualify_reason}). Defaulting to safe human escalation."
        )
        state.final_autonomy_tier = "ESCALATE"
        state.final_escalation_required = True
        state.safety_rule_triggered = None
        state.safety_override = (state.provisional_autonomy_tier != "ESCALATE")
        state.decision_source = "Deterministic_Safety_Rule"
        state.decision_rationale = rationale_detail
        state.escalation_reason = fallback_reason
        state.escalation_dossier = self._build_escalation_dossier(state, "NONE", fallback_reason)
        state.status = "ESCALATED"
        state.updated_at = datetime.now(timezone.utc).isoformat()
        return state

    def _check_mandatory_escalation_rules(
        self, state: OrchestrationState, confidence: float
    ) -> Optional[Tuple[str, str]]:
        """
        Evaluates mandatory escalation criteria in priority order:
        CRIT_01 -> RISK_VERIF_02 -> EVID_CONF_03 -> EVID_INSUF_04 -> LOW_CONF_05 -> SENS_FIN_06.
        Returns (rule_id, reason) if any rule triggers, else None.
        """
        norm_risk = str(state.risk).strip().capitalize() if state.risk else None

        for criterion in self.trigger_criteria:
            r_id = criterion.get("rule_id")
            reason = criterion.get("reason", "Safety Policy Escalation")

            if r_id == RULE_CRIT_01:
                # condition: risk == 'Critical' or severe security breach indicators in query
                is_crit_security = False
                if state.query:
                    q_low = state.query.lower()
                    if re.search(r"\b(sql injection|database credentials leaked|data breach|database compromised|zero-day exploit|ransomware)\b", q_low):
                        is_crit_security = True
                if norm_risk == "Critical" or is_crit_security:
                    return r_id, reason

            elif r_id == RULE_RISK_VERIF_02:
                # condition: risk == 'High' and verification_required == true and verification_score < 0.80
                if norm_risk == "High" and state.verification_required:
                    # Missing verification score when verification is required is treated conservatively as failed (< 0.80)
                    if state.verification_score is None:
                        return r_id, reason
                    try:
                        v_score = float(state.verification_score)
                        if np.isnan(v_score) or v_score < 0.80:
                            return r_id, reason
                    except (ValueError, TypeError):
                        return r_id, reason

            elif r_id == RULE_EVID_CONF_03:
                # condition: evidence_quality == 'Conflicting' and risk in ['Medium', 'High']
                is_conflicting = (
                    state.evidence_quality == "Conflicting" or state.has_conflicting_evidence
                )
                if is_conflicting and norm_risk in ("Medium", "High"):
                    return r_id, reason

            elif r_id == RULE_EVID_INSUF_04:
                # condition: evidence_quality == 'Insufficient' and risk in ['Medium', 'High']
                if state.evidence_quality == "Insufficient" and norm_risk in ("Medium", "High"):
                    return r_id, reason

            elif r_id == RULE_LOW_CONF_05:
                # condition: final_confidence < 0.65
                if confidence < 0.65:
                    return r_id, reason

            elif r_id == RULE_SENS_FIN_06:
                # condition: sensitive_action == true and financial_impact == true and risk == 'High'
                if state.sensitive_action and state.financial_impact and norm_risk == "High":
                    return r_id, reason

        return None

    def _qualifies_for_answer(
        self, state: OrchestrationState, confidence: float
    ) -> Tuple[bool, str]:
        """
        Validates whether state satisfies all conditions for ANSWER tier:
        - max_risk: 'Low'
        - max_complexity: 'Medium' (Low or Medium)
        - min_confidence: 0.75
        - required_evidence_quality: ['Strong']
        - verification_required: false
        - sensitive_action: false
        """
        norm_risk = str(state.risk).strip().capitalize() if state.risk else None
        norm_comp = str(state.complexity).strip().capitalize() if state.complexity else None

        if norm_risk != "Low":
            return False, f"Risk '{norm_risk}' exceeds max_risk 'Low'"

        if norm_comp not in ("Low", "Medium"):
            return False, f"Complexity '{norm_comp}' exceeds max_complexity 'Medium'"

        min_conf = float(self.answer_conditions.get("min_confidence", 0.75))
        if confidence < min_conf:
            return False, f"Confidence {confidence:.4f} below min_confidence {min_conf}"

        req_evidence = self.answer_conditions.get("required_evidence_quality", ["Strong"])
        if state.evidence_quality not in req_evidence:
            return False, f"Evidence quality '{state.evidence_quality}' not in required {req_evidence}"

        if state.verification_required:
            return False, "verification_required is True"

        if state.sensitive_action:
            return False, "sensitive_action is True"

        return True, "All ANSWER criteria satisfied"

    def _qualifies_for_orchestrate(
        self, state: OrchestrationState, confidence: float
    ) -> Tuple[bool, str]:
        """
        Validates whether state satisfies all conditions for ORCHESTRATE tier:
        - allowed_risks: ['Low', 'Medium', 'High']
        - required_complexity: 'High'
        - min_confidence: 0.70
        - required_evidence_quality: ['Strong', 'Insufficient']
        - policy_restriction: false
        - action_requires_escalation: false
        """
        norm_risk = str(state.risk).strip().capitalize() if state.risk else None
        norm_comp = str(state.complexity).strip().capitalize() if state.complexity else None

        allowed_risks = self.orchestrate_conditions.get(
            "allowed_risks", ["Low", "Medium", "High"]
        )
        if norm_risk not in allowed_risks:
            return False, f"Risk '{norm_risk}' not in allowed_risks {allowed_risks}"

        req_comp = self.orchestrate_conditions.get("required_complexity", "High")
        if norm_comp != req_comp:
            return False, f"Complexity '{norm_comp}' does not match required_complexity '{req_comp}'"

        min_conf = float(self.orchestrate_conditions.get("min_confidence", 0.70))
        if confidence < min_conf:
            return False, f"Confidence {confidence:.4f} below min_confidence {min_conf}"

        req_evidence = self.orchestrate_conditions.get(
            "required_evidence_quality", ["Strong", "Insufficient"]
        )
        if state.evidence_quality not in req_evidence:
            return False, f"Evidence quality '{state.evidence_quality}' not in required {req_evidence}"

        has_policy_restriction = bool(
            state.metadata.get("policy_restriction", False)
            or getattr(state, "policy_restriction", False)
        )
        if has_policy_restriction:
            return False, "policy_restriction is True"

        has_action_escalation = bool(
            state.metadata.get("action_requires_escalation", False)
            or getattr(state, "action_requires_escalation", False)
        )
        if has_action_escalation:
            return False, "action_requires_escalation is True"

        return True, "All ORCHESTRATE criteria satisfied"
