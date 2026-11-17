"""
Automated Test Suite: Deterministic Safety Guardrails, M6 False Negative Overrides & HITL Safety Gating
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation
File: tests/test_safety.py
"""

import pytest
from src.orchestration.state import OrchestrationState
from src.orchestration.safety_policy import SafetyPolicyEngine
from src.agents.policy_agent import PolicyAgent
from src.agents.supervisor_agent import SupervisorAgent


@pytest.fixture
def safety_components():
    """Initializes policy agent and supervisor agent with default safety engine."""
    engine = SafetyPolicyEngine()
    policy_agent = PolicyAgent(safety_engine=engine)
    supervisor_agent = SupervisorAgent()
    return {
        "engine": engine,
        "policy_agent": policy_agent,
        "supervisor_agent": supervisor_agent,
    }


# ===========================================================================
# 1. M6 Statistical Classifier False-Negative Override Tests
# ===========================================================================

def test_m6_false_negative_critical_risk_override(safety_components):
    """
    Verifies that if statistical classifier M6 provisionally outputs ANSWER,
    but risk is Critical, Rule CRIT_01 deterministically overrides M6 to ESCALATE.
    """
    policy_agent = safety_components["policy_agent"]
    supervisor_agent = safety_components["supervisor_agent"]

    state = OrchestrationState(
        query="System database compromised, customer records exported without auth",
        domain="E-commerce",
        risk="Critical",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",  # M6 false negative
        retrieved_chunks=[{"chunk_id": "chk_sec_01", "chunk_text": "Security incident response protocol"}],
        evidence_quality="Strong",
        retrieval_similarity_score=0.92,
    )

    state = policy_agent.execute(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True
    assert state.safety_rule_triggered == "CRIT_01"
    assert state.decision_source == "Deterministic_Safety_Rule"
    assert state.escalation_dossier is not None
    dossier = state.escalation_dossier
    dossier_rule = dossier.get("triggered_rule") if isinstance(dossier, dict) else getattr(dossier, "triggered_rule", None)
    assert dossier_rule == "CRIT_01"

    # Verify customer-facing response is populated and safe
    state = supervisor_agent.execute(state)
    esc_notice = state.decision_rationale or (state.escalation_dossier.get("escalation_notice") if isinstance(state.escalation_dossier, dict) else None)
    assert esc_notice is not None
    assert len(esc_notice.strip()) > 0
    assert "human" in esc_notice.lower() or "specialist" in esc_notice.lower() or "support" in esc_notice.lower()


def test_m6_false_negative_sensitive_financial_override(safety_components):
    """
    Verifies that if M6 provisionally outputs ANSWER on a high-risk financial dispute,
    Rule SENS_FIN_06 deterministically overrides to ESCALATE when verification passes.
    """
    policy_agent = safety_components["policy_agent"]
    supervisor_agent = safety_components["supervisor_agent"]

    state = OrchestrationState(
        query="Immediate chargeback required for $1500 unauthorized transaction",
        domain="E-commerce",
        risk="High",
        complexity="Low",
        sensitive_action=True,
        financial_impact=True,
        verification_required=True,
        metadata={"verification_score": 0.95},  # Verification passed, so RISK_VERIF_02 does not trigger
        provisional_autonomy_tier="ANSWER",  # M6 false negative
        retrieved_chunks=[{"chunk_id": "chk_fin_01", "chunk_text": "Chargeback dispute policy for unauthorized transactions."}],
        evidence_quality="Strong",
        retrieval_similarity_score=0.88,
    )

    state = policy_agent.execute(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.safety_rule_triggered == "SENS_FIN_06"
    assert state.decision_source == "Deterministic_Safety_Rule"

    state = supervisor_agent.execute(state)
    esc_notice = state.decision_rationale or (state.escalation_dossier.get("escalation_notice") if isinstance(state.escalation_dossier, dict) else None)
    assert esc_notice is not None
    assert "escalat" in esc_notice.lower() or "specialist" in esc_notice.lower()


def test_m6_false_negative_conflicting_evidence_override(safety_components):
    """
    Verifies that if M6 provisionally outputs ANSWER, but knowledge base evidence
    is conflicting for medium risk, Rule EVID_CONF_03 forces ESCALATE.
    """
    policy_agent = safety_components["policy_agent"]

    state = OrchestrationState(
        query="What is the refund eligibility period?",
        domain="E-commerce",
        risk="Medium",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",  # M6 false negative
        retrieved_chunks=[
            {"chunk_id": "chk_ref_01", "chunk_text": "Refunds within 14 days", "conflict_flag": True},
            {"chunk_id": "chk_ref_02", "chunk_text": "Refunds within 30 days"}
        ],
        evidence_quality="Conflicting",
        has_conflicting_evidence=True,
        retrieval_similarity_score=0.78,
    )

    state = policy_agent.execute(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.safety_rule_triggered == "EVID_CONF_03"
    assert state.decision_source == "Deterministic_Safety_Rule"


def test_m6_false_negative_insufficient_evidence_override(safety_components):
    """
    Verifies that if M6 provisionally outputs ANSWER, but evidence is Insufficient
    for medium risk, Rule EVID_INSUF_04 forces ESCALATE.
    """
    policy_agent = safety_components["policy_agent"]

    state = OrchestrationState(
        query="Can I return an item without packaging?",
        domain="E-commerce",
        risk="Medium",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",  # M6 false negative
        retrieved_chunks=[{"chunk_id": "chk_01", "chunk_text": "General store information"}],
        evidence_quality="Insufficient",
        retrieval_similarity_score=0.38,  # Below 0.45 threshold
    )

    state = policy_agent.execute(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.safety_rule_triggered == "EVID_INSUF_04"
    assert state.decision_source == "Deterministic_Safety_Rule"


def test_m6_false_negative_verification_failure_override(safety_components):
    """
    Verifies that if verification is required for High risk and verification score
    is below 0.80, Rule RISK_VERIF_02 forces ESCALATE despite provisional ANSWER.
    """
    policy_agent = safety_components["policy_agent"]

    state = OrchestrationState(
        query="Verify and cancel high-value order #89212",
        domain="E-commerce",
        risk="High",
        complexity="Low",
        verification_required=True,
        metadata={"verification_score": 0.45},  # Failed verification (< 0.80)
        provisional_autonomy_tier="ANSWER",  # M6 false negative
        retrieved_chunks=[{"chunk_id": "chk_01", "chunk_text": "Cancellation policy"}],
        evidence_quality="Strong",
        retrieval_similarity_score=0.82,
    )

    state = policy_agent.execute(state)
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.safety_rule_triggered == "RISK_VERIF_02"
    assert state.decision_source == "Deterministic_Safety_Rule"


# ===========================================================================
# 2. Boundary and Confidence Invariants
# ===========================================================================

def test_low_confidence_boundary_065(safety_components):
    """
    Verifies the exact 0.65 confidence boundary:
    - final_confidence < 0.65 triggers LOW_CONF_05
    - final_confidence >= 0.65 does NOT trigger LOW_CONF_05
    """
    engine = safety_components["engine"]

    # Case A: Low confidence (< 0.65)
    state_low = OrchestrationState(
        query="General store policy question",
        domain="E-commerce",
        risk="Low",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        retrieval_similarity_score=0.40,
        evidence_quality="Strong",
    )
    state_low.metadata["observable_confidence"] = 0.649
    state_low = engine.evaluate(state_low)
    assert state_low.final_autonomy_tier == "ESCALATE"
    assert state_low.safety_rule_triggered == "LOW_CONF_05"

    # Case B: Sufficient confidence (>= 0.65)
    state_ok = OrchestrationState(
        query="General store policy question",
        domain="E-commerce",
        risk="Low",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        retrieval_similarity_score=0.85,
        evidence_quality="Strong",
    )
    state_ok.metadata["observable_confidence"] = 0.651
    state_ok = engine.evaluate(state_ok)
    assert state_ok.safety_rule_triggered != "LOW_CONF_05"


# ===========================================================================
# 3. Determinism and Idempotence
# ===========================================================================

def test_safety_evaluation_determinism(safety_components):
    """
    Verifies that evaluating the exact same query state 50 times yields
    strictly identical decisions, scores, and triggered rules.
    """
    policy_agent = safety_components["policy_agent"]

    decisions = []
    rules = []
    confidences = []

    for _ in range(50):
        state = OrchestrationState(
            query="Repeatability verification test query",
            domain="E-commerce",
            risk="High",
            complexity="Medium",
            sensitive_action=True,
            financial_impact=True,
            verification_required=True,
            metadata={"verification_score": 0.95},
            retrieved_chunks=[{"chunk_id": "chk_repeat_01", "chunk_text": "Financial transaction policy"}],
            evidence_quality="Strong",
            retrieval_similarity_score=0.72,
        )
        evaluated = policy_agent.execute(state)
        decisions.append(evaluated.final_autonomy_tier)
        rules.append(evaluated.safety_rule_triggered)
        confidences.append(evaluated.final_confidence)

    assert len(set(decisions)) == 1, "Non-deterministic autonomy tier detected!"
    assert len(set(rules)) == 1, "Non-deterministic safety rule trigger detected!"
    assert len(set(confidences)) == 1, "Non-deterministic confidence calculation detected!"
    assert decisions[0] == "ESCALATE"
    assert rules[0] == "SENS_FIN_06"


# ===========================================================================
# 4. Fail-Safe Behavior on Corrupted / Missing State Data
# ===========================================================================

def test_safety_fail_safe_missing_risk(safety_components):
    """
    Verifies that if risk is missing (None or empty), the engine safely disqualifies
    autonomous execution and defaults conservatively to ESCALATE.
    """
    engine = safety_components["engine"]

    state = OrchestrationState(
        query="Query with missing risk assessment",
        domain="E-commerce",
        risk=None,  # Missing risk
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        evidence_quality="Strong",
    )

    evaluated = engine.evaluate(state)
    assert evaluated.final_autonomy_tier == "ESCALATE"
    assert evaluated.final_escalation_required is True
    assert evaluated.escalation_reason is not None
