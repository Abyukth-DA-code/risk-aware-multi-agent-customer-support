"""
Comprehensive Test Suite for Deterministic Safety Policy Engine
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation

Destination: tests/test_safety_policy.py
"""

import pytest
from pathlib import Path
from src.orchestration.state import OrchestrationState
from src.orchestration.safety_policy import (
    RULE_CRIT_01,
    RULE_EVID_CONF_03,
    RULE_EVID_INSUF_04,
    RULE_LOW_CONF_05,
    RULE_RISK_VERIF_02,
    RULE_SENS_FIN_06,
    SafetyPolicyEngine,
)


@pytest.fixture
def engine():
    """Initializes standard SafetyPolicyEngine with canonical autonomy_rules.yaml."""
    return SafetyPolicyEngine()


def _create_base_state(**kwargs) -> OrchestrationState:
    """Helper to construct a baseline OrchestrationState for test cases."""
    defaults = {
        "query": "Where is my replacement package?",
        "domain": "E-commerce & Retail",
        "intent": "Order & Shipping Inquiries",
        "sentiment": "Neutral",
        "risk": "Low",
        "complexity": "Low",
        "provisional_autonomy_tier": "ANSWER",
        "evidence_quality": "Strong",
        "retrieval_similarity_score": 0.85,
        "verification_required": False,
        "sensitive_action": False,
        "financial_impact": False,
    }
    defaults.update(kwargs)
    return OrchestrationState(**defaults)


# ===========================================================================
# 1. Configuration Loading & Invariants
# ===========================================================================

def test_engine_initialization_default_and_custom_paths(engine):
    """Verify engine loads default config and resolves paths properly."""
    assert engine.config_path.is_file()
    assert "autonomy_tiers" in engine.config
    assert "observable_confidence_weights" in engine.config

    # Check exact YAML trigger criteria
    rule_ids = [c["rule_id"] for c in engine.trigger_criteria]
    assert rule_ids == [
        RULE_CRIT_01,
        RULE_RISK_VERIF_02,
        RULE_EVID_CONF_03,
        RULE_EVID_INSUF_04,
        RULE_LOW_CONF_05,
        RULE_SENS_FIN_06,
    ]


def test_engine_initialization_invalid_path():
    """Verify engine raises FileNotFoundError for non-existent config path."""
    with pytest.raises(FileNotFoundError):
        SafetyPolicyEngine("non_existent_rules.yaml")


# ===========================================================================
# 2. Mandatory Escalation Rules (CRIT_01 to SENS_FIN_06)
# ===========================================================================

def test_rule_crit_01_triggers_on_critical_risk(engine):
    """CRIT_01: risk == 'Critical' triggers immediate escalation and overrides M6."""
    state = _create_base_state(
        risk="Critical",
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ESCALATE"
    assert result.final_escalation_required is True
    assert result.safety_rule_triggered == RULE_CRIT_01
    assert result.safety_override is True
    assert result.decision_source == "Deterministic_Safety_Rule"
    assert result.escalation_reason == "Critical autonomy risk or severe security concern"
    assert result.status == "ESCALATED"
    assert result.escalation_dossier is not None
    assert result.escalation_dossier["triggered_rule"] == RULE_CRIT_01


def test_rule_risk_verif_02_triggers_when_high_risk_and_verification_fails(engine):
    """RISK_VERIF_02: risk == 'High', verification_required == True, verification_score < 0.80."""
    state = _create_base_state(
        risk="High",
        verification_required=True,
        verification_score=0.72,
        provisional_autonomy_tier="ORCHESTRATE",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ESCALATE"
    assert result.final_escalation_required is True
    assert result.safety_rule_triggered == RULE_RISK_VERIF_02
    assert result.safety_override is True
    assert result.escalation_reason == "High Risk + Verification/Evidence Issue"
    assert result.status == "ESCALATED"


def test_rule_evid_conf_03_triggers_on_conflicting_evidence_medium_or_high_risk(engine):
    """EVID_CONF_03: evidence_quality == 'Conflicting' and risk in ['Medium', 'High']."""
    # Test with Medium risk
    state_med = _create_base_state(
        risk="Medium",
        evidence_quality="Conflicting",
        provisional_autonomy_tier="ANSWER",
    )
    res_med = engine.evaluate(state_med)
    assert res_med.final_autonomy_tier == "ESCALATE"
    assert res_med.safety_rule_triggered == RULE_EVID_CONF_03
    assert res_med.escalation_reason == "Insufficient/Conflicting Evidence"

    # Test with High risk via has_conflicting_evidence flag
    state_high = _create_base_state(
        risk="High",
        has_conflicting_evidence=True,
        evidence_quality="Pending",
        provisional_autonomy_tier="ORCHESTRATE",
    )
    res_high = engine.evaluate(state_high)
    assert res_high.final_autonomy_tier == "ESCALATE"
    assert res_high.safety_rule_triggered == RULE_EVID_CONF_03


def test_rule_evid_insuf_04_triggers_on_insufficient_evidence_medium_or_high_risk(engine):
    """EVID_INSUF_04: evidence_quality == 'Insufficient' and risk in ['Medium', 'High']."""
    state = _create_base_state(
        risk="Medium",
        evidence_quality="Insufficient",
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ESCALATE"
    assert result.final_escalation_required is True
    assert result.safety_rule_triggered == RULE_EVID_INSUF_04
    assert result.escalation_reason == "Insufficient/Conflicting Evidence"


def test_rule_low_conf_05_triggers_on_low_confidence(engine):
    """LOW_CONF_05: final_confidence < 0.65."""
    # Low similarity score causing composite confidence to fall below 0.65
    state = _create_base_state(
        risk="Low",
        complexity="Low",
        retrieval_similarity_score=0.40,
        metadata={"document_freshness_score": 0.0},  # Lowers composite confidence
        evidence_quality="Pending",
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    assert result.final_confidence < 0.65
    assert result.final_autonomy_tier == "ESCALATE"
    assert result.safety_rule_triggered == RULE_LOW_CONF_05
    assert result.escalation_reason == "Low Confidence Review"


def test_rule_sens_fin_06_triggers_on_sensitive_financial_high_risk(engine):
    """SENS_FIN_06: sensitive_action == True, financial_impact == True, risk == 'High'."""
    state = _create_base_state(
        risk="High",
        complexity="High",
        sensitive_action=True,
        financial_impact=True,
        retrieval_similarity_score=0.90,
        verification_required=False,
        evidence_quality="Strong",
        provisional_autonomy_tier="ORCHESTRATE",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ESCALATE"
    assert result.final_escalation_required is True
    assert result.safety_rule_triggered == RULE_SENS_FIN_06
    assert result.escalation_reason == "Human Approval Required"


# ===========================================================================
# 3. Threshold Boundaries
# ===========================================================================

def test_boundary_risk_verif_02(engine):
    """Boundary test for verification_score threshold 0.80."""
    # 0.799 (< 0.80) -> MUST trigger RISK_VERIF_02
    state_below = _create_base_state(
        risk="High",
        complexity="High",
        verification_required=True,
        verification_score=0.799,
        retrieval_similarity_score=0.90,
        evidence_quality="Strong",
    )
    res_below = engine.evaluate(state_below)
    assert res_below.safety_rule_triggered == RULE_RISK_VERIF_02
    assert res_below.final_autonomy_tier == "ESCALATE"

    # 0.800 (>= 0.80) -> MUST NOT trigger RISK_VERIF_02
    state_at = _create_base_state(
        risk="High",
        complexity="High",
        verification_required=True,
        verification_score=0.800,
        retrieval_similarity_score=0.90,
        evidence_quality="Strong",
        provisional_autonomy_tier="ORCHESTRATE",
    )
    res_at = engine.evaluate(state_at)
    assert res_at.safety_rule_triggered != RULE_RISK_VERIF_02
    assert res_at.final_autonomy_tier == "ORCHESTRATE"


def test_boundary_low_conf_05(engine):
    """Boundary test for final_confidence threshold 0.65."""
    # Metadata injection to control confidence boundary precisely
    state_below = _create_base_state(
        risk="Low",
        complexity="Low",
        metadata={
            "retrieval_similarity": 0.649,
            "source_authority_score": 0.649,
            "document_freshness_score": 0.649,
            "evidence_agreement_score": 0.649,
        },
        retrieval_similarity_score=0.649,
        evidence_quality="Strong",
    )
    res_below = engine.evaluate(state_below)
    assert res_below.final_confidence < 0.65
    assert res_below.safety_rule_triggered == RULE_LOW_CONF_05
    assert res_below.final_autonomy_tier == "ESCALATE"

    # At or above threshold (0.650)
    state_above = _create_base_state(
        risk="Low",
        complexity="High",
        metadata={
            "retrieval_similarity": 0.72,
            "source_authority_score": 0.72,
            "document_freshness_score": 0.72,
            "evidence_agreement_score": 0.72,
        },
        retrieval_similarity_score=0.72,
        evidence_quality="Strong",
        provisional_autonomy_tier="ORCHESTRATE",
    )
    res_above = engine.evaluate(state_above)
    assert res_above.final_confidence >= 0.70
    assert res_above.safety_rule_triggered != RULE_LOW_CONF_05


# ===========================================================================
# 4. Missing / Invalid Inputs & Conservative Fail-Safe
# ===========================================================================

def test_missing_verification_score_when_verification_required_triggers_rule(engine):
    """When verification is required but score is None, do not assume safe."""
    state = _create_base_state(
        risk="High",
        verification_required=True,
        verification_score=None,  # Missing verification signal
        provisional_autonomy_tier="ORCHESTRATE",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ESCALATE"
    assert result.safety_rule_triggered == RULE_RISK_VERIF_02


def test_missing_evidence_and_chunks_fails_safe_to_zero_confidence(engine):
    """When no retrieval chunks or similarity exist, confidence fails safe to 0.0."""
    state = OrchestrationState(
        query="Unanswerable or unindexed query",
        retrieved_chunks=[],
        retrieval_similarity_score=None,
        evidence_quality="Pending",
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    assert result.final_confidence == 0.0
    assert result.final_autonomy_tier == "ESCALATE"
    assert result.safety_rule_triggered == RULE_LOW_CONF_05


def test_missing_risk_or_complexity_disqualifies_autonomous_tiers(engine):
    """When risk or complexity is None, state fails autonomous qualification and safely escalates."""
    state = _create_base_state(
        risk=None,  # Missing risk classification
        complexity="Low",
        retrieval_similarity_score=0.90,
        evidence_quality="Strong",
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    # Missing risk does not trigger CRIT_01 or LOW_CONF_05, but disqualifies ANSWER / ORCHESTRATE
    assert result.final_autonomy_tier == "ESCALATE"
    assert result.safety_rule_triggered is None  # Does not invent a YAML rule ID
    assert "Disqualified" in result.escalation_reason


# ===========================================================================
# 5. Observable Confidence Calculation & Dynamic Weight Normalization
# ===========================================================================

def test_confidence_weight_renormalization_over_available_signals(engine):
    """Only authentic signals are used; weights are re-normalized over available signals."""
    chunks = [
        {
            "chunk_id": "CHK-001",
            "similarity": 0.80,
            "authority": "High",
            "freshness_status": "Current",
        },
        {
            "chunk_id": "CHK-002",
            "similarity": 0.85,
            "authority": "High",
            "freshness_status": "Current",
        },
    ]
    state = _create_base_state(
        retrieved_chunks=chunks,
        retrieval_similarity_score=0.85,
        evidence_quality="Strong",
    )
    breakdown = engine.get_confidence_breakdown(state)

    # Signals present: retrieval_similarity (0.25), source_authority (0.15), document_freshness (0.15), evidence_agreement (0.10)
    # Total available weight: 0.65 (reranker and db consistency are unavailable)
    assert breakdown["total_available_weight"] == 0.65
    assert "reranker_score" in breakdown["unavailable_signals"]
    assert "database_consistency" in breakdown["unavailable_signals"]

    # Weighted sum: 0.25*0.85 + 0.15*1.0 + 0.15*1.0 + 0.10*1.0 = 0.2125 + 0.15 + 0.15 + 0.10 = 0.6125
    # Composite: 0.6125 / 0.65 = 0.9423
    assert breakdown["composite_confidence"] == pytest.approx(0.9423, abs=1e-3)


def test_confidence_outdated_document_lowers_freshness(engine):
    """When chunks have outdated evidence, document_freshness signal becomes 0.0."""
    chunks = [
        {
            "chunk_id": "CHK-OLD",
            "similarity": 0.80,
            "authority": "High",
            "freshness_status": "Outdated",
        }
    ]
    state = _create_base_state(
        retrieved_chunks=chunks,
        retrieval_similarity_score=0.80,
        has_outdated_evidence=True,
        evidence_quality="Strong",
    )
    breakdown = engine.get_confidence_breakdown(state)
    assert breakdown["available_signals"]["document_freshness"]["value"] == 0.0


def test_confidence_all_six_signals_present(engine):
    """When all six signals are provided, total weight is 1.00."""
    state = _create_base_state(
        retrieval_similarity_score=0.80,
        evidence_quality="Strong",
        metadata={
            "reranker_score": 0.85,
            "source_authority_score": 1.0,
            "document_freshness_score": 1.0,
            "evidence_agreement_score": 1.0,
            "database_consistency_score": 0.90,
        },
    )
    breakdown = engine.get_confidence_breakdown(state)
    assert breakdown["total_available_weight"] == 1.00
    assert len(breakdown["unavailable_signals"]) == 0
    # Expected: 0.25*0.80 + 0.25*0.85 + 0.15*1.0 + 0.15*1.0 + 0.10*1.0 + 0.10*0.90
    # = 0.20 + 0.2125 + 0.15 + 0.15 + 0.10 + 0.09 = 0.9025
    assert breakdown["composite_confidence"] == pytest.approx(0.9025, abs=1e-3)


# ===========================================================================
# 6. Tier Qualification (ANSWER vs ORCHESTRATE vs Fallback)
# ===========================================================================

def test_answer_tier_qualification(engine):
    """State satisfying all ANSWER conditions qualifies for ANSWER."""
    state = _create_base_state(
        risk="Low",
        complexity="Low",
        retrieval_similarity_score=0.90,
        evidence_quality="Strong",
        verification_required=False,
        sensitive_action=False,
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ANSWER"
    assert result.final_escalation_required is False
    assert result.safety_rule_triggered is None
    assert result.safety_override is False
    assert result.decision_source == "Model_Verified"
    assert result.status == "VERIFIED"
    assert result.escalation_dossier is None


def test_orchestrate_tier_qualification(engine):
    """State satisfying all ORCHESTRATE conditions qualifies for ORCHESTRATE."""
    state = _create_base_state(
        risk="High",
        complexity="High",
        retrieval_similarity_score=0.90,
        evidence_quality="Strong",
        verification_required=False,
        sensitive_action=False,
        provisional_autonomy_tier="ORCHESTRATE",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ORCHESTRATE"
    assert result.final_escalation_required is False
    assert result.safety_rule_triggered is None
    assert result.safety_override is False
    assert result.decision_source == "Model_Verified"
    assert result.status == "VERIFIED"


def test_tier_disqualification_conservative_fallback(engine):
    """When state qualifies for neither ANSWER nor ORCHESTRATE, fall back conservatively to ESCALATE."""
    # Medium risk (disqualifies ANSWER), Medium complexity (disqualifies ORCHESTRATE)
    state = _create_base_state(
        risk="Medium",
        complexity="Medium",
        retrieval_similarity_score=0.90,
        evidence_quality="Strong",
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ESCALATE"
    assert result.final_escalation_required is True
    assert result.safety_rule_triggered is None  # Must NOT invent a YAML rule ID
    assert result.safety_override is True
    assert result.decision_source == "Deterministic_Safety_Rule"
    assert "Disqualified" in result.escalation_reason
    assert result.escalation_dossier is not None


# ===========================================================================
# 7. Provisional M6 Overrides & Invariants
# ===========================================================================

def test_m6_provisional_answer_overridden_by_mandatory_rule(engine):
    """Provisional ANSWER cannot bypass a mandatory safety rule."""
    state = _create_base_state(
        risk="Critical",
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ESCALATE"
    assert result.safety_override is True
    assert result.decision_source == "Deterministic_Safety_Rule"


def test_m6_provisional_escalate_preserved_when_no_rule_triggers(engine):
    """Existing provisional ESCALATE is conservatively preserved if no specific rule fires."""
    state = _create_base_state(
        risk="Low",
        complexity="Low",
        retrieval_similarity_score=0.90,
        evidence_quality="Strong",
        provisional_autonomy_tier="ESCALATE",
        escalation_reason="M6 model triage recommended escalation",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ESCALATE"
    assert result.final_escalation_required is True
    assert result.safety_rule_triggered is None
    assert result.safety_override is False
    assert result.decision_source == "Model_Verified"
    assert result.status == "ESCALATED"


def test_m6_provisional_tier_promoted_when_complexity_requires_orchestration(engine):
    """Provisional ANSWER promoted to ORCHESTRATE when complexity is High."""
    state = _create_base_state(
        risk="Low",
        complexity="High",
        retrieval_similarity_score=0.90,
        evidence_quality="Strong",
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ORCHESTRATE"
    assert result.final_escalation_required is False
    assert result.safety_override is True
    assert result.decision_source == "Deterministic_Safety_Rule"


# ===========================================================================
# 8. Escalation Dossier Creation & Structure
# ===========================================================================

def test_escalation_dossier_contains_all_traceable_fields(engine):
    """Escalation dossier captures full query context and chunk IDs."""
    chunks = [
        {"chunk_id": "CHUNK-101", "similarity": 0.85},
        {"chunk_id": "CHUNK-102", "similarity": 0.82},
    ]
    state = _create_base_state(
        risk="Critical",
        retrieved_chunks=chunks,
    )
    result = engine.evaluate(state)

    dossier = result.escalation_dossier
    assert isinstance(dossier, dict)
    assert dossier["triggered_rule"] == RULE_CRIT_01
    assert dossier["reason"] == "Critical autonomy risk or severe security concern"
    assert dossier["query"] == state.query
    assert dossier["domain"] == state.domain
    assert dossier["retrieved_chunk_ids"] == ["CHUNK-101", "CHUNK-102"]
    assert "timestamp" in dossier
    assert "recommended_action" in dossier
    assert "context_summary" in dossier


def test_escalation_dossier_is_none_for_autonomous_resolutions(engine):
    """Escalation dossier and reason remain None when autonomous tiers are selected."""
    state = _create_base_state(
        risk="Low",
        complexity="Low",
        retrieval_similarity_score=0.90,
        evidence_quality="Strong",
        provisional_autonomy_tier="ANSWER",
    )
    result = engine.evaluate(state)

    assert result.final_autonomy_tier == "ANSWER"
    assert result.escalation_dossier is None
    assert result.escalation_reason is None
