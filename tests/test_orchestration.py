"""
Automated Test Suite: Multi-Agent Orchestration Pipeline, Agent Coordination & State Transitions
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation
File: tests/test_orchestration.py
"""

import pytest
from src.orchestration.graph import OrchestrationPipeline, get_orchestration_pipeline
from src.orchestration.state import OrchestrationState
from src.orchestration.model_service import ModelService
from src.orchestration.safety_policy import SafetyPolicyEngine
from src.agents.triage_agent import TriageAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.supervisor_agent import SupervisorAgent


@pytest.fixture(scope="module")
def pipeline():
    """Initializes a shared OrchestrationPipeline with all specialist agents."""
    return get_orchestration_pipeline()


# ===========================================================================
# 1. Multi-Agent Coordination & Trace Integrity
# ===========================================================================

def test_multi_agent_pipeline_trace_integrity(pipeline):
    """
    Verifies that all 4 specialist agents execute sequentially and record
    structured traces in state.agent_execution_trace.
    """
    query = "What is the return window for clothing items?"
    state = pipeline.run(
        query=query,
        domain="E-commerce",
        generate_grounded_response=False,
    )

    assert state.query == query
    assert state.domain == "E-commerce"
    assert state.intent is not None
    assert state.risk is not None
    assert state.complexity is not None
    assert state.final_autonomy_tier in ["ANSWER", "ORCHESTRATE", "ESCALATE", "CLARIFY"]

    # Verify that agents logged execution traces
    trace_agents = [t.get("agent") for t in state.agent_execution_trace]
    assert "TriageAgent" in trace_agents
    assert "RetrievalAgent" in trace_agents
    assert "PolicyAgent" in trace_agents
    assert "SupervisorAgent" in trace_agents


# ===========================================================================
# 2. Pronoun & Reference Disambiguation Across Turns
# ===========================================================================

def test_contextual_pronoun_resolution_in_pipeline(pipeline):
    """
    Verifies that a follow-up query containing pronouns ('them', 'it') is
    disambiguated using conversation_history before retrieval.
    """
    history = [
        {"role": "user", "content": "What is the warranty policy for laptop computers?"},
        {"role": "assistant", "content": "Laptops come with a 1-year manufacturer warranty covering hardware defects."},
    ]

    follow_up = "Does it cover accidental water damage or screen drops?"
    state = pipeline.run(
        query=follow_up,
        domain="E-commerce",
        conversation_history=history,
    )

    assert state.query == follow_up
    assert state.resolved_query is not None
    assert "warranty" in state.resolved_query.lower() or "laptop" in state.resolved_query.lower()


# ===========================================================================
# 3. Dynamic Clarification on Ambiguous Low-Information Queries
# ===========================================================================

def test_ambiguous_query_routing_to_clarify_or_escalate(pipeline):
    """
    Verifies that vague queries without clear domain actions do not hallucinate
    answers and route to CLARIFY or conservative ESCALATE.
    """
    state = pipeline.run(
        query="Help me please",
        domain="E-commerce",
    )

    # Vague query with no evidence should either request clarification or escalate safely
    assert state.final_autonomy_tier in ["CLARIFY", "ESCALATE"]
    if state.final_autonomy_tier == "CLARIFY":
        assert state.final_response is not None
        assert "?" in state.final_response
    else:
        assert state.final_escalation_required is True


# ===========================================================================
# 4. Critical Risk & Financial Override in Orchestration
# ===========================================================================

def test_pipeline_critical_security_escalation(pipeline):
    """
    Verifies that critical risk security events force ESCALATE tier through the entire pipeline.
    """
    state = pipeline.run(
        query="Emergency: SQL injection detected on customer portal, database credentials leaked",
        domain="E-commerce",
    )

    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True
    assert state.escalation_dossier is not None
    assert state.status == "ESCALATED"


def test_pipeline_sensitive_financial_override(pipeline):
    """
    Verifies that high-risk sensitive financial operations trigger escalation.
    """
    state = pipeline.run(
        query="I demand an immediate chargeback of $2500 for unauthorized billing",
        domain="E-commerce",
        sensitive_action=True,
        financial_impact=True,
    )

    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True
    assert state.safety_rule_triggered in ["CRIT_01", "SENS_FIN_06", "RISK_VERIF_02"]


# ===========================================================================
# 5. Fail-Safe Execution on Missing / Empty Retrieval
# ===========================================================================

def test_pipeline_empty_retrieval_fail_safe(pipeline):
    """
    Verifies that when retrieval finds zero chunks, the pipeline sets
    evidence_quality = 'Insufficient' and safely refrains from autonomous ANSWER.
    """
    # Create custom state with no chunks
    state = OrchestrationState(
        query="Specific query with no matching documentation xyz123456",
        domain="E-commerce",
        risk="Medium",
        complexity="Low",
        provisional_autonomy_tier="ANSWER",
        retrieved_chunks=[],
        retrieval_similarity_score=None,
    )

    # Execute policy agent
    state = pipeline.policy_agent.execute(state)
    assert state.evidence_quality == "Insufficient"
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.safety_rule_triggered == "EVID_INSUF_04"


# ===========================================================================
# 6. State Serialization Round-Trip Integrity
# ===========================================================================

def test_orchestration_state_json_serialization(pipeline):
    """
    Verifies that OrchestrationState can be serialized to JSON safely without
    losing fields or encountering NumPy/datetime serialization errors.
    """
    state = pipeline.run(
        query="How do I track my delivery status?",
        domain="E-commerce",
    )

    json_dict = state.to_dict()
    assert isinstance(json_dict, dict)
    assert json_dict["query"] == "How do I track my delivery status?"
    assert "agent_execution_trace" in json_dict
    assert isinstance(json_dict["agent_execution_trace"], list)
