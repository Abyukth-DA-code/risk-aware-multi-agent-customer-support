"""
Triage Specialist Agent
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Role:
- Executes M1–M6 multi-task query analysis via ModelService.
- Evaluates domain, intent, sentiment, risk, and complexity signals.
- Flags sensitive actions and financial impacts early.
- Records structured audit trace in OrchestrationState.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional

from src.orchestration.model_service import ModelService
from src.orchestration.state import OrchestrationState

logger = logging.getLogger("agents.triage")


class TriageAgent:
    """Specialist agent responsible for initial inquiry triage and classification."""

    def __init__(self, model_service: Optional[ModelService] = None):
        self.model_service = model_service or ModelService.get_instance()

    def execute(self, state: OrchestrationState) -> OrchestrationState:
        """
        Executes query triage using M1–M6 models and updates OrchestrationState.
        """
        start_time = datetime.now(timezone.utc).isoformat()
        if "TriageAgent" not in state.selected_agents:
            state.selected_agents.append("TriageAgent")

        try:
            # If domain or triage is missing, run model service analysis
            if not state.intent or not state.domain:
                sens_str = "Yes" if state.sensitive_action else "No"
                fin_str = "Yes" if state.financial_impact else "No"
                analysis = self.model_service.analyze_query(
                    query=state.query,
                    domain=state.domain,
                    sensitive_action=sens_str,
                    financial_impact=fin_str,
                    action_type=state.action_type,
                )

                # Populate state attributes if not already set
                state.domain = state.domain or analysis.get("domain")
                state.domain_confidence = state.domain_confidence or analysis.get("domain_confidence")
                state.intent = state.intent or analysis.get("intent")
                state.intent_confidence = state.intent_confidence or analysis.get("intent_confidence")
                state.sentiment = state.sentiment or analysis.get("sentiment")
                state.sentiment_confidence = state.sentiment_confidence or analysis.get("sentiment_confidence")
                state.risk = state.risk or analysis.get("risk")
                state.risk_confidence = state.risk_confidence or analysis.get("risk_confidence")
                state.risk_class_probabilities = state.risk_class_probabilities or analysis.get("risk_class_probabilities", {})
                state.complexity = state.complexity or analysis.get("complexity")
                state.complexity_confidence = state.complexity_confidence or analysis.get("complexity_confidence")
                state.complexity_class_probabilities = state.complexity_class_probabilities or analysis.get("complexity_class_probabilities", {})
                state.provisional_autonomy_tier = state.provisional_autonomy_tier or analysis.get("provisional_autonomy_tier")
                state.provisional_escalation_required = state.provisional_escalation_required or analysis.get("provisional_escalation_required")
                state.provisional_confidence = state.provisional_confidence or analysis.get("provisional_confidence")
                state.provisional_class_probabilities = state.provisional_class_probabilities or analysis.get("provisional_class_probabilities", {})

            # Check for ambiguity in query phrasing
            cleaned_query = state.query.strip().lower()
            words = cleaned_query.split()
            is_underspecified = len(words) <= 3 and any(
                p in cleaned_query for p in ["cancel", "return", "refund", "status", "deadline", "fees", "exam"]
            )
            if is_underspecified and float(state.intent_confidence or 0.0) < 0.35:
                state.metadata["is_ambiguous"] = True

            state.agent_execution_trace.append({
                "agent": "TriageAgent",
                "timestamp": start_time,
                "action": "QUERY_TRIAGE",
                "status": "COMPLETED",
                "findings": {
                    "domain": state.domain,
                    "intent": state.intent,
                    "intent_confidence": state.intent_confidence,
                    "risk": state.risk,
                    "complexity": state.complexity,
                    "sentiment": state.sentiment,
                    "provisional_autonomy_tier": state.provisional_autonomy_tier,
                    "is_ambiguous": state.metadata.get("is_ambiguous", False),
                },
            })
            state.status = "CLASSIFIED"

        except Exception as exc:
            logger.error(f"TriageAgent execution encountered error: {exc}", exc_info=True)
            state.errors.append(f"TriageAgent error: {str(exc)}")
            state.agent_execution_trace.append({
                "agent": "TriageAgent",
                "timestamp": start_time,
                "action": "QUERY_TRIAGE",
                "status": "FAILED",
                "error": str(exc),
            })

        state.updated_at = datetime.now(timezone.utc).isoformat()
        return state
