"""
Supervisor Specialist Agent
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Role:
- Evaluates collective triage, retrieval, and policy findings.
- Executes adaptive resolution:
  1. ANSWER: Synthesizes citation-grounded response via GroundedAnswerGenerator.
  2. CLARIFY: Generates targeted clarifying prompt on ambiguous inquiries without safety breaches.
  3. ESCALATE: Refuses autonomous answer, persists complete EscalationDossier to SQLite queue,
     and formats structured human escalation notice with verifiable escalation_id.
- Populates state.final_response consistently.
- Records full execution trace in OrchestrationState.
"""

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, Optional
import uuid

from src.agents.grounded_generator import GroundedAnswerGenerator
from src.orchestration.state import OrchestrationState
from src.workspaces.db import WorkspaceDB

logger = logging.getLogger("agents.supervisor")


class SupervisorAgent:
    """Master supervisor agent coordinating final decision synthesis and escalation queuing."""

    def __init__(self, db: Optional[WorkspaceDB] = None):
        self.db = db

    def _get_db(self) -> WorkspaceDB:
        if self.db is None:
            self.db = WorkspaceDB()
        return self.db

    def execute(
        self,
        state: OrchestrationState,
        workspace_id: Optional[str] = None,
        generate_grounded_response: bool = True,
    ) -> OrchestrationState:
        """
        Executes adaptive decision synthesis:
        Routes to ANSWER, CLARIFY, or ESCALATE.
        """
        start_time = datetime.now(timezone.utc).isoformat()
        t_sup_start = time.perf_counter()
        state.metadata.setdefault("latency_generation_ms", 0.0)

        def _finalize(st: OrchestrationState) -> OrchestrationState:
            lat_gen = float(st.metadata.get("latency_generation_ms", 0.0) or 0.0)
            lat_sup = max(0.0, (time.perf_counter() - t_sup_start) * 1000 - lat_gen)
            st.metadata["latency_supervisor_ms"] = round(lat_sup, 3)
            return st

        if "SupervisorAgent" not in state.selected_agents:
            state.selected_agents.append("SupervisorAgent")

        ws_id = workspace_id or state.metadata.get("workspace_id")

        try:
            # -------------------------------------------------------------
            # Pathway 1: Mandatory Escalation (Safety Policy or Evidence Gate)
            # -------------------------------------------------------------
            if state.final_escalation_required or state.final_autonomy_tier == "ESCALATE":
                escalation_id = f"esc_{uuid.uuid4().hex[:12]}"
                esc_reason = state.escalation_reason or state.decision_rationale or "Safety policy escalation triggered."
                rec_action = (
                    state.escalation_dossier.get("recommended_action")
                    if isinstance(state.escalation_dossier, dict)
                    else "Human specialist review and manual resolution."
                )

                escalation_notice = (
                    f"Your request has been routed to a human customer support specialist for review "
                    f"(Escalation ID: {escalation_id}).\n"
                    f"Reason: {esc_reason}\n"
                    f"A representative will review the case details and follow up promptly."
                )

                # Persist to SQLite escalation queue if workspace context is present
                if ws_id:
                    db = state.metadata.get("db") or self._get_db()
                    tenant_id = state.metadata.get("tenant_id")
                    if not tenant_id:
                        ws_record = db.get_workspace(ws_id)
                        tenant_id = ws_record["tenant_id"] if ws_record else "ten_unknown"

                    chunks_json = json.dumps(state.retrieved_chunks, default=str)
                    dossier_record = {
                        "escalation_id": escalation_id,
                        "workspace_id": ws_id,
                        "tenant_id": tenant_id,
                        "session_id": state.session_id,
                        "customer_id": state.customer_id,
                        "query": state.query,
                        "domain": state.domain or "General",
                        "intent": state.intent,
                        "intent_confidence": state.intent_confidence,
                        "risk": state.risk,
                        "risk_confidence": state.risk_confidence,
                        "complexity": state.complexity,
                        "provisional_autonomy_tier": state.provisional_autonomy_tier,
                        "final_autonomy_tier": state.final_autonomy_tier or "ESCALATE",
                        "evidence_quality": state.evidence_quality,
                        "retrieval_similarity_score": state.retrieval_similarity_score,
                        "retrieved_chunks_json": chunks_json,
                        "safety_rule_triggered": state.safety_rule_triggered,
                        "escalation_reason": esc_reason,
                        "recommended_action": rec_action,
                        "status": "PENDING",
                    }
                    try:
                        db.create_escalation(dossier_record)
                        logger.info(f"Escalation dossier '{escalation_id}' enqueued for workspace '{ws_id}'.")
                    except Exception as db_exc:
                        logger.warning(f"Failed to persist escalation dossier: {db_exc}")

                # Ensure escalation_dossier has escalation_id and structured notice
                if isinstance(state.escalation_dossier, dict):
                    state.escalation_dossier["escalation_id"] = escalation_id
                    state.escalation_dossier["escalation_notice"] = escalation_notice
                else:
                    state.escalation_dossier = {
                        "escalation_id": escalation_id,
                        "reason": esc_reason,
                        "escalation_notice": escalation_notice,
                    }

                state.status = "ESCALATED"
                state.final_response = None
                state.decision_rationale = escalation_notice

                state.agent_execution_trace.append({
                    "agent": "SupervisorAgent",
                    "timestamp": start_time,
                    "action": "ESCALATE_ROUTING",
                    "status": "COMPLETED",
                    "decision": "ESCALATE",
                    "escalation_id": escalation_id,
                    "reason": esc_reason,
                })
                return _finalize(state)

            # -------------------------------------------------------------
            # Pathway 2: Targeted Clarification (Ambiguous Inquiries)
            # -------------------------------------------------------------
            is_ambiguous = bool(state.metadata.get("is_ambiguous", False))
            if is_ambiguous and not state.final_escalation_required and generate_grounded_response:
                clarification_msg = (
                    f"To help you with your inquiry regarding {state.intent or 'your account'}, "
                    f"could you please clarify your request or provide additional details "
                    f"(such as the order number, item description, or relevant date)?"
                )
                state.final_autonomy_tier = "CLARIFY"
                state.final_response = clarification_msg
                state.status = "RESOLVED"
                state.decision_rationale = "Inquiry requires additional customer clarification."

                state.agent_execution_trace.append({
                    "agent": "SupervisorAgent",
                    "timestamp": start_time,
                    "action": "CLARIFY_ROUTING",
                    "status": "COMPLETED",
                    "decision": "CLARIFY",
                    "clarification_prompt": clarification_msg,
                })
                return _finalize(state)

            # -------------------------------------------------------------
            # Pathway 3: Grounded Direct Answer (ANSWER Tier)
            # -------------------------------------------------------------
            if state.final_autonomy_tier == "ANSWER" and not state.final_escalation_required:
                if generate_grounded_response:
                    t_gen0 = time.perf_counter()
                    grounded_ans = GroundedAnswerGenerator.generate_response(state)
                    state.metadata["latency_generation_ms"] = round((time.perf_counter() - t_gen0) * 1000, 3)
                    if grounded_ans is not None:
                        state.final_response = grounded_ans
                        state.status = "RESOLVED"
                    else:
                        # Fail-closed fallback: if generator refuses (e.g. unverified entailment), escalate safely
                        escalation_id = f"esc_{uuid.uuid4().hex[:12]}"
                        esc_reason = (
                            state.escalation_reason
                            or "Grounded answer could not be synthesized from retrieved evidence."
                        )
                        escalation_notice = (
                            f"Your request has been routed to a human customer support specialist for review "
                            f"(Escalation ID: {escalation_id}).\n"
                            f"Reason: {esc_reason}\n"
                            f"A representative will review the case details and follow up promptly."
                        )
                        if ws_id:
                            db = state.metadata.get("db") or self._get_db()
                            tenant_id = state.metadata.get("tenant_id")
                            if not tenant_id:
                                ws_record = db.get_workspace(ws_id)
                                tenant_id = ws_record["tenant_id"] if ws_record else "ten_unknown"

                            chunks_json = json.dumps(state.retrieved_chunks, default=str)
                            dossier_record = {
                                "escalation_id": escalation_id,
                                "workspace_id": ws_id,
                                "tenant_id": tenant_id,
                                "session_id": state.session_id,
                                "customer_id": state.customer_id,
                                "query": state.query,
                                "domain": state.domain or "General",
                                "intent": state.intent,
                                "intent_confidence": state.intent_confidence,
                                "risk": state.risk,
                                "risk_confidence": state.risk_confidence,
                                "complexity": state.complexity,
                                "provisional_autonomy_tier": state.provisional_autonomy_tier,
                                "final_autonomy_tier": "ESCALATE",
                                "evidence_quality": state.evidence_quality,
                                "retrieval_similarity_score": state.retrieval_similarity_score,
                                "retrieved_chunks_json": chunks_json,
                                "safety_rule_triggered": state.safety_rule_triggered or "EVID_INSUF_04",
                                "escalation_reason": esc_reason,
                                "recommended_action": "Specialist review of policy documentation and customer records",
                                "status": "PENDING",
                            }
                            try:
                                db.create_escalation(dossier_record)
                                logger.info(f"Escalation dossier '{escalation_id}' enqueued for workspace '{ws_id}'.")
                            except Exception as db_exc:
                                logger.warning(f"Failed to persist escalation dossier: {db_exc}")

                        if isinstance(state.escalation_dossier, dict):
                            state.escalation_dossier["escalation_id"] = escalation_id
                            state.escalation_dossier["escalation_notice"] = escalation_notice
                        else:
                            state.escalation_dossier = {
                                "escalation_id": escalation_id,
                                "reason": esc_reason,
                                "escalation_notice": escalation_notice,
                            }
                        state.final_autonomy_tier = "ESCALATE"
                        state.final_escalation_required = True
                        state.status = "ESCALATED"
                        state.final_response = None
                        state.decision_rationale = escalation_notice
                else:
                    state.final_response = None
                    state.status = "VERIFIED"

                state.agent_execution_trace.append({
                    "agent": "SupervisorAgent",
                    "timestamp": start_time,
                    "action": "ANSWER_SYNTHESIS",
                    "status": "COMPLETED",
                    "decision": state.final_autonomy_tier,
                    "has_response": state.final_response is not None,
                })
                return _finalize(state)

            # -------------------------------------------------------------
            # Pathway 4: Dynamic Orchestration or Conservative Fallback
            # -------------------------------------------------------------
            if state.final_autonomy_tier == "ORCHESTRATE" and not generate_grounded_response:
                state.status = "VERIFIED"
                state.final_response = None
            else:
                state.status = "RESOLVED"
                if generate_grounded_response and not state.final_response:
                    t_gen0 = time.perf_counter()
                    grounded_ans = GroundedAnswerGenerator.generate_response(state)
                    state.metadata["latency_generation_ms"] = round((time.perf_counter() - t_gen0) * 1000, 3)
                    state.final_response = grounded_ans

            state.agent_execution_trace.append({
                "agent": "SupervisorAgent",
                "timestamp": start_time,
                "action": "DEFAULT_FINALIZATION",
                "status": "COMPLETED",
                "decision": state.final_autonomy_tier or "PENDING",
            })

        except Exception as exc:
            logger.error(f"SupervisorAgent error: {exc}", exc_info=True)
            state.errors.append(f"SupervisorAgent error: {str(exc)}")
            state.status = "FAILED"
            state.final_response = "An error occurred while evaluating your request. Please contact support."
            state.agent_execution_trace.append({
                "agent": "SupervisorAgent",
                "timestamp": start_time,
                "action": "SUPERVISOR_ERROR",
                "status": "FAILED",
                "error": str(exc),
            })

        state.updated_at = datetime.now(timezone.utc).isoformat()
        return _finalize(state)
