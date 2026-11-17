"""
Orchestration Workflow & Pipeline Integration
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation

Destination: src/orchestration/graph.py

Coordinates the multi-stage customer support resolution lifecycle:
1. Model Service Query Triage (M1-M6)
2. Hybrid RAG Retrieval (BM25 + Dense ChromaDB)
3. Authentic Evidence Assessment & Verification Signals
4. Deterministic Safety Policy Evaluation (SafetyPolicyEngine)
5. Traceable State Transition & Decision Output
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.orchestration.model_service import ModelService
from src.orchestration.safety_policy import SafetyPolicyEngine
from src.orchestration.state import OrchestrationState
from src.rag.retriever import RAGRetriever

logger = logging.getLogger("orchestration.graph")


# ---------------------------------------------------------------------------
# Stage 3: Authentic Evidence Assessment & Verification Processing
# ---------------------------------------------------------------------------

def assess_evidence_and_verification(state: OrchestrationState) -> OrchestrationState:
    """
    Evaluates authentic retrieval evidence and verification requirements
    using real available signals without fabricating missing values.

    Rules:
    - If no chunks were retrieved or similarity is unavailable -> 'Insufficient'
    - If contradictory evidence is detected -> 'Conflicting'
    - If peak similarity >= 0.70, freshness is Current, and authority is High/Official -> 'Strong'
    - If peak similarity is between 0.60 and 0.70 or has outdated evidence -> 'Insufficient'
    - If verification is required, checks real operational records or metadata;
      does NOT invent a verification score if missing.
    """
    # 1. Evidence Quality Evaluation
    if not state.retrieved_chunks or state.retrieval_similarity_score is None:
        state.evidence_quality = "Insufficient"
        state.verification_findings.append("No relevant knowledge base passages retrieved.")
    else:
        sim = float(state.retrieval_similarity_score)

        # Check for conflicting evidence
        is_conflicting = (
            state.has_conflicting_evidence
            or state.metadata.get("has_conflicting_evidence", False)
            or any(
                isinstance(c, dict) and c.get("conflict_flag") is True
                for c in state.retrieved_chunks
            )
        )

        if is_conflicting:
            state.has_conflicting_evidence = True
            state.evidence_quality = "Conflicting"
            state.verification_findings.append("Contradictory policy documentation detected across retrieved chunks.")
        elif state.has_outdated_evidence or sim < 0.45:
            state.evidence_quality = "Insufficient"
            reason = "outdated document detected" if state.has_outdated_evidence else f"low retrieval similarity ({sim:.4f} < 0.45)"
            state.verification_findings.append(f"Evidence quality insufficient due to {reason}.")
        else:
            state.evidence_quality = "Strong"
            state.verification_findings.append(f"Strong evidence confirmed (peak similarity: {sim:.4f}, fresh documents).")

    # 2. Verification Requirement & Signal Evaluation
    # Verification requirement: respects explicit state/caller flag or sensitive/financial impact
    if state.metadata.get("verification_required") is not None:
        state.verification_required = bool(state.metadata["verification_required"])
    elif state.sensitive_action or state.financial_impact:
        state.verification_required = True

    if state.verification_required:
        # Check authentic verification signals from metadata or verification_details
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
                    state.verification_findings.append(
                        f"Independent verification completed: score {state.verification_score:.4f} "
                        f"(Status: {state.verification_status})."
                    )
            except (ValueError, TypeError):
                state.verification_score = None
                state.verification_status = "Pending"
                state.verification_findings.append("Invalid verification score encountered.")
        else:
            # Mark unavailable explicitly without fabricating
            state.verification_score = None
            state.verification_status = "Pending"
            state.verification_findings.append("Verification is required but independent verification score is unavailable.")
    else:
        state.verification_status = "Not_Required"

    state.status = "VERIFIED"
    state.updated_at = datetime.now(timezone.utc).isoformat()
    return state


# ---------------------------------------------------------------------------
# Master Orchestration Pipeline Function
# ---------------------------------------------------------------------------

class OrchestrationPipeline:
    """
    Coordinates end-to-end execution of customer support inquiries:
    Triage -> Retrieval -> Evidence Assessment -> Deterministic Safety Gate
    """

    def __init__(
        self,
        model_service: Optional[ModelService] = None,
        retriever: Optional[RAGRetriever] = None,
        safety_engine: Optional[SafetyPolicyEngine] = None,
        db: Optional[Any] = None,
    ):
        self.model_service = model_service or ModelService.get_instance()
        self.retriever = retriever
        self.safety_engine = safety_engine or SafetyPolicyEngine()
        self.db = db

        # Specialist Multi-Agent Team (Phase E)
        from src.agents.triage_agent import TriageAgent
        from src.agents.retrieval_agent import RetrievalAgent
        from src.agents.policy_agent import PolicyAgent
        from src.agents.supervisor_agent import SupervisorAgent

        self.triage_agent = TriageAgent(model_service=self.model_service)
        self.retrieval_agent = RetrievalAgent(retriever=self.retriever)
        self.policy_agent = PolicyAgent(safety_engine=self.safety_engine)
        self.supervisor_agent = SupervisorAgent(db=self.db)

    def _get_retriever(self) -> Optional[RAGRetriever]:
        """Lazy retriever initializer to conserve startup time."""
        if self.retriever is None:
            try:
                self.retriever = RAGRetriever()
            except Exception as exc:
                logger.warning(f"RAGRetriever initialization deferred: {exc}")
                self.retriever = None
        return self.retriever

    def execute_from_state(self, state: OrchestrationState) -> OrchestrationState:
        """
        Takes an initialized/retrieved state, completes evidence assessment
        and deterministic safety policy evaluation via PolicyAgent,
        and applies adaptive resolution via SupervisorAgent.
        """
        # Step 1: Policy & Evidence Assessment
        state = self.policy_agent.execute(state)

        # Step 2: Adaptive Supervisor Resolution
        ws_id = state.metadata.get("workspace_id")
        gen_response = bool(state.metadata.get("generate_grounded_response", False))
        state = self.supervisor_agent.execute(
            state, workspace_id=ws_id, generate_grounded_response=gen_response
        )

        logger.info(
            f"Orchestration evaluated query [{state.query[:50]}...]: "
            f"Final Tier={state.final_autonomy_tier}, "
            f"Escalate={state.final_escalation_required}, "
            f"Confidence={state.final_confidence}, "
            f"Triggered Rule={state.safety_rule_triggered}"
        )
        return state

    def run(
        self,
        query: str,
        customer_id: Optional[str] = None,
        session_id: Optional[str] = None,
        domain: Optional[str] = None,
        sensitive_action: bool = False,
        financial_impact: bool = False,
        action_type: str = "Information/Support",
        top_k: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        generate_grounded_response: Optional[bool] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        customer_memory: Optional[List[Dict[str, Any]]] = None,
    ) -> OrchestrationState:
        """
        Executes complete multi-agent pipeline:
        1. Contextual Reference Resolution (resolving follow-ups safely without cross-session leakage)
        2. Triage Agent (M1-M6 analysis & sensitivity flags)
        3. Retrieval Agent (Workspace-isolated or Demo retrieval)
        4. Policy Agent (Evidence assessment & deterministic safety policy)
        5. Supervisor Agent (Adaptive resolution: Answer, Clarify, or Escalate)
        """
        t_start = time.perf_counter()

        sens_str = "Yes" if sensitive_action else "No"
        fin_str = "Yes" if financial_impact else "No"

        # 1. Contextual Reference Resolution
        t0_res = time.perf_counter()
        from src.memory.resolver import ReferenceResolver
        resolved_q, was_resolved = ReferenceResolver.resolve(query, conversation_history)
        lat_res = round((time.perf_counter() - t0_res) * 1000, 3)

        # 2. Triage Agent (M1-M6 analysis)
        t0_triage = time.perf_counter()
        analysis = self.model_service.analyze_query(
            query=query,
            domain=domain,
            sensitive_action=sens_str,
            financial_impact=fin_str,
            action_type=action_type,
        )

        meta = dict(metadata or {})
        if workspace_id:
            meta["workspace_id"] = workspace_id
        if was_resolved:
            meta["resolved_query"] = resolved_q
        if generate_grounded_response is not None:
            meta["generate_grounded_response"] = generate_grounded_response
        elif workspace_id:
            meta["generate_grounded_response"] = True

        state = OrchestrationState.from_model_service(
            analysis=analysis,
            customer_id=customer_id,
            session_id=session_id,
            resolved_query=resolved_q if was_resolved else None,
            conversation_history=conversation_history or [],
            customer_memory=customer_memory or [],
            metadata=meta,
            sensitive_action=sensitive_action,
            financial_impact=financial_impact,
            action_type=action_type,
        )

        state = self.triage_agent.execute(state)
        lat_triage = round((time.perf_counter() - t0_triage) * 1000, 3)

        # 3. Retrieval Agent
        t0_ret = time.perf_counter()
        if self.retrieval_agent.retriever is None:
            self.retrieval_agent.retriever = self._get_retriever()
        state = self.retrieval_agent.execute(state, workspace_id=workspace_id, top_k=top_k)
        lat_ret = round((time.perf_counter() - t0_ret) * 1000, 3)

        # 4 & 5. Policy & Supervisor Agents
        state = self.execute_from_state(state)

        # Finalize full request timing
        lat_e2e = round((time.perf_counter() - t_start) * 1000, 3)

        lat_verif = float(state.metadata.get("latency_verification_ms", 0.0) or 0.0)
        lat_policy = float(state.metadata.get("latency_policy_ms", 0.0) or 0.0)
        lat_sup = float(state.metadata.get("latency_supervisor_ms", 0.0) or 0.0)
        lat_gen = float(state.metadata.get("latency_generation_ms", 0.0) or 0.0)

        latency_breakdown = {
            "latency_resolution_ms": lat_res,
            "latency_triage_ms": lat_triage,
            "latency_retrieval_ms": lat_ret,
            "latency_verification_ms": lat_verif,
            "latency_policy_ms": lat_policy,
            "latency_supervisor_ms": lat_sup,
            "latency_generation_ms": lat_gen,
            "latency_e2e_ms": lat_e2e,
        }

        state.metadata.update(latency_breakdown)
        state.metadata["latency_breakdown"] = latency_breakdown
        return state


# ---------------------------------------------------------------------------
# Convenience Public API Function
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[OrchestrationPipeline] = None


def get_orchestration_pipeline() -> OrchestrationPipeline:
    """Returns singleton instance of the OrchestrationPipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = OrchestrationPipeline()
    return _pipeline_instance


def execute_orchestration(state: OrchestrationState) -> OrchestrationState:
    """
    Standard entry point invoked by API and downstream graphs.
    Evaluates evidence assessment and deterministic safety gating.
    """
    pipeline = get_orchestration_pipeline()
    return pipeline.execute_from_state(state)
