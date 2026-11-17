"""
Retrieval Specialist Agent
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Role:
- Manages isolated knowledge evidence retrieval bounded strictly to the authenticated workspace.
- Enforces strict zero-fallback protection: empty tenant collections never leak global demo data.
- Normalizes retrieved passage metadata (chunk_id, authority, freshness, similarity).
- Records retrieval metrics in the execution trace.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from src.orchestration.state import OrchestrationState
from src.rag.retriever import RAGRetriever

logger = logging.getLogger("agents.retrieval")


class RetrievalAgent:
    """Specialist agent responsible for tenant-authorized vector knowledge retrieval."""

    def __init__(self, retriever: Optional[RAGRetriever] = None):
        self.retriever = retriever

    def _get_retriever(self) -> Optional[RAGRetriever]:
        if self.retriever is None:
            try:
                self.retriever = RAGRetriever()
            except Exception as exc:
                logger.warning(f"RAGRetriever lazy initialization failed: {exc}")
                self.retriever = None
        return self.retriever

    def execute(
        self,
        state: OrchestrationState,
        workspace_id: Optional[str] = None,
        top_k: int = 3,
    ) -> OrchestrationState:
        """
        Executes knowledge retrieval bounded strictly to workspace_id or demo collection.
        """
        start_time = datetime.now(timezone.utc).isoformat()
        if "RetrievalAgent" not in state.selected_agents:
            state.selected_agents.append("RetrievalAgent")

        retriever = self._get_retriever()
        if retriever is None:
            state.errors.append("RetrievalAgent: RAGRetriever is unavailable.")
            state.evidence_quality = "Insufficient"
            state.agent_execution_trace.append({
                "agent": "RetrievalAgent",
                "timestamp": start_time,
                "action": "KNOWLEDGE_RETRIEVAL",
                "status": "DEGRADED",
                "error": "RAGRetriever unavailable",
            })
            return state

        try:
            ws_target = workspace_id or state.metadata.get("workspace_id")
            retrieval_query = (
                state.resolved_query
                or state.metadata.get("resolved_query")
                or state.query
            )
            if ws_target:
                # Query strictly the workspace's dedicated collection
                chunks = retriever.retrieve(
                    query=retrieval_query,
                    top_k=top_k,
                    workspace_id=ws_target,
                )
            else:
                # Unauthenticated demonstration retrieval against capstone_knowledge_v1
                chunks = retriever.retrieve(
                    query=retrieval_query,
                    domain=state.domain,
                    top_k=top_k,
                    distinct_documents=True,
                    candidate_multiplier=4,
                    hybrid=False,
                )

            q_emb = getattr(retriever, "last_query_embedding", None)
            state.add_retrieved_chunks(chunks, query_embedding=q_emb)

            # Record detailed trace
            retrieval_meta = {
                "workspace_id": ws_target or "demo",
                "retrieved_count": len(chunks),
                "peak_similarity": state.retrieval_similarity_score,
                "chunk_ids": [c.get("chunk_id") for c in chunks if isinstance(c, dict)],
                "documents": list(dict.fromkeys([c.get("title") for c in chunks if isinstance(c, dict)])),
            }

            state.agent_execution_trace.append({
                "agent": "RetrievalAgent",
                "timestamp": start_time,
                "action": "KNOWLEDGE_RETRIEVAL",
                "status": "COMPLETED",
                "findings": retrieval_meta,
            })
            state.status = "RETRIEVED"

        except Exception as exc:
            logger.error(f"RetrievalAgent error: {exc}", exc_info=True)
            state.errors.append(f"RetrievalAgent error: {str(exc)}")
            state.evidence_quality = "Insufficient"
            state.agent_execution_trace.append({
                "agent": "RetrievalAgent",
                "timestamp": start_time,
                "action": "KNOWLEDGE_RETRIEVAL",
                "status": "FAILED",
                "error": str(exc),
            })

        state.updated_at = datetime.now(timezone.utc).isoformat()
        return state
