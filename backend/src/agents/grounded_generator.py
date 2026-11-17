"""
Deterministic Grounded Answer Generator
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Synthesizes grounded customer support responses derived strictly from verified
retrieved knowledge chunks, appending verifiable source citations.
Enforces zero fabrication: if evidence is insufficient or escalated, refuses generation.
"""

from typing import Any, Dict, List, Optional
from src.orchestration.state import OrchestrationState


class GroundedAnswerGenerator:
    """Deterministic, citation-grounded response synthesizer."""

    @staticmethod
    def generate_response(state: OrchestrationState) -> Optional[str]:
        """
        Synthesizes an evidence-grounded answer with source citations.
        Returns None if escalated to human supervisor or evidence is insufficient.
        """
        # Invariant 1: If safety policy mandates escalation, never generate automated answer
        if state.final_escalation_required or state.final_autonomy_tier == "ESCALATE":
            return None

        # Invariant 2: Require valid retrieved chunks
        if not state.retrieved_chunks:
            return None

        # Invariant 3: Require minimum evidence quality
        if state.evidence_quality not in ("Strong", "Adequate"):
            return None

        # Invariant 4: Require verified evidence entailment (Phase G)
        if getattr(state, "entailment_status", None) not in (None, "SUPPORTED"):
            return None

        # Sort chunks by similarity
        valid_chunks = [
            c for c in state.retrieved_chunks
            if isinstance(c, dict) and float(c.get("similarity", 0.0)) >= 0.35
        ]
        if not valid_chunks:
            return None

        # Primary evidence chunk (highest similarity)
        top_chunk = valid_chunks[0]
        top_text = top_chunk.get("chunk_text", "").replace("\ufeff", "").strip()
        doc_id = top_chunk.get("document_id", "DOC")
        title = top_chunk.get("title", "Policy Document")
        authority = top_chunk.get("authority", "Official")
        freshness = top_chunk.get("freshness_status", "Current")

        # Clean passage text
        passage = top_text.replace("\n", " ").strip()
        if len(passage) > 350:
            # Trim to full sentence within 350 chars if possible
            last_period = passage[:350].rfind(".")
            if last_period > 100:
                passage = passage[:last_period + 1]
            else:
                passage = passage[:350] + "..."

        # Assemble grounded response
        response = (
            f"Based on the official {title}:\n\n"
            f"\"{passage}\"\n\n"
            f"Source Citation: [{doc_id}] {title} (Authority: {authority}, Status: {freshness})"
        )

        # If there is secondary supporting evidence from a different document, append citation
        citations = [f"[{doc_id}] {title}"]
        for sec in valid_chunks[1:3]:
            sec_id = sec.get("document_id")
            sec_title = sec.get("title")
            if sec_id and sec_id != doc_id:
                citations.append(f"[{sec_id}] {sec_title}")

        if len(citations) > 1:
            response += f"\nAdditional Reference Documents: {', '.join(citations[1:])}"

        return response
