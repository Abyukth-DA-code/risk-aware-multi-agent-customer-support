"""
CLI Query Inspector & Multi-Agent Verification Tool
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Usage:
  python scripts/inspect_query.py "Can I get a refund for my order?"
  python scripts/inspect_query.py "I suspect fraudulent charges on my credit card!" --sensitive --financial
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestration.graph import OrchestrationPipeline


def inspect(
    query: str,
    domain: str = None,
    sensitive_action: bool = False,
    financial_impact: bool = False,
    verification_score: float = None,
    conflicting_evidence: bool = False,
    top_k: int = 3,
):
    print("=" * 80)
    print("      MULTI-AGENT QUERY INSPECTION & SAFETY VERIFICATION REPORT")
    print("=" * 80)
    print(f"QUERY: \"{query}\"")
    print(f"FLAGS: SensitiveAction={sensitive_action} | FinancialImpact={financial_impact}")
    print("-" * 80)

    pipeline = OrchestrationPipeline()

    metadata = {}
    if verification_score is not None:
        metadata["verification_required"] = True
        metadata["verification_score"] = float(verification_score)
    if conflicting_evidence:
        metadata["has_conflicting_evidence"] = True

    state = pipeline.run(
        query=query,
        domain=domain,
        sensitive_action=sensitive_action,
        financial_impact=financial_impact,
        top_k=top_k,
        metadata=metadata,
    )

    # 1. Model Feature Evaluation (M1 - M6)
    print("\n[1] MODEL TRIAGE EVALUATION (M1 - M6):")
    print(f"  * M5 Domain:     {state.domain} (Confidence: {round(state.domain_confidence or 0, 4)})")
    print(f"  * M1 Intent:     {state.intent} (Confidence: {round(state.intent_confidence or 0, 4)})")
    print(f"  * M2 Sentiment:  {state.sentiment} (Confidence: {round(state.sentiment_confidence or 0, 4)})")
    print(f"  * M3 Risk:       {state.risk} (Confidence: {round(state.risk_confidence or 0, 4)})")
    if state.risk_class_probabilities:
        probs = {k: round(v, 3) for k, v in state.risk_class_probabilities.items()}
        print(f"                   Probs: {probs}")
    print(f"  * M4 Complexity: {state.complexity} (Confidence: {round(state.complexity_confidence or 0, 4)})")
    if state.complexity_class_probabilities:
        probs = {k: round(v, 3) for k, v in state.complexity_class_probabilities.items()}
        print(f"                   Probs: {probs}")
    print(f"  * M6 Provisional Autonomy: {state.provisional_autonomy_tier} (Escalate={state.provisional_escalation_required})")

    # 2. Knowledge Extraction (RAG)
    print("\n[2] EXTRACTED KNOWLEDGE PASSAGES (RAG RETRIEVAL):")
    print(f"  * Total Chunks Retrieved: {len(state.retrieved_chunks)}")
    print(f"  * Peak Similarity Score:  {round(state.retrieval_similarity_score or 0, 4)}")
    print(f"  * Evidence Quality:       {state.evidence_quality}")
    print(f"  * Outdated Flag:          {state.has_outdated_evidence}")
    print(f"  * Conflict Flag:          {state.has_conflicting_evidence}")

    if not state.retrieved_chunks:
        print("  [WARN] NO PASSAGES FOUND IN KNOWLEDGE BASE!")
    else:
        for idx, chunk in enumerate(state.retrieved_chunks, start=1):
            doc_id = chunk.get("document_id", "N/A")
            title = chunk.get("title", "N/A")
            authority = chunk.get("authority", "N/A")
            sim = chunk.get("similarity", 0.0)
            text = chunk.get("chunk_text", "")
            snippet = text[:150] + ("..." if len(text) > 150 else "")
            print(f"  [{idx}] Doc {doc_id} ('{title}') | Authority: {authority} | Sim: {round(sim, 4)}")
            print(f"      Text: \"{snippet}\"")

    # 3. Escalation & Safety Policy Decision
    print("\n[3] DETERMINISTIC SAFETY POLICY & ESCALATION DECISION:")
    print(f"  * Final Autonomy Tier:       {state.final_autonomy_tier}")
    print(f"  * Final Escalation Required: {state.final_escalation_required}")
    print(f"  * Composite Confidence:      {round(state.final_confidence or 0, 4)}")
    print(f"  * Decision Source:           {state.decision_source}")
    print(f"  * Safety Override:           {state.safety_override}")
    print(f"  * Safety Rule Triggered:     {state.safety_rule_triggered or 'None (Passed All Invariant Checks)'}")
    print(f"  * Decision Rationale:        {state.decision_rationale}")

    if state.final_escalation_required:
        print("\n  [!] CASE ESCALATED TO HUMAN SUPERVISOR (HITL):")
        print(f"      Reason: {state.escalation_reason}")
        if state.escalation_dossier:
            dossier = state.escalation_dossier
            print(f"      Recommended Action: {dossier.get('recommended_action')}")
            print(f"      Context Summary:    {dossier.get('context_summary')}")
    else:
        print(f"\n  [OK] CASE RESOLVED AUTONOMOUSLY: {state.decision_rationale}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect query classification, RAG retrieval, and safety escalation.")
    parser.add_argument("query", type=str, help="Customer query text to inspect")
    parser.add_argument("--domain", type=str, default=None, help="Explicit domain override (E-commerce, Education, Small Business)")
    parser.add_argument("--sensitive", action="store_true", help="Mark inquiry as sensitive action")
    parser.add_argument("--financial", action="store_true", help="Mark inquiry as financial impact")
    parser.add_argument("--verification-score", type=float, default=None, help="Mock independent verification score (0.0 to 1.0)")
    parser.add_argument("--conflicting", action="store_true", help="Simulate conflicting policy evidence")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    inspect(
        query=args.query,
        domain=args.domain,
        sensitive_action=args.sensitive,
        financial_impact=args.financial,
        verification_score=args.verification_score,
        conflicting_evidence=args.conflicting,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
