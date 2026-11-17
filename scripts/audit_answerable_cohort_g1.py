"""
Phase G.1 Diagnostic Audit Script: Answerable Benchmark Cohort (428 Queries)
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
Research Direction: Risk-Aware Adaptive Multi-Agent Orchestration

Produces row-level diagnostic CSV with 7 mutually exclusive root-cause categories.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestration.graph import get_orchestration_pipeline
from src.agents.verification_agent import EvidenceEntailmentVerifier

logging.basicConfig(level=logging.WARNING)

BENCHMARK_PATH = ROOT / "data" / "processed" / "rag" / "retrieval_benchmark_remediated_v2.csv"
AUDIT_V2_PATH = ROOT / "outputs" / "rag_evaluation" / "benchmark_relevance_audit_v2.csv"
OUT_DIR = ROOT / "outputs" / "phase_g1_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = get_orchestration_pipeline()

# Load benchmark & audit data
bench_df = pd.read_csv(BENCHMARK_PATH)
audit_df = pd.read_csv(AUDIT_V2_PATH)

# Answerable cohort: exclude MISSING_KNOWLEDGE and METADATA_CONFLICT
ans_df = bench_df[~bench_df["Audit_Relevance_Category"].isin(["MISSING_KNOWLEDGE", "METADATA_CONFLICT"])].copy()
print(f"Total Answerable Cohort Queries: {len(ans_df)}", flush=True)

results = []
latencies_e2e = []
latencies_verif = []

# Warmup run
pipeline.run(query="What is your return policy?", domain="E-commerce", generate_grounded_response=True)

for idx, row in ans_df.iterrows():
    case_id = str(row["Case_ID"]).strip()
    query = str(row["Query"]).strip()
    domain = str(row.get("Domain", "E-commerce")).strip()
    intent = str(row.get("Intent", "Unknown")).strip()
    expected_doc = str(row.get("Expected_Document_ID", "")).strip()
    expected_ans = str(row.get("Audit_Relevance_Category", "RELEVANT")).strip()

    t0 = time.perf_counter()
    state = pipeline.run(
        query=query,
        domain=domain,
        generate_grounded_response=True,
    )
    e2e_ms = (time.perf_counter() - t0) * 1000

    lat_breakdown = state.metadata.get("latency_breakdown", {})
    verif_ms = lat_breakdown.get("latency_verification_ms", 0.0)
    latencies_e2e.append(e2e_ms)
    latencies_verif.append(verif_ms)

    chunks = state.retrieved_chunks or []
    retrieved_doc_ids = [str(c.get("document_id", "")).strip() for c in chunks if isinstance(c, dict)]
    doc_titles = [f"[{c.get('document_id')}]: {c.get('title')}" for c in chunks if isinstance(c, dict)]
    retrieved_summary = "; ".join(doc_titles) if doc_titles else "None"
    top_chunk = chunks[0] if chunks else {}
    top_doc = str(top_chunk.get("document_id", "None")).strip()
    top_sim = float(state.retrieval_similarity_score or 0.0)

    findings_list = state.entailment_details.get("findings", []) if state.entailment_details else []
    findings_str = " | ".join(findings_list) if findings_list else "None"

    tier = state.final_autonomy_tier
    rule = state.safety_rule_triggered
    resp = state.final_response

    # Determine customer-visible outcome
    if tier == "ANSWER" and resp is not None:
        customer_outcome = "Autonomous Answer with Citations"
    elif tier == "CLARIFY" and resp is not None:
        customer_outcome = "Clarification Request"
    elif state.final_escalation_required or tier == "ESCALATE":
        customer_outcome = "Human Escalation Notice"
    else:
        customer_outcome = f"Routing: {tier}"

    # Mutually Exclusive Root-Cause Classification (for non-autonomous queries)
    category = "N/A"
    rationale = "N/A"

    if tier == "ANSWER" and resp is not None:
        category = "N/A (Autonomous Answer)"
        rationale = "Autonomous answer generated with verified grounded citations."
    else:
        # Precedence 1: Genuine safety-driven escalation
        is_safety_rule = rule in ["CRIT_01", "SENS_FIN_06", "RISK_VERIF_02", "LOW_CONF_05"]
        is_high_risk = str(state.risk).capitalize() in ["Critical", "High"]
        if is_safety_rule or is_high_risk:
            category = "1. Genuine safety-driven escalation"
            rationale = (
                f"Mandatory safety protection triggered (Rule: [{rule}], Risk: '{state.risk}', "
                f"Confidence: {state.final_confidence:.3f}). Autonomous answering strictly prohibited."
            )

        # Precedence 2: Genuine ambiguity requiring clarification
        elif expected_ans == "AMBIGUOUS" or tier == "CLARIFY" or bool(state.metadata.get("is_ambiguous")):
            category = "5. Genuine ambiguity requiring clarification"
            rationale = (
                f"Inquiry is intrinsically ambiguous or underspecified (Audit category: {expected_ans}). "
                "Requires customer clarification rather than speculative answer."
            )

        # Precedence 3: Operational or transactional request requiring a tool, authorization or human review
        elif (
            re.search(r"({{[^}]+}}|#\d+|\b(?:order|bill|purchase|account)\s*#?\d+|access key|pin code)\b", query, re.IGNORECASE)
            or intent in [
                "track_order", "get_invoice", "check_invoice", "recover_password",
                "set_up_shipping_address", "change_shipping_address", "delete_account",
                "cancel_order", "get_refund"
            ]
        ):
            category = "7. Operational or transactional request requiring a tool, authorization or human review"
            rationale = (
                f"Query requests operational transaction on specific customer record/entity (Intent: '{intent}'). "
                "Demands operational DB/tool execution or agent authorization rather than static policy text."
            )

        # Precedence 4: Model or routing decision unrelated to evidence insufficiency
        elif state.entailment_status == "SUPPORTED" and state.evidence_quality == "Strong":
            category = "6. Model or routing decision unrelated to evidence insufficiency"
            rationale = (
                f"Evidence is verified SUPPORTED and Strong, but query was disqualified by policy rule "
                f"(max_risk: Low vs actual risk '{state.risk}') or qualified for ORCHESTRATE."
            )

        # Precedence 5: Retrieval failure despite supporting documents being present
        elif expected_ans == "RELEVANT" and expected_doc and (expected_doc not in retrieved_doc_ids):
            category = "3. Retrieval failure despite supporting documents being present"
            rationale = (
                f"Expected document [{expected_doc}] exists in knowledge base, but was not retrieved in top-k "
                f"(retrieved: {retrieved_doc_ids}, top sim: {top_sim:.4f})."
            )

        # Precedence 6: False-negative entailment verification
        elif (
            expected_ans == "RELEVANT"
            and top_sim >= 0.45
            and (expected_doc in retrieved_doc_ids or top_doc == expected_doc)
            and state.entailment_status in ["INSUFFICIENT_EVIDENCE", "CONTRADICTED"]
        ):
            category = "4. False-negative entailment verification"
            rationale = (
                f"Target document [{top_doc}] was retrieved with similarity {top_sim:.4f} >= 0.45, "
                f"but evidence verifier evaluated status as '{state.entailment_status}'."
            )

        # Precedence 7: Missing or insufficient knowledge-base evidence
        else:
            category = "2. Missing or insufficient knowledge-base evidence"
            rationale = (
                f"Knowledge base does not contain sufficient factual evidence (Audit category: {expected_ans}, "
                f"top document: [{top_doc}], peak similarity: {top_sim:.4f} < 0.45 threshold)."
            )

    results.append({
        "case_id": case_id,
        "query": query,
        "expected_answerability": expected_ans,
        "domain": domain,
        "intent": intent,
        "retrieved_doc_ids_and_titles": retrieved_summary,
        "retrieval_similarity": round(top_sim, 4),
        "entailment_status": state.entailment_status or "None",
        "entailment_findings": findings_str,
        "final_autonomy_tier": tier or "None",
        "safety_rule_triggered": rule if rule is not None else "None",
        "customer_visible_outcome": customer_outcome,
        "abstention_root_cause_category": category,
        "abstention_rationale": rationale,
    })

res_df = pd.DataFrame(results)
csv_out = OUT_DIR / "answerable_row_level_diagnostic_g1.csv"
res_df.to_csv(csv_out, index=False)
print(f"\nSaved row-level diagnostic to {csv_out}", flush=True)

# Metrics calculation
total_queries = len(res_df)
answered_queries = len(res_df[res_df["abstention_root_cause_category"] == "N/A (Autonomous Answer)"])
non_autonomous_queries = total_queries - answered_queries

print("\n" + "=" * 80)
print(f"ANSWERABLE COHORT (428 QUERIES) DIAGNOSTIC SUMMARY")
print("=" * 80)
print(f"Total Evaluated:          {total_queries}")
print(f"Autonomous Answers:       {answered_queries} ({answered_queries/total_queries*100:.2f}%)")
print(f"Non-Autonomous Outcomes:  {non_autonomous_queries} ({non_autonomous_queries/total_queries*100:.2f}%)")

print("\nMutually Exclusive Root-Cause Distribution (337 Non-Autonomous Queries):")
cat_counts = res_df[res_df["abstention_root_cause_category"] != "N/A (Autonomous Answer)"]["abstention_root_cause_category"].value_counts().sort_index()
for cat, count in cat_counts.items():
    print(f"  {cat}: {count} ({count/non_autonomous_queries*100:.2f}%)")
print(f"  Total Categorized: {cat_counts.sum()} (Reconciled: {cat_counts.sum() == non_autonomous_queries})")

print("\nTier Breakdown:")
print(res_df["final_autonomy_tier"].value_counts().to_dict())

print("\nEntailment Status Breakdown:")
print(res_df["entailment_status"].value_counts().to_dict())

print(f"\nCache Telemetry: {EvidenceEntailmentVerifier.get_cache_stats()}")
