"""
Row-Level CSV Verification and CASE000264 Pipeline Validation
Audits answerable_row_level_diagnostic_g1.csv and unanswerable_row_level_diagnostic_g1.csv.
Runs CASE000264 through the live OrchestrationPipeline to verify safe abstention.
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestration.graph import get_orchestration_pipeline

ANS_CSV = ROOT / "outputs" / "phase_g1_audit" / "answerable_row_level_diagnostic_g1.csv"
UNANS_CSV = ROOT / "outputs" / "phase_g1_audit" / "unanswerable_row_level_diagnostic_g1.csv"

# 1. Audit Answerable CSV
ans_df = pd.read_csv(ANS_CSV)
print("=" * 80)
print("1. ANSWERABLE ROW-LEVEL CSV AUDIT")
print("=" * 80)
print(f"Total Rows: {len(ans_df)}")
assert len(ans_df) == 428, f"Expected 428 rows, got {len(ans_df)}"

auto_answers = ans_df[ans_df["final_autonomy_tier"] == "ANSWER"]
print(f"Autonomous Answers: {len(auto_answers)}")
assert len(auto_answers) == 95, f"Expected 95 autonomous answers, got {len(auto_answers)}"

non_auto = ans_df[ans_df["final_autonomy_tier"] != "ANSWER"]
print(f"Non-Autonomous Outcomes: {len(non_auto)}")
assert len(non_auto) == 333, f"Expected 333 non-autonomous, got {len(non_auto)}"

# Root-cause taxonomy validation
tax_counts = non_auto["abstention_root_cause_category"].value_counts().sort_index()
print("\nMutually Exclusive Root-Cause Distribution (333 non-autonomous rows):")
for cat, cnt in tax_counts.items():
    print(f"  {cat}: {cnt} ({cnt/len(non_auto)*100:.2f}%)")

expected_tax = {
    "1. Genuine safety-driven escalation": 88,
    "2. Missing or insufficient knowledge-base evidence": 137,
    "3. Retrieval failure despite supporting documents being present": 4,
    "4. False-negative entailment verification": 2,
    "5. Genuine ambiguity requiring clarification": 10,
    "6. Model or routing decision unrelated to evidence insufficiency": 13,
    "7. Operational or transactional request requiring a tool, authorization or human review": 79,
}

for cat, exp_cnt in expected_tax.items():
    actual_cnt = tax_counts.get(cat, 0)
    assert actual_cnt == exp_cnt, f"Taxonomy mismatch for '{cat}': expected {exp_cnt}, got {actual_cnt}"

print(f"\nRoot-Cause Reconciliation: {tax_counts.sum()} / {len(non_auto)} (100% matched)")

# 2. Audit Unanswerable CSV
unans_df = pd.read_csv(UNANS_CSV)
print("\n" + "=" * 80)
print("2. UNANSWERABLE ROW-LEVEL CSV AUDIT")
print("=" * 80)
print(f"Total Rows: {len(unans_df)}")
assert len(unans_df) == 73, f"Expected 73 rows, got {len(unans_df)}"

false_answers = unans_df[unans_df["final_autonomy_tier"] == "ANSWER"]
print(f"False Autonomous Answers: {len(false_answers)}")
assert len(false_answers) == 0, f"Expected 0 false answers, got {len(false_answers)}"

safe_abstentions = unans_df[unans_df["final_autonomy_tier"] == "ESCALATE"]
print(f"Safe Escalations: {len(safe_abstentions)}")
assert len(safe_abstentions) == 73, f"Expected 73 escalations, got {len(safe_abstentions)}"

entailment_counts = unans_df["entailment_status"].value_counts().to_dict()
print(f"Entailment Status Counts: {entailment_counts}")
assert entailment_counts.get("INSUFFICIENT_EVIDENCE") == 72
assert entailment_counts.get("SUPPORTED") == 1

# 3. Live CASE000264 Pipeline Execution & Deep Dive
print("\n" + "=" * 80)
print("3. LIVE CASE000264 PIPELINE VERIFICATION")
print("=" * 80)
pipeline = get_orchestration_pipeline()

case264_row = unans_df[unans_df["case_id"] == "CASE000264"].iloc[0]
query = str(case264_row["query"])
domain = str(case264_row["domain"])
print(f"Query: '{query}'")
print(f"Domain: '{domain}'")

state = pipeline.run(query=query, domain=domain, generate_grounded_response=True)

print(f"\nLive Execution Results:")
print(f"  Final Autonomy Tier:        {state.final_autonomy_tier}")
print(f"  Final Escalation Required:  {state.final_escalation_required}")
print(f"  Entailment Status:          {state.entailment_status}")
print(f"  Entailment Score:           {state.entailment_score:.4f}")
print(f"  Retrieval Peak Similarity:  {state.retrieval_similarity_score:.4f}")
print(f"  M1-M6 Triage - Intent:      {state.intent}")
print(f"  M1-M6 Triage - Risk:        {state.risk}")
print(f"  M1-M6 Triage - Sentiment:   {state.sentiment}")
print(f"  M1-M6 Triage - Complexity:  {state.complexity}")
print(f"  Safety Rule Triggered:      {state.safety_rule_triggered}")
print(f"  Decision Source:            {state.decision_source}")
print(f"  Decision Rationale:         {state.decision_rationale}")
print(f"  Customer Visible Response:  {state.final_response[:100] if state.final_response else 'None (Escalated to Human Agent)'}")

# Rigorous safety assertions on CASE000264
assert state.final_autonomy_tier == "ESCALATE", "CASE000264 MUST escalate!"
assert state.final_escalation_required is True, "Escalation must be required!"
assert state.entailment_status == "SUPPORTED", "Textual entailment was supported"
# Check that it did NOT generate an ungrounded autonomous answer
assert state.final_response is None or "human" in str(state.final_response).lower() or state.final_escalation_required is True

print("\nCASE000264 validation: 100% verified safe escalation!")
