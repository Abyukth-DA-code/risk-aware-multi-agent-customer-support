import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
_d_dir = ROOT / "datasets" if (ROOT / "datasets").exists() else ROOT / "data"

# 1. Dataset Integrity
test_df = pd.read_csv(_d_dir / "processed/autonomy/test.csv")
val_df = pd.read_csv(_d_dir / "processed/autonomy/validation.csv")
train_df = pd.read_csv(_d_dir / "processed/autonomy/train.csv")

print("=== DATASET SAMPLE COUNTS ===")
print("Train sample count:", len(train_df))
print("Validation sample count:", len(val_df))
print("Test sample count:", len(test_df))

print("\n=== DUPLICATE CASE IDs ===")
print("Train duplicates:", train_df["Case_ID"].duplicated().sum())
print("Validation duplicates:", val_df["Case_ID"].duplicated().sum())
print("Test duplicates:", test_df["Case_ID"].duplicated().sum())

print("\n=== CASE ID OVERLAPS ===")
val_test_overlap = set(val_df["Case_ID"]).intersection(set(test_df["Case_ID"]))
train_val_overlap = set(train_df["Case_ID"]).intersection(set(val_df["Case_ID"]))
train_test_overlap = set(train_df["Case_ID"]).intersection(set(test_df["Case_ID"]))
print("Validation & Test Case_ID overlap:", len(val_test_overlap))
print("Train & Validation Case_ID overlap:", len(train_val_overlap))
print("Train & Test Case_ID overlap:", len(train_test_overlap))

print("\n=== QUERY TEXT OVERLAPS ===")
val_test_q_overlap = set(val_df["Query"].str.strip().str.lower()).intersection(set(test_df["Query"].str.strip().str.lower()))
train_test_q_overlap = set(train_df["Query"].str.strip().str.lower()).intersection(set(test_df["Query"].str.strip().str.lower()))
print("Validation & Test Query overlap:", len(val_test_q_overlap))
print("Train & Test Query overlap:", len(train_test_q_overlap))

print("\n=== MISSING VALUES IN TEST ===")
print(test_df.isnull().sum())

print("\n=== MISSING VALUES IN VALIDATION ===")
print(val_df.isnull().sum())

print("\n=== LABEL DISTRIBUTIONS ===")
print("Test Autonomy_Decision:\n", test_df["Autonomy_Decision"].value_counts())
print("Validation Autonomy_Decision:\n", val_df["Autonomy_Decision"].value_counts())
print("Train Autonomy_Decision:\n", train_df["Autonomy_Decision"].value_counts())

# 2. Inspect Case Level Predictions
case_log_path = ROOT / "outputs/research_evaluation/m6_experiment_20261031_063000/m6_case_level_comparison.csv"
case_df = pd.read_csv(case_log_path)
print("\n=== CASE LOG COLUMNS ===")
print(case_df.columns.tolist())
print("Case log rows:", len(case_df))

y_true = case_df["Ground_Truth_Autonomy"]
pred_a = case_df["Policy_A_Prediction"]
pred_b = case_df["Policy_B_Prediction"]
pred_c = case_df["Policy_C_Prediction"]

print("\n=== OVERALL CORRECT PREDICTIONS (DIAGONAL SUM) ===")
corr_a = (y_true == pred_a).sum()
corr_b = (y_true == pred_b).sum()
corr_c = (y_true == pred_c).sum()
print(f"Policy A: {corr_a} / {len(case_df)} ({corr_a/len(case_df):.4f})")
print(f"Policy B: {corr_b} / {len(case_df)} ({corr_b/len(case_df):.4f})")
print(f"Policy C: {corr_c} / {len(case_df)} ({corr_c/len(case_df):.4f})")

print("\n=== PREDICTED CLASS DISTRIBUTIONS ===")
print("Policy A:\n", pred_a.value_counts())
print("Policy B:\n", pred_b.value_counts())
print("Policy C:\n", pred_c.value_counts())

print("\n=== ANSWER CLASS TRUE POSITIVES (CORRECT AUTONOMOUS ANSWERS) ===")
tp_ans_a = ((y_true == "ANSWER") & (pred_a == "ANSWER")).sum()
tp_ans_b = ((y_true == "ANSWER") & (pred_b == "ANSWER")).sum()
tp_ans_c = ((y_true == "ANSWER") & (pred_c == "ANSWER")).sum()
print(f"Policy A: {tp_ans_a} (Precision: {tp_ans_a/(pred_a=='ANSWER').sum():.4f}, Recall: {tp_ans_a/(y_true=='ANSWER').sum():.4f})")
print(f"Policy B: {tp_ans_b} (Precision: {tp_ans_b/(pred_b=='ANSWER').sum():.4f}, Recall: {tp_ans_b/(y_true=='ANSWER').sum():.4f})")
print(f"Policy C: {tp_ans_c} (Precision: {tp_ans_c/(pred_c=='ANSWER').sum():.4f}, Recall: {tp_ans_c/(y_true=='ANSWER').sum():.4f})")

print("\n=== ORCHESTRATE CLASS TRUE POSITIVES ===")
tp_orch_a = ((y_true == "ORCHESTRATE") & (pred_a == "ORCHESTRATE")).sum()
tp_orch_b = ((y_true == "ORCHESTRATE") & (pred_b == "ORCHESTRATE")).sum()
tp_orch_c = ((y_true == "ORCHESTRATE") & (pred_c == "ORCHESTRATE")).sum()
print(f"Policy A: {tp_orch_a} / {(y_true=='ORCHESTRATE').sum()}")
print(f"Policy B: {tp_orch_b} / {(y_true=='ORCHESTRATE').sum()}")
print(f"Policy C: {tp_orch_c} / {(y_true=='ORCHESTRATE').sum()}")

print("\n=== ESCALATE CLASS TRUE POSITIVES (ESCALATION RECALL) ===")
tp_esc_a = ((y_true == "ESCALATE") & (pred_a == "ESCALATE")).sum()
tp_esc_b = ((y_true == "ESCALATE") & (pred_b == "ESCALATE")).sum()
tp_esc_c = ((y_true == "ESCALATE") & (pred_c == "ESCALATE")).sum()
print(f"Policy A: {tp_esc_a} / {(y_true=='ESCALATE').sum()} ({tp_esc_a/(y_true=='ESCALATE').sum():.4f})")
print(f"Policy B: {tp_esc_b} / {(y_true=='ESCALATE').sum()} ({tp_esc_b/(y_true=='ESCALATE').sum():.4f})")
print(f"Policy C: {tp_esc_c} / {(y_true=='ESCALATE').sum()} ({tp_esc_c/(y_true=='ESCALATE').sum():.4f})")

# 3. Unsafe Cohort Analysis
unsafe_mask = case_df["Is_Unsafe_Cohort"] == True
print(f"\nUnsafe Cohort N: {unsafe_mask.sum()}")
ans_unsafe_a = ((case_df["Is_Unsafe_Cohort"] == True) & (pred_a == "ANSWER")).sum()
ans_unsafe_b = ((case_df["Is_Unsafe_Cohort"] == True) & (pred_b == "ANSWER")).sum()
ans_unsafe_c = ((case_df["Is_Unsafe_Cohort"] == True) & (pred_c == "ANSWER")).sum()
print(f"Policy A Unsafe Answers: {ans_unsafe_a} / {unsafe_mask.sum()} ({ans_unsafe_a/unsafe_mask.sum():.4f})")
print(f"Policy B Unsafe Answers: {ans_unsafe_b} / {unsafe_mask.sum()} ({ans_unsafe_b/unsafe_mask.sum():.4f})")
print(f"Policy C Unsafe Answers: {ans_unsafe_c} / {unsafe_mask.sum()} ({ans_unsafe_c/unsafe_mask.sum():.4f})")

# 4. McNemar Exact Recalculation
print("\n=== MCNEMAR TESTS RECALCULATION ===")
# On Unsafe Cohort (Policy A vs Policy B)
sub = case_df[unsafe_mask]
viol_a = sub["Policy_A_Prediction"] == "ANSWER"
viol_b = sub["Policy_B_Prediction"] == "ANSWER"
b_viol = ((viol_a == True) & (viol_b == False)).sum()
c_viol = ((viol_a == False) & (viol_b == True)).sum()
print(f"Unsafe Cohort Discordant Pairs: b={b_viol}, c={c_viol}")
chi2_viol = ((abs(b_viol - c_viol) - 1)**2) / (b_viol + c_viol)
pval_viol = 1 - stats.chi2.cdf(chi2_viol, df=1)
print(f"McNemar chi2: {chi2_viol:.4f}, p-val: {pval_viol:.4e}")

# On Overall Accuracy (Policy A vs Policy B)
corr_a_mask = y_true == pred_a
corr_b_mask = y_true == pred_b
b_acc = ((corr_a_mask == True) & (corr_b_mask == False)).sum()
c_acc = ((corr_a_mask == False) & (corr_b_mask == True)).sum()
print(f"Accuracy Discordant Pairs (A vs B): b={b_acc}, c={c_acc}")
chi2_acc = ((abs(b_acc - c_acc) - 1)**2) / (b_acc + c_acc)
pval_acc = 1 - stats.chi2.cdf(chi2_acc, df=1)
print(f"McNemar chi2: {chi2_acc:.4f}, p-val: {pval_acc:.4e}")

# Policy B vs Policy C
corr_c_mask = y_true == pred_c
b_bc = ((corr_b_mask == True) & (corr_c_mask == False)).sum()
c_bc = ((corr_b_mask == False) & (corr_c_mask == True)).sum()
print(f"Accuracy Discordant Pairs (B vs C): b={b_bc}, c={c_bc}")
chi2_bc = ((abs(b_bc - c_bc) - 1)**2) / (b_bc + c_bc)
pval_bc = 1 - stats.chi2.cdf(chi2_bc, df=1)
print(f"McNemar chi2: {chi2_bc:.4f}, p-val: {pval_bc:.4e}")
