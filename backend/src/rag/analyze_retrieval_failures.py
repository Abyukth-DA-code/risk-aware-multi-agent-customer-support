
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT / "outputs" / "rag_evaluation"
    / "retrieval_test_predictions.csv"
)

df = pd.read_csv(RESULTS)

misses = df[df["Recall_at_5"] == 0].copy()

print("\n========== FAILURE SUMMARY ==========")
print(f"Total evaluated: {len(df)}")
print(f"Expected document found in top 5: {len(df) - len(misses)}")
print(f"Expected document missing: {len(misses)}")

print("\n========== MISSES BY DOMAIN ==========")
print(
    misses.groupby("Domain")
    .size()
    .rename("Misses")
    .to_frame()
)

print("\n========== MOST FREQUENT MISSED DOCUMENTS ==========")
print(
    misses["Expected_Document_ID"]
    .value_counts()
    .head(15)
    .to_string()
)

print("\n========== SAMPLE FAILED QUERIES ==========")
columns = [
    "Case_ID",
    "Domain",
    "Query",
    "Expected_Document_ID",
    "Retrieved_Document_IDs",
]

print(
    misses[columns]
    .head(15)
    .to_string(index=False, max_colwidth=100)
)

print("\nNo dataset or vector-store files were modified.")
