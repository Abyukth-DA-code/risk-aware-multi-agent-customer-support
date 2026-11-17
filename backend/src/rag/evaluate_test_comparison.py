"""
Held-out Test Benchmark Evaluation & Final Comparison.

Evaluates Pure Dense Raw, Pure Dense Distinct, and Hybrid Distinct
on the official remediated test benchmark (retrieval_benchmark_remediated_v2.csv).
Runs ONCE for final validation and reports the test metrics.
"""

from pathlib import Path
import sys
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEST_BENCHMARK = ROOT / "data" / "processed" / "rag" / "retrieval_benchmark_remediated_v2.csv"
OUTPUT_DIR = ROOT / "outputs" / "rag_evaluation"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_retrieval import evaluate


def run_test_evaluation():
    if not TEST_BENCHMARK.exists():
        raise FileNotFoundError(f"Test benchmark not found: {TEST_BENCHMARK}")

    experiments = [
        {
            "name": "test_pure_dense_raw",
            "desc": "Pure Dense (Raw Chunks)",
            "distinct": False,
            "hybrid": False,
            "rrf_k": 60,
            "multiplier": 4,
        },
        {
            "name": "test_pure_dense_distinct_m4",
            "desc": "Pure Dense (Distinct Docs, m=4)",
            "distinct": True,
            "hybrid": False,
            "rrf_k": 60,
            "multiplier": 4,
        },
        {
            "name": "test_hybrid_distinct_k60_m2",
            "desc": "Hybrid Distinct (k=60, m=2)",
            "distinct": True,
            "hybrid": True,
            "rrf_k": 60,
            "multiplier": 2,
        },
        {
            "name": "test_hybrid_distinct_k60_m4",
            "desc": "Hybrid Distinct (k=60, m=4)",
            "distinct": True,
            "hybrid": True,
            "rrf_k": 60,
            "multiplier": 4,
        },
    ]

    summary_rows = []

    print("======================================================================")
    print("RUNNING FINAL HELD-OUT TEST BENCHMARK VALIDATION (501 queries, 428 ans)")
    print("======================================================================")

    for exp in experiments:
        print(f"\n---> Running Test: {exp['desc']} ({exp['name']})")
        t0 = time.time()
        results_df, overall, by_domain = evaluate(
            benchmark_path=TEST_BENCHMARK,
            output_dir=OUTPUT_DIR,
            eval_name=exp["name"],
            top_k=5,
            distinct_documents=exp["distinct"],
            candidate_multiplier=exp["multiplier"],
            hybrid=exp["hybrid"],
            rrf_k=exp["rrf_k"],
        )
        total_time = time.time() - t0

        summary_rows.append({
            "Experiment": exp["desc"],
            "Config_ID": exp["name"],
            "Distinct_Docs": exp["distinct"],
            "Hybrid": exp["hybrid"],
            "RRF_k": exp["rrf_k"] if exp["hybrid"] else "N/A",
            "Candidate_Multiplier": exp["multiplier"],
            "Ans_Recall@1": overall["Answerable_Recall@1"],
            "Ans_Recall@3": overall["Answerable_Recall@3"],
            "Ans_Recall@5": overall["Answerable_Recall@5"],
            "Ans_MRR": overall["Answerable_MRR"],
            "Overall_Recall@5": overall["Recall@5"],
            "Overall_MRR": overall["MRR"],
            "Mean_Latency_ms": overall["Mean_Latency_ms"],
            "P95_Latency_ms": overall["P95_Latency_ms"],
            "Total_Wall_s": round(total_time, 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "test_benchmark_final_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n======================================================================")
    print("TEST BENCHMARK FINAL COMPARISON RESULTS:")
    print("======================================================================")
    display_cols = [
        "Experiment",
        "Ans_Recall@1",
        "Ans_Recall@3",
        "Ans_Recall@5",
        "Ans_MRR",
        "Mean_Latency_ms",
        "P95_Latency_ms",
    ]
    print(summary_df[display_cols].to_string(index=False))

    print(f"\nSaved test comparison to: {summary_path}")
    return summary_df


if __name__ == "__main__":
    run_test_evaluation()
