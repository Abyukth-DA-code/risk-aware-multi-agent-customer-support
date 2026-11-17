"""
Development Benchmark Tuning & Hybrid Retrieval Comparison Suite.

Executes controlled retrieval experiments across:
 - Pure Dense Raw Chunks
 - Pure Dense Distinct Documents
 - Hybrid Lexical-Dense Chunks (RRF k=60)
 - Hybrid Lexical-Dense Distinct Documents (RRF k in [20, 60, 100])
 - Multiplier sweeps (m in [2, 4, 8])

Strictly uses data/processed/rag/retrieval_dev_benchmark.csv for tuning.
Outputs comparison tables and selects candidate for final test validation.
"""

from pathlib import Path
import sys
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEV_BENCHMARK = ROOT / "data" / "processed" / "rag" / "retrieval_dev_benchmark.csv"
OUTPUT_DIR = ROOT / "outputs" / "rag_evaluation"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_retrieval import evaluate


def run_dev_tuning():
    if not DEV_BENCHMARK.exists():
        raise FileNotFoundError(f"Dev benchmark not found: {DEV_BENCHMARK}")

    experiments = [
        {
            "name": "dev_pure_dense_raw",
            "desc": "Pure Dense (Raw Chunks)",
            "distinct": False,
            "hybrid": False,
            "rrf_k": 60,
            "multiplier": 4,
        },
        {
            "name": "dev_pure_dense_distinct_m4",
            "desc": "Pure Dense (Distinct Docs, m=4)",
            "distinct": True,
            "hybrid": False,
            "rrf_k": 60,
            "multiplier": 4,
        },
        {
            "name": "dev_hybrid_raw_k60",
            "desc": "Hybrid (Raw Chunks, k=60)",
            "distinct": False,
            "hybrid": True,
            "rrf_k": 60,
            "multiplier": 4,
        },
        {
            "name": "dev_hybrid_distinct_k20_m4",
            "desc": "Hybrid (Distinct, k=20, m=4)",
            "distinct": True,
            "hybrid": True,
            "rrf_k": 20,
            "multiplier": 4,
        },
        {
            "name": "dev_hybrid_distinct_k60_m4",
            "desc": "Hybrid (Distinct, k=60, m=4)",
            "distinct": True,
            "hybrid": True,
            "rrf_k": 60,
            "multiplier": 4,
        },
        {
            "name": "dev_hybrid_distinct_k100_m4",
            "desc": "Hybrid (Distinct, k=100, m=4)",
            "distinct": True,
            "hybrid": True,
            "rrf_k": 100,
            "multiplier": 4,
        },
        {
            "name": "dev_hybrid_distinct_k60_m2",
            "desc": "Hybrid (Distinct, k=60, m=2)",
            "distinct": True,
            "hybrid": True,
            "rrf_k": 60,
            "multiplier": 2,
        },
        {
            "name": "dev_hybrid_distinct_k60_m8",
            "desc": "Hybrid (Distinct, k=60, m=8)",
            "distinct": True,
            "hybrid": True,
            "rrf_k": 60,
            "multiplier": 8,
        },
    ]

    summary_rows = []
    domain_dfs = {}

    print("======================================================================")
    print("STARTING RETRIEVAL DEVELOPMENT BENCHMARK TUNING (501 queries, 429 ans)")
    print("======================================================================")

    for exp in experiments:
        print(f"\n---> Running Experiment: {exp['desc']} ({exp['name']})")
        t0 = time.time()
        results_df, overall, by_domain = evaluate(
            benchmark_path=DEV_BENCHMARK,
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
        domain_dfs[exp["name"]] = by_domain

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "dev_benchmark_tuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n======================================================================")
    print("DEV BENCHMARK TUNING SUMMARY RESULTS:")
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

    print(f"\nSaved dev tuning summary to: {summary_path}")
    return summary_df, domain_dfs


if __name__ == "__main__":
    run_dev_tuning()
