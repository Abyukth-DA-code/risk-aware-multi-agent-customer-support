"""
RAG Retrieval Evaluation Engine.

Evaluates retrieval performance (Recall@1, Recall@3, Recall@5, MRR, latency)
against designated benchmark datasets (baseline v1 or remediated v2).
Supports per-domain breakdown and explicit reporting of knowledge gaps.
"""

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = (
    ROOT / "data" / "processed" / "rag" / "retrieval_benchmark.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "rag_evaluation"


def evaluate(
    benchmark_path: Path = BENCHMARK_PATH,
    output_dir: Path = OUTPUT_DIR,
    eval_name: str = "",
    top_k: int = 5,
    distinct_documents: bool = False,
    candidate_multiplier: int = 4,
    candidate_pool_size: int = None,
    hybrid: bool = False,
    rrf_k: int = 60,
):
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}")

    # Import the retriever from src/rag
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from retriever import RAGRetriever

    df = pd.read_csv(benchmark_path)

    required = {
        "Case_ID",
        "Query",
        "Domain",
        "Evaluation_Split",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark is missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Benchmark contains no rows.")

    if df["Query"].isna().any():
        raise ValueError("Benchmark contains missing queries.")

    # Verification of evaluation split
    splits = df["Evaluation_Split"].astype(str).str.strip().unique()

    if hybrid and distinct_documents:
        mode_label = f"Hybrid Distinct (RRF k={rrf_k})"
    elif hybrid:
        mode_label = f"Hybrid Chunks (RRF k={rrf_k})"
    elif distinct_documents:
        mode_label = "Distinct Documents"
    else:
        mode_label = "Raw Chunks"

    print(f"\n=======================================================")
    print(f"RAG RETRIEVAL EVALUATION: {eval_name or benchmark_path.name} [{mode_label}]")
    print(f"=======================================================")
    print(f"Benchmark path:     {benchmark_path}")
    print(f"Benchmark rows:     {len(df)}")
    print(f"Evaluation split:   {', '.join(splits)}")
    print(f"Distinct documents: {distinct_documents} (multiplier: {candidate_multiplier})")
    print(f"Hybrid mode:        {hybrid} (RRF k: {rrf_k})")
    print("Initializing retriever...")

    retriever = RAGRetriever()

    rows = []
    start_all = time.perf_counter()

    for index, row in df.iterrows():
        query = str(row["Query"]).strip()
        domain = str(row["Domain"]).strip()
        expected = str(row.get("Expected_Document_ID", "")).strip()
        if expected.lower() in ("nan", "none", ""):
            expected = ""

        start = time.perf_counter()
        results = retriever.retrieve(
            query=query,
            domain=domain,
            top_k=top_k,
            distinct_documents=distinct_documents,
            candidate_multiplier=candidate_multiplier,
            candidate_pool_size=candidate_pool_size,
            hybrid=hybrid,
            rrf_k=rrf_k,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = [str(item["document_id"]) for item in results]

        # Verify no duplicate document IDs in distinct mode
        if distinct_documents:
            assert len(retrieved_ids) == len(set(retrieved_ids)), (
                f"Duplicate document IDs found in distinct mode: {retrieved_ids}"
            )

        # Relevance matching: expected ID must be non-empty and present in retrieved_ids
        if expected:
            ranks = [
                rank
                for rank, document_id in enumerate(retrieved_ids, start=1)
                if document_id == expected
            ]
            first_rank = ranks[0] if ranks else None
        else:
            first_rank = None

        rows.append({
            "Case_ID": row["Case_ID"],
            "Query": query,
            "Domain": domain,
            "Expected_Document_ID": expected or "NONE_KNOWLEDGE_GAP",
            "Retrieved_Document_IDs": " | ".join(retrieved_ids),
            "Expected_Rank": first_rank,
            "Recall_at_1": int(first_rank == 1),
            "Recall_at_3": int(first_rank is not None and first_rank <= 3),
            "Recall_at_5": int(first_rank is not None and first_rank <= 5),
            "Reciprocal_Rank": (1.0 / first_rank if first_rank is not None else 0.0),
            "Top1_Similarity": results[0]["similarity"] if results else None,
            "Latency_ms": round(elapsed_ms, 3),
            "Is_Answerable": int(bool(expected)),
        })

        if (index + 1) % 100 == 0 or index + 1 == len(df):
            print(f"Evaluated {index + 1}/{len(df)} queries")

    elapsed_total = time.perf_counter() - start_all
    results_df = pd.DataFrame(rows)

    def summarize(group):
        ans_count = group["Is_Answerable"].sum()
        return pd.Series({
            "Total_Queries": len(group),
            "Answerable_Queries": int(ans_count),
            "Unanswerable_Gaps": int(len(group) - ans_count),
            "Recall@1": group["Recall_at_1"].mean(),
            "Recall@3": group["Recall_at_3"].mean(),
            "Recall@5": group["Recall_at_5"].mean(),
            "MRR": group["Reciprocal_Rank"].mean(),
            "Answerable_Recall@1": (group[group["Is_Answerable"] == 1]["Recall_at_1"].mean() if ans_count > 0 else 0.0),
            "Answerable_Recall@3": (group[group["Is_Answerable"] == 1]["Recall_at_3"].mean() if ans_count > 0 else 0.0),
            "Answerable_Recall@5": (group[group["Is_Answerable"] == 1]["Recall_at_5"].mean() if ans_count > 0 else 0.0),
            "Answerable_MRR": (group[group["Is_Answerable"] == 1]["Reciprocal_Rank"].mean() if ans_count > 0 else 0.0),
            "Mean_Latency_ms": group["Latency_ms"].mean(),
            "P95_Latency_ms": group["Latency_ms"].quantile(0.95),
        })

    overall = summarize(results_df)
    by_domain = results_df.groupby("Domain", dropna=False).apply(summarize, include_groups=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{eval_name}" if eval_name else ""
    details_path = output_dir / f"retrieval_test_predictions{suffix}.csv"
    domain_path = output_dir / f"retrieval_metrics_by_domain{suffix}.csv"
    summary_path = output_dir / f"retrieval_metrics_overall{suffix}.csv"

    results_df.to_csv(details_path, index=False)
    by_domain.to_csv(domain_path)
    overall.to_frame(name="Value").to_csv(summary_path)

    # If this is the default run without eval_name, also preserve default files
    if not eval_name:
        results_df.to_csv(output_dir / "retrieval_test_predictions.csv", index=False)
        by_domain.to_csv(output_dir / "retrieval_metrics_by_domain.csv")
        overall.to_frame(name="Value").to_csv(output_dir / "retrieval_metrics_overall.csv")

    print(f"\n========== OVERALL RETRIEVAL METRICS ({eval_name or 'Default'}) [{mode_label}] ==========")
    print(f"Total Queries:       {int(overall['Total_Queries'])}")
    print(f"Answerable Queries:  {int(overall['Answerable_Queries'])}")
    print(f"Unanswerable Gaps:   {int(overall['Unanswerable_Gaps'])}")
    print(f"Overall Recall@1:    {overall['Recall@1']:.4f}")
    print(f"Overall Recall@3:    {overall['Recall@3']:.4f}")
    print(f"Overall Recall@5:    {overall['Recall@5']:.4f}")
    print(f"Overall MRR:         {overall['MRR']:.4f}")
    print(f"Answerable Recall@1: {overall['Answerable_Recall@1']:.4f}")
    print(f"Answerable Recall@3: {overall['Answerable_Recall@3']:.4f}")
    print(f"Answerable Recall@5: {overall['Answerable_Recall@5']:.4f}")
    print(f"Answerable MRR:      {overall['Answerable_MRR']:.4f}")
    print(f"Mean Latency:        {overall['Mean_Latency_ms']:.2f} ms")
    print(f"P95 Latency:         {overall['P95_Latency_ms']:.2f} ms")
    print(f"Elapsed Time:        {elapsed_total:.2f} s")

    print("\n========== PER-DOMAIN METRICS ==========")
    cols_to_show = ["Total_Queries", "Answerable_Queries", "Recall@5", "Answerable_Recall@5", "MRR", "Answerable_MRR"]
    print(by_domain[cols_to_show].round(4).to_string())

    misses = results_df[(results_df["Is_Answerable"] == 1) & (results_df["Recall_at_5"] == 0)]
    print(f"\nAnswerable queries missing expected document in top 5: {len(misses)}")

    print(f"\nReports saved to: {output_dir}")
    return results_df, overall, by_domain


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval Benchmark")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH, help="Path to benchmark CSV")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--eval_name", type=str, default="", help="Evaluation run name identifier")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K retrieval depth")
    parser.add_argument("--distinct_documents", action="store_true", help="Enable distinct document retrieval mode")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid BM25 + dense retrieval with RRF")
    parser.add_argument("--rrf_k", type=int, default=60, help="Reciprocal Rank Fusion constant k (default: 60)")
    parser.add_argument("--candidate_multiplier", type=int, default=4, help="Candidate pool multiplier")
    parser.add_argument("--candidate_pool_size", type=int, default=None, help="Explicit candidate pool size")
    args = parser.parse_args()

    evaluate(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        eval_name=args.eval_name,
        top_k=args.top_k,
        distinct_documents=args.distinct_documents,
        candidate_multiplier=args.candidate_multiplier,
        candidate_pool_size=args.candidate_pool_size,
        hybrid=args.hybrid,
        rrf_k=args.rrf_k,
    )


if __name__ == "__main__":
    main()
