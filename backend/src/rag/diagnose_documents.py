"""
RAG Knowledge Base & Benchmark Retrieval Diagnostic Tool.

Performs deep analysis of retrieval failures, most frequently missed documents,
root causes, and knowledge gap patterns across baseline and remediated benchmarks.
"""

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DOCS_PATH = ROOT / "data" / "processed" / "rag" / "documents.csv"
CHUNKS_PATH = ROOT / "data" / "processed" / "rag" / "chunks.csv"
BENCHMARK_PATH = ROOT / "data" / "processed" / "rag" / "retrieval_benchmark.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "rag_evaluation" / "retrieval_test_predictions.csv"
AUDIT_PATH = ROOT / "outputs" / "rag_evaluation" / "benchmark_relevance_audit_v2.csv"


def diagnose(
    docs_path: Path = DOCS_PATH,
    chunks_path: Path = CHUNKS_PATH,
    benchmark_path: Path = BENCHMARK_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    audit_path: Path = AUDIT_PATH,
):
    for path in [docs_path, chunks_path, benchmark_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    docs = pd.read_csv(docs_path).fillna("")
    chunks = pd.read_csv(chunks_path).fillna("")
    benchmark = pd.read_csv(benchmark_path).fillna("")

    print("\n=======================================================")
    print("RAG BENCHMARK & KNOWLEDGE BASE DIAGNOSTICS")
    print("=======================================================")
    print(f"Total documents: {len(docs)} across domains {list(docs['Domain'].unique())}")
    print(f"Total chunks:    {len(chunks)}")
    print(f"Total queries:   {len(benchmark)}")

    # Audit integration if available
    if audit_path.exists():
        audit_df = pd.read_csv(audit_path).fillna("")
        print(f"\nAudit records loaded from: {audit_path}")
        print("\n--- Relevance Category Distribution ---")
        print(audit_df["Relevance_Category"].value_counts().to_string())

        print("\n--- Root Cause Breakdown ---")
        print(audit_df["Root_Cause"].value_counts().to_string())

        print("\n--- Most Frequent Knowledge Gap Topics ---")
        gaps = audit_df[audit_df["Relevance_Category"] == "MISSING_KNOWLEDGE"]
        print(gaps.groupby(["Query_Domain", "Query_Intent"]).size().to_string())

        print("\n--- Verified Mappings (Remapped Documents) ---")
        remapped = audit_df[audit_df["Relevance_Category"] == "MISMATCHED"]
        remap_summary = remapped.groupby(
            ["Query_Domain", "Query_Intent", "Original_Expected_Document_ID", "Proposed_Document_ID", "Proposed_Title"]
        ).size().reset_index(name="Count")
        print(remap_summary.to_string(index=False))

    # Retrieval misses diagnosis
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path).fillna("")
        misses = predictions[predictions["Recall_at_5"] == 0]
        print(f"\n=======================================================")
        print(f"RETRIEVAL MISSES ANALYSIS (Total missed in Top 5: {len(misses)})")
        print("=======================================================")

        frequent = misses["Expected_Document_ID"].value_counts().head(5)
        print("\n--- Top 5 Most Missed Expected Document IDs ---")
        for doc_id, count in frequent.items():
            matching_docs = docs[docs["Document_ID"].astype(str) == str(doc_id)]
            matching_chunks = chunks[chunks["Document_ID"].astype(str) == str(doc_id)]

            print(f"\n{'-' * 60}")
            print(f"Document ID: {doc_id} | Misses: {count}")
            if matching_docs.empty:
                print("  Status: Absent from documents.csv (or UNANSWERABLE GAP)")
            else:
                doc = matching_docs.iloc[0]
                print(f"  Title:     {doc.get('Title')}")
                print(f"  Domain:    {doc.get('Domain')}")
                print(f"  Authority: {doc.get('Authority')} | Freshness: {doc.get('Freshness_Status')}")
                print(f"  Content:   {str(doc.get('Content', ''))[:200]}...")

            print(f"  Chunks count: {len(matching_chunks)}")
            sample_queries = misses[misses["Expected_Document_ID"].astype(str) == str(doc_id)]["Query"].head(3).tolist()
            for sq in sample_queries:
                print(f"    - Sample query: \"{sq}\"")

    # Reference integrity check
    doc_ids = set(docs["Document_ID"].astype(str))
    chunk_doc_ids = set(chunks["Document_ID"].astype(str))
    expected_ids = set(benchmark["Expected_Document_ID"].astype(str)) - {"", "nan", "NONE_KNOWLEDGE_GAP"}

    print("\n=======================================================")
    print("REFERENCE INTEGRITY VERIFICATION")
    print("=======================================================")
    missing_in_docs = sorted(expected_ids - doc_ids)
    missing_in_chunks = sorted(expected_ids - chunk_doc_ids)
    print(f"Expected IDs absent from documents.csv: {missing_in_docs if missing_in_docs else 'None (100% Valid)'}")
    print(f"Expected IDs absent from chunks.csv:    {missing_in_chunks if missing_in_chunks else 'None (100% Valid)'}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose RAG Knowledge Base and Benchmark")
    parser.add_argument("--docs", type=Path, default=DOCS_PATH, help="Path to documents CSV")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH, help="Path to chunks CSV")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH, help="Path to benchmark CSV")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH, help="Path to predictions CSV")
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH, help="Path to audit CSV")
    args = parser.parse_args()

    diagnose(
        docs_path=args.docs,
        chunks_path=args.chunks,
        benchmark_path=args.benchmark,
        predictions_path=args.predictions,
        audit_path=args.audit,
    )


if __name__ == "__main__":
    main()