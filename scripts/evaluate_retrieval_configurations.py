"""
Retrieval Configuration Benchmark Comparison
Evaluates Pure Dense vs Hybrid BM25 on the 501-query benchmark.
Computes Recall@1, Recall@3, Recall@5, MRR, and Latency for Answerable and Overall cohorts.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.retriever import RAGRetriever

BENCHMARK_PATH = ROOT / "data" / "processed" / "rag" / "retrieval_benchmark_remediated_v2.csv"
df = pd.read_csv(BENCHMARK_PATH)

retriever = RAGRetriever()

configs = [
    {
        "name": "Pure Dense (Raw Chunks)",
        "hybrid": False,
        "distinct_documents": False,
        "candidate_multiplier": 1,
    },
    {
        "name": "Pure Dense (Distinct Docs, m=4)",
        "hybrid": False,
        "distinct_documents": True,
        "candidate_multiplier": 4,
    },
    {
        "name": "Hybrid Lexical-Dense (Current Demo Mode)",
        "hybrid": True,
        "distinct_documents": False,
        "candidate_multiplier": 4,
    },
    {
        "name": "Hybrid Distinct (k=60, m=4)",
        "hybrid": True,
        "distinct_documents": True,
        "candidate_multiplier": 4,
    },
]

# Warm up
retriever.retrieve("What is your refund policy?", top_k=5, hybrid=False)
retriever.retrieve("What is your refund policy?", top_k=5, hybrid=True)

results = []

for cfg in configs:
    name = cfg["name"]
    is_hybrid = cfg["hybrid"]
    is_distinct = cfg["distinct_documents"]
    m = cfg["candidate_multiplier"]

    latencies = []

    # Track for Answerable cohort (Expected_Document_ID is present and not unanswerable)
    ans_hits_1, ans_hits_3, ans_hits_5 = 0, 0, 0
    ans_mrr_sum = 0.0
    ans_count = 0

    # Track for all queries with an Expected_Document_ID
    all_hits_5 = 0
    all_mrr_sum = 0.0
    all_count = 0

    t_start = time.perf_counter()

    for idx, row in df.iterrows():
        query = str(row["Query"]).strip()
        domain = str(row.get("Domain", "E-commerce")).strip()
        exp_doc = str(row.get("Expected_Document_ID", "")).strip()
        rel_cat = str(row.get("Audit_Relevance_Category", "")).strip()

        t0 = time.perf_counter()
        retrieved = retriever.retrieve(
            query=query,
            domain=domain,
            top_k=5,
            distinct_documents=is_distinct,
            candidate_multiplier=m,
            hybrid=is_hybrid,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        ret_doc_ids = [str(c.get("document_id", "")).strip() for c in retrieved]

        is_answerable = rel_cat not in ["MISSING_KNOWLEDGE", "METADATA_CONFLICT"] and bool(exp_doc)

        if is_answerable:
            ans_count += 1
            if len(ret_doc_ids) >= 1 and exp_doc == ret_doc_ids[0]:
                ans_hits_1 += 1
            if exp_doc in ret_doc_ids[:3]:
                ans_hits_3 += 1
            if exp_doc in ret_doc_ids[:5]:
                ans_hits_5 += 1

            # Reciprocal rank
            if exp_doc in ret_doc_ids:
                rank = ret_doc_ids.index(exp_doc) + 1
                ans_mrr_sum += 1.0 / rank

        if exp_doc:
            all_count += 1
            if exp_doc in ret_doc_ids[:5]:
                all_hits_5 += 1
            if exp_doc in ret_doc_ids:
                rank = ret_doc_ids.index(exp_doc) + 1
                all_mrr_sum += 1.0 / rank

    wall_s = time.perf_counter() - t_start

    ans_r1 = (ans_hits_1 / ans_count) if ans_count > 0 else 0.0
    ans_r3 = (ans_hits_3 / ans_count) if ans_count > 0 else 0.0
    ans_r5 = (ans_hits_5 / ans_count) if ans_count > 0 else 0.0
    ans_mrr = (ans_mrr_sum / ans_count) if ans_count > 0 else 0.0

    all_r5 = (all_hits_5 / all_count) if all_count > 0 else 0.0
    all_mrr = (all_mrr_sum / all_count) if all_count > 0 else 0.0

    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    results.append({
        "Configuration": name,
        "Hybrid": is_hybrid,
        "Distinct_Docs": is_distinct,
        "Ans_Recall@1": round(ans_r1, 4),
        "Ans_Recall@3": round(ans_r3, 4),
        "Ans_Recall@5": round(ans_r5, 4),
        "Ans_MRR": round(ans_mrr, 4),
        "Overall_Recall@5": round(all_r5, 4),
        "Overall_MRR": round(all_mrr, 4),
        "Mean_Latency_ms": round(mean_lat, 2),
        "P95_Latency_ms": round(p95_lat, 2),
        "Total_Wall_s": round(wall_s, 2),
    })

res_df = pd.DataFrame(results)
print("\n" + "=" * 110)
print("RETRIEVAL CONFIGURATION EMPIRICAL COMPARISON (501 Benchmark Queries)")
print("=" * 110)
print(res_df.to_string(index=False))
print("=" * 110)

out_csv = ROOT / "outputs" / "phase_g1_audit" / "retrieval_configuration_comparison.csv"
res_df.to_csv(out_csv, index=False)
print(f"\nSaved comparison to {out_csv}")
