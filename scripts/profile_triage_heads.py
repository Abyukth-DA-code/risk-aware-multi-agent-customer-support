"""
Triage Model Heads Individual Profiling Script
Measures latency (Mean, Median, P95, Max) for each of M1-M6 individually across 100 queries.
Investigates the root cause of the high P95 triage latency.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestration.model_service import ModelService

BENCHMARK_PATH = ROOT / "data" / "processed" / "rag" / "retrieval_benchmark_remediated_v2.csv"
bench_df = pd.read_csv(BENCHMARK_PATH)
sample_queries = bench_df.head(100).copy()

svc = ModelService.get_instance()

# Warmup run
svc.analyze_query("Test warmup query for triage models", domain="E-commerce")

m_latencies = {
    "M1_Intent": [],
    "M2_Sentiment": [],
    "M3_Risk": [],
    "M4_Complexity": [],
    "M5_Domain": [],
    "M6_Escalation": [],
    "Total_Sequential_Sum": [],
}

for idx, row in sample_queries.iterrows():
    q = str(row["Query"])
    d = str(row.get("Domain", "E-commerce"))

    # M5: Domain
    t0 = time.perf_counter()
    dom_res = svc.predict_domain(q)
    m5_ms = (time.perf_counter() - t0) * 1000
    m_latencies["M5_Domain"].append(m5_ms)
    resolved_d = dom_res["domain"] or d

    # M1: Intent
    t0 = time.perf_counter()
    intent_res = svc.predict_intent(q)
    m1_ms = (time.perf_counter() - t0) * 1000
    m_latencies["M1_Intent"].append(m1_ms)

    # M2: Sentiment
    t0 = time.perf_counter()
    sent_res = svc.predict_sentiment(q)
    m2_ms = (time.perf_counter() - t0) * 1000
    m_latencies["M2_Sentiment"].append(m2_ms)

    # M3: Risk
    t0 = time.perf_counter()
    risk_res = svc.predict_risk(q, domain=resolved_d)
    m3_ms = (time.perf_counter() - t0) * 1000
    m_latencies["M3_Risk"].append(m3_ms)

    # M4: Complexity
    t0 = time.perf_counter()
    comp_res = svc.predict_complexity(q, domain=resolved_d)
    m4_ms = (time.perf_counter() - t0) * 1000
    m_latencies["M4_Complexity"].append(m4_ms)

    # M6: Escalation
    t0 = time.perf_counter()
    esc_res = svc.predict_escalation(
        query=q,
        domain=resolved_d,
        risk=risk_res["risk"],
        complexity=comp_res["complexity"],
        sentiment=sent_res["sentiment"],
    )
    m6_ms = (time.perf_counter() - t0) * 1000
    m_latencies["M6_Escalation"].append(m6_ms)

    m_latencies["Total_Sequential_Sum"].append(m1_ms + m2_ms + m3_ms + m4_ms + m5_ms + m6_ms)

def calc_stats(arr):
    a = np.array(arr)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }

summary = {head: calc_stats(m_latencies[head]) for head in m_latencies}

print("\n" + "=" * 90)
print("INDIVIDUAL TRIAGE MODEL HEADS LATENCY BREAKDOWN (100 Queries)")
print("=" * 90)
print(f"{'MODEL HEAD':<24} | {'MEAN (ms)':<12} | {'MEDIAN (ms)':<12} | {'P95 (ms)':<12} | {'MAX (ms)':<12}")
print("-" * 90)
for head, stats in summary.items():
    print(f"{head:<24} | {stats['mean']:<12.2f} | {stats['median']:<12.2f} | {stats['p95']:<12.2f} | {stats['max']:<12.2f}")
print("=" * 90)

out_file = ROOT / "outputs" / "phase_g1_audit" / "triage_heads_latency_profile.json"
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved triage heads profile to {out_file}")
