"""
Enhanced Latency Profiling Script (v2)
Measures Cold vs Warm latency distributions (Mean, Median, P95, Max) across all 8 pipeline stages
over 100 benchmark queries with consistent measurement boundaries and orchestration overhead tracking.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestration.graph import OrchestrationPipeline, get_orchestration_pipeline
from src.agents.verification_agent import EvidenceEntailmentVerifier

BENCHMARK_PATH = ROOT / "data" / "processed" / "rag" / "retrieval_benchmark_remediated_v2.csv"
bench_df = pd.read_csv(BENCHMARK_PATH)
sample_queries = bench_df.head(100).copy()

stages = [
    "latency_resolution_ms",
    "latency_triage_ms",
    "latency_retrieval_ms",
    "latency_verification_ms",
    "latency_policy_ms",
    "latency_supervisor_ms",
    "latency_generation_ms",
    "latency_orchestration_overhead_ms",
    "latency_e2e_ms",
]

pipeline = get_orchestration_pipeline()

# 1. Cold Profiling
print("Running Cold Evaluation across 100 queries...", flush=True)
cold_metrics = {stage: [] for stage in stages}

for idx, row in sample_queries.iterrows():
    q = str(row["Query"])
    d = str(row.get("Domain", "E-commerce"))
    EvidenceEntailmentVerifier.clear_cache()

    t0 = time.perf_counter()
    state = pipeline.run(query=q, domain=d, generate_grounded_response=True)
    e2e_actual = (time.perf_counter() - t0) * 1000

    bd = state.metadata.get("latency_breakdown", {})
    sum_stages = 0.0
    for stage in stages:
        if stage == "latency_e2e_ms":
            cold_metrics[stage].append(bd.get("latency_e2e_ms", e2e_actual))
        elif stage == "latency_orchestration_overhead_ms":
            pass
        else:
            val = bd.get(stage, 0.0)
            cold_metrics[stage].append(val)
            sum_stages += val

    overhead = max(0.0, e2e_actual - sum_stages)
    cold_metrics["latency_orchestration_overhead_ms"].append(overhead)

# 2. Warm Profiling
print("Running Warm Evaluation across 100 queries (Populated Cache + Query Vector Reuse)...", flush=True)
# Pre-warm with 15 queries
for idx, row in sample_queries.head(15).iterrows():
    pipeline.run(query=str(row["Query"]), domain=str(row.get("Domain", "E-commerce")), generate_grounded_response=True)

warm_metrics = {stage: [] for stage in stages}

for idx, row in sample_queries.iterrows():
    q = str(row["Query"])
    d = str(row.get("Domain", "E-commerce"))

    t0 = time.perf_counter()
    state = pipeline.run(query=q, domain=d, generate_grounded_response=True)
    e2e_actual = (time.perf_counter() - t0) * 1000

    bd = state.metadata.get("latency_breakdown", {})
    sum_stages = 0.0
    for stage in stages:
        if stage == "latency_e2e_ms":
            warm_metrics[stage].append(bd.get("latency_e2e_ms", e2e_actual))
        elif stage == "latency_orchestration_overhead_ms":
            pass
        else:
            val = bd.get(stage, 0.0)
            warm_metrics[stage].append(val)
            sum_stages += val

    overhead = max(0.0, e2e_actual - sum_stages)
    warm_metrics["latency_orchestration_overhead_ms"].append(overhead)

def calc_stats(arr):
    if not arr:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    a = np.array(arr)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }

cold_summary = {stage: calc_stats(cold_metrics[stage]) for stage in stages}
warm_summary = {stage: calc_stats(warm_metrics[stage]) for stage in stages}

out_data = {
    "sample_size": len(sample_queries),
    "cold_profile": cold_summary,
    "warm_profile": warm_summary,
    "cache_stats": EvidenceEntailmentVerifier.get_cache_stats(),
}

out_file = ROOT / "outputs" / "phase_g1_audit" / "latency_profile_v2.json"
with open(out_file, "w") as f:
    json.dump(out_data, f, indent=2)

print("\n" + "=" * 120)
print(f"{'STAGE':<34} | {'COLD MEAN':<10} | {'COLD MED':<10} | {'COLD P95':<10} | {'COLD MAX':<10} | {'WARM MEAN':<10} | {'WARM MED':<10} | {'WARM P95':<10} | {'WARM MAX':<10}")
print("-" * 120)
for stage in stages:
    cm = cold_summary[stage]["mean"]
    cmd = cold_summary[stage]["median"]
    cp = cold_summary[stage]["p95"]
    cx = cold_summary[stage]["max"]
    wm = warm_summary[stage]["mean"]
    wmd = warm_summary[stage]["median"]
    wp = warm_summary[stage]["p95"]
    wx = warm_summary[stage]["max"]
    print(f"{stage:<34} | {cm:<10.2f} | {cmd:<10.2f} | {cp:<10.2f} | {cx:<10.2f} | {wm:<10.2f} | {wmd:<10.2f} | {wp:<10.2f} | {wx:<10.2f}")
print("=" * 120)

print(f"\nSaved v2 latency profile to {out_file}")
