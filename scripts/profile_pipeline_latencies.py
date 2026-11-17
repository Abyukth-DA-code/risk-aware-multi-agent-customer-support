"""
Latency Profiling Script for Phase G.1
Measures Cold vs Warm latency distributions (Mean, Median, P95, Max) across all 8 pipeline stages.
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
sample_queries = bench_df.head(60).copy()

stages = [
    "latency_resolution_ms",
    "latency_triage_ms",
    "latency_retrieval_ms",
    "latency_verification_ms",
    "latency_policy_ms",
    "latency_supervisor_ms",
    "latency_generation_ms",
    "latency_e2e_ms",
]

# 1. Cold Profiling (Fresh Verifier Cache for each query or cold run)
print("Running Cold Evaluation (Cache Cleared)...", flush=True)
EvidenceEntailmentVerifier.clear_cache()
pipeline = get_orchestration_pipeline()

cold_metrics = {stage: [] for stage in stages}
for idx, row in sample_queries.iterrows():
    q = str(row["Query"])
    d = str(row.get("Domain", "E-commerce"))
    # Clear cache before query to simulate cold execution
    EvidenceEntailmentVerifier.clear_cache()
    t0 = time.perf_counter()
    state = pipeline.run(query=q, domain=d, generate_grounded_response=True)
    e2e_actual = (time.perf_counter() - t0) * 1000

    bd = state.metadata.get("latency_breakdown", {})
    for stage in stages:
        if stage == "latency_e2e_ms":
            cold_metrics[stage].append(bd.get("latency_e2e_ms", e2e_actual))
        else:
            cold_metrics[stage].append(bd.get(stage, 0.0))

# 2. Warm Profiling (Populated Cache + Query Vector Reuse)
print("Running Warm Evaluation (Populated Cache + Query Vector Reuse)...", flush=True)
# Pre-warm with a few queries
for idx, row in sample_queries.head(10).iterrows():
    pipeline.run(query=str(row["Query"]), domain=str(row.get("Domain", "E-commerce")), generate_grounded_response=True)

warm_metrics = {stage: [] for stage in stages}
for idx, row in sample_queries.iterrows():
    q = str(row["Query"])
    d = str(row.get("Domain", "E-commerce"))
    t0 = time.perf_counter()
    state = pipeline.run(query=q, domain=d, generate_grounded_response=True)
    e2e_actual = (time.perf_counter() - t0) * 1000

    bd = state.metadata.get("latency_breakdown", {})
    for stage in stages:
        if stage == "latency_e2e_ms":
            warm_metrics[stage].append(bd.get("latency_e2e_ms", e2e_actual))
        else:
            warm_metrics[stage].append(bd.get(stage, 0.0))

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
    "cold_profile": cold_summary,
    "warm_profile": warm_summary,
    "cache_stats": EvidenceEntailmentVerifier.get_cache_stats(),
}

out_file = ROOT / "outputs" / "phase_g1_audit" / "latency_profile_g1.json"
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    json.dump(out_data, f, indent=2)

print("\nSaved Latency Profile to:", out_file)
print("\n" + "=" * 90)
print(f"{'STAGE':<26} | {'COLD MEAN (ms)':<14} | {'COLD P95 (ms)':<14} | {'WARM MEAN (ms)':<14} | {'WARM P95 (ms)':<14}")
print("-" * 90)
for stage in stages:
    cm = cold_summary[stage]["mean"]
    cp = cold_summary[stage]["p95"]
    wm = warm_summary[stage]["mean"]
    wp = warm_summary[stage]["p95"]
    print(f"{stage:<26} | {cm:<14.2f} | {cp:<14.2f} | {wm:<14.2f} | {wp:<14.2f}")
print("=" * 90)
