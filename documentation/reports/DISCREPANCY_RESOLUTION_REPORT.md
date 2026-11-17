# Master Discrepancy Resolution and Research Results Freeze Report

**Project Title:** Autonomous Multi-Agent Customer Support Resolution System with Risk-Aware Adaptive Orchestration and Human-in-the-Loop Escalation  
**Degree:** Bachelor of Technology in Computer Science & Engineering (AI & ML) — Final Year Major Project  
**Base Research Paper:** STELLAR: A Structured, Trustworthy, and Explainable LLM-Led Architecture for Reliable Customer Support (ACL 2024)  
**Document Purpose:** Definitive resolution of numerical, architectural, and dataset discrepancies identified during research audit prior to final document freeze.  
**Audit Date:** November 2026  
**Artifact Directory:** `FINAL_RESEARCH_DOCUMENTATION/`  

---

## 1. Executive Summary of Audited Inconsistencies & Corrective Actions

This report audits every numerical inconsistency, dataset discrepancy, and phrasing issue identified in the research evaluation. All findings have been verified directly against raw execution logs and source CSVs (`outputs/research_evaluation/m6_experiment_20261031_063000/` and `data/processed/autonomy/`).

### Master Inconsistency Reconciliation Matrix

| # | Discrepancy Area | Original / Ambiguous Text | Corrected Verified Value | Source Log / Script Supporting Correction | Root Cause & Resolution |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Policy B Dual Results** | Conflated Original Policy B (25.55% Acc, 409 Esc) and Optimized Policy B (29.54% Acc, 379 Esc). | **Separated into two distinct configurations:**<br>1. *Original Policy B (Uncalibrated Baseline):* Acc: 25.55% (128/501), Esc: 409 (81.64%), Unsafe Answers: 6.16% (21/341).<br>2. *Optimized Policy B (Validation-Calibrated Gate):* Acc: 29.54% (148/501), Esc: 379 (75.65%), Unsafe Answers: 8.21% (28/341). | `m6_fixed_vs_adaptive_metrics.csv`<br>`m6_case_level_comparison.csv` | Original Policy B used rigid $C \ge 0.75$ gate; Optimized Policy B used validation-calibrated $C \ge 0.65$ gate for Low Risk. Clearly demarcated. |
| **2** | **Correct-Answer Count Ambiguity** | Alternating between "128 to 148" and "64 to 89" without clear definitions. | **Defined exact metrics and denominators:**<br>• *Total 3-Class Correct Predictions ($N=501$):* 128 (Original B) vs. 148 (Optimized B).<br>• *Direct Autonomous Answer True Positives ($N=373$):* 64 (Original B) vs. 89 (Optimized B). | `m6_confusion_matrices.csv`<br>`m6_case_level_comparison.csv` | Clarified denominator distinction: 128/501 (25.55%) is overall accuracy; 64/373 (17.16%) is ANSWER class recall. |
| **3** | **Validation Dataset Size** | Stated as "N = 375" in some text. | **Exact sample count is $N = 501$ rows** (file: `data/processed/autonomy/validation.csv`). | `data/processed/autonomy/validation.csv` | Corrected historical typo; validation dataset contains exactly 501 unique cases with zero duplicate IDs and zero overlap with test set. |
| **4** | **Validation Script Error** | Scratch script `test_val_optimization.py` exited with code 1. | **Resolved column name KeyError:** `retrieval_dev_benchmark.csv` did not contain `Audit_Relevance_Category`. | `scratch/audit_full_resolution.py` | Column in dev benchmark is `Expected_Status` / `Expected_Document_ID`. Fixed in audit script; full data integrity verified. |
| **5** | **Arbitrary Latency Boundary** | Cited an "arbitrary 507 ms boundary" and "500 ms SLA". | **Reported exact measured percentiles:** Mean: 112.77 ms, Median: 87.23 ms, P95: 287.70 ms, P99: 444.16 ms, Max: 506.08 ms. | `m6_latency_comparison.csv` | Removed speculative SLA claims; evaluated measured percentiles against standard interactive web latency (1.0–2.0 s). |
| **6** | **Statistical p-value Reporting** | Reported p-values as `p = 0.0000e+00`. | **Formatted to standard academic convention:** $p < 0.001$ with exact $\chi^2$ statistics ($\chi^2 = 249.004$ on safety, $\chi^2 = 248.062$ on accuracy). | `m6_statistical_analysis.md`<br>`scipy.stats` | Avoided reporting absolute zero for asymptotic continuous test distributions. |
| **7** | **Dataset Independence Scope** | Claimed "independent clinical validation". | **Characterized as an "exploratory held-out benchmark" on 501 cases.** | `data/processed/autonomy/test.csv` | Disclosed that static pre-retrieval ground truth differs from dynamic post-retrieval evidence evaluation. |
| **8** | **Generative Hallucination Scope** | Ambiguous claims regarding "hallucination elimination". | **Strictly scoped to Decision Routing Safety:** Benchmark ran with `generate_grounded_response=False`. | `scripts/run_m6_controlled_experiment.py` | Clarified that the experiment measures routing decisions and retrieval precision, not LLM token generation. |

---

## 2. Rigorous Dataset Integrity and Partitioning Audit

A comprehensive code-level inspection was conducted across all partitions in `data/processed/autonomy/`:

### 2.1 Sample Counts & Data Hygiene
* **Training Set (`train.csv`):** Exactly **2,333 rows**, 0 duplicate `Case_ID`s, 0 missing cells across all 12 columns.
* **Validation Set (`validation.csv`):** Exactly **501 rows**, 0 duplicate `Case_ID`s, 0 missing cells across all 12 columns.
* **Held-Out Test Set (`test.csv`):** Exactly **501 rows**, 0 duplicate `Case_ID`s, 0 missing cells across all 12 columns.
* **Total Corpus Size:** $2,333 + 501 + 501 = \mathbf{3,335}$ unique cases.

### 2.2 Leakage & Overlap Verification
* **Case_ID Overlap:**
  * $\text{Validation} \cap \text{Test} = \mathbf{0}$ cases.
  * $\text{Train} \cap \text{Validation} = \mathbf{0}$ cases.
  * $\text{Train} \cap \text{Test} = \mathbf{0}$ cases.
* **Normalized Query Text Overlap:**
  * $\text{Validation Query} \cap \text{Test Query} = \mathbf{0}$ cases.
  * $\text{Train Query} \cap \text{Test Query} = \mathbf{0}$ cases.
* **Test Set Label Independence:** No test set labels or test prediction logs were accessed during the tuning of the validation threshold gates. The validation tuning used exclusively `data/processed/autonomy/validation.csv` ($N = 501$).

### 2.3 Label Distributions Across Splits

| Partition | Total Cases ($N$) | Ground Truth ANSWER | Ground Truth ORCHESTRATE | Ground Truth ESCALATE |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 2,333 | 1,712 (73.38%) | 407 (17.45%) | 214 (9.17%) |
| **Validation Set** | 501 | 384 (76.65%) | 76 (15.17%) | 41 (8.18%) |
| **Held-Out Test Set** | 501 | 373 (74.45%) | 79 (15.77%) | 49 (9.78%) |

---

## 3. Policy B Result Discrepancy Resolution: Original vs. Optimized

To ensure complete academic transparency, Policy B is presented under its **Original Uncalibrated Baseline** configuration, with the **Optimized (Validation-Calibrated)** configuration reported as a rigorous sensitivity exploration:

### 3.1 Policy Configurations Defined
1. **Policy A (Fixed Baseline):**
   - Routing: Pre-retrieval argmax classification ($R = \arg\max P(y|x)$).
   - Knowledge retrieval: Disabled ($k=0$).
   - Safety checks: None.
2. **Original Policy B (Uncalibrated Adaptive Policy):**
   - Routing: Fast-path cascade (M1–M6) $\to$ ChromaDB retrieval $\to$ Observable Composite Confidence $C$.
   - Safety Engine: Canonical rules in `configs/autonomy_rules.yaml`.
   - ANSWER Gate: `max_risk: Low`, `max_complexity: Medium`, `evidence_quality: Strong`, **uncalibrated threshold $C \ge 0.75$**.
   - ORCHESTRATE Gate: `required_complexity: High`, $C \ge 0.70$.
   - Fallback: Defaults to ESCALATE if gates are not met.
3. **Optimized Policy B (Validation-Calibrated Adaptive Policy):**
   - Safety Engine: Preserves all 6 mandatory invariants (`CRIT_01`, `RISK_VERIF_02`, `EVID_CONF_03`, `EVID_INSUF_04`, `LOW_CONF_05`, `SENS_FIN_06`).
   - ANSWER Gate: Relaxed threshold for `Risk: Low` to **$C \ge 0.65$** (aligned with `LOW_CONF_05`). Allowed safe `Risk: Medium` queries with Strong evidence and high confidence ($C \ge 0.75$) to be answered autonomously.
   - Fallback: Defaults to ESCALATE.

### 3.2 Exact Comparative Metrics Across Configurations ($N = 501$ Test Cases)

| Metric | Policy A (Fixed M6) | Original Policy B (Uncalibrated) | Optimized Policy B (Validation-Calibrated) | Source Column / Exact Definition |
| :--- | :---: | :---: | :---: | :--- |
| **Total Correct Predictions** | **444 / 501 (88.62%)** | **128 / 501 (25.55%)** | **148 / 501 (29.54%)** | Trace of $3 \times 3$ confusion matrix ($\text{TP}_{\text{ANS}} + \text{TP}_{\text{ORCH}} + \text{TP}_{\text{ESC}}$) |
| **Accuracy 95% Wilson CI** | $[85.54\%, 91.11\%]$ | $[21.93\%, 29.54\%]$ | $[25.71\%, 33.68\%]$ | Asymmetric Wilson score interval |
| **Macro-Averaged F1** | **0.6316** | **0.2882** | **0.3142** | Unweighted mean of class F1 scores |
| **Correct Autonomous Answers** | **365 / 373 (97.86%)** | **64 / 373 (17.16%)** | **89 / 373 (23.86%)** | True Positives for class `ANSWER` alone |
| **Answer Class Precision** | 91.25% (365/400) | **96.97%** (64/66) | 92.71% (89/96) | Precision among queries routed to ANSWER |
| **Correct Multi-Agent Dispatches**| **77 / 79 (97.47%)** | 20 / 79 (25.32%) | 20 / 79 (25.32%) | True Positives for class `ORCHESTRATE` |
| **Orchestrate Class Precision** | 79.38% (77/97) | 76.92% (20/26) | 76.92% (20/26) | Precision among queries routed to ORCHESTRATE |
| **True Human Escalations Captured**| 2 / 49 (4.08%) | **44 / 49 (89.80%)** | 39 / 49 (79.59%) | True Positives for class `ESCALATE` (Escalation Recall) |
| **Escalation Class Precision** | **50.00%** (2/4) | 10.76% (44/409) | 10.29% (39/379) | Precision on human escalation queue |
| **Total Human Escalation Workload**| **4 (0.80%)** | 409 (81.64%) | 379 (75.65%) | Total cases routed to human review |
| **Unsafe Cohort Answers ($N=341$)**| 272 / 341 (79.77%) | **21 / 341 (6.16%)** | 28 / 341 (8.21%) | Primary Safety KPI: Lower is Better |
| **Relative Risk Reduction** | Baseline (0.0%) | **$-92.28\%$ ($p < 0.001$)** | **$-89.71\%$ ($p < 0.001$)** | Relative reduction vs. Policy A on unsafe cohort |
| **High/Critical Breaches ($N=30$)** | 2 / 30 (6.67%) | **0 / 30 (0.00%)** | **0 / 30 (0.00%)** | Zero breaches observed under invariant `CRIT_01` |
| **Knowledge-Gap Breaches ($N=73$)**| 73 / 73 (100.0%) | **0 / 73 (0.00%)** | **0 / 73 (0.00%)** | Zero breaches observed under invariant `EVID_INSUF_04` |
| **Unnecessary Escalations on Safe** | **0 / 94 (0.00%)** | 48 / 94 (51.06%) | 48 / 94 (51.06%) | Safe FAQ queries unnecessarily escalated |
| **Mean Latency (ms)** | **55.17 ms** | 112.77 ms | 112.78 ms | Monotonic CPU `time.perf_counter()` |
| **Median Latency (P50, ms)** | **39.76 ms** | 87.23 ms | 87.23 ms | Monotonic CPU `time.perf_counter()` |
| **95th Percentile Latency (P95)** | **163.19 ms** | 287.70 ms | 287.71 ms | Monotonic CPU `time.perf_counter()` |
| **99th Percentile Latency (P99)** | **182.54 ms** | 444.16 ms | 444.16 ms | Monotonic CPU `time.perf_counter()` |
| **Maximum Latency (Max, ms)** | **410.29 ms** | 506.08 ms | 506.09 ms | Worst-case single query execution |

---

## 4. Mathematical Clarification of Answer Counts

To eliminate confusion between the two measurements:
1. **Total Correct Predictions (Accuracy Numerator):**
   $$\text{Total Correct} = \text{TP}_{\text{ANSWER}} + \text{TP}_{\text{ORCHESTRATE}} + \text{TP}_{\text{ESCALATE}}$$
   - **Original Policy B:** $64 + 20 + 44 = \mathbf{128} \implies \frac{128}{501} = \mathbf{25.55\%}$
   - **Optimized Policy B:** $89 + 20 + 39 = \mathbf{148} \implies \frac{148}{501} = \mathbf{29.54\%}$
   - Denominator: All $N = 501$ test queries.
2. **Correct Autonomous Answers (ANSWER Class Recall Numerator):**
   $$\text{Correct Autonomous Answers} = \text{TP}_{\text{ANSWER}}$$
   - **Original Policy B:** $\mathbf{64} \implies \frac{64}{373} = \mathbf{17.16\%}$
   - **Optimized Policy B:** $\mathbf{89} \implies \frac{89}{373} = \mathbf{23.86\%}$
   - Denominator: The $N = 373$ queries where ground truth is `ANSWER`.
   - **Yield Improvement:** Calibrating Policy B unlocked $89 - 64 = \mathbf{25}$ additional safe autonomous answers, representing a **$+38.8\%$ relative increase** in autonomous answering capability.

---

## 5. Recalculated Statistical Significance Tests

All paired statistical tests were recomputed directly from the row-level predictions in `m6_case_level_comparison.csv`:

### 5.1 Unsafe Cohort Safety Difference (Policy A vs. Original Policy B)
* Unsafe Cohort Sample: $N = 341$ queries.
* Discordant Pairs:
  * $b$ (Policy A unsafe, Policy B safe) = **251**
  * $c$ (Policy A safe, Policy B unsafe) = **0**
* Edwards Continuity-Corrected McNemar Statistic:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(251 - 1)^2}{251} = \frac{62,500}{251} = \mathbf{249.0040} \quad (df = 1)$$
* **p-value:** $p = 1 - F_{\chi^2}(249.0040, 1) = 0.0000e+00 \implies \mathbf{p < 0.001}$.
* **Odds Ratio:** $\frac{b}{c} = \frac{251}{0} \to \infty$.
* **Conclusion:** The reduction in unsafe answering is statistically definitive.

### 5.2 3-Class Routing Accuracy Difference (Policy A vs. Original Policy B)
* Evaluation Sample: $N = 501$ queries.
* Discordant Pairs:
  * $b$ (Policy A correct, Policy B incorrect) = **358**
  * $c$ (Policy A incorrect, Policy B correct) = **42**
* Edwards Continuity-Corrected McNemar Statistic:
  $$\chi^2 = \frac{(|358 - 42| - 1)^2}{358 + 42} = \frac{315^2}{400} = \frac{99,225}{400} = \mathbf{248.0625} \quad (df = 1)$$
* **p-value:** $p < 0.001$.
* **Conclusion:** Policy A has statistically higher static accuracy against offline labels ($p < 0.001$), but does so by guessing ANSWER on unanswerable and unsafe queries.

### 5.3 Optimization Accuracy Improvement (Original B vs. Optimized B)
* Evaluation Sample: $N = 501$ queries.
* Discordant Pairs:
  * $b$ (Original B correct, Optimized B incorrect) = **5**
  * $c$ (Original B incorrect, Optimized B correct) = **25**
* Edwards Continuity-Corrected McNemar Statistic:
  $$\chi^2 = \frac{(|5 - 25| - 1)^2}{5 + 25} = \frac{19^2}{30} = \frac{361}{30} = \mathbf{12.0333} \quad (df = 1)$$
* **p-value:** $p = 5.2258 \times 10^{-4} \implies \mathbf{p < 0.001}$.
* **Conclusion:** The calibration of qualification gates produces a statistically significant accuracy improvement (+3.99 percentage points, $p < 0.001$).

### 5.4 Paired Latency Analysis (Policy A vs. Policy B)
* Paired Student's $t$-test: $t = -24.8851, p = 1.5451 \times 10^{-89} \implies \mathbf{p < 0.001}$.
* Wilcoxon Signed-Rank Test: $W = 455.0, p = 1.3172 \times 10^{-82} \implies \mathbf{p < 0.001}$.
* Overhead Magnitude: $+57.60$ ms mean, $+47.47$ ms median (P50).

---

## 6. Hardware, CPU, and Latency Execution Profiling

All latency measurements were captured using the following standardized protocol:
* **Host Platform:** Local x86_64 workstation (AMD Ryzen 5 processor, 6 cores / 12 threads) running Windows 11.
* **Runtime Environment:** Python 3.11.9, Scikit-learn 1.4.2, PyTorch 2.3.0 (CPU execution mode), ChromaDB 0.5.0 with Cosine HNSW indexing.
* **Warm-up Protocol:** Executed 3 priming queries through all models and ChromaDB vector indices prior to benchmarking to eliminate initial module-load and disk-cache cold-start latency.
* **Timing Method:** Monotonic wall-clock profiling using `time.perf_counter()`.
* **Execution Mode:** Single-threaded, synchronous sequential execution across the 501 held-out inquiries.

---

## 7. Remaining Methodological Limitations

1. **Grounded Text Generation Evaluation Incomplete:** The 501-case benchmark set `generate_grounded_response=False`. This research evaluates **pre-generation routing safety and retrieval precision**, NOT generative LLM token hallucination or answer factuality.
2. **Generative LLM Selection Pending (Review 1 Feedback):** The Review 1 feedback requested comparing LLM backends. While response time and retrieval comparisons were completed, evaluating generative backends (e.g., Llama-3 vs. Mistral vs. Gemini) remains future work.
3. **Exploratory Nature of Held-Out Evaluation:** Ground-truth labels reflect pre-retrieval intent. When live adaptive routing evaluates post-retrieval evidence, divergence from static labels is expected; shadowed live testing is required for production validation.
4. **Finite Sample Bounds:** Zero observed breaches on 30 High/Critical cases reflects deterministic invariant enforcement within the sample, not a mathematical guarantee of open-world zero risk.
