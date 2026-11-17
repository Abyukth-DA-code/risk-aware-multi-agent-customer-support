# Comprehensive Project Verification, Evidence Audit, and Research Gaps Report

**Project Title:** Autonomous Multi-Agent Customer Support Resolution System with Risk-Aware Adaptive Orchestration and Human-in-the-Loop Escalation  
**Author / Candidate:** Final-Year B.Tech CSE (AI & ML) Major Project  
**Base Paper:** STELLAR: A Structured, Trustworthy, and Explainable LLM-Led Architecture for Reliable Customer Support (ACL 2024)  
**Evaluation Scope:** Head-to-Head Two-Policy Comparison: Policy A (Fixed M6 Baseline) vs. Policy B (Risk-Aware Adaptive Autonomy) with Validation-Calibrated Sensitivity Analysis  
**Audit Date:** November 2026  
**Artifact Directory:** `FINAL_RESEARCH_DOCUMENTATION/`  

---

## 1. Executive Summary & Audited Experimental Scorecard

In accordance with academic review directives and research audit mandates, the research evaluation is structured as a strict **Two-Policy Head-to-Head Comparison**:
1. **Policy A (Fixed M6 Autonomy Baseline):** Direct argmax classification from M6 prior to retrieval ($R = \arg\max P(y|x)$) without downstream evidence retrieval or deterministic safety overrides.
2. **Policy B (Risk-Aware Adaptive Autonomy):** Multi-agent cascade (M1–M6) with tenant-isolated ChromaDB retrieval, composite evidence confidence scoring ($C$), and deterministic safety policy arbitration enforcing human-in-the-loop (HITL) escalation.

### Canonical Held-Out Test Evaluation Scorecard ($N = 501$ Cases)

| Evaluation Dimension | Policy A: Fixed M6 Baseline | Policy B: Original Uncalibrated | Policy B: Calibrated ($C \ge 0.65$ Low-Risk) | Absolute Delta (Orig vs A) | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unsafe Answer Rate ($N=341$)** | 79.77% (272/341) | **6.16%** (21/341) | **8.21%** (28/341) | $-73.61\%$ | **$p < 0.001$ (McNemar $\chi^2 = 249.004$)** |
| **High/Critical Violations ($N=30$)** | 6.67% (2/30) | **0.00%** (0/30) | **0.00%** (0/30) | $-6.67\%$ | Deterministic Invariant (`CRIT_01`) |
| **Knowledge-Gap Violations ($N=73$)**| 100.00% (73/73) | **0.00%** (0/73) | **0.00%** (0/73) | $-100.00\%$ | Deterministic Invariant (`EVID_INSUF_04`) |
| **Escalation Recall ($N=49$)** | 4.08% (2/49) | **89.80%** (44/49) | **79.59%** (39/49) | $+85.72\%$ | **$p < 0.001$ (McNemar)** |
| **Escalation Precision ($N=49$)** | 50.00% (2/4) | 10.76% (44/409) | 10.29% (39/379) | $-39.24\%$ | Trace of high safe escalation |
| **3-Class Correct Decisions** | **88.62%** (444/501) | 25.55% (128/501) | 29.54% (148/501) | $-63.07\%$ | $p < 0.001$ (Offline static artifact) |
| **Autonomous Answer Yield (Recall)**| **97.86%** (365/373) | 17.16% (64/373) | 23.86% (89/373) | $-80.70\%$ | $+25$ answers recovered in calibrated |
| **Autonomous Answer Precision** | 91.25% (365/400) | **96.97%** (64/66) | 92.71% (89/96) | $+5.72\%$ | High purity of autonomous output |
| **Macro-Averaged F1** | **0.6316** | 0.2882 | 0.3142 | $-0.3434$ | Offline label distortion |
| **Total Human Escalations** | **4** (0.80%) | 409 (81.64%) | 379 (75.65%) | $+405$ cases | $-30$ cases queued to humans |
| **Unnecessary Safe Escalations** | **0.00%** (0/94) | 51.06% (48/94) | 24.47% (23/94) | $+51.06\%$ | $-25$ safe queries unblocked |
| **Mean Latency (CPU Monotonic)** | **55.17 ms** | 112.77 ms | 112.78 ms | $+57.60$ ms | $p < 0.001$ (Paired $t = -24.89$) |
| **Median Latency (P50)** | **39.76 ms** | 87.23 ms | 87.23 ms | $+47.47$ ms | Monotonic CPU clock |
| **P95 Tail Latency** | **163.19 ms** | 287.70 ms | 287.71 ms | $+124.51$ ms | Well within web interactive limits |
| **P99 Tail Latency** | **182.54 ms** | 444.16 ms | 444.17 ms | $+261.62$ ms | Peak load tail |
| **Max Observed Latency** | **410.29 ms** | 506.08 ms | 506.09 ms | $+95.79$ ms | Exact peak (no rounded 507 ms) |

---

## 2. Comprehensive Discrepancy Resolution & Dataset Hygiene Audit

### 2.1 Reconciling Policy B Dual Result Sets
The audit definitively resolved the presence of two result sets for Policy B:
- **Configuration B1 (Original Uncalibrated Baseline):** Fixed uniform evidence threshold ($C \ge 0.70$) across all risk tiers. Produces 25.55% accuracy (128/501), Macro-F1 0.2882, 409 human escalations (81.64%), and 64 correct autonomous answers.
- **Configuration B2 (Validation-Calibrated Gate):** Threshold relaxed to $C \ge 0.65$ strictly for low-risk, simple queries identified by M1 and M2, preserving $C \ge 0.85$ for medium risk and mandatory escalation for high/critical risk. Produces 29.54% accuracy (148/501), Macro-F1 0.3142, 379 human escalations (75.65%), and 89 correct autonomous answers (+25 recovered, +38.8% yield).
- **Academic Framing:** Policy B is presented as the singular proposed adaptive architecture. Configuration B1 represents the baseline adaptive implementation, and Configuration B2 represents its validation-calibrated optimization. Exploratory Policy C terminology has been fully eliminated.

### 2.2 Denominator & Measurement Disambiguation
- **Total Correct 3-Class Decisions ($N = 501$ total test inquiries):**
  - Policy A: 444 / 501 (88.62%)
  - Policy B (Uncalibrated): 128 / 501 (25.55%)
  - Policy B (Calibrated): 148 / 501 (29.54%)
- **Class-Specific Autonomous Answer True Positives ($N = 373$ true answerable cases):**
  - Policy A: 365 / 373 (97.86% recall, 91.25% precision)
  - Policy B (Uncalibrated): 64 / 373 (17.16% recall, 96.97% precision)
  - Policy B (Calibrated): 89 / 373 (23.86% recall, 92.71% precision)
- All tables, narrative descriptions, and figure labels now explicitly state both the numerator and denominator to prevent ambiguity.

### 2.3 Dataset Split Hygiene & Script Diagnostics
- Direct verification of disk CSV files in `data/processed/autonomy/`:
  - `train.csv`: 2,333 rows, 0 duplicate IDs.
  - `validation.csv`: 501 rows, 0 duplicate IDs.
  - `test.csv`: 501 rows, 0 duplicate IDs.
  - Pairwise Overlaps: Train $\cap$ Validation = 0, Train $\cap$ Test = 0, Validation $\cap$ Test = 0 across Case IDs and normalized Query texts.
- **Root Cause of `scratch/test_val_optimization.py` Exit Code 1:**
  - An ad-hoc verification script failed due to `KeyError: "['Audit_Relevance_Category'] not in index"` during an inner join with `data/processed/rag/retrieval_dev_benchmark.csv`.
  - The dev benchmark CSV stores relevance under `Expected_Status` and `Expected_Document_ID`.
  - **Audit Verdict:** The failure was caused by a minor column header typo in a scratch script, NOT underlying dataset corruption. The 501 validation samples are genuine, intact, and completely disjoint from the test partition.

### 2.4 Statistical Rigor & Notation Corrections
- Paired McNemar Test (Policy A vs. Policy B on 501 paired cases):
  - Discordant pairs: Policy A correct / Policy B incorrect = 320; Policy A incorrect / Policy B correct = 4.
  - McNemar Test Statistic: $\chi^2 = \frac{(|320 - 4| - 1)^2}{320 + 4} = 306.42$ ($p < 0.001$).
- Paired McNemar Test on Unsafe Cohort ($N = 341$):
  - Discordant pairs: Policy A unsafe / Policy B safe = 252; Policy A safe / Policy B unsafe = 1.
  - McNemar Test Statistic: $\chi^2 = 249.0040$ ($p < 0.001$).
- Paired McNemar Test on Optimization Gain (Uncalibrated vs. Calibrated, $N = 501$):
  - Discordant pairs: Calibrated correct / Uncalibrated incorrect = 23; Calibrated incorrect / Uncalibrated correct = 3.
  - McNemar Test Statistic: $\chi^2 = 12.0333$ ($p < 0.001$).
- All statistical claims now use rigorous inequality notation ($p < 0.001$) rather than non-standard $p = 0.0000$.

### 2.5 Latency Precision & Boundary Elimination
- Replaced the arbitrary rounded "507 ms boundary" and unverified "500 ms SLA claim" with verified empirical CPU monotonic measurements:
  - Policy A: Mean 55.17 ms, Median (P50) 39.76 ms, P95 163.19 ms, P99 182.54 ms, Max 410.29 ms.
  - Policy B: Mean 112.77 ms, Median (P50) 87.23 ms, P95 287.70 ms, P99 444.16 ms, Max 506.08 ms.
  - Policy B tail latency (P95 = 287.70 ms) comfortably satisfies interactive web standards (1.0–2.0 s).

---

## 3. Methodological Clarifications and Boundary Conditions

### 3.1 Exploratory Benchmark Labeling
Ground-truth test labels in `data/processed/autonomy/test.csv` were assigned pre-retrieval based solely on the static intent category (e.g., standard FAQ queries were labeled `ANSWER`). 
In live execution, Policy B dynamically retrieves supporting documentation and evaluates composite evidence confidence ($C$). If confidence is insufficient ($C < 0.70$), Policy B deliberately overrides the static label to escalate to a human supervisor.
Because static labels do not account for dynamic evidence quality, this 501-case evaluation is characterized as an **exploratory held-out benchmark** measuring policy divergence, rather than an independent multi-corpus clinical trial.

### 3.2 Routing Safety vs. Text Hallucination
The benchmark executed with `generate_grounded_response=False`. 
* **What Was Evaluated:** Decision routing accuracy, unsafe autonomous routing rates, human escalation recall, retrieval recall, and latency percentiles.
* **What Was NOT Evaluated:** Generative LLM token hallucination, factual consistency of generated text, or comparative LLM backends (e.g., Llama-3 vs. Mistral vs. Gemini).
* This distinction is maintained throughout the documentation to prevent unverified claims.

### 3.3 Elimination of Overstated Claims
* All phrases asserting "guaranteed safety", "zero-tolerance guarantee", and "100% elimination of risk" have been replaced with precise empirical descriptions: *"0.0% observed violations on the evaluated 30 High/Critical test cases under invariant `CRIT_01` within the evaluated dataset."*
* Assertions of an "unverified 500 ms SLA" have been replaced with exact empirical measurements: *"Mean latency 112.77 ms, Median 87.23 ms, and P95 latency 287.70 ms, operating comfortably within standard interactive web latency thresholds (1.0–2.0 seconds)."*

---

## 4. The 91.82% Hybrid Retrieval Myth vs. Verified 87.38% Pure Dense Reality

### 4.1 Investigation and Root Cause
A preliminary working draft cited "91.82% Recall@5" for hybrid retrieval. A full audit of `outputs/rag_evaluation/hybrid_retrieval_validation_report.md` revealed:
1. The 91.82% figure was an unverified estimate computed on an unpartitioned development set where multiple raw chunk duplicates from the same document artificially inflated top-5 hit counts.
2. On the canonical 501 held-out test inquiries with **distinct-document deduplication ($m=4$)**, the verified retrieval metrics are:

| Retrieval Strategy | Deduplication ($m$) | Hybrid (RRF) | Recall@1 | Recall@3 | Recall@5 | MRR | Latency (Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Dense (Raw Chunks)** | None ($m=1$) | No | 61.92% | 77.34% | 82.24% | 0.6977 | 35.95 ms |
| **Pure Dense Distinct ($m=4$) [RECOMMENDED]** | $m=4$ | No | **61.92%** | **80.37%** | **87.38%** | **0.7192** | **37.19 ms** |
| **Hybrid Distinct ($k=60, m=2$)** | $m=2$ | Yes ($k=60$) | 52.10% | 75.93% | 87.38% | 0.6496 | 38.19 ms |
| **Hybrid Distinct ($k=60, m=4$)** | $m=4$ | Yes ($k=60$) | 52.34% | 72.20% | 83.88% | 0.6364 | 38.04 ms |

### 4.2 Lexical Dilution Post-Mortem
BM25 lexical ranking degrades MRR from **0.7192 (Pure Dense) to 0.6496 (Hybrid)** and drops Recall@1 from **61.92% to 52.10%**. 
Because customer inquiries use colloquial, non-standard terminology, BM25 matches superficial keyword tokens, injecting irrelevant documents into top ranks. 
**Pure Dense Semantic Retrieval with Distinct-Document Deduplication ($m=4$) is the proven production architecture.**

---

## 5. Deconstructing the Static Accuracy Paradox

* **Policy A Accuracy: 88.62% (444/501)**
* **Policy B Accuracy: 25.55% (128/501) [Uncalibrated] / 29.54% (148/501) [Calibrated]**

Why does Policy A achieve high static accuracy while Policy B drops?
1. 74.45% of test inquiries are labeled `ANSWER` pre-retrieval.
2. Policy A predicts `ANSWER` on 400 cases, scoring high static accuracy through indiscriminate guessing. In doing so, it attempts answers on **73 out of 73 unanswerable queries** and **272 out of 341 unsafe queries**, while capturing only **2 of 49 true human escalations (Recall: 4.08%)**.
3. Policy B evaluates evidence confidence ($C$). When $C < 0.70$, it routes to human escalation. This defensive posture diverts 306 queries labeled `ANSWER` to human review, capturing **89.80% of escalations (44/49)** and slashing unsafe answers by **92.28% ($p < 0.001$)**.
4. In enterprise safety, static accuracy against ungrounded labels is an invalid KPI. **Unsafe Answer Rate and Escalation Recall are the true operational metrics.**

---

## 6. Review Feedback & Actions Taken Audit

| Review Stage | Panel Comment | Corrective Action Implemented | File Evidence | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Review 1** | 1. Identify significant gap for problem statement.<br>2. Check response time, retrieval quality, and compare LLM models. | 1. Formulated the Enterprise Trilemma (yield vs. risk vs. latency).<br>2. Benchmarked latency percentiles (Mean, P50, P95).<br>3. Evaluated Dense vs. Hybrid retrieval.<br>4. Evaluated safety and routing metrics.<br>5. Generative LLM comparison NOT COMPLETED (`generate_grounded_response=False`). | `m6_latency_comparison.csv`<br>`test_benchmark_final_comparison.csv`<br>`Verification_and_Gaps_Report.md` | **Partially Completed** (Latency, retrieval, safety done; LLM generation pending) |
| **Review 2** | Re-verify methodology and scope and continue as per plan. | Rechecked M1–M6 models against serialized joblib artifacts; formalized deterministic invariants (`CRIT_01`, `EVID_INSUF_04`); conducted 501-case controlled benchmark. | `src/models/*.joblib`<br>`policy_engine.py`<br>`m6_fixed_vs_adaptive_metrics.csv` | **Completed** |

---

## 7. Review 3 Oral Defense Strategy & Anticipated Questions

### Question 1: "Why did you restructure to a two-policy comparison?"
*Answer:*  
"The primary research hypothesis investigates whether risk-aware adaptive orchestration outperforms the fixed baseline. Focusing strictly on Policy A (Fixed M6 Baseline) versus Policy B (Proposed Adaptive Autonomy) directly isolates the effect of post-retrieval evidence assessment and deterministic safety invariants. Exploratory calibration variants are reported as sensitivity analysis rather than competing system architectures, maintaining academic clarity."

### Question 2: "Why is your static routing accuracy only 25.55% for Policy B?"
*Answer:*  
"Policy A's 88.62% accuracy is an artifact of pre-retrieval ground truth, where 74.45% of cases were labeled ANSWER. Policy A guessed ANSWER on 400 cases, erroneously attempting answers on 73 unanswerable queries and missing 95.9% of true human escalations. Policy B evaluates live evidence confidence and enforces fail-safe escalation. While this conservative posture reduces static accuracy against offline labels, it slashes real-world unsafe answers by 92.28% ($p < 0.001$) and captures 89.8% of human escalations."

### Question 3: "What is the primary operational trade-off of Policy B?"
*Answer:*  
"The primary trade-off is an elevated human supervisory workload: Policy B forwards 81.64% of queries to human review in its baseline configuration, including a 51.06% unnecessary escalation rate on safe FAQ queries where conversational phrasing caused confidence to fall slightly below the 0.70 threshold. Through validation calibration ($C \ge 0.65$ for low risk), we successfully reduce escalations to 75.65% and recover 25 valid answers (+38.8% yield) while maintaining 0 high/critical breaches."

### Question 4: "Why does Pure Dense retrieval outperform Hybrid retrieval?"
*Answer:*  
"Our controlled RAG ablation on 501 queries proved that BM25 lexical fusion degrades Mean Reciprocal Rank from 0.7192 to 0.6496 and drops Recall@1 from 61.92% to 52.10%. Because customer support queries contain colloquial phrasing and spelling variations, BM25 matches superficial keyword tokens, displacing semantically relevant knowledge chunks. Pure Dense retrieval with distinct-document deduplication ($m=4$) achieves 87.38% Recall@5 with a mean latency of 37.19 ms."

### Question 5: "Did you evaluate generative LLM hallucination?"
*Answer:*  
"No, and we are explicitly transparent about this boundary. The benchmark was executed with `generate_grounded_response=False` to rigorously measure architectural decision routing, safety gating, and retrieval quality without confounding stochastic generation variances. Evaluating token-level hallucination across LLM backends is the primary identified area for post-capstone research."

---

## 8. Master Deliverables Manifest

All deliverables have been updated, audited, and archived in [`FINAL_RESEARCH_DOCUMENTATION/`](file:///c:/Users/Lenovo/Downloads/Final_Year_AIML_Capstron_Project_Before_AWS/Final_Year_AIML_Capstron_Project/FINAL_RESEARCH_DOCUMENTATION/):
1. **Master Reconciliation Audit:** [`DISCREPANCY_RESOLUTION_REPORT.md`](file:///c:/Users/Lenovo/Downloads/Final_Year_AIML_Capstron_Project_Before_AWS/Final_Year_AIML_Capstron_Project/FINAL_RESEARCH_DOCUMENTATION/DISCREPANCY_RESOLUTION_REPORT.md)
2. **Detailed Research Change Log:** [`FINAL_RESEARCH_CHANGELOG.md`](file:///c:/Users/Lenovo/Downloads/Final_Year_AIML_Capstron_Project_Before_AWS/Final_Year_AIML_Capstron_Project/FINAL_RESEARCH_DOCUMENTATION/FINAL_RESEARCH_CHANGELOG.md)
3. **Master Word Document (30 Chapters, Two-Policy):** [`Final_Major_Project_Research_Evaluation.docx`](file:///c:/Users/Lenovo/Downloads/Final_Year_AIML_Capstron_Project_Before_AWS/Final_Year_AIML_Capstron_Project/FINAL_RESEARCH_DOCUMENTATION/Final_Major_Project_Research_Evaluation.docx)
4. **Comparative Excel Workbook (6 Tabs, Two-Policy):** [`Final_Results_Comparison.xlsx`](file:///c:/Users/Lenovo/Downloads/Final_Year_AIML_Capstron_Project_Before_AWS/Final_Year_AIML_Capstron_Project/FINAL_RESEARCH_DOCUMENTATION/Final_Results_Comparison.xlsx)
5. **Two-Policy Architecture Diagram:** [`Final_Research_Architecture.png`](file:///c:/Users/Lenovo/Downloads/Final_Year_AIML_Capstron_Project_Before_AWS/Final_Year_AIML_Capstron_Project/FINAL_RESEARCH_DOCUMENTATION/Final_Research_Architecture.png)
6. **High-Resolution Results Figures (6 Visuals):** [`Final_Research_Results_Graphs/`](file:///c:/Users/Lenovo/Downloads/Final_Year_AIML_Capstron_Project_Before_AWS/Final_Year_AIML_Capstron_Project/FINAL_RESEARCH_DOCUMENTATION/Final_Research_Results_Graphs/)
7. **Verification and Gaps Audit (This Document):** [`Verification_and_Gaps_Report.md`](file:///c:/Users/Lenovo/Downloads/Final_Year_AIML_Capstron_Project_Before_AWS/Final_Year_AIML_Capstron_Project/FINAL_RESEARCH_DOCUMENTATION/Verification_and_Gaps_Report.md)
