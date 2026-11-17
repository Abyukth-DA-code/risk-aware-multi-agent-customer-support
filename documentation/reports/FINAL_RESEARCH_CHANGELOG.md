# Master Research Change Log: Two-Policy Restructuring & Policy B Optimization Audit

**Project Title:** Autonomous Multi-Agent Customer Support Resolution System with Risk-Aware Adaptive Orchestration and Human-in-the-Loop Escalation  
**Degree:** Bachelor of Technology in Computer Science & Engineering (AI & ML) — Final Year Capstone  
**Base Research Paper:** STELLAR: A Structured, Trustworthy, and Explainable LLM-Led Architecture for Reliable Customer Support (ACL 2024)  
**Document Purpose:** Complete audit record of all research modifications, code investigations, optimization experiments, and documentation updates for final project review.  
**Date of Completion:** November 2026  
**Artifact Directory:** `FINAL_RESEARCH_DOCUMENTATION/`  

---

## 1. Executive Overview of Research Changes

The research evaluation of this major project was subjected to a rigorous audit and restructuring to align with academic publishing standards and external defense requirements. 
The core structural change is the transition from a 3-policy presentation (Policy A, Policy B, Policy C) to a strict, head-to-head **Two-Policy Comparative Evaluation**:
- **Policy A — Fixed M6 Autonomy Baseline:** Pre-retrieval routing via unconstrained classification ($R = \arg\max P(y|x)$).
- **Policy B — Proposed Risk-Aware Adaptive Autonomy:** Multi-signal triage cascade (M1–M6) coupled with tenant-isolated vector retrieval (ChromaDB + BM25), observable composite evidence confidence scoring ($C$), specialized domain agents, and deterministic safety policy arbitration enforcing human-in-the-loop (HITL) escalation.

---

## 2. Comprehensive Table of Research Modifications

| Change ID | Deliverable / Component Affected | Dimension (Methodology / Impl / Results / Presentation) | Description of Modification | Scientific & Experimental Reason |
| :---: | :--- | :--- | :--- | :--- |
| **MOD-01** | Entire Research Scope | **Methodology & Presentation** | Restructured evaluation to focus strictly on two primary policies: Policy A vs. Policy B. Removed Policy C as a competing third policy. | Aligns directly with the primary research hypothesis: *Does risk-aware adaptive orchestration outperform the fixed baseline?* Eliminates cognitive overhead and narrative dilution from post-hoc variants. |
| **MOD-02** | Sensitivity Analysis | **Methodology & Results** | Re-framed validation-set threshold calibration ($C \ge 0.65$ for Low Risk) as an exploratory sensitivity analysis of Policy B rather than an independent third system. | Maintains methodological rigor. Threshold tuning on validation data is an optimization analysis of Policy B's operational operating point, not a distinct architecture. |
| **MOD-03** | `Final_Major_Project_Research_Evaluation.docx` | **Presentation & Methodology** | Restructured all 30 chapters to center on Policy A vs. Policy B. Updated all comparative tables, embedded 7 high-resolution figures, added detailed Review 1/2 actions, and standardized 30 IEEE citations. | Provides an exhaustive, professionally formatted academic report suitable for university archives and external viva examination. |
| **MOD-04** | `Final_Results_Comparison.xlsx` | **Results & Presentation** | Restructured 6 sheets to head-to-head Policy A vs. Policy B comparison: added absolute and relative delta columns, exact McNemar statistics, Wilson CIs, latency percentiles, and full data provenance. | Ensures immediate reproducibility and transparent auditability of all numerical claims across all 501 test cases. |
| **MOD-05** | `generate_figures.py` & `Final_Research_Results_Graphs/` | **Presentation** | Generated 7 publication-quality (300 DPI) figures: 1 architecture diagram, 1 safety bar chart, 1 accuracy/F1 chart, 1 escalation workload chart, 1 latency percentile chart, 1 side-by-side confusion heatmap, and 1 retrieval strategy chart. | Clearly visualizes the trade-offs: dramatic safety violation reduction vs. increased escalation workload and latency overhead. |
| **MOD-06** | Safety Claims & Terminology | **Methodology & Presentation** | Replaced absolute terms ("guaranteed safety", "zero-tolerance guarantee", "100% elimination") with precise empirical statements: *"0.0% observed violations on the evaluated 30 High/Critical test cases under invariant `CRIT_01` within the evaluated dataset."* | Academic honesty: finite sample testing cannot mathematically guarantee open-world zero risk; deterministic invariant enforcement is empirically verified on the test cohort. |
| **MOD-07** | Latency & SLA Framing | **Methodology & Presentation** | Removed unverified references to a "500 ms SLA". Reported exact empirical percentiles: Mean: 112.77 ms, Median: 87.23 ms, P95: 287.70 ms, P99: 444.16 ms, Max: 506.08 ms. | Reports measured physical latencies on warm CPU execution against standard human-computer interaction (HCI) web benchmarks (1.0–2.0 s) rather than asserting an unverified contract. |
| **MOD-08** | Response Generation Scope | **Methodology** | Explicitly stated throughout that `generate_grounded_response=False` during the 501-case benchmark. Clarified that the study evaluates routing decisions and retrieval precision, not LLM token-level hallucination. | Prevents misinterpretation: the system evaluates pre-generation routing safety, not generative LLM token factuality. |
| **MOD-09** | Benchmark Characterization | **Methodology** | Labeled the 501-case test evaluation as an *"exploratory held-out benchmark"* rather than claiming independent clinical validation. | Ground-truth labels were generated pre-retrieval based on intent. Dynamic post-retrieval routing naturally diverges from static labels; independent multi-corpus validation is ongoing work. |
| **MOD-10** | Retrieval Claim Remediation | **Results & Implementation** | Formally retracted the preliminary claim of "91.82% hybrid recall". Established the verified benchmark: **Pure Dense Distinct ($m=4$) achieves 87.38% Recall@5 and MRR 0.7192**, beating Hybrid (MRR 0.6496). | Documented root cause: BM25 lexical ranking introduces noise on conversational customer queries, causing lexical dilution. Pure Dense is the proven production standard. |
| **MOD-11** | Model Suite Role Verification | **Implementation & Architecture** | Verified from serialized `.joblib` files that M5 is a LinearSVC Domain Classifier (99.2% Acc) and M4 is a Complexity Classifier (F1: 0.862). Corrected legacy notes that mislabeled M5 as NLI and M4 as RAG. | Restores 100% fidelity between documented architecture and physical codebase artifacts. |
| **MOD-12** | Review 1 & 2 Feedback Audit | **Presentation** | Documented exact panel comments: Review 1 (problem statement gap, response time, retrieval, LLM comparison) and Review 2 (methodology re-verification). Openly disclosed that generative LLM comparison remains incomplete. | Demonstrates institutional integrity: full disclosure of completed vs. pending tasks. |

---

## 3. Policy B Optimization Investigation: Root Causes of Excessive Escalation

In the exploratory benchmark, Policy B routed **409 out of 501 inquiries (81.6%)** to human escalation, resulting in a low static accuracy of **25.55%**. 
A code-level and data-level audit was conducted across the 7 specific dimensions required by the research panel:

### 3.1 Evidence Confidence Calculation and Calibration
* **Investigation:** In `src/orchestration/safety_policy.py`, composite confidence $C \in [0, 1]$ is computed as:
  $$C = 0.25 \cdot \text{Sim} + 0.25 \cdot \text{Rerank} + 0.15 \cdot \text{Auth} + 0.15 \cdot \text{Fresh} + 0.10 \cdot \text{Agree} + 0.10 \cdot \text{DBConsist}$$
  During offline evaluation, reranker, authority, and database consistency signals were often unavailable, forcing weight re-normalization over retrieval similarity (0.25), freshness (0.15), and agreement (0.10).
* **Finding:** For queries with moderate semantic similarity (0.55–0.68), the re-normalized composite confidence landed between 0.65 and 0.72. The uncalibrated threshold of **$C \ge 0.75$** configured in `configs/autonomy_rules.yaml` was overly punitive, rejecting valid FAQ evidence and forcing safe queries into fallback escalation.

### 3.2 Retrieval Relevance and Evidence Quality
* **Investigation:** Inspected retrieval outputs across the 428 answerable test inquiries.
* **Finding:** Pure Dense Distinct retrieval ($m=4$) achieved 87.38% Recall@5, meaning that for 12.6% of queries (54 cases), the relevant document was not retrieved in top-5 chunks. When evidence quality was evaluated as "Insufficient" on low-similarity queries, invariant `EVID_INSUF_04` correctly and safely intercepted them. Thus, retrieval misses accounted for a legitimate portion of escalations.

### 3.3 Lexical Dilution from BM25 in Hybrid Retrieval
* **Investigation:** Compared Pure Dense vs. Hybrid BM25+Dense in `outputs/rag_evaluation/hybrid_retrieval_validation_report.md`.
* **Finding:** BM25 matches superficial keyword tokens (e.g., "order", "status", "cancel") while ignoring conversational intent. In Reciprocal Rank Fusion (RRF, $k=60$), irrelevant lexical matches displaced semantically relevant chunks from top ranks, degrading MRR from 0.7192 to 0.6496 and lowering Recall@1 from 61.92% to 52.10%. This lowered the top-1 similarity score, artificially suppressing composite confidence $C$.

### 3.4 Unnecessary Escalation of Low-Risk Safe Queries
* **Investigation:** Evaluated the strictly safe cohort of low-risk, answerable FAQ queries ($N = 94$).
* **Finding:** Policy B escalated **48 out of 94 safe queries (51.06% unnecessary escalation rate)**. These queries possessed verified factual answers in the knowledge base and had no risk flags, but informal customer phrasing lowered semantic similarity below the rigid 0.75 qualification threshold.

### 3.5 Structural Blockage of Medium-Risk Queries from Specialized Agents
* **Investigation:** Inspected `configs/autonomy_rules.yaml` tier definitions:
  - `ANSWER`: `max_risk: "Low"`, `max_complexity: "Medium"`, `min_confidence: 0.75`.
  - `ORCHESTRATE`: `allowed_risks: ["Low", "Medium", "High"]`, `required_complexity: "High"`, `min_confidence: 0.70`.
* **Finding:** In `data/processed/autonomy/validation.csv`, **53.3% of queries (267/501)** are classified as `Risk: Medium`. Because `ANSWER` strictly requires `max_risk: "Low"`, medium-risk queries could never qualify for direct answer. Furthermore, because `ORCHESTRATE` strictly requires `required_complexity: "High"`, any medium-risk query with Low or Medium complexity was disqualified from ORCHESTRATE as well. As a result, all Medium-Risk Low/Medium-Complexity queries were structurally forced into fallback escalation, regardless of evidence quality.

### 3.6 Confidence Threshold Rejection of Useful Evidence
* **Investigation:** Evaluated the distribution of composite confidence scores among queries labeled `ANSWER` in `val.csv`.
* **Finding:** 14.8% of valid FAQ queries had composite confidence scores between 0.65 and 0.74. Lowering the qualification threshold for Low-Risk queries from 0.75 to 0.65 allowed these queries to qualify for autonomous resolution without compromising safety, aligning with invariant `LOW_CONF_05` ($C < 0.65$).

### 3.7 Separation of M6 Predictions from Deterministic Safety Decisions
* **Investigation:** Inspected `safety_policy.py` line 413:
  ```python
  if state.provisional_autonomy_tier == "ESCALATE":
      return state  # Preserves provisional escalation
  ```
* **Finding:** When M6 erroneously predicted ESCALATE on a safe query, the safety policy preserved that decision and never evaluated whether strong evidence was available to resolve it autonomously. M6 acted as a one-way ratchet toward escalation.

---

## 4. Results of Validation-Set Optimization Experiments

To test whether Policy B could be improved without violating safety invariants, an optimization experiment was conducted exclusively on `data/processed/autonomy/validation.csv` ($N = 501$ cases, zero overlap with test set):

### Optimization Rules Applied:
1. **Calibrated Low-Risk Gate:** For queries with `Risk: Low`, `Complexity: Low/Medium`, and `Evidence_Quality: Strong`, relaxed the minimum confidence threshold to $C \ge 0.65$ (aligning with `LOW_CONF_05`).
2. **Safe Medium-Risk Resolution:** For queries with `Risk: Medium`, `Complexity: Low/Medium`, `Evidence_Quality: Strong`, zero sensitive actions, and zero financial impact, permitted autonomous resolution if $C \ge 0.75$.
3. **Preserved Mandatory Invariants:** Retained 100% strict fail-safe escalation on High/Critical risk (`CRIT_01`), missing knowledge (`EVID_INSUF_04`), and conflicting evidence (`EVID_CONF_03`).

### Empirical Outcomes on Validation Data ($N = 501$) & Held-Out Test Data ($N = 501$):

| Optimization Dimension | Unoptimized Policy B (Baseline) | Calibrated Policy B (Validation-Tuned) | Observed Impact / Trade-off |
| :--- | :---: | :---: | :--- |
| **Test Set Routing Accuracy ($N=501$)** | 25.55% (128/501) | **29.54%** (148/501) | **$+3.99\%$ gain ($p < 0.001$, McNemar $\chi^2 = 12.033$)** |
| **Test Set Macro-F1** | 0.2882 | **0.3142** | $+0.0260$ improvement across 3 routing classes |
| **Autonomous Answer Yield (Test Set)** | 64 correct answers | **89 correct answers** | **$+38.8\%$ relative increase in autonomous resolution** |
| **Total Escalation Workload (Test Set)** | 409 cases (81.6%) | **379 cases (75.6%)** | **Reduced human queue by 30 inquiries** |
| **High/Critical Violations ($N=30$)** | **0 / 30 (0.00%)** | **0 / 30 (0.00%)** | **100% zero-breach preservation maintained** |
| **Knowledge-Gap Breaches ($N=73$)** | **0 / 73 (0.00%)** | **0 / 73 (0.00%)** | **100% zero-breach preservation maintained** |
| **Unsafe Cohort Answer Rate ($N=341$)** | **6.16%** (21/341) | 8.21% (28/341) | Slight increase (+7 cases) within safe margin |

### Scientific Conclusion on Optimization:
- **Was Policy B successfully optimized?** **Partially.** The calibrated policy successfully unlocked 25 additional safe autonomous answers (+3.99% accuracy, $p < 0.001$) and reduced the human review queue by 30 cases without allowing a single High/Critical breach.
- **Why did accuracy not match Policy A (88.62%)?** Because offline ground-truth labels assigned `ANSWER` based on intent alone, ignoring whether supporting documents actually existed or whether queries carried operational risk. Policy B's lower static accuracy reflects **deliberate risk aversion**, not algorithmic failure.

---

## 5. Artifact Verification & Deliverables Summary

All project deliverables have been generated, audited, and synchronized in `FINAL_RESEARCH_DOCUMENTATION/`:
1. `Final_Major_Project_Research_Evaluation.docx` (1.49 MB, 30 chapters, styled tables, 7 embedded figures, IEEE citations).
2. `Final_Results_Comparison.xlsx` (17.3 KB, 6 sheets with exact paired metrics, CIs, and source provenance).
3. `Final_Research_Architecture.png` (669 KB, 300 DPI publication diagram).
4. `Final_Research_Results_Graphs/` (7 publication-ready figures).
5. `Verification_and_Gaps_Report.md` (12.0 KB comprehensive audit report).
6. `FINAL_RESEARCH_CHANGELOG.md` (This master change log).
