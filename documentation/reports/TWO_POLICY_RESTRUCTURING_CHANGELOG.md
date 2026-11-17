# Two-Policy Evaluation Restructuring: Change Log and Scientific Justification

**Document Title:** Change Log and Methodological Audit for Two-Policy Evaluation Restructuring  
**Project:** Autonomous Multi-Agent Customer Support Resolution System with Risk-Aware Adaptive Orchestration and Human-in-the-Loop Escalation  
**Date:** November 2026  
**Auditor / Author:** Final-Year B.Tech CSE (AI & ML) Research Evaluation Audit  
**Artifact Directory:** `FINAL_RESEARCH_DOCUMENTATION/`  

---

## 1. Executive Summary of Restructuring

In response to research review directives and academic evaluation guidelines, the research evaluation of the capstone project is formally restructured from a three-policy configuration (Policy A, Policy B, Policy C) to a rigorous **Two-Policy Head-to-Head Comparison**:

1. **Policy A — Fixed M6 Autonomy Baseline:** Direct model inference using the M6 autonomy classifier ($R = \arg\max P(y|x)$) without downstream knowledge retrieval, composite evidence confidence, or deterministic safety overrides.
2. **Policy B — Risk-Aware Adaptive Autonomy Policy:** The complete multi-agent orchestration architecture incorporating fast-path classification (M1–M6), tenant-isolated semantic retrieval (ChromaDB HNSW + BM25Okapi), observable composite evidence confidence scoring ($C$), and deterministic safety policy arbitration enforcing human-in-the-loop (HITL) escalation.

This change log documents every modification made across documentation, comparison workbooks, and visualization figures, accompanied by its formal experimental and methodological justification.

---

## 2. Table of Changes Across Deliverables

| Deliverable | Original Formulation | Restructured Formulation | Scientific & Experimental Justification |
| :--- | :--- | :--- | :--- |
| **Primary Research Scope** | Three-policy comparison (Policy A, Policy B, Policy C). | **Strict Two-Policy Comparison (Policy A vs. Policy B).** | Aligns directly with the primary research hypothesis: *Does risk-aware adaptive orchestration outperform the fixed baseline?* Eliminates cognitive overhead and narrative dilution from post-hoc policy variants. |
| **Policy C Treatment** | Presented as a separate third candidate policy in all comparison tables and charts. | **Removed as a separate competitive policy.** Discussed exclusively as an exploratory sensitivity analysis on validation threshold tuning in Chapter 23. | Policy C was a threshold-calibrated variant ($C \ge 0.65$ vs. $0.70$). Treating it as a distinct system obscured the primary contrast between Fixed and Adaptive orchestration. |
| **Excel Workbook (`Final_Results_Comparison.xlsx`)** | Contained 3-column comparisons across all tabs. | **Restructured into a clean 2-policy comparison (Policy A vs. Policy B) with delta columns, 95% CIs, and McNemar test statistics.** | Provides a definitive, readable side-by-side benchmark for evaluators, with exact differences and p-values clearly visible. |
| **Word Report (`Final_Major_Project_Research_Evaluation.docx`)** | 30 chapters with recurring tripartite comparisons. | **30 chapters restructured to focus on Policy A vs. Policy B.** Includes updated tables, updated embedded figures, and expanded trade-off discussions. | Focuses the academic narrative on the core scientific tension: Safety vs. Operational Efficiency vs. Static Label Accuracy. |
| **Figures & Graphs (`Final_Research_Results_Graphs/`)** | 3-bar charts and 3-panel confusion matrices (A, B, C). | **2-bar comparative charts and 2-panel confusion heatmaps (Policy A vs. Policy B).** | Clear, high-contrast visual comparison that highlights the dramatic reduction in unsafe answering and the trade-off in escalation volume. |
| **Safety Claims & Phrasing** | Claims of "guaranteed safety", "zero-tolerance guarantee", and "100% elimination of risk". | **Replaced with precise empirical phrasing:** "0.0% observed violations on the evaluated 30 High/Critical test cases under invariant `CRIT_01` within the evaluated dataset." | Academic rigor requires distinguishing deterministic rule enforcement on an empirical cohort from universal, open-world safety guarantees. |
| **Latency & SLA Phrasing** | Assertions of meeting an "unverified 500 ms production SLA". | **Replaced with verified empirical latency percentiles:** Mean: 112.77 ms, Median (P50): 87.23 ms, P95: 287.70 ms, with standard web interaction benchmarks (1.0–2.0 s). | No formal SLA contract exists for the academic prototype; reporting exact measured percentiles against standard HCI interaction thresholds is truthful and defensible. |
| **Grounded Response & Hallucination Scope** | Ambiguous statements regarding "hallucination elimination". | **Explicitly scoped to routing safety:** Clearly states that `generate_grounded_response=False` during the 501-case benchmark; the system evaluates routing decisions, not token-level LLM factuality. | Prevents reviewer challenge regarding missing LLM token evaluation. |
| **Dataset & Evaluation Characterization** | Described as "independent validation" of adaptive routing. | **Characterized as an "exploratory held-out benchmark" on 501 canonical cases.** | Ground-truth labels were generated pre-retrieval based on intent. Because live adaptive routing evaluates post-retrieval evidence, the benchmark measures policy divergence rather than independent clinical validation. |
| **STELLAR Base Paper Comparison** | General references to STELLAR. | **Rigorous architectural delta articulated:** STELLAR relies on post-generation LLM reflection; our architecture implements pre-generation triage (M1–M6) and deterministic invariants prior to generation. | Establishes genuine research novelty while acknowledging foundational debt to the base paper. |

---

## 3. Detailed Audit of Policy B Implementation & Trade-Offs

### 3.1 Audited Strengths of Policy B
1. **Empirical Safety Interception:** On the 341-case unsafe cohort, Policy B slashes unsafe autonomous answers from **79.77% (Policy A, 272 cases)** down to **6.16% (21 cases)**, representing an **empirical relative risk reduction of 92.28%** ($p < 0.001$, McNemar $\chi^2 = 249.004$).
2. **Deterministic Enforcement on Critical Queries:** For all 30 High/Critical operational risk inquiries (account compromise, legal threats, fraud), Policy B achieves **0.0% autonomous answering (0/30)** by triggering invariant `CRIT_01`, compared to Policy A which erroneously answered 2 cases (6.67%).
3. **Defense Against Knowledge Gaps:** For all 73 inquiries where the knowledge base contains zero supporting documentation, Policy B achieves **0.0% autonomous answering (0/73)** via invariant `EVID_INSUF_04`, whereas Policy A attempted autonomous answers on **100.0% (73/73)**.
4. **Human Escalation Capture:** Policy B successfully captures **89.80% (44/49)** of true ground-truth human escalations, compared to Policy A which captured only **4.08% (2/49)**, missing 95.9% of escalations.

### 3.2 Audited Trade-Offs & Failure Modes of Policy B
1. **Severe Degradation of Static Routing Accuracy:** Against pre-retrieval ground truth, Policy B achieves only **25.55% accuracy (128/501)** compared to Policy A's **88.62% (444/501)**. This occurs because Policy B diverts 306 queries labeled `ANSWER` and 59 queries labeled `ORCHESTRATE` to human escalation due to strict evidence thresholds ($C < 0.70$).
2. **Elevated Human Supervisory Workload:** Policy B routes **409 out of 501 inquiries (81.6%)** to human escalation, creating a substantial human review burden.
3. **Unnecessary Escalations on Safe Inquiries:** On the strictly safe cohort of low-risk, answerable FAQ queries ($N = 94$), Policy B unnecessarily escalates **48 cases (51.06%)** because colloquial user phrasing resulted in composite evidence confidence falling slightly below the rigid $0.70$ threshold.
4. **Operational Latency Overhead:** Policy B introduces a statistically significant latency overhead of $+57.60$ ms on average ($t = -24.89, p < 0.001$), increasing Mean latency from 55.17 ms to 112.77 ms and P95 latency from 163.19 ms to 287.70 ms.

---

## 4. Preservation of Experimental Integrity

To comply with the highest standards of scientific ethics:
1. **Original Data Unaltered:** `data/processed/autonomy/test.csv`, `val.csv`, and `train.csv` remain completely untouched.
2. **Benchmark Logs Preserved:** The original benchmark directory `outputs/research_evaluation/m6_experiment_20261031_063000/` and its CSV logs (`m6_fixed_vs_adaptive_metrics.csv`, `m6_case_level_comparison.csv`, `m6_latency_comparison.csv`) remain intact.
3. **No Synthetic Numbers:** Every number cited in the restructured deliverables is drawn directly from the verified execution logs.

---

## 5. Summary of Deliverable Actions

1. **Change Log (This Document):** Created and archived in `FINAL_RESEARCH_DOCUMENTATION/TWO_POLICY_RESTRUCTURING_CHANGELOG.md`.
2. **Regenerate Visualization Figures:** Update `generate_figures.py` to produce clean two-policy figures (Policy A vs. Policy B) for architecture, safety violations, accuracy/F1, latency percentiles, and confusion matrices.
3. **Regenerate Comparative Excel Workbook:** Update `generate_excel.py` to produce `Final_Results_Comparison.xlsx` focused strictly on Policy A vs. Policy B.
4. **Compile Restructured Academic Word Document:** Update `build_docx.py` to generate `Final_Major_Project_Research_Evaluation.docx` structured around the two-policy narrative with all academic disclosures.
5. **Update Verification and Gaps Report:** Refresh `Verification_and_Gaps_Report.md` to reflect the two-policy framework.
