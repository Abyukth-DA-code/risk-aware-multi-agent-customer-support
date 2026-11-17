# Comprehensive Audit & Methodology Report: Sentiment Dataset v2

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation
**Research Theme**: Risk-Aware Adaptive Multi-Agent Orchestration
**Component**: Model 2 — Customer Sentiment Analysis
**Dataset Version**: v2_refined
**Audit Date**: October 13, 2026
**Auditor**: Antigravity Dataset Governance & ML Engineering Unit

---

## 1. Executive Summary

This report documents the systematic audit, controlled data refinement, and validation of **Sentiment Dataset v2**. Following baseline training on v1 (which achieved 99.82% test accuracy using Linear SVM), deep error inspection and academic peer-review identified two structural opportunities for dataset strengthening:
1. **Label Taxonomical Noise in Synthetic Non-Ecommerce Data**: 20 records representing calm, procedural administrative queries (such as asking for ID card replacement or adding a team member) were tagged as `Negative` due to conflating adverse operational situations with negative affective emotion.
2. **Evaluation Realism on Positive Sentiment**: All positive validation and test queries in v1 were generated using repetitive grammatical templates featuring explicit gratitude tokens, potentially creating an optimistic evaluation ceiling.

Dataset v2 systematically resolves both challenges while strictly preserving Baseline v1 for reproducible comparative analysis.

---

## 2. Before-and-After Class Distributions

| Split | Metric | Baseline v1 | Refined v2 | Absolute Change | Percentage Distribution (v2) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Train** | **Total Rows** | **2,518** | **2,504** | -14 | 100.0% |
| | Negative | 568 | 554 | -14 | 22.12% |
| | Neutral | 1,760 | 1,760 | 0 | 70.29% |
| | Positive | 190 | 190 | 0 | 7.59% |
| **Validation** | **Total Rows** | **563** | **570** | +7 | 100.0% |
| | Negative | 120 | 117 | -3 | 20.53% |
| | Neutral | 381 | 391 | +10 | 68.60% |
| | Positive | 62 | 62 | 0 | 10.88% |
| **Test** | **Total Rows** | **563** | **570** | +7 | 100.0% |
| | Negative | 116 | 113 | -3 | 19.82% |
| | Neutral | 385 | 395 | +10 | 69.30% |
| | Positive | 62 | 62 | 0 | 10.88% |
| **Overall** | **Total Cases** | **3,644** | **3,644** | **0** | **100.0%** |

---

## 3. Detailed Audit of the 20 Suspected Label Errors

Each suspected case was audited against Section 2 of [`reports/sentiment_data_dictionary.md`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/reports/sentiment_data_dictionary.md):
- **Neutral Definition**: *"Factual inquiries, informational requests, operational queries, routine task execution without clear affective valence."*
- **Negative Definition**: *"Explicit customer dissatisfaction, frustration, complaints, product/service failure, or annoyance."*
- **Negative Exclusion**: *"Factual neutral procedural questions... are NOT negative unless negative emotion is stated."*

### Template Group Determinations:
1. **Lost Student ID (5 cases)**: Corrected from `Negative` to `Neutral`. Losing an ID card is an operational mishap, but asking *"What should I do?"* is a factual request for replacement instructions without anger or grievance.
2. **ID Issuance Status (5 cases)**: Corrected from `Negative` to `Neutral`. Asking *"When will my student ID be issued?"* is an ordinary status inquiry.
3. **Add Employee (5 cases)**: Corrected from `Negative` to `Neutral`. Inquiring how to add a staff member to a business profile is an administrative feature inquiry.
4. **Re-order Products (5 cases)**: Corrected from `Negative` to `Neutral`. Asking how to purchase identical items indicates commercial affinity, possessing zero negative valence.

All 20 corrections were logged with full justification in [`reports/sentiment_v2_label_corrections.csv`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/reports/sentiment_v2_label_corrections.csv).

---

## 4. Elimination of Cross-Split Template Family Leakage

In Baseline v1, queries within these four template families were partitioned across Train, Validation, and Test with different affix variations. To guarantee **zero leakage** and create an authentic test of **out-of-template generalization**:
- **Train Split**: All 14 training queries from these families were moved into evaluation splits. Train contains **0 instances** of these families.
- **Validation Split**: Assigned 100% of the *ID issuance status* and *Re-order products* families (10 cases).
- **Test Split**: Assigned 100% of the *Lost student ID* and *Add employee* families (10 cases).

When Model 2 is evaluated on the test set, it must correctly predict `Neutral` on queries like *"I lost the student ID card. What should I do?"* without having seen a single variant during training.

---

## 5. Curated Positive Evaluation Benchmark

The 62 positive queries in Validation and 62 positive queries in Test were replaced with high-quality, human-curated customer-support expressions representing five authentic communication modalities:
1. **Brief Gratitude** (e.g., *"Thanks, got the link now."*, *"Much appreciated, have a great day!"*)
2. **Indirect Relief & Satisfaction** (e.g., *"That's a huge relief, thought my transcript was lost forever."*, *"Phew, the payment finally went through properly."*)
3. **Confirmation of Issue Resolution** (e.g., *"Followed the steps and the portal let me enroll successfully."*, *"The error code disappeared after clearing cache."*)
4. **Praise for Support Quality** (e.g., *"Customer service here is top notch."*, *"The agent walked me through the setup very patiently."*)
5. **Colloquial & Conversational Positive Feedback** (e.g., *"You're an absolute lifesaver, thanks a ton!"*, *"Worked like a charm, cheers!"*, *"Spot on, exactly what I needed!"*)

All curated benchmark queries are explicitly tracked with `Data_Origin = 'CURATED_BENCHMARK'`.

---

## 6. Disjointness & Verification Assertions

Automated checks confirmed:
- `Case_ID` overlap across all splits: **0**
- Exact `Query` overlap across all splits: **0**
- Normalized `Query` overlap across all splits: **0**
- Super-normalized `Query` overlap across all splits: **0**
- Template-family leakage across splits: **0**

Dataset v2 is fully validated, scientifically defensible, and frozen for experimental training.
