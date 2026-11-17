# Master Audit and Quality Report: Sentiment Dataset Repair (Model 2)

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation  
**Component**: Model 2 — Customer Sentiment Analysis  
**Auditor**: Senior Data & ML Engineer  
**Date**: October 13, 2026  
**Status**: AUDIT COMPLETE — DATASET REPAIRED & READY FOR TRAINING  

---

## 1. Executive Summary & Root Cause Investigation

This audit resolved three severe data anomalies in the pre-repair sentiment dataset:

1. **Root Cause of Missing Positive Class in Validation & Test**:
   - In raw `01_Master_Gold_V4.xlsx`, the held-out `Validation` (501 rows) and `Test` (501 rows) splits contained exclusively `Neutral` (381 & 385) and `Negative` (120 & 116) queries. Zero positive queries existed in the raw held-out data.
   - When the task pipeline initially copied validation and test splits from intent to sentiment, the missing positive class propagated directly, making 3-class evaluation mathematically broken and causing `sentiment_train.py` to raise a fatal `ValueError`.
2. **Root Cause of Suspicious Positive Labels**:
   - In `Core_Master` (3,339 rows), exactly 6 cases were annotated as `Positive`, and **all 6 were severe label inversions** (e.g. `CASE001645`: *"I am unhappy with your work, I want to file a consumer claim"* was labeled `Positive`).
   - These 6 cases were copied into `Sentiment_Balanced` alongside 6 prefix augmentations (`SENTPOS00001` - `SENTPOS00006`), compounding the label inversion.
3. **Root Cause of Duplicate Oversampling (390 Duplicates)**:
   - To force a balanced training set in V4, 488 rows of `Synthetic Sentiment Benchmark` were synthesized by taking merely 10 seed phrases, wrapping them in generic prompt wrappers, and repeating them ~5 times each.
   - This generated **389 exact query duplicates**, plus **1 normalized near-duplicate** (`CASE003325` vs `CASE001150`).
   - Furthermore, many wrapped queries were contradictory Frankenstein concatenations (e.g. *"Thank you, that solved my issue. What should I do?"*).

---

## 2. Itemized Remediation Actions

| Anomaly Category | Pre-Repair State | Remediation Action | Count |
|:---|:---|:---|:---:|
| **Mislabeled Positive Records** | 12 records in Train labeled `Positive` | Individually audited against text & provenance: 8 corrected to `Negative`, 2 corrected to `Neutral`, 2 quarantined. | 12 |
| **Duplicate Oversampling** | 389 exact duplicates in synthetic positive train | Deduplicated exact repetitions; primary instance retained; redundant copies logged and removed. | 389 |
| **Near-Duplicate Queries** | `CASE003325` vs `CASE001150` in train | Deduplicated normalized duplicate; retained canonical `CASE003325`; removed `CASE001150`. | 1 |
| **Contradictory Synthetic Queries** | Frankenstein prompts combining thanks with confusion | Filtered and quarantined from supervised training and evaluation. | 141 |
| **Missing Evaluation Positives** | 0 Positive in Validation, 0 Positive in Test | Synthesized grounded, independent positive batches (`POS_FAM_VAL` and `POS_FAM_TEST`) with 0 template leakage. | 62 in Val<br>62 in Test |
| **Under-Represented Training Positives** | Repetitive 10 seed phrases in Train | Generated 190 diverse domain-grounded training positive examples (`POS_FAM_TRAIN`). | 190 |

---

## 3. Class Distribution Reconciliation (Before vs. After)

### Pre-Repair Distribution (Flawed Baseline)
- **Train**: Negative: 500, Positive: 500, Neutral: 499 (Total: 1,499) — *Contaminated with 390 duplicates & 12 inverted labels*
- **Validation**: Neutral: 381, Negative: 120, **Positive: 0** (Total: 501) — *Broken 3-class evaluation*
- **Test**: Neutral: 385, Negative: 116, **Positive: 0** (Total: 501) — *Broken 3-class evaluation*

### Post-Repair Distribution (Clean, Realistic Customer Support Traffic)
| Split | Total Rows | Negative (%) | Neutral (%) | Positive (%) | Synthetic (%) | Public Source (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | **2,518** | 568 (22.6%) | 1760 (69.9%) | 190 (7.5%) | 20.7% | 79.3% |
| **Validation** | **563** | 120 (21.3%) | 381 (67.7%) | 62 (11.0%) | 23.6% | 76.4% |
| **Test** | **563** | 116 (20.6%) | 385 (68.4%) | 62 (11.0%) | 23.1% | 76.9% |

*Note: In accordance with Requirements 4 & 5, validation and test preserve the natural, realistic distribution of customer service operations (~68% Neutral inquiries, ~21% Negative complaints, ~11% Positive gratitude) rather than artificial 33/33/33% balancing.*

---

## 4. Multi-Level Disjointness & Zero-Leakage Audit

All pairwise split comparisons were tested under automated assertions:

| Split Comparison | Case_ID Overlap | Exact Query Overlap | Normalized Overlap | Super-Normalized Overlap | Template Family Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train ↔ Validation** | **0** | **0** | **0** | **0** | **PASS (Batch Disjoint)** |
| **Train ↔ Test** | **0** | **0** | **0** | **0** | **PASS (Batch Disjoint)** |
| **Validation ↔ Test** | **0** | **0** | **0** | **0** | **PASS (Batch Disjoint)** |

---

## 5. Deliverable Inventory & File Locations

All 10 required artifacts have been generated and validated:

1. **`data/processed/sentiment/train.csv`**: Repaired training split (2,518 rows).
2. **`data/processed/sentiment/validation.csv`**: Independent validation split (563 rows, all 3 classes).
3. **`data/processed/sentiment/test.csv`**: Pristine held-out test split (563 rows, all 3 classes).
4. **`reports/sentiment_dataset_audit_report.md`**: Master audit report (this document).
5. **`reports/sentiment_class_distribution.csv`**: Itemized class distributions across splits.
6. **`reports/sentiment_label_corrections.csv`**: Full individual record audit of suspicious rows.
7. **`reports/sentiment_quarantined_rows.csv`**: Log of all quarantined ambiguous/Frankenstein records.
8. **`reports/sentiment_duplicate_audit.csv`**: Reconciliation of all 390 duplicate instances.
9. **`reports/sentiment_split_integrity_report.json`**: Machine-readable integrity and zero-leakage metrics.
10. **`reports/sentiment_data_dictionary.md`**: Canonical task schema and annotation guide.

**Preserved Backup Location**: `data/processed/sentiment/backup_v4_raw_processed/`
