# Duplicate and Near-Duplicate Analysis Report

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation  
**Date**: September 28, 2026  

---

## 1. Overview of Duplicate Analysis

This audit evaluated three levels of duplication:
1. **Exact Row / Case_ID Duplication**
2. **Exact Query String Duplication**
3. **Normalized Near-Duplicate Duplication** (case-folding, punctuation stripping, whitespace normalization).

---

## 2. Split-by-Split Duplication Metrics

| Split / Sheet | Total Rows | Unique Case_IDs | Duplicate Rows | Exact Duplicate Queries | Normalized Duplicate Queries | Explanation |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `Core_Master` | 3,339 | 3,339 | 0 | 0 | 0 | Natural unaugmented master cases |
| `Train_Balanced` | 5,640 | 5,640 | 0 | 809 | 809 | Duplicates caused strictly by `Augmented == 'Yes'` replication |
| `Validation` | 501 | 501 | 0 | 0 | 0 | Pristine held-out validation set |
| `Test` | 501 | 501 | 0 | 0 | 0 | Pristine held-out final test set |
| `Critical_Safety` | 180 | 180 | 0 | 0 | 0 | Dedicated critical safety benchmark |
| `Risk_Balanced` | 2,400 | 2,400 | 0 | 297 | 297 | Training balance replication |
| `Complexity_Balanced` | 2,100 | 2,100 | 0 | 0 | 0 | Non-replicated balanced cases |
| `Sentiment_Balanced` | 1,500 | 1,500 | 0 | 389 | 389 | Training balance replication |
| `Domain_Balanced` | 3,600 | 3,600 | 0 | 1,120 | 1,120 | Training balance replication |

---

## 3. Cross-Split Overlap & Contamination Audit

### 3.1 Raw Split Overlap & Contamination Audit

- **Raw Train_Balanced vs. Test**:
  - Case_ID Overlap: **0**
  - Exact Query Overlap: **0**
  - Standard Normalized Overlap: **0**
  - Super-Normalized / Space-Collapsed Overlap: **2** (Missing-space template typos identified: `knowhow`, `accountfor`)
- **Validation vs. Test**:
  - Case_ID Overlap: **0**
  - Exact Query Overlap: **0**
  - Normalized Query Overlap: **0**
  - Super-Normalized Overlap: **0**
- **Raw Train_Balanced vs. Validation**:
  - Case_ID Overlap: **0**
  - Exact Query Overlap: **0**
  - Normalized Query Overlap: **2** (Near-Duplicates identified)
  - Super-Normalized Overlap: **2**

### 3.2 Specific Contaminations Discovered in Raw V4

1. **Train vs. Validation Punctuation/Casing Duplicates**:
   - Query: `"I don't know what I need to do to speak to customer service"` (`CASE002854` in Train vs `CASE000761` in Validation).
   - Query: `"i dont know what i have to do to delete my platinum account"` (`CASE002853` in Train vs `CASE000963` in Validation).
2. **Train vs. Test Missing-Space Template Contaminations**:
   - Query: `"i dont know how to cancel purchase {Order Number}"` (`CASE002313` in Train vs `CASE002283` in Test: `"I don't knowhow to cancel purchase {Order Number}"`).
   - Query: `"create {Account Category} account for wife"` (`CASE002396` in Train vs `CASE001713` in Test: `"create {Account Category} accountfor wife"`).
   - Child augmentation in Train: `AUG002464` derived from `CASE002313`.

### 3.3 Strict Quarantine Enforcement & Final Clean Status

All 5 contaminated records (`CASE002853`, `CASE002854`, `CASE002313`, `CASE002396`, `AUG002464`) were strictly quarantined from training splits across all 7 operational tasks (`intent`, `sentiment`, `risk`, `complexity`, `escalation`, `autonomy`, `domain`).
Held-out Validation (501) and Test (501) remain 100% untouched.

| Split Comparison | Exact Query Overlap | Normalized Query Overlap | Super-Normalized Overlap | Status |
|:---|:---:|:---:|:---:|:---:|
| **Train ↔ Test** | **0** | **0** | **0** | **PASS (100% Isolated)** |
| **Train ↔ Validation** | **0** | **0** | **0** | **PASS (100% Isolated)** |
| **Validation ↔ Test** | **0** | **0** | **0** | **PASS (100% Isolated)** |
| **Critical Safety ↔ Train** | **0** | **0** | **0** | **PASS (100% Isolated)** |
| **Critical Safety ↔ Test** | **0** | **0** | **0** | **PASS (100% Isolated)** |

### 3.4 Critical Safety Isolation Audit

- `Critical_Safety` (180 cases) has **0 overlap** with `Core_Master`, `Validation`, or `Test`.
- However, the raw `Risk_Balanced` sheet contained these 180 cases oversampled to 600.
- **Leakage Prevention Action**: `Critical_Safety` is isolated in `data/processed/safety/critical_safety.csv` and strictly excluded from training data.

---

## 4. Operational & Relational Duplicate Checks

- **Customers**: 1,000 unique `Customer_ID`s (0 duplicates).
- **Orders**: 1,500 unique `Order_ID`s (0 duplicates).
- **Transactions**: 1,803 unique `Transaction_ID`s (0 duplicates).
- **Support Tickets**: 1,000 unique `Ticket_ID`s (0 duplicates).
- **Students**: 500 unique `Student_ID`s (0 duplicates).
- **Business Records**: 500 unique `Business_Record_ID`s (0 duplicates).
- **Memory**: 1,500 unique `Memory_ID`s (0 duplicates).
