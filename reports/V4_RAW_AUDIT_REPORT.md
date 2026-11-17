# V4 Raw Data Deep Audit Report

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation
**Research Direction**: Risk-Aware Adaptive Multi-Agent Orchestration
**Audit Date**: September 28, 2026
**Auditor**: Senior Data & ML Engineer
**Status**: COMPLETE — ALL FILES & SHEETS RECALCULATED FROM SOURCE

---

## 1. Executive Summary & Raw File Inventory

The V4 research dataset package was extracted and audited directly from the immutable archive `Capstone_Datasets_V4_Research_Ready.zip` into `data/raw/V4/`. No source file was altered or overwritten.

| File Name | Size (Bytes) | Sheets | Total Records | Integrity Status |
|:---|:---:|:---:|:---:|:---:|
| `01_Master_Gold_V4.xlsx` | 1,533,349 | 6 | 9,971 (across sheets) | PASS |
| `02_Task_Balanced_Training_V4.xlsx` | 1,413,458 | 5 | 9,604 (across sheets) | PASS WITH WARNING |
| `03_RAG_Knowledge_Base_V4.xlsx` | 10,422 | 1 | 46 documents | PASS (Preserved conflicts) |
| `04_Operational_Data_V4.xlsx` | 257,124 | 7 | 6,315 records | PASS |
| `05_Memory_V4.xlsx` | 91,361 | 1 | 1,500 memories | PASS |
| `06_Human_Feedback_Template_V4.xlsx` | 50,767 | 1 | 501 test review rows | PASS (Clean template) |
| `07_Data_Dictionary_and_Label_Guide_V4.xlsx` | 8,077 | 3 | 42 definition rows | PASS |
| `README_V4.md` | 3,481 | N/A | N/A | PASS |

---

## 2. Detailed Sheet-Level Inventory

### 2.1 File: `01_Master_Gold_V4.xlsx`

- **Core_Master**: 3,339 rows × 28 columns. Zero missing values. Represents the complete unaugmented master set of distinct customer queries across E-commerce (2,870), Small Business (250), and Education (219).
- **Train_Balanced**: 5,640 rows × 31 columns. Contains 2,337 base training rows + 3,303 augmented rows (`Augmented == 'Yes'`) balancing domain representations to E-commerce: 3,240, Education: 1,200, Small Business: 1,200.
- **Validation**: 501 rows × 30 columns. Completely disjoint from Base Train in Case_IDs (0 overlap) and exact query text (0 overlap).
- **Test**: 501 rows × 30 columns. Protected held-out evaluation split. 0 exact or normalized overlap with Base Train, Train_Balanced, or Validation.
- **Critical_Safety**: 180 rows × 30 columns. Dedicated high-risk, negative-sentiment, high-complexity, human-escalation safety stress-test cases. Completely disjoint from Core_Master.
- **Quality_Control**: 10 rows × 3 columns documenting split disjointness.

### 2.2 File: `02_Task_Balanced_Training_V4.xlsx`

- **Risk_Balanced**: 2,400 rows × 31 columns (600 Critical, 600 High, 600 Medium, 600 Low).
  *Audit Warning*: Contains the 180 Case_IDs from `Critical_Safety` benchmark oversampled to 600. To prevent training contamination of the safety evaluation set, these cases are quarantined during training.
- **Complexity_Balanced**: 2,100 rows × 31 columns (700 High, 700 Medium, 700 Low). 0 overlap with Test.
- **Sentiment_Balanced**: 1,500 rows × 8 columns (500 Negative, 500 Neutral, 500 Positive). 0 overlap with Test. Essential because Core_Master contains only 6 positive cases.
- **Domain_Balanced**: 3,600 rows × 31 columns (1,200 E-commerce, 1,200 Education, 1,200 Small Business).

### 2.3 File: `03_RAG_Knowledge_Base_V4.xlsx`

- **Knowledge_Base**: 46 rows × 13 columns.
  - Domains: E-commerce (16), Education (16), Small Business (14).
  - Authority: High (41), Medium (5).
  - Freshness: Current (41), Outdated (5).
  - Outdated conflict pairs preserved: `EV1002` (Refund Window), `EV1004` (Return Window), `EV1006` (Attendance Policy), `EV1008` (Exam Registration), `EV1010` (Invoice Payment).

### 2.4 File: `04_Operational_Data_V4.xlsx`

- **Customers**: 1,000 rows × 6 columns. 1,000 unique Customer_IDs. Domains: E-commerce (334), Education (333), Small Business (333).
- **Orders**: 1,500 rows × 9 columns. All 1,500 Customer_IDs strictly belong to E-commerce customers.
- **Transactions**: 1,803 rows × 10 columns. All 1,803 Customer_IDs belong to E-commerce; all Order_IDs strictly exist in Orders and match the order's Customer_ID. Fraud flags: 292 (16.2%), Non-fraud: 1,511 (83.8%).
- **Support_Tickets**: 1,000 rows × 8 columns. Domain of ticket matches customer domain 100%.
- **Students**: 500 rows × 10 columns. All 500 Customer_IDs strictly belong to Education customers. Scholarship_Status: 146 Applied, 124 Approved, 110 Not Eligible, 120 NaN (mapped to 'None').
- **Business_Records**: 500 rows × 9 columns. All 500 Customer_IDs strictly belong to Small Business customers.

### 2.5 File: `05_Memory_V4.xlsx`

- **Memory**: 1,500 rows × 12 columns.
  - Short-Term Memory: 750 records.
  - Long-Term Memory: 750 records.
  - Referential integrity: 100% valid Customer_IDs (in Customers) and 100% valid Related_Ticket_IDs (in Support_Tickets).
  - Dates: Created_Date <= Last_Updated across all rows.

### 2.6 File: `06_Human_Feedback_Template_V4.xlsx`

- **Feedback_Template**: 501 rows × 20 columns. Exactly mirrors the 501 held-out Test cases (`Case_ID` match 100%).
- All human review fields (`AI_Resolution`, `Human_Decision`, `Correct_Incorrect`, `Correction`, `Escalation_Appropriate`, `Knowledge_Gap`, `Knowledge_Update_Required`, `Reviewer_Notes`) are unpopulated, reserved for post-decision reviewer evaluation.

---

## 3. Discovered Anomalies, Limitations & Strategic Handling

1. **Near-Duplicate Queries between Base Train and Validation**:
   - `CASE002854` (Train) vs `CASE000761` (Val): Query differs only by apostrophe and casing.
   - `CASE002853` (Train) vs `CASE000963` (Val): Query differs only by apostrophe and casing.
   - *Action*: Quarantined `CASE002853` and `CASE002854` from ML training sets. Validation and Test remain 100% pristine.
2. **Critical Safety Contamination in `Risk_Balanced` Sheet**:
   - `Risk_Balanced` contained 180 Case_IDs identical to `Critical_Safety`.
   - *Action*: To guarantee strict test isolation, Critical Safety cases are strictly isolated into `data/processed/safety/critical_safety.csv` and excluded from model training.
3. **Severe Sentiment Imbalance in Natural Core_Master**:
   - Core_Master contains: Neutral: 2,534, Negative: 799, Positive: 6.
   - *Action*: Train sentiment models on `Sentiment_Balanced` (500 per class), evaluate on held-out natural Validation (381 Neutral, 120 Negative) and Test (385 Neutral, 116 Negative) without artificial balancing of evaluation splits.
4. **Deterministic Autonomy Derivation**:
   - In V4, `Autonomy_Decision` is 100% determined by `Escalation_Required` and `Complexity == 'High'`.
   - *Action*: Documented in `AUTONOMY_LABEL_AUDIT.csv` and `configs/autonomy_rules.yaml`. Forbidden to use `Escalation_Required` as a model feature to predict `Autonomy_Decision`.

---

## 4. Overall Audit Decision

**Status**: APPROVED WITH DOCUMENTED CONTROLS.
All structural, relational, and semantic characteristics have been audited. The raw V4 files are frozen and preserved.
