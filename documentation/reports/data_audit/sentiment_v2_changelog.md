# Sentiment Dataset v2 Changelog: Controlled Data Refinement

**Project**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation  
**Component**: Model 2 — Customer Sentiment Analysis  
**Release**: Dataset v2 (Refined & Leakage-Free)  
**Date**: October 13, 2026  

---

## 1. Summary of Changes from Baseline v1 to Refined v2

| Item | Baseline v1 | Refined v2 | Rationale |
|:---|:---|:---|:---|
| **Storage Location** | `data/processed/sentiment/` | `data/processed/sentiment/v2_refined/` | Strict preservation of v1 baseline assets. |
| **Train Row Count** | 2,518 | 2,504 (-14 rows) | 14 cases moved to evaluation splits to eliminate cross-split template leakage. |
| **Validation Row Count** | 563 | 570 (+7 rows) | 7 cases added from Train (ID issuance & Re-order templates). |
| **Test Row Count** | 563 | 570 (+7 rows) | 7 cases added from Train (Lost ID & Add employee templates). |
| **Total Ecosystem Rows** | 3,644 | 3,644 (Zero net row change) | Strict row conservation; zero data fabricated or discarded. |
| **Non-Ecommerce Label Errors** | 20 records labeled `Negative` | 20 records corrected to `Neutral` | Conformance with Canonical Data Dictionary (calm procedural queries). |
| **Template Family Leakage** | 4 families scattered across Train & Eval | 0 cross-split leakage for audited families | Families isolated exclusively to single evaluation splits (out-of-template test). |
| **Positive Evaluation Source** | 100% Rigid Synthetic Formulas | 100% Human-Curated Diverse Benchmark | 5 emotion categories: brief gratitude, indirect relief, resolved confirmation, praise, and colloquial expressions. |
| **Positive Evaluation Origin Tag** | `SYNTHETIC` | `CURATED_BENCHMARK` | Accurate academic provenance governance. |

---

## 2. Forensic Log of the 20 Relabeled Records

All 20 records belonging to the 4 suspect template families were individually inspected and corrected from `Negative` to `Neutral`:

1. **Lost student ID (5 cases)**: `CASE000002`, `CASE000392`, `CASE000827`, `CASE000657`, `CASE000754`
   - *Previous Label*: `Negative` -> *New Label*: `Neutral`
   - *Allocation*: 100% allocated to **Test** (Train = 0).
   - *Rationale*: Routine assistance request following credential misplacement. No anger or dissatisfaction.
2. **ID issuance status (5 cases)**: `CASE000041`, `CASE000235`, `CASE000573`, `CASE001334`, `CASE000074`
   - *Previous Label*: `Negative` -> *New Label*: `Neutral`
   - *Allocation*: 100% allocated to **Validation** (Train = 0).
   - *Rationale*: Administrative timeline status request.
3. **Add employee (5 cases)**: `CASE000888`, `CASE000304`, `CASE000365`, `CASE000400`, `CASE000152`
   - *Previous Label*: `Negative` -> *New Label*: `Neutral`
   - *Allocation*: 100% allocated to **Test** (Train = 0).
   - *Rationale*: Standard business account configuration question.
4. **Re-order products (5 cases)**: `CASE000617`, `CASE000219`, `CASE000351`, `CASE000587`, `CASE001213`
   - *Previous Label*: `Negative` -> *New Label*: `Neutral`
   - *Allocation*: 100% allocated to **Validation** (Train = 0).
   - *Rationale*: Commercial repurchase inquiry showing positive buying intent.

---

## 3. Disjointness & Generalization Guarantees

1. **Zero Contamination**:
   - Exact query overlap: 0 across all pairs.
   - Normalized query overlap: 0 across all pairs.
   - Super-normalized query overlap: 0 across all pairs.
2. **True Out-of-Template Evaluation**:
   - The training split contains **zero** instances of *"I lost my student ID card"* or *"How can I add another employee"*.
   - When evaluated on the test set, the model's accuracy on these items measures genuine **out-of-template semantic comprehension**.
3. **Realistic Positive Diversity**:
   - Evaluation sets now test varied customer language including conversational praise (*"You're an absolute lifesaver"*, *"Worked like a charm"*), indirect relief (*"Oh perfect, the refund showed up"*, *"Phew, the payment went through"*), and succinct gratitude (*"Thanks, got the link"*, *"All sorted, cheers"*).
