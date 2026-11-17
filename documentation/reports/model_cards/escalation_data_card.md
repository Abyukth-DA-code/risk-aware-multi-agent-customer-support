# Human Escalation Prediction Dataset Data Card

**Task**: Human-in-the-Loop Escalation Requirement Prediction  
**Classes**: `Yes`, `No` (Binary Classification)  
**Date**: September 28, 2026  

---

## 1. Dataset Summary & Sizes

- **Training Set (`escalation/train.csv`)**: 2,335 instances (Pure unaugmented base training set; 2 near-duplicates quarantined).
  - Class Distribution: `No`: 2,122 (90.9%), `Yes`: 213 (9.1%).
- **Validation Set (`escalation/validation.csv`)**: 501 instances (`No`: 460, `Yes`: 41).
- **Test Set (`escalation/test.csv`)**: 501 instances (`No`: 452, `Yes`: 49).
- **Critical Safety Benchmark (`safety/critical_safety.csv`)**: 180 instances (`Yes`: 180, 100%).

---

## 2. Input Features & Leakage Controls

- **Allowed Pre-Decision Observable Signals**:
  - `Query` (Customer query text)
  - `Domain` (Business domain)
  - `Risk` (Predicted or observable operational risk)
  - `Complexity` (Predicted execution complexity)
  - `Evidence_Quality` (`Strong`, `Insufficient`, `Conflicting`)
  - `Verification_Required` (`Yes`, `No`)
  - `Sensitive_Action` (`Yes`, `No`)
  - `Financial_Impact` (`Yes`, `No`)
  - `Sentiment` (`Negative`, `Neutral`, `Positive`)
- **Strictly Forbidden Leakage Features**:
  - `Escalation_Reason` (prohibited: explicitly states escalation rationale)
  - `Autonomy_Decision` (prohibited: in V4, `ESCALATE` is deterministically 1:1 with `Escalation_Required == 'Yes'`)
  - All post-decision feedback (`Human_Decision`, `Correction`, `Escalation_Appropriate`, `Reviewer_Notes`).

---

## 3. Evaluation Protocol

- Evaluated using **Escalation Precision**, **Escalation Recall**, **F1-Score**, and **Cost of False Negatives** (failing to escalate a high-risk or conflicting case).
- Critical Safety set is evaluated with a strict 100% recall requirement (zero high-risk disputes resolved without human oversight).
