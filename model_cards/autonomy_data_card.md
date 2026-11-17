# Adaptive Autonomy Decision Dataset Data Card

**Task**: Risk-Aware Adaptive Autonomy Tier Assignment
**Classes**: `ANSWER`, `ORCHESTRATE`, `ESCALATE` (3 classes)
**Date**: September 28, 2026

---

## 1. Dataset Summary & Sizes

- **Training Set (`autonomy/train.csv`)**: 2,335 instances (Pure base train; 2 near-duplicates quarantined).
  - Distribution: `ANSWER`: 1,715 (73.4%), `ORCHESTRATE`: 407 (17.4%), `ESCALATE`: 213 (9.1%).
- **Validation Set (`autonomy/validation.csv`)**: 501 instances (`ANSWER`: 384, `ORCHESTRATE`: 76, `ESCALATE`: 41).
- **Test Set (`autonomy/test.csv`)**: 501 instances (`ANSWER`: 373, `ORCHESTRATE`: 79, `ESCALATE`: 49).
- **Critical Safety Benchmark (`safety/critical_safety.csv`)**: 180 instances (`ESCALATE`: 180, 100%).

---

## 2. Input Features & Leakage Controls

- **Allowed Pre-Decision Input Signals**:
  - `Risk` (Low, Medium, High, Critical)
  - `Complexity` (Low, Medium, High)
  - `Evidence_Quality` (Strong, Insufficient, Conflicting)
  - `Verification_Required` (Yes, No)
  - `Sensitive_Action` (Yes, No)
  - `Financial_Impact` (Yes, No)
  - `Sentiment` (Negative, Neutral, Positive)
  - `Query` and `Domain`
- **Strict Leakage Controls**:
  - `Escalation_Required` is strictly forbidden from the input feature set because in V4 `ESCALATE` is 100% correlated with `Escalation_Required == 'Yes'`.
  - Post-decision resolution and human feedback fields are strictly forbidden.

---

## 3. Operational Research Significance

The autonomy engine operationalizes the primary capstone research contribution:
1. **ANSWER Tier**: Direct resolution for low-risk, well-supported inquiries.
2. **ORCHESTRATE Tier**: Dynamic multi-agent planner activation for high-complexity inquiries.
3. **ESCALATE Tier**: Human-in-the-loop escalation dossier generation for high-risk, ambiguous, conflicting, or safety-critical inquiries.
