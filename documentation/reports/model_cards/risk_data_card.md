# Operational Risk Classification Dataset Data Card

**Task**: Operational Risk Level Determination for Autonomy Gating  
**Classes**: `Low`, `Medium`, `High`, `Critical` (4 classes)  
**Date**: September 28, 2026  

---

## 1. Dataset Summary & Sizes

- **Training Set (`risk/train.csv`)**: 1,800 instances (Balanced 600 Low, 600 Medium, 600 High; Critical cases quarantined).
- **Validation Set (`risk/validation.csv`)**: 501 instances (267 Medium, 203 Low, 31 High).
- **Test Set (`risk/test.csv`)**: 501 instances (268 Medium, 203 Low, 30 High).
- **Critical Safety Benchmark (`safety/critical_safety.csv`)**: 180 instances (180 Critical Risk, 100% held-out for safety evaluation).

---

## 2. Input Features & Leakage Controls

- **Allowed Pre-Decision Features**:
  - `Query` (Text)
  - `Domain` (Categorical)
  - `Sensitive_Action` (`Yes`, `No`)
  - `Financial_Impact` (`Yes`, `No`)
  - `Action_Type` (`Inquiry`, `Modification`, `Cancellation`, `Access`, `Complaint`)
- **Strictly Forbidden Leakage Features**:
  - `Risk_Reason` (prohibited: explicitly states why the label was assigned)
  - `Autonomy_Decision` (prohibited: downstream policy output)
  - `Escalation_Required` & `Escalation_Reason` (prohibited: consequence of risk)
  - `Final_Confidence_Gold` & `Verification_Score_Gold` (synthetic evaluation signals)
  - Post-decision human feedback / reviews.

---

## 3. Critical Safety Isolation & Limitations

- **Safety Benchmark Isolation**: The raw `Risk_Balanced` sheet originally replicated the 180 Critical Safety cases into the training split. In our pipeline, all Critical Safety cases are strictly isolated from training to ensure a pristine, zero-shot safety evaluation benchmark.
- **Correlation with Complexity**: In V4, all High Risk cases in `Core_Master` also carry High Complexity due to scenario design. Models must be evaluated on independent stress queries to avoid spurious correlation.
