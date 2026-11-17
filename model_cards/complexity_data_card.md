# Query Complexity Classification Dataset Data Card

**Task**: Execution Complexity Estimation for Dynamic Agent Planning
**Classes**: `Low`, `Medium`, `High` (3 classes)
**Date**: September 28, 2026

---

## 1. Dataset Summary & Sizes

- **Training Set (`complexity/train.csv`)**: 2,100 instances (Balanced 700 Low, 700 Medium, 700 High).
- **Validation Set (`complexity/validation.csv`)**: 501 instances (243 Medium, 157 Low, 101 High).
- **Test Set (`complexity/test.csv`)**: 501 instances (244 Medium, 161 Low, 96 High).

---

## 2. Preprocessing & Leakage Controls

- **Input Features**: `Query`, `Domain`, observable query structural indicators.
- **Target Variable**: `Complexity`.
- **Strict Leakage & Circularity Controls**:
  - `Complexity_Reason` is strictly forbidden.
  - `Required_Agents`, `Required_Tools`, `Required_Capabilities`, and `Required_Data_Sources` are strictly forbidden. Using actual agent counts or tool invocations to predict complexity would create circular logic when complexity is used to select agents.
  - `Autonomy_Decision` and `Escalation_Required` are strictly excluded.

---

## 3. Operational Meaning & Research Notes

- **Low Complexity**: Single-lookup or direct RAG knowledge queries (e.g. policy questions).
- **Medium Complexity**: Operational database lookups (order status, student transcript, invoice lookup).
- **High Complexity**: Multi-system coordination (cross-system payment and order verification, complex portal access troubleshooting).
