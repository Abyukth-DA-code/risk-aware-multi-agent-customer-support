# V4 Data Leakage Report & Formal Leakage Matrix

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation
**Date**: September 28, 2026

---

## 1. Principles of Target Leakage Prevention

In a research-oriented multi-agent customer support system, target leakage occurs when:
1. Features derived from downstream decisions or post-decision events are fed into upstream classifiers.
2. Synthetic benchmark confidence scores are used as input features to predict the very confidence or decision they represent.
3. Near-duplicate queries are shared between training and evaluation splits.
4. One target variable is deterministically derived from another and used as a shortcut feature.

---

## 2. Formal Task-by-Task Data Leakage Matrix

| ML Task / Target Variable | Allowed Input Features | Strictly Forbidden Features | Reason for Prohibition |
|:---|:---|:---|:---|
| **Intent Classification**<br>`Intent` (47 classes) | `Query`, `Domain` (or domain configuration) | `Autonomy_Decision`, `Escalation_Required`, `Required_Agents`, `Required_Tools`, `Action_Type`, `Risk_Reason`, `Complexity_Reason`, `Intent_Confidence_Gold`, `Final_Confidence_Gold`, human feedback | Downstream operational outcomes or direct consequence of intent. |
| **Sentiment Analysis**<br>`Sentiment` (3 classes) | `Query`, pre-decision customer conversation context | `Autonomy_Decision`, `Escalation_Required`, `Risk`, `Risk_Reason`, `Human_Decision`, `Reviewer_Notes` | Customer sentiment must be inferred solely from customer expression. |
| **Operational Risk**<br>`Risk` (Low, Medium, High, Critical) | `Query`, `Domain`, observable pre-decision features (`Financial_Impact`, `Sensitive_Action`, `Action_Type` if observable) | `Risk_Reason`, `Autonomy_Decision`, `Escalation_Required`, `Escalation_Reason`, `Verification_Score_Gold`, `Final_Confidence_Gold`, post-decision human feedback | `Risk_Reason` explicitly states why the label was assigned; Autonomy & Escalation are downstream consequences of Risk. |
| **Query Complexity**<br>`Complexity` (Low, Medium, High) | `Query`, `Domain`, observable query characteristics (entity count, syntax) | `Complexity_Reason`, `Required_Agents`, `Required_Tools`, `Required_Capabilities`, `Required_Data_Sources`, `Autonomy_Decision`, `Escalation_Required` | Agents and tools are downstream planning actions; using them to predict complexity causes circularity. |
| **Escalation Prediction**<br>`Escalation_Required` (Yes, No) | Observable pre-decision signals: predicted risk, predicted complexity, evidence quality, verifier score, policy restrictions | `Escalation_Reason`, `Autonomy_Decision`, `AI_Resolution`, `Human_Decision`, `Correction`, `Escalation_Appropriate`, `Reviewer_Notes` | `Escalation_Reason` directly leaks the escalation justification; `Autonomy_Decision` ESCALATE is identical to Escalation_Required; feedback is post-decision. |
| **Autonomy Level Engine**<br>`Autonomy_Decision` (ANSWER, ORCHESTRATE, ESCALATE) | Upstream observable signals: `Risk`, `Complexity`, `Evidence_Quality`, `Verification_Required`, `Sensitive_Action`, `Financial_Impact` | Direct shortcut from `Escalation_Required` as sole feature; post-decision resolution fields; reviewer feedback | The autonomy engine must dynamically arbitrate from multifaceted operational signals. |
| **Domain Classification**<br>`Domain` (3 classes, optional demo) | `Query` | All other fields | Domain must be predicted solely from raw query text. |

---

## 3. Discovered Leakage Vectors & Quarantine Actions

1. **Risk_Balanced Contamination with Critical_Safety Benchmark**:
   - `Risk_Balanced` sheet contained 180 cases identical to the held-out `Critical_Safety` benchmark.
   - *Resolution*: Quarantined and isolated `Critical_Safety` exclusively into `data/processed/safety/critical_safety.csv`.
2. **Train-Validation Near-Duplicates**:
   - `CASE002854` and `CASE002853` in Train matched `CASE000761` and `CASE000963` in Validation under normalized string matching.
   - *Resolution*: Quarantined `CASE002854` and `CASE002853` from all ML training sets.
3. **Train-Test Spacing Near-Duplicates (Typo Template Contamination)**:
   - `CASE002313` (`i dont know how to cancel purchase {Order Number}`) matched Test `CASE002283` (`I don't knowhow to cancel purchase {Order Number}`) due to the missing-space typo `knowhow`.
   - `CASE002396` (`create {Account Category} account for wife`) matched Test `CASE001713` (`create {Account Category} accountfor wife`) due to the missing-space typo `accountfor`.
   - `AUG002464` in Train derived from `CASE002313`.
   - *Resolution*: Quarantined `CASE002313`, `CASE002396`, and `AUG002464` from all training sets. Train-Test normalized and super-normalized overlap is now rigorously **0** across all tasks.
4. **Escalation Reason Population**:
   - `Escalation_Reason` is populated across 100% of rows (even non-escalated rows contain values like `'Can Resolve'` or `'Low Risk / Sufficient Confidence'`).
   - *Resolution*: Strictly excluded `Escalation_Reason` from all feature feature matrices.
