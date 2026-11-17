# Master Canonical Data Dictionary

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation
**Version**: 4.0 (Final Clean Canonical)
**Date**: September 28, 2026

---

## 1. Case-Level Schema (`FINAL_MASTER_GOLD.csv`)

| Field Name | Data Type | Meaning | Allowed / Expected Values | Source | Input / Target Role | Post-Decision? | Leakage Risk | Missing Value Policy |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `Case_ID` | String | Unique surrogate primary key for support case | `CASE000001` - `CASE003519` | Master Gold / Critical Safety | Identifier | No | None | Required. 0 nulls allowed. |
| `Query` | String | Raw text of customer inquiry | Natural language string (8-157 chars) | Bitext / Domain Benchmarks | Model Input | No | None | Required. 0 nulls allowed. |
| `Domain` | Categorical | Business domain of customer interaction | `E-commerce`, `Education`, `Small Business` | Deployment Config / Gold Label | Context / Target (Demo) | No | Target in multi-domain demo | Required. |
| `Intent` | Categorical | Customer task intent | 47 canonical intents (27 E-comm, 10 Edu, 10 SmallBiz) | Master Gold | Target | No | Primary target | Required. |
| `Sentiment` | Categorical | Customer emotional polarity | `Negative`, `Neutral`, `Positive` | Master Gold / Sentiment Benchmark | Target | No | Target | Required. |
| `Risk` | Categorical | Operational risk level | `Low`, `Medium`, `High`, `Critical` | Master Gold / Safety Benchmark | Target | No | Target | Required. |
| `Risk_Reason` | String | Justification for assigned risk | Explanatory text | Master Gold | Reference / Explainability | No | **HIGH LEAKAGE** (Forbidden in ML) | Required for reference. |
| `Complexity` | Categorical | System execution complexity | `Low`, `Medium`, `High` | Master Gold | Target | No | Target | Required. |
| `Complexity_Reason` | String | Justification for complexity | Explanatory text | Master Gold | Reference / Explainability | No | **HIGH LEAKAGE** (Forbidden in ML) | Required for reference. |
| `Required_Agents` | Multi-label String | Specialized agents required | Semicolon-delimited capability agents | Master Gold | Target (Dynamic Planner) | No | Downstream of Complexity | Required. |
| `Required_Tools` | Multi-label String | Data tools/databases needed | Semicolon-delimited tools | Master Gold | Target (Dynamic Planner) | No | Downstream of Complexity | Required. |
| `Required_Data_Sources` | Multi-label String | Underlying data tables | Semicolon-delimited data sources | Master Gold | Target (Dynamic Planner) | No | Downstream of Complexity | Required. |
| `Sensitive_Action` | Binary | Involves account/financial modification | `Yes`, `No` | Master Gold | Input Feature (Pre-decision) | No | Allowed if observable | Required. |
| `Financial_Impact` | Binary | Involves monetary transaction | `Yes`, `No` | Master Gold | Input Feature (Pre-decision) | No | Allowed if observable | Required. |
| `Action_Type` | Categorical | Type of action requested | `Inquiry`, `Modification`, `Cancellation`, `Access`, `Complaint` | Master Gold | Input Feature (Pre-decision) | No | Allowed if observable | Required. |
| `Evidence_Quality` | Categorical | Quality of retrieved support evidence | `Strong`, `Insufficient`, `Conflicting` | Master Gold / RAG Verifier | Input Feature (Decision Engine) | No | Allowed in Escalation/Autonomy | Required. |
| `Verification_Required` | Binary | Whether independent verification is required | `Yes`, `No` | Master Gold / Verifier | Input Feature (Decision Engine) | No | Allowed in Escalation/Autonomy | Required. |
| `Escalation_Required` | Binary | Ground truth escalation requirement | `Yes`, `No` | Master Gold | Target (Escalation Model) | No | **LEAKAGE** if used to predict Autonomy | Required. |
| `Escalation_Reason` | String | Ground truth reason for escalation | Categorical justification | Master Gold | Reference / Audit | No | **HIGH LEAKAGE** (Forbidden in ML) | Required for reference. |
| `Autonomy_Decision` | Categorical | Prescribed autonomy tier | `ANSWER`, `ORCHESTRATE`, `ESCALATE` | Master Gold | Target (Autonomy Engine) | No | Policy Target | Required. |
| `Data_Origin` | Categorical | Provenance category | `PUBLIC_SOURCE`, `SYNTHETIC`, `DERIVED` | Provenance Audit | Metadata | No | None | Required. |
| `Source_Type` | String | Original upstream source name | `Public Bitext`, `Synthetic`, `Synthetic Safety Benchmark` | Provenance Audit | Metadata | No | None | Required. |
| `Split` | Categorical | Evaluation split designation | `train_base`, `validation`, `test`, `critical_safety` | Master Gold | Partitioning | No | None | Required. |
| `Quality_Status` | Categorical | Data verification status | `VERIFIED_GOLD`, `QUARANTINED` | Quality Control Pipeline | Quality Gate | No | None | Required. |

---

## 2. Operational Database Schema

- **Customers**: `Customer_ID` (PK), `Domain`, `Customer_Type`, `Account_Status`, `Verification_Status`, `Dataset_Version`
- **Orders**: `Order_ID` (PK), `Customer_ID` (FK), `Order_Status`, `Order_Amount_INR`, `Payment_Status`, `Delivery_Status`, `Return_Status`, `Dataset_Version`, `Domain`
- **Transactions**: `Transaction_ID` (PK), `Customer_ID` (FK), `Order_ID` (FK), `Transaction_Type`, `Amount_INR`, `Payment_Method`, `Transaction_Status`, `Fraud_Flag`, `Dataset_Version`, `Domain`
- **Support_Tickets**: `Ticket_ID` (PK), `Customer_ID` (FK), `Query_Type`, `Customer_Query`, `Ticket_Status`, `Previous_Resolution`, `Dataset_Version`, `Domain`
- **Students**: `Student_ID` (PK), `Customer_ID` (FK), `Program`, `Year`, `Attendance_Percent`, `Fee_Status`, `Exam_Status`, `Scholarship_Status`, `Dataset_Version`, `Domain`
- **Business_Records**: `Business_Record_ID` (PK), `Customer_ID` (FK), `Record_Type`, `Reference_ID`, `Amount_INR`, `Record_Status`, `Approval_Status`, `Dataset_Version`, `Domain`

---

## 3. RAG Knowledge Base Schema

- **Document_ID** (PK): Unique document code (`E001`-`E012`, `E013`-`E024`, `E025`-`E036`, `EV1001`-`EV1010`)
- **Domain**: `E-commerce`, `Education`, `Small Business`
- **Document_Type**: Policy, Rules, Terms, Guidelines
- **Title**: Descriptive title of the knowledge document
- **Source**: Authoritative entity (e.g., `Official E-commerce Policy`)
- **Authority**: `High` (official current policy), `Medium` (superseded/outdated version)
- **Version**: Version tag (`v1.0`, `v2.0`)
- **Effective_Date**: Date when policy took effect (YYYY-MM-DD)
- **Last_Updated**: Date of last modification (YYYY-MM-DD or null if initial)
- **Freshness_Status**: `Current` vs. `Outdated`
- **Retrieval_Priority**: Integer priority weighting (1 = Primary, 2 = Secondary)
- **Content**: Clean text of policy statement
