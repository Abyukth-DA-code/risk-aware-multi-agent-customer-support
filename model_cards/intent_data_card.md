# Intent Classification Dataset Data Card

**Task**: Intent Classification across Multi-Domain Customer Inquiries
**Supported Domains**: E-commerce (27 intents), Education (10 intents), Small Business (10 intents) — Total: 47 intents
**Date**: September 28, 2026

---

## 1. Dataset Summary & Sizes

- **Training Set (`intent/train.csv`)**: 5,638 instances (Balanced multi-domain training set derived from `Train_Balanced` after quarantining 2 train-val near-duplicates).
- **Validation Set (`intent/validation.csv`)**: 501 instances (Natural held-out evaluation set).
- **Test Set (`intent/test.csv`)**: 501 instances (Protected held-out evaluation benchmark).

---

## 2. Label Taxonomy & Classes

Total Distinct Classes: **47**
- **E-commerce (27)**: `cancel_order`, `change_order`, `change_shipping_address`, `check_cancellation_fee`, `check_invoice`, `check_payment_methods`, `check_refund_policy`, `complaint`, `contact_customer_service`, `contact_human_agent`, `create_account`, `delete_account`, `delivery_options`, `delivery_period`, `edit_account`, `get_invoice`, `get_refund`, `newsletter_subscription`, `payment_issue`, `place_order`, `recover_password`, `registration_problems`, `review`, `set_up_shipping_address`, `switch_account`, `track_order`, `track_refund`.
- **Education (10)**: `Academic Records`, `Attendance Policy`, `Course Registration`, `Exam Registration`, `Exam Schedule`, `Fees and Payment`, `Library`, `Placement Support`, `Scholarship`, `Student ID`.
- **Small Business (10)**: `Account Management`, `Business Payment`, `Customer Support`, `Delivery Inquiry`, `Invoice Inquiry`, `Order Management`, `Product Information`, `Return and Refund`, `Subscription Management`, `Vendor Support`.

---

## 3. Preprocessing & Leakage Controls

- **Input Features**: `Query`, `Domain` (or domain configuration context).
- **Target Variable**: `Intent`.
- **Text Normalization**: Unicode NFKC, whitespace compaction, preserving special identifiers (e.g. order numbers, amounts).
- **Strict Leakage Controls**: All downstream execution variables (`Autonomy_Decision`, `Escalation_Required`, `Required_Agents`, `Required_Tools`, `Action_Type`, `Risk_Reason`, `Complexity_Reason`, `Final_Confidence_Gold`, human feedback) are strictly excluded from input matrices.
- **Deduplication**: Quarantined 2 near-duplicate queries (`CASE002853`, `CASE002854`) that overlapped with Validation. Test overlap is verified at exactly **0**.

---

## 4. Limitations & Known Weaknesses

- In natural deployment, E-commerce accounts for ~86% of inquiries in unaugmented data (`Core_Master`), while Education and Small Business represent smaller shares.
- In known single-domain deployment, models should be evaluated within the specific domain partition rather than across the full 47-class space.
