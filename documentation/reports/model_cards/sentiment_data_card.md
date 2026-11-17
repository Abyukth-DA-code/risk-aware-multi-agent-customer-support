# Sentiment Analysis Dataset Data Card

**Task**: Customer Sentiment Classification  
**Classes**: `Negative`, `Neutral`, `Positive` (3 classes)  
**Date**: September 28, 2026  

---

## 1. Dataset Summary & Sizes

- **Training Set (`sentiment/train.csv`)**: 1,500 instances (Balanced 500 Negative, 500 Neutral, 500 Positive).
- **Validation Set (`sentiment/validation.csv`)**: 501 instances (Natural customer support distribution: 381 Neutral, 120 Negative, 0 Positive).
- **Test Set (`sentiment/test.csv`)**: 501 instances (Natural customer support distribution: 385 Neutral, 116 Negative, 0 Positive).

---

## 2. Preprocessing & Leakage Controls

- **Input Features**: `Query`.
- **Target Variable**: `Sentiment`.
- **Leakage Controls**: Customer sentiment must be determined solely from customer text. Post-decision human reviews, agent resolution outcomes, and escalation status are strictly prohibited from the feature set.
- **Split Isolation**: 0 overlap in `Case_ID` or query text between training and evaluation sets.

---

## 3. Critical Research Findings & Limitations

- **Severe Natural Sentiment Imbalance**: In authentic customer support (`Core_Master`), customers reach out when facing issues; hence Positive sentiment is extremely rare (only 6 positive cases out of 3,339 in master data, and 0 in natural held-out test splits).
- **Training vs. Evaluation Protocol**: The balanced training set enables models to learn positive linguistic markers, but evaluation must be reported on both balanced benchmarks and the natural held-out test set to maintain scientific integrity.
