# Sentiment Task Canonical Data Dictionary & Annotation Guide

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation
**Component**: Model 2 — Customer Sentiment Analysis
**Date**: October 13, 2026
**Status**: Repaired, Verified & Leakage-Free

---

## 1. Task Specification

The Sentiment Analysis component predicts customer affective polarity from the user utterance to dynamically modulate escalation urgency, agent empathy, and response priority in the multi-agent orchestrator.

- **Input**: `Query` (natural language text of customer utterance)
- **Target**: `Sentiment` (3 categorical classes: `Negative`, `Neutral`, `Positive`)
- **Metadata**: `Case_ID` (unique identifier), `Domain` (business context), `Data_Origin` (provenance tracker)

---

## 2. Canonical Sentiment Label Policy

| Class Label | Operational Definition | Inclusion Criteria | Exclusion Criteria | Canonical Examples |
|:---|:---|:---|:---|:---|
| **`Negative`** | Explicit customer dissatisfaction, frustration, complaints, product/service failure, or annoyance. | Customer mentions anger, defects, payment failures, delay complaints, poor service, or files a formal dispute. | Factual neutral procedural questions (e.g. asking where to find a return policy) are NOT negative unless negative emotion is stated. | *"I am unhappy with your work, I want to file a consumer claim."*<br>*"My fee payment has failed. What should I do?"*<br>*"you aren't helpful, how can I chat with someone?"* |
| **`Neutral`** | Factual inquiries, informational requests, operational queries, routine task execution without clear affective valence. | Routine status lookups, policy questions, password resets, address configurations, course registrations, order modifications. | Formulaic greetings (*"Hello"*, *"Please help me"*) or closings (*"Thank you in advance"*) attached to a routine inquiry remain Neutral. | *"i need assistance to set my new delivery address up"*<br>*"How can I download my invoice?"*<br>*"When is the scholarship application deadline?"* |
| **`Positive`** | Explicit customer praise, gratitude, satisfaction, enthusiastic feedback, or confirmed issue resolution. | Statements expressing thankfulness for prompt help, satisfaction with resolution, praise for agent assistance, delivery speed compliments. | Mere politeness (*"Please"*, *"Kindly"*) without gratitude or praise does NOT qualify as Positive. Contradictory help requests are excluded. | *"Thank you, that solved my issue."*<br>*"My replacement item arrived today in perfect condition, thank you so much!"*<br>*"The portal login issue was resolved within an hour, great support!"* |

---

## 3. Dataset Schema (`train.csv`, `validation.csv`, `test.csv`)

| Column Name | Data Type | Nullability | Permitted Values | Provenance | Description |
|:---|:---:|:---:|:---|:---|:---|
| `Case_ID` | String | NOT NULL | Alphanumeric (e.g. `CASE...`, `SYN...`) | Master Gold / Provenance Engine | Unique primary key. Disjoint across splits. |
| `Query` | String | NOT NULL | Clean text string | Public Bitext / Grounded Synthetic | Customer utterance without leakage tags. |
| `Domain` | Categorical | NOT NULL | `E-commerce`, `Education`, `Small Business` | Deployment Architecture | Business domain context. |
| `Sentiment` | Categorical | NOT NULL | `Negative`, `Neutral`, `Positive` | Gold Label Policy | Ground truth sentiment label. |
| `Data_Origin` | Categorical | NOT NULL | `PUBLIC_SOURCE`, `SYNTHETIC` | Data Provenance Tracker | Origin indicator to track synthetic data ratio. |

---

## 4. Evaluation Constraints & Zero-Leakage Policy

1. **Target Feature Isolation**: Upstream models predict sentiment solely from `Query` text. Downstream outcomes (`Risk`, `Autonomy_Decision`, `Escalation_Required`, `Reviewer_Notes`) are strictly forbidden as model features.
2. **Template Family Disjointness**: All synthetic positive evaluation queries were synthesized under disjoint template batches (`POS_FAM_VAL` and `POS_FAM_TEST`), guaranteeing zero lexical, phrase, or template leakage from `POS_FAM_TRAIN`.
3. **Realistic Evaluation Proportions**: Validation and Test splits preserve authentic customer-support distribution (Neutral ~68%, Negative ~21%, Positive ~11%) rather than artificial 33/33/33% quotas.
