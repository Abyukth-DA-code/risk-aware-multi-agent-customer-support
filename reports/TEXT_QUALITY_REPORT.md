# Text Quality and Query Distribution Report

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation
**Date**: September 28, 2026

---

## 1. Text Quality Overview

Natural language customer queries across `Core_Master` (3,339), `Validation` (501), `Test` (501), and `Critical_Safety` (180) were evaluated for length, token count, noise, control characters, empty text, and vocabulary richness.

---

## 2. Text Statistics Summary

| Dataset Split | Rows | Empty Queries | Min Chars | Max Chars | Mean Chars | Min Words | Max Words | Mean Words | Noise / Corrupted Text |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Core_Master` | 3,339 | 0 | 8 | 91 | 48.0 | 1 | 16 | 8.9 | 0 |
| `Validation` | 501 | 0 | 8 | 89 | 47.6 | 1 | 15 | 8.9 | 0 |
| `Test` | 501 | 0 | 15 | 89 | 48.1 | 2 | 15 | 9.0 | 0 |
| `Critical_Safety` | 180 | 0 | 58 | 157 | 108.1 | 9 | 26 | 17.4 | 0 |
| `Train_Balanced` | 5,640 | 0 | 10 | 125 | 65.7 | 2 | 22 | 12.0 | 0 |

---

## 3. Linguistic Observations & Cleaning Directives

1. **Short Queries**:
   - Queries as short as 8 characters (e.g., `"help me"`, `"order status"`) represent authentic real-world customer shorthand and are intentionally preserved.
2. **Critical Safety Scenarios**:
   - Critical Safety queries have substantially longer lengths (Mean: 108.1 chars, Max: 157 chars) because they describe detailed, multi-step sensitive scenarios involving financial fraud, unauthorized account lockouts, or high-value charge discrepancies.
3. **Punctuation & Capitalization**:
   - Varied casing (`"i need help"` vs `"I need help"`) and colloquial punctuation (missing apostrophes in `"dont"`) reflect natural customer typing habits.
   - Text cleaning applies Unicode normalization (`NFKC`) and whitespace trimming while strictly preserving meaningful alphanumeric identifiers, transaction codes, and monetary figures.
