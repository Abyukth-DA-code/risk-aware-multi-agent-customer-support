# Privacy & Synthetic Data Governance Report

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation  
**Date**: September 28, 2026  

---

## 1. Academic Privacy Compliance

This system is developed as an academic university capstone research prototype.
Under institutional research guidelines and global data protection standards (GDPR / DPDP Act), no real customer Personally Identifiable Information (PII) is included.

---

## 2. PII Audit of Operational Data

- **Customer Identifiers**: Generated using synthetic surrogate keys (`CUST0001` to `CUST1000`).
- **Names, Phone Numbers, Physical Addresses**: Not present in operational schemas.
- **Account Status & Verification**: Simulated categorical indicators (`Active`, `Suspended`, `Closed`, `Verified`, `Pending`).
- **Financial Transactions**: Synthetic card / UPI transactions with randomized non-real identifiers (`TXN00001` to `TXN01803`).
- **Student Data**: Synthetic IDs (`STU00001` to `STU00500`) associated with standard university degree programs.
- **Small Business Data**: Synthetic business record IDs (`BIZ00001` to `BIZ00500`).
- **Memory Entries**: All 1,500 memory records are classified with `Privacy_Level = 'Internal'`.

---

## 3. Data Origin & Provenance Summary

| Entity / Dataset | Total Records | Primary Source | Synthetic Ratio | Provenance Status |
|:---|:---:|:---|:---:|:---|
| `Core_Master` (E-commerce) | 2,870 | Bitext Customer Support Dataset (Hugging Face) | 0.0% | Public Source Verified |
| `Core_Master` (Small Business) | 250 | Synthetic Domain Query Benchmark | 100.0% | Marked as Synthetic |
| `Core_Master` (Education) | 219 | Smart University Query Classification / Synthetic | 100.0% | Marked as Synthetic |
| `Critical_Safety` | 180 | Synthetic Safety Benchmark | 100.0% | Marked as Synthetic |
| `Customers` | 1,000 | Synthetic Operational Database Generator | 100.0% | Marked as Synthetic |
| `Orders` | 1,500 | Synthetic Operational Database Generator | 100.0% | Marked as Synthetic |
| `Transactions` | 1,803 | Financial Transactions / Fraud Generator | 100.0% | Marked as Synthetic |
| `Support_Tickets` | 1,000 | Synthetic Support Ticket System | 100.0% | Marked as Synthetic |
| `Students` | 500 | Smart University Operational Generator | 100.0% | Marked as Synthetic |
| `Business_Records` | 500 | Small Business Operational Generator | 100.0% | Marked as Synthetic |
| `Memory` | 1,500 | Synthetic Support Memory Generator | 100.0% | Marked as Synthetic |
| `RAG Knowledge Base` | 46 | Official Domain Policy Generator | 100.0% | Marked as Synthetic |

All synthetic records in processed exports carry explicit provenance flags (`data_origin = 'SYNTHETIC'`).
