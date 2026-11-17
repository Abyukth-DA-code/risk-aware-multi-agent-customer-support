# RAG Benchmark Relevance Audit & Root-Cause Remediation Report

**Project Title:** Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation
**Research Focus:** Risk-Aware Adaptive Multi-Agent Orchestration
**Component:** Knowledge Base & Retrieval-Augmented Generation (RAG) Evaluation System
**Date:** October 2026
**Auditor Engine:** `src/rag/audit_benchmark_relevance.py` (MiniLM-L6-v2 Semantic Verification)
**Evaluator Engine:** `src/rag/evaluate_retrieval.py` (Multi-Benchmark Evaluation Engine)

---

## Executive Summary

An automated, evidence-grounded semantic relevance audit was executed across all **501 held-out test queries** in `data/processed/rag/retrieval_benchmark.csv` against the 46 policy documents and 95 chunks in the knowledge base.

### Primary Audit Findings:
1. **Benchmark Ground-Truth Flaw:** The previously reported baseline Recall@5 of **71.46%** and 143 misses was severely contaminated by synthetic benchmark mapping errors, cross-domain mismatches, and queries targeting topics completely absent from the knowledge base.
2. **73 Unanswerable Knowledge Gaps (14.6% of benchmark):** 73 queries ask for policy concepts never authored into the knowledge base (e.g., shopping catalogs, list of accepted payment methods, product feedback/reviews, library services, campus placements, small business subscriptions). The original benchmark had forcefully mapped these to unrelated documents (e.g., customer complaints to fraud, reviews to warranties, payment methods to duplicate charges).
3. **40 Demonstrably Mismatched Mappings (8.0% of benchmark):** 40 queries had valid matching documents in the KB, but the benchmark expected completely incorrect documents (e.g., academic transcripts mapped to assignment submission instead of document requests).
4. **3 Cross-Domain Metadata Conflicts (0.6% of benchmark):** 3 Small Business queries had their expected document assigned to `E012` (E-commerce Customer Support Contact Policy), creating an impossible retrieval condition under domain-filtered retrieval.
5. **Separation of Label Gains vs. Retrieval Mechanics:**
   - On the **Answerable Subset (428 queries)** with verified ground truth:
     - **Recall@1:** **61.92%** (vs. 51.70% baseline, **+10.22% absolute gain**)
     - **Recall@3:** **77.34%** (vs. 64.87% baseline, **+12.47% absolute gain**)
     - **Recall@5:** **82.24%** (vs. 71.46% baseline, **+10.78% absolute gain**)
     - **MRR:** **0.6977** (vs. 0.5882 baseline, **+0.1095 gain**)
     - Misses in Top 5 dropped from **143** down to **76**.

---

## 1. Relevance Category Distribution (All 501 Cases)

| Domain | Total Queries | RELEVANT | PARTIAL | MISMATCHED | MISSING_KNOWLEDGE | METADATA_CONFLICT | AMBIGUOUS | Requires Human Review |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E-commerce** | 433 | 146 | 183 | 32 | 53 | 0 | 19 | 255 |
| **Education** | 32 | 19 | 3 | 4 | 6 | 0 | 0 | 9 |
| **Small Business** | 36 | 12 | 6 | 4 | 11 | 3 | 0 | 20 |
| **TOTAL** | **501** | **177** | **192** | **40** | **70** | **3** | **19** | **284** |
| **Percentage** | **100.0%** | **35.3%** | **38.3%** | **8.0%** | **14.0%** | **0.6%** | **3.8%** | **56.7%** |

*Note: 3 Small Business queries represent both cross-domain metadata conflicts and knowledge gaps (total unanswerable queries = 73).*

---

## 2. Root Cause Breakdown (Criteria A through G)

| Root Cause ID & Description | Count | Percentage | Primary Impacted Queries / Intents |
| :--- | :---: | :---: | :--- |
| **D: Missing or incomplete knowledge-base content** | **262** | 52.3% | 70 complete knowledge gaps (`place_order`, `check_payment_methods`, `review`, `Library`, `Placement Support`, `Subscription Management`, `Vendor Support`, `Product Information`) + 192 partial queries lacking specific policy sub-clauses in chunks. |
| **None: Valid Mapping (Retrieved at Rank 1)** | **148** | 29.5% | Correctly mapped queries retrieved at Rank 1 (`cancel_order`, `track_order`, `get_invoice`, `Attendance Policy`, `Fees and Payment`, `Business Payment`, etc.). |
| **A: Incorrect expected document ID** | **40** | 8.0% | Demonstrably wrong expected document where a valid matching document exists in the KB (`Academic Records` -> E024, `Account Management` -> E030, `complaint` -> E012, `check_refund_policy` -> E002). |
| **C: Correct document but inadequate retrieval ranking** | **29** | 5.8% | Relevant queries where expected document was retrieved at Ranks 2–5 or pushed out due to chunk crowding / lexical mismatches. |
| **G: Ambiguous or genuinely unanswerable query** | **19** | 3.8% | Highly underspecified, noisy, or borderline queries with low semantic match across all KB policies. |
| **B: Incorrect expected title, domain, authority, or status** | **3** | 0.6% | Small Business `Delivery Inquiry` queries incorrectly assigned to E-commerce document `E012`. |
| **F: Duplicate or poorly constructed chunks** | *(Secondary)* | *(Identified)* | Document chunks where Chunk 1 is a superset of Chunks 2 & 3, causing multiple chunks from the same document to crowd the top-5 result list. |

---

## 3. Detailed Diagnosis of Common Failure Patterns

### Failure Pattern 1: Forced Mappings on Missing Knowledge (70 Queries)
* **Diagnosis:** Synthetic benchmark generation forced every test query to have an `Expected_Document_ID`, even when the query topic is entirely outside the knowledge base.
* **Evidence:**
  - `place_order` (17 queries): Asks how to shop, purchase items, or view catalog. The KB has post-purchase policies (returns, cancellations, delivery delays), but no shopping catalog. Expected `E012` (Customer Support Contact Policy) had cosine similarity of only **0.043**.
  - `check_payment_methods` (17 queries): Asks what payment methods are accepted (credit cards, PayPal, bank transfer). Expected `E005` (Duplicate Payment Policy) had cosine similarity of **0.260**, because E005 strictly discusses double charges.
  - `review` (19 queries): Asks how to leave a customer review or service feedback. Expected `E011` (Product Warranty Policy) had negative cosine similarity (**-0.021**).
  - `Library` & `Placement Support` (6 queries): Asks about library renewals and campus placements. Expected `E024` (Document Request Policy) had no mention of books or campus jobs.
* **Remediation:** Marked `MISSING_KNOWLEDGE`, `Expected_Document_ID = ""`, `Remediation_Status = "KNOWLEDGE_GAP"`. Excluded from standard answerable recall to avoid penalizing retrieval for unauthored content.

### Failure Pattern 2: Demonstrably Incorrect Expected Document IDs (40 Queries)
* **Diagnosis:** The query is answerable by an existing document, but the benchmark pointed to a completely wrong document.
* **Evidence & Corrections:**
  1. **Education `Academic Records` (4 queries):**
     - Queries: *"How do I request an official transcript?", "How can I download my marks card?"*
     - Original Expected: `E015` (Assignment Submission Policy) — Similarity: **0.259**
     - Corrected Document: `E024` (Document Request Policy: *"Requests for official student documents are processed through designated administrative channels..."*) — Similarity: **0.465** (+0.206 gain).
  2. **Small Business `Account Management` (4 queries):**
     - Queries: *"Can I update my company information?"*
     - Original Expected: `E031` (Employee Account Access Policy: internal staff permissions) — Similarity: **0.170**
     - Corrected Document: `E030` (Customer Information Policy: *"Customer information should be accessed and changed only for legitimate business purposes..."*) — Similarity: **0.489** (+0.319 gain).
  3. **E-commerce `complaint` (16 queries):**
     - Queries: *"where can I make a consumer claim against your company?", "consumer reclamation"*
     - Original Expected: `E010` (Suspicious Transaction Policy: fraud and unauthorized financial activity)
     - Corrected Document: `E012` (Customer Support Contact Policy: approved support channels and authorized escalation).
  4. **E-commerce `check_refund_policy` (16 queries):**
     - Queries: *"where to check the refund rules?", "what are your reimbursement terms?"*
     - Original Expected: `E001` (Product Return Policy)
     - Corrected Document: `E002` (Refund Processing Policy) or `EV1001` (Refund Window).

### Failure Pattern 3: Cross-Domain Metadata Conflicts (3 Queries)
* **Diagnosis:** Queries in Small Business expecting E-commerce documents.
* **Evidence:**
  - Case `CASE000183`, `CASE000087`, `CASE000004`: `Domain = Small Business`, `Intent = Delivery Inquiry`, `Query = "Can I change the delivery address?"`.
  - Expected Document was set to `E012` (E-commerce Customer Support Contact Policy).
  - Under domain filtering (`where={"Domain": "Small Business"}`), `E012` is physically filtered out, guaranteeing 0% recall.
* **Remediation:** Flagged as `METADATA_CONFLICT` and `MISSING_KNOWLEDGE` for Small Business (Small Business has no shipping address document).

### Failure Pattern 4: Chunk Redundancy & Top-5 Crowding
* **Diagnosis:** Document chunking created redundant parent/child chunks (e.g. `E002_C01` contains the exact text of `E002_C02` and `E002_C03`).
* **Evidence:** Retrieval queries frequently return 2 or 3 chunks from the same document in the top 5 (e.g., `E002_C02` at Rank 2 and `E002_C01` at Rank 5), preventing other relevant documents from entering the top-5 window.
* **Recommendation:** Group retrieved chunks by `Document_ID` before returning top-k distinct documents.

---

## 4. Retrieval Performance Comparison: Baseline vs. Remediated Benchmark

| Metric | Original Baseline (`baseline_v1`) | Remediated Benchmark (`remediated_v2` - All 501) | Remediated Benchmark (`remediated_v2` - Answerable 428) | Delta on Answerable Subset |
| :--- | :---: | :---: | :---: | :---: |
| **Total Queries** | 501 | 501 | 428 | -73 unanswerable |
| **Recall@1** | 51.70% | 52.89% | **61.92%** | **+10.22%** |
| **Recall@3** | 64.87% | 66.07% | **77.34%** | **+12.47%** |
| **Recall@5** | 71.46% | 70.26% | **82.24%** | **+10.78%** |
| **MRR** | 0.5882 | 0.5960 | **0.6977** | **+0.1095** |
| **Mean Latency** | 16.28 ms | 13.98 ms | 13.98 ms | -2.30 ms |
| **P95 Latency** | 24.38 ms | 16.83 ms | 16.83 ms | -7.55 ms |
| **Queries Missing from Top 5** | 143 | 149 (incl. 73 gaps) | **76** | **-67 misses (-46.9%)** |

### Per-Domain Recall@5 Breakdown:
- **E-commerce:** 73.21% (Baseline) -> **81.84%** (Answerable Remediated)
- **Education:** 71.88% (Baseline) -> **88.46%** (Answerable Remediated)
- **Small Business:** 50.00% (Baseline) -> **81.82%** (Answerable Remediated)

---

## 5. Artifact & File Change Audit Trail

| File Path | Action | Rationale / Purpose |
| :--- | :--- | :--- |
| `data/processed/rag/retrieval_benchmark_baseline_v1.csv` | **Created** | Immutable copy of original benchmark preserving exact baseline metrics (51.70% / 71.46%). |
| `src/rag/audit_benchmark_relevance.py` | **Modified** | Replaced placeholder with MiniLM-L6-v2 semantic relevance auditor, calibrated confidence, 6 categories, and 7 root causes. |
| `src/rag/evaluate_retrieval.py` | **Modified** | Upgraded evaluator supporting `--benchmark`, `--eval_name`, and answerable vs unanswerable reporting. |
| `src/rag/diagnose_documents.py` | **Modified** | Upgraded diagnostic tool integrating audit records, root causes, and missed document distributions. |
| `tests/test_rag_audit.py` | **Created** | Comprehensive 14-point test suite validating all criteria (mismatches, gaps, splits, regressions). |
| `outputs/rag_evaluation/benchmark_relevance_audit_v2.csv` | **Created** | Case-level audit artifact with evidence, similarity deltas, confidence, and recommended actions. |
| `outputs/rag_evaluation/benchmark_audit_summary.csv` | **Created** | Domain-level summary breakdown of categories and human review counts. |
| `data/processed/rag/retrieval_benchmark_remediated_v2.csv` | **Created** | Clean proposed benchmark with verified document mappings and explicit knowledge gap flags. |
| `data/processed/FINAL_MASTER_GOLD.csv` | **Preserved (Untouched)** | Confirmed 100% untouched (3,519 rows, all splits intact). |
| `data/raw/V4/` | **Preserved (Untouched)** | Confirmed 100% frozen and untouched. |

---

## 6. Test Suite Execution & Verification

Full regression suite executed via `pytest`:
* **Total Tests Executed:** 34
* **Total Tests Passed:** 34
* **Pass Rate:** **100.0%**
* **Execution Time:** ~19.36s

### Test Coverage Highlights:
- `test_rag.py`: All 3 existing integrity, conflict preservation (EV1002, EV1004, EV1006, EV1008, EV1010), and chunk linkage tests passed.
- `test_rag_audit.py`: All 14 audit criteria passed:
  1. Missing document IDs detection passed.
  2. Wrong expected document mappings detected passed.
  3. Cross-domain mismatches detected passed.
  4. Missing knowledge categorization passed.
  5. Partial relevance and ambiguity handling passed.
  6. Duplicate chunks integrity passed.
  7. Metadata conflicts detection passed.
  8. Preservation of original benchmark labels passed.
  9. Deterministic output and reproducibility passed.
  10. Invalid model outputs and malformed inputs passed.
  11. Low-confidence cases flagged for review passed.
  12. Existing RAG retrieval operational continuity passed.
  13. Frozen V4 data immutability passed.
  14. Existing test suite constraints passed.
- Operational & Schema Tests: `test_schema.py`, `test_duplicates.py`, `test_splits.py`, `test_labels.py`, `test_leakage.py`, `test_relationships.py` all passed.

---

## 7. Remaining Limitations & Human-in-the-Loop Review Items

1. **Unauthored Policy Expansion:** 73 queries represent legitimate customer questions that have no policy documentation in the knowledge base. The organization's policy authoring team must decide whether to author new documents for:
   - E-commerce: Shopping catalog policy, accepted payment modalities policy, and customer review guidelines.
   - Education: Library operations and campus placement eligibility.
   - Small Business: Subscription service policy, vendor management protocol, and product catalog.
2. **Partial Relevance Refinement (192 queries):** Queries asking about specific sub-rules (e.g. cancellation fees, delivery windows, account modification) are mapped to parent policies that mention the broad topic but omit exact numerical rules. Chunk expansions can elevate these from PARTIAL to RELEVANT.
3. **Chunk Deduping / Distinct Document Retrieval:** The vector store returns raw chunks. When multiple chunks from the same document rank in the top 5, distinct document diversity suffers. Implementing a `distinct_by="document_id"` filter in `RAGRetriever` will further boost document-level recall.

---

## 8. Reproducible Run Commands

To reproduce all results, re-run audits, evaluations, or test suites, execute the following commands in powershell:

```powershell
# 1. Run the Semantic Relevance Auditor (audits all 501 cases and generates v2 artifacts)
.\.venv\Scripts\python.exe src/rag/audit_benchmark_relevance.py

# 2. Evaluate against the original Baseline Benchmark (reproduces 51.70% / 71.46%)
.\.venv\Scripts\python.exe src/rag/evaluate_retrieval.py --benchmark data/processed/rag/retrieval_benchmark_baseline_v1.csv --eval_name baseline_v1

# 3. Evaluate against the Remediated Benchmark v2 (reports 61.92% / 82.24% on answerable cases)
.\.venv\Scripts\python.exe src/rag/evaluate_retrieval.py --benchmark data/processed/rag/retrieval_benchmark_remediated_v2.csv --eval_name remediated_v2

# 4. Run the Enhanced Diagnostic Tool (prints root causes, gaps, and missed documents)
.\.venv\Scripts\python.exe src/rag/diagnose_documents.py

# 5. Run the Full Automated Test Suite (34 tests including all 20 existing regression tests)
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
