# Distinct-Document Retrieval Implementation & Validation Report

**Project Title:** Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation
**Research Focus:** Risk-Aware Adaptive Multi-Agent Orchestration
**Component:** RAG Retriever & Document-Level Relevance Evaluation
**Date:** October 2026
**Module:** [`src/rag/retriever.py`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/src/rag/retriever.py)
**Evaluator:** [`src/rag/evaluate_retrieval.py`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/src/rag/evaluate_retrieval.py)

---

## 1. Executive Summary

An optional **distinct-document retrieval mode** was implemented and validated across both a separate development dataset and the held-out test benchmark.

### Problem Addressed:
In standard raw chunk retrieval, documents split into multiple chunks (e.g. `E002_C01`, `E002_C02`, `E002_C03`) frequently crowd the top-5 search results. For example, a query about refund status returned `E002` at both Rank 2 and Rank 5. This chunk redundancy displaced other genuinely relevant documents from entering the top-5 window, artificially deflating Recall@3, Recall@5, and document-level diversity.

### Solution Implemented:
1. **Candidate Pool Expansion:** In distinct-document mode, the retriever queries a candidate pool of `fetch_k = min(max(top_k, top_k * candidate_multiplier), available_chunks)` chunks (default multiplier = 4, fetching up to 20 candidates).
2. **Deduplication with Rank Preservation:** Chunks are inspected in descending similarity order. If a document ID has already been seen from a higher-scoring chunk, subsequent redundant chunks are skipped. Only the single highest-similarity chunk for each distinct document is retained.
3. **Exact Schema & Domain Preservation:** All 11 metadata fields are preserved intact. Exact cosine similarities are reported without converting them into pseudo-probabilities. Domain filtering (`where={"Domain": domain}`) is strictly enforced.
4. **Development Set Tuning:** Candidate multipliers (2, 4, 6) were evaluated strictly on a separate held-out development dataset (501 validation queries from `FINAL_MASTER_GOLD.csv`), selecting multiplier 4 as optimal before running once on the test benchmark.

---

## 2. Evaluation Results on Development Dataset (`retrieval_dev_benchmark.csv`)

*Dataset:* 501 queries from `FINAL_MASTER_GOLD.csv` (`Split == 'validation'`), completely disjoint from test cases. 429 answerable queries, 72 knowledge gaps.

| Metric | Dev Raw Chunks | Dev Distinct Docs ($m=2$) | Dev Distinct Docs ($m=4$, Tuned) | Dev Distinct Docs ($m=6$) | Delta ($m=4$ vs Raw) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Queries** | 501 | 501 | 501 | 501 | — |
| **Answerable Queries** | 429 | 429 | 429 | 429 | — |
| **Unanswerable Gaps** | 72 | 72 | 72 | 72 | — |
| **Answerable Recall@1** | 66.67% | 66.67% | 66.67% | 66.67% | **0.00%** (Rank 1 unchanged) |
| **Answerable Recall@3** | 82.75% | 85.31% | **85.31%** | 85.31% | **+2.56%** |
| **Answerable Recall@5** | 88.58% | 92.54% | **93.01%** | 93.01% | **+4.43%** |
| **Answerable MRR** | 0.7500 | 0.7683 | **0.7693** | 0.7693 | **+0.0193** |
| **Mean Latency** | 13.58 ms | 12.06 ms | 12.74 ms | 12.57 ms | **-0.84 ms** |
| **P95 Latency** | 16.38 ms | 14.05 ms | 15.05 ms | 14.69 ms | **-1.33 ms** |
| **Misses in Top 5** | 49 | 32 | **30** | 30 | **-19 misses (-38.8%)** |

*Development Conclusion:* Multiplier $m=4$ delivers peak recall (93.01%) and peak MRR (0.7693). Multiplier $m=6$ offers zero additional recall gain at slightly higher vector query cost. Therefore, $m=4$ was frozen for the final test benchmark run.

---

## 3. Measured Results on Final Held-Out Test Benchmark (`retrieval_benchmark_remediated_v2.csv`)

*Dataset:* 501 held-out test queries, 428 answerable, 73 knowledge gaps.

| Metric | Raw Chunks Mode | Distinct Documents Mode ($m=4$) | Measured Delta |
| :--- | :---: | :---: | :---: |
| **Total Test Queries** | 501 | 501 | — |
| **Answerable Queries** | 428 | 428 | — |
| **Unanswerable Gaps** | 73 | 73 | — |
| **Answerable Recall@1** | 61.92% | **61.92%** | **0.00%** |
| **Answerable Recall@3** | 77.34% | **80.37%** | **+3.03% absolute** |
| **Answerable Recall@5** | 82.24% | **87.38%** | **+5.14% absolute** |
| **Answerable MRR** | 0.6977 | **0.7192** | **+0.0215** |
| **Mean Latency** | 12.25 ms | **17.03 ms** | +4.78 ms |
| **P95 Latency** | 14.27 ms | **27.32 ms** | +13.05 ms |
| **Misses in Top 5 (Answerable)** | 76 | **54** | **-22 misses (-28.9%)** |

### Per-Domain Recall@5 on Answerable Test Queries:
* **E-commerce (380 answerable):** **81.84%** $\rightarrow$ **87.63%** (**+5.79% absolute gain**)
* **Education (26 answerable):** **88.46%** $\rightarrow$ **88.46%** (23/26 resolved)
* **Small Business (22 answerable):** **81.82%** $\rightarrow$ **81.82%** (18/22 resolved)

---

## 4. Overall Project Progression (From Initial Baseline to Current State)

| Milestone / Architecture Phase | Recall@1 | Recall@3 | Recall@5 | MRR | Misses in Top 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Original Unaudited Baseline** (`baseline_v1` on contaminated benchmark) | 51.70% | 64.87% | 71.46% | 0.5882 | 143 misses |
| **2. Post-Audit Benchmark Remediation** (Raw chunks on answerable subset) | 61.92% | 77.34% | 82.24% | 0.6977 | 76 misses |
| **3. Distinct-Document Retrieval** (Tuned $m=4$ on answerable subset) | **61.92%** | **80.37%** | **87.38%** | **0.7192** | **54 misses** |
| **Cumulative Impact** | **+10.22%** | **+15.50%** | **+15.92%** | **+0.1310** | **-89 misses (-62.2%)** |

---

## 5. Automated Test Suite Verification

Full test suite executed: **42 passed, 0 failed (100% pass rate)**.
* **8 New Distinct Retriever Tests** ([`tests/test_distinct_retriever.py`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/tests/test_distinct_retriever.py)):
  1. `test_1_distinct_documents_deduplication`: PASSED
  2. `test_2_highest_similarity_chunk_preserved`: PASSED
  3. `test_3_domain_filter_preservation`: PASSED
  4. `test_4_empty_results_and_invalid_inputs`: PASSED
  5. `test_5_candidate_pool_limits_and_multipliers`: PASSED
  6. `test_6_deterministic_ranking_stability`: PASSED
  7. `test_7_result_schema_conformance`: PASSED
  8. `test_8_no_double_counting_in_evaluation_metrics`: PASSED
* **14 RAG Audit Tests** ([`tests/test_rag_audit.py`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/tests/test_rag_audit.py)): PASSED (all 14 criteria)
* **20 System Regression Tests**: PASSED (`test_duplicates`, `test_labels`, `test_leakage`, `test_rag`, `test_relationships`, `test_schema`, `test_splits`).

---

## 6. Exact Metric Definitions

* **Recall@$k$:** Defined as $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{rank}_i \le k)$, where $\text{rank}_i$ is the 1-indexed position of the expected document in the returned results list. If the expected document is absent from top-$k$, $\mathbb{I} = 0$.
* **Mean Reciprocal Rank (MRR):** Defined as $\frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}$ if the expected document is found within top-$k$, else $0$.
* **No Double Counting Guarantee:** In distinct mode, each document appears at most once in `retrieved_ids`. Therefore, an expected document occupies strictly one position $\text{rank}_i$, preventing duplicate chunks of the same document from artificially inflating precision or crowding out subsequent documents.
* **Unanswerable Knowledge Gaps:** For queries where the topic is absent from the knowledge base (`Expected_Document_ID = ""`), recall is reported separately on the Answerable Subset ($N=428$) vs. All Queries ($N=501$).

---

## 7. Remaining Limitations

1. **Top-1 Recall Unchanged (61.92%):** Distinct-document retrieval changes ranks 2 through 5 by removing duplicates of higher-ranked documents. It does not alter Rank 1 because Rank 1 is already the single highest scoring chunk. To improve Rank 1 recall, hybrid lexical-dense retrieval (BM25 + Dense) or cross-encoder reranking should be explored.
2. **54 Remaining Misses in Top 5:** These represent legitimate semantic ranking gaps where the MiniLM bi-encoder scored an adjacent policy higher than the expected policy due to keyword overlap (e.g. queries using the word "restitution" or "fee" where the chunk text lacks exact synonyms).
3. **Slight Latency Tradeoff:** Fetching 20 candidates and deduplicating increased P95 latency from 14.27 ms to 27.32 ms, which remains well within acceptable production SLA boundaries (< 50 ms).
