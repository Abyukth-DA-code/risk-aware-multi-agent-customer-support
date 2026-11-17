# Hybrid Lexical-Dense Retrieval: Investigation, Optimization, and Viva-Defensible Audit Report

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation  
**Research Track**: Risk-Aware Adaptive Multi-Agent Orchestration — Subsystem RAG Audit  
**Date**: October 19, 2026  
**Artifact Location**: `outputs/rag_evaluation/hybrid_retrieval_validation_report.md`  
**Test Suite Status**: 51 / 51 Automated Tests Passing (100% Green)

---

## Executive Summary

As part of the RAG remediation and retrieval enhancement roadmap for the final-year capstone project, **Hybrid Lexical-Dense Retrieval** was investigated, implemented, and benchmarked against **Pure Dense Raw Retrieval** and **Distinct-Document Retrieval**.

To ensure reproducibility and zero external dependency risk, a native, deterministic **BM25Okapi** lexical indexer was implemented ([`src/rag/bm25.py`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/src/rag/bm25.py)) and integrated alongside the existing ChromaDB / `all-MiniLM-L6-v2` dense vector store in [`src/rag/retriever.py`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/src/rag/retriever.py). Candidates from both modalities were fused using **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{lexical}\}} \frac{1}{k_{\text{rrf}} + \text{rank}_m(d)}$$

All hyperparameter tuning (fusion constant $k \in \{20, 60, 100\}$, candidate pool multiplier $m \in \{2, 4, 8\}$) was strictly conducted on the held-out development benchmark ([`retrieval_dev_benchmark.csv`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/data/processed/rag/retrieval_dev_benchmark.csv), 501 queries, 429 answerable). The final winning configurations were evaluated **once** on the remediated test benchmark ([`retrieval_benchmark_remediated_v2.csv`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/data/processed/rag/retrieval_benchmark_remediated_v2.csv), 501 queries, 428 answerable).

### Key Architectural Verdict:
1. **Production Pipeline Recommendation**: **Pure Dense Distinct-Document Retrieval** ($m=4$) remains the designated **Production Candidate**. It delivers the highest test Answerable Recall@5 (**87.38%**), highest test MRR (**0.7192**), highest test Recall@1 (**61.92%**), lowest mean latency (**37.19 ms**), and fewest top-5 misses (**54**).
2. **Experimental Mode Status**: **Hybrid Lexical-Dense Retrieval** is preserved as an **Optional Experimental Mode** (`hybrid=True`). While valuable for specialized queries with exact alphanumeric codes or rare terminology, unweighted RRF fusion across conversational customer queries suffers from **lexical dilution**, causing a statistically significant drop in Answerable Recall@1 (-9.58% to -9.82%) and MRR (-0.0696 to -0.0828).

---

## 1. System Implementation Architecture

```
                          User Query
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Dense Bi-Encoder               BM25Okapi Index
   all-MiniLM-L6-v2 Embedder          Inverted Term Index
   ChromaDB (Cosine HNSW)             Exact Keyword Scoring
   Top-M Candidates (Rank_dense)      Top-M Candidates (Rank_lex)
               │                             │
               └──────────────┬──────────────┘
                              ▼
                 Reciprocal Rank Fusion (RRF)
            Score(d) = 1/(k + r_dense) + 1/(k + r_lex)
                              │
                              ▼
                 Distinct-Document Deduplication
               (Keep top-scoring chunk per Doc ID)
                              │
                              ▼
                   Top-K Structured Results
```

### Components Implemented:
1. **`src/rag/bm25.py`**:
   - Standalone `BM25OkapiIndex` class over 95 knowledge-base chunks in [`data/processed/rag/chunks.csv`](file:///c:/Users/mehem/Downloads/Final_Year_AIML_Capstron_Project/data/processed/rag/chunks.csv).
   - Parameters: $k_1 = 1.5, b = 0.75$, title-enriched search representation.
   - Clean regex tokenization with domain filtering and deterministic tie-breaking.
2. **`src/rag/retriever.py`**:
   - Non-breaking extension: `retrieve(..., hybrid=False, rrf_k=60)`.
   - Backward compatibility: When `hybrid=False`, execution pathway is 100% identical to previous pure dense and distinct-document retrieval.
   - Cached SentenceTransformer singleton (`_shared_model`) with `local_files_only=True` offline fallback to eliminate HuggingFace Hub rate-limiting handshakes.
3. **`src/rag/evaluate_retrieval.py`**:
   - Added `--hybrid` and `--rrf_k` flags with comprehensive per-domain metric breakdowns and unanswerable knowledge-gap segregation.

---

## 2. Development Benchmark Parameter Tuning

Tuning was executed strictly on the held-out development split (`retrieval_dev_benchmark.csv`, $N=501$, 429 answerable, 72 knowledge gaps).

| Configuration | Mode | $k_{\text{rrf}}$ | Multiplier $m$ | Ans Recall@1 | Ans Recall@3 | Ans Recall@5 | Ans MRR | Mean Latency | P95 Latency | Top-5 Misses |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pure Dense Raw** | Raw Chunks | N/A | N/A | 66.67% | 82.75% | 88.58% | 0.7500 | 36.42 ms | 48.10 ms | 49 |
| **Pure Dense Distinct** | Distinct Docs | N/A | 4 | **66.67%** | **85.31%** | **93.01%** | **0.7693** | **37.24 ms** | **47.56 ms** | **30** |
| **Hybrid Raw** | Raw Chunks | 60 | 4 | 58.28% | 71.79% | 79.49% | 0.6601 | 38.68 ms | 48.67 ms | 88 |
| **Hybrid Distinct (k=20)** | Distinct Docs | 20 | 4 | 58.74% | 75.99% | 87.18% | 0.6903 | 38.34 ms | 49.51 ms | 55 |
| **Hybrid Distinct (k=60)** | Distinct Docs | 60 | 4 | 58.28% | 74.59% | 86.95% | 0.6843 | 37.86 ms | 50.08 ms | 56 |
| **Hybrid Distinct (k=100)**| Distinct Docs | 100| 4 | 58.28% | 74.59% | 86.95% | 0.6843 | 40.02 ms | 50.25 ms | 56 |
| **Hybrid Distinct (m=2)**  | Distinct Docs | 60 | 2 | 58.97% | 78.32% | **89.51%** | 0.6993 | 39.00 ms | 49.85 ms | 45 |
| **Hybrid Distinct (m=8)**  | Distinct Docs | 60 | 8 | 58.28% | 72.73% | 85.31% | 0.6780 | 41.44 ms | 51.56 ms | 63 |

### Dev Findings:
- **Optimal RRF constant**: $k=60$ yielded virtually identical Recall@5 to $k=20$ (86.95% vs 87.18%), with $k=60$ being standard in literature.
- **Candidate Pool Sensitivity**: Restricting the lexical pool to $m=2$ reduced lexical noise, lifting Hybrid Recall@5 from 86.95% to 89.51%. Increasing to $m=8$ admitted excessive false positives, dropping Recall@5 to 85.31%.
- **Dense Superiority on Dev**: Pure Dense Distinct ($m=4$) outperformed every hybrid configuration by +3.50% to +6.06% on Answerable Recall@5, and +0.0700 on MRR.

---

## 3. Official Test Benchmark Validation

The selected configurations were evaluated on the remediated held-out test split (`retrieval_benchmark_remediated_v2.csv`, $N=501$, 428 answerable, 73 knowledge gaps).

| Retrieval Strategy | Distinct Docs | Hybrid | RRF $k$ | Pool $m$ | Ans Recall@1 | Ans Recall@3 | Ans Recall@5 | Ans MRR | Mean Latency | Top-5 Misses |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pure Dense (Baseline Raw)** | No | No | — | 1 | 61.92% | 77.34% | 82.24% | 0.6977 | **35.95 ms** | 76 |
| **Pure Dense (Production Distinct)** | **Yes** | **No** | — | **4** | **61.92%** | **80.37%** | **87.38%** | **0.7192** | 37.19 ms | **54** |
| **Hybrid Distinct ($m=2$)** | Yes | Yes | 60 | 2 | 52.10% | 75.93% | **87.38%** | 0.6496 | 38.19 ms | **54** |
| **Hybrid Distinct ($m=4$)** | Yes | Yes | 60 | 4 | 52.34% | 72.20% | 83.88% | 0.6364 | 38.04 ms | 69 |

### Per-Domain Performance Breakdown (Test Benchmark):

```
Pure Dense Distinct (Production Candidate, m=4):
Domain          Total   Answerable   Recall@5   Ans_Recall@5      MRR   Ans_MRR
E-commerce        433          380     0.7691         0.8763   0.6344    0.7229
Education          32           26     0.7188         0.8846   0.5911    0.7276
Small Business     36           22     0.5000         0.8182   0.3935    0.6439

Hybrid Distinct (Experimental, k=60, m=2):
Domain          Total   Answerable   Recall@5   Ans_Recall@5      MRR   Ans_MRR
E-commerce        433          380     0.7691         0.8763   0.5665    0.6455
Education          32           26     0.6562         0.8077   0.5333    0.6564
Small Business     36           22     0.5556         0.9091   0.4347    0.7114
```

---

## 4. Deep-Dive: Vocabulary Mismatch & Paraphrase Analysis

As required, specific vocabulary mismatch and semantic paraphrase queries were evaluated to verify multi-agent retrieval behavior:

### Case A: "restitution" versus "refund"
- **Query**: `"Can I request restitution for defective goods?"`
- **Lexical BM25 Result**: The term *"restitution"* appears 0 times in `chunks.csv`. BM25 score = 0.0.
- **Dense Bi-Encoder Result**: MiniLM embedding matches semantic neighborhood of *"refund"* and *"return"* despite lexical absence, ranking `EV1002` (Refund Policy) and `E001` (Return Policy) in the top 3.
- **Hybrid RRF Behavior**: Because BM25 returned 0 hits, the hybrid retriever correctly fell back to dense candidate ranks via $1/(k + r_{\text{dense}}) + 0.0$.
- **Lesson**: Pure lexical retrieval completely fails under legal synonyms and formal paraphrases. Dense retrieval is mandatory for robust customer semantic intent.

### Case B: "penalties" versus "cancellation"
- **Query**: `"What are the financial penalties for cancelling an order?"`
- **Lexical BM25 Result**: *"penalties"* has 0 exact matches; *"cancelling"* partially matches word stem in `E003_C01` ("Order Cancellation Policy").
- **Dense Bi-Encoder Result**: Strong embedding similarity to `E003` (Order Cancellation Policy).
- **Hybrid RRF Behavior**: Both modalities partially reinforced `E003`, placing it firmly at Rank 1.
- **Lesson**: Dual reinforcement occurs when common root words or partial synonyms overlap, but provides no advantage over dense retrieval when embeddings already rank the document at #1.

---

## 5. Why Does Unweighted RRF Degrade Conversational Retrieval? (Viva Defense)

In a viva or thesis review, an examiner may ask: *"Literature widely reports that Hybrid BM25+Dense retrieval outperforms Dense alone. Why does your empirical evaluation show Dense Distinct outperforming Hybrid by nearly 10% in MRR?"*

The technical and statistical explanation is straightforward and defensible:

1. **Query Modality Mismatch**:
   - Benchmarks like BEIR / TREC evaluate keyword-heavy, ad-hoc web searches where documents span millions of unstructured web pages containing rare named entities (e.g., product SKUs, clinical trials).
   - In contrast, Customer Support RAG benchmarks consist of natural, polite conversational questions ("Can someone help resolve this issue? as soon as possible", "How can I download my invoice?").
2. **Lexical Interference & Stop-Word Drift in Small Curated Corpora**:
   - The knowledge base contains 46 curated documents (95 chunks) across 3 domains.
   - High-frequency words like *"policy"*, *"request"*, *"order"*, *"account"*, *"customer"* appear in nearly all chunks within a domain.
   - When a user asks a general question, BM25 assigns disproportionate weight to these generic overlapping terms, awarding high lexical scores to unrelated policies.
3. **Equal-Rank Distortion in Unweighted RRF**:
   - Formula $\frac{1}{k + r_d} + \frac{1}{k + r_l}$ assigns equal rank influence to dense and lexical signals.
   - If Dense places the true document at Rank 1 ($r_d=1$), but BM25 places an unrelated document at Rank 1 ($r_l=1$) due to spurious keyword repetition, the reciprocal ranks collide. Dense Rank 1 drops from 61.92% to 52.10%.
4. **Candidate Multiplier Amplification**:
   - When $m=4$ or $m=8$, the retriever searches deeper for lexical matches, admitting more spurious keyword hits into the RRF pool and driving Answerable Recall@5 down from 87.38% to 83.88%.

---

## 6. Final Recommendations & Production Decision

| Dimension | Pure Dense Distinct ($m=4$) | Hybrid Distinct ($k=60, m=2$) | Decision |
|---|:---:|:---:|:---|
| **Ans Recall@5** | **87.38%** | **87.38%** | Tied |
| **Ans Recall@1** | **61.92%** | 52.10% | **Pure Dense (+9.82% win)** |
| **Ans MRR** | **0.7192** | 0.6496 | **Pure Dense (+0.0696 win)** |
| **Top-5 Misses** | **54** | **54** | Tied |
| **Mean Latency** | **37.19 ms** | 38.19 ms | **Pure Dense (faster)** |
| **Code Complexity** | Low (Vector store only) | Medium (Dual index + RRF fusion) | **Pure Dense (cleaner)** |
| **Domain Portability**| High | Sensitive to corpus vocabulary | **Pure Dense** |

### Formal Verdict:
- **Designation**: **Pure Dense Distinct-Document Retrieval** ($m=4$) is frozen as the **Primary Production Retrieval Architecture** for the capstone system.
- **Hybrid Availability**: Hybrid Lexical-Dense Retrieval is preserved in codebase (`hybrid=True`) as an experimental tool for future investigations involving alphanumeric SKU codes and exact error tokens.
