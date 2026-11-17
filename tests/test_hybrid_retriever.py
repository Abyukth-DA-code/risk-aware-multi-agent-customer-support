"""
Automated Test Suite: Hybrid Lexical-Dense RAG Retrieval Mode, RRF Fusion, & Vocabulary Mismatch.

Covers:
 1. BM25Okapi lexical index correctness, tokenization, and scoring.
 2. Reciprocal Rank Fusion (RRF) logic and rank aggregation.
 3. Distinct-document deduplication under hybrid mode.
 4. Strict domain filter preservation across domains in hybrid mode.
 5. Empty lexical fallback to dense candidates.
 6. Edge cases and input validation (empty query, whitespace, invalid top_k, non-existent domain).
 7. Vocabulary mismatch & paraphrase handling ("restitution" vs "refund", "penalties" vs "cancellation").
 8. Schema conformance and type safety for hybrid retrieval outputs.
 9. Deterministic ranking stability and bit-identical reproducibility.
"""

from pathlib import Path
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def get_retriever():
    pytest.importorskip("chromadb")
    from src.rag.retriever import RAGRetriever
    return RAGRetriever()


def get_bm25():
    from src.rag.bm25 import BM25OkapiIndex
    return BM25OkapiIndex()


def test_1_bm25_index_basic_retrieval_and_scoring():
    """Verify BM25 index produces positive scores for matching keywords and filters by domain."""
    bm25 = get_bm25()
    results = bm25.retrieve("order return refund", domain="E-commerce", top_k=5)
    assert len(results) > 0
    for r in results:
        assert r["domain"] == "E-commerce"
        assert r["bm25_score"] > 0.0
        assert "chunk_id" in r
        assert "document_id" in r


def test_2_rrf_scoring_and_fusion_logic():
    """Verify that RRF combines ranks according to 1/(k + rank) and rewards dual-modality hits."""
    retriever = get_retriever()
    query = "How do I return an item and get a refund?"
    results = retriever.retrieve(
        query=query,
        domain="E-commerce",
        top_k=5,
        distinct_documents=False,
        hybrid=True,
        rrf_k=60,
    )

    assert len(results) > 0
    # Scores must be sorted descending
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    # Check RRF formula for top item
    top = results[0]
    r_dense = top["dense_rank"]
    r_lex = top["lexical_rank"]
    expected_rrf = (1.0 / (60 + r_dense) if r_dense else 0.0) + (1.0 / (60 + r_lex) if r_lex else 0.0)
    assert top["rrf_score"] == pytest.approx(expected_rrf, abs=1e-6)


def test_3_distinct_documents_in_hybrid_mode():
    """Verify that distinct_documents=True guarantees unique document IDs in hybrid mode."""
    retriever = get_retriever()
    query = "order cancellation and return policy"
    results = retriever.retrieve(
        query=query,
        domain="E-commerce",
        top_k=5,
        distinct_documents=True,
        hybrid=True,
    )

    assert len(results) > 0
    doc_ids = [r["document_id"] for r in results]
    assert len(doc_ids) == len(set(doc_ids)), f"Duplicate document IDs found in distinct hybrid mode: {doc_ids}"


def test_4_domain_isolation_in_hybrid_mode():
    """Verify that domain filtering strictly isolates results to the specified domain in hybrid mode."""
    retriever = get_retriever()
    domains = ["E-commerce", "Education", "Small Business"]

    for d in domains:
        results = retriever.retrieve(
            query="policy terms and guidelines",
            domain=d,
            top_k=4,
            distinct_documents=True,
            hybrid=True,
        )
        assert len(results) > 0
        for item in results:
            assert item["domain"] == d, f"Domain leak: expected {d}, got {item['domain']}"


def test_5_empty_lexical_fallback_to_dense():
    """Verify that when query terms have zero lexical matches, dense retrieval provides fallback candidates."""
    retriever = get_retriever()
    # Query with words completely absent from the lexical corpus
    query = "xenomorphic quantum entanglement synchronizer"
    results = retriever.retrieve(
        query=query,
        domain="E-commerce",
        top_k=3,
        distinct_documents=True,
        hybrid=True,
    )

    assert len(results) > 0
    # All items should have bm25_score == 0 and lexical_rank is None, but valid dense ranks and similarities
    for r in results:
        assert r["bm25_score"] == 0.0
        assert r["lexical_rank"] is None
        assert r["dense_rank"] is not None
        assert r["rrf_score"] > 0.0


def test_6_edge_cases_and_input_validation():
    """Verify input validation under hybrid mode."""
    retriever = get_retriever()

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("", domain="E-commerce", top_k=5, hybrid=True)

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("   ", domain="E-commerce", top_k=5, hybrid=True)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve("valid query", domain="E-commerce", top_k=0, hybrid=True)

    empty_res = retriever.retrieve("valid query", domain="NON_EXISTENT_DOMAIN_XYZ", top_k=5, hybrid=True)
    assert empty_res == []


def test_7_vocabulary_mismatch_and_paraphrase_analysis():
    """
    Test specific vocabulary mismatch and paraphrase cases:
    - 'restitution' versus 'refund'
    - 'penalties' versus 'cancellation'
    """
    retriever = get_retriever()

    # Case 1: 'restitution' (legal synonym for refund)
    query_synonym = "Can I request restitution for defective goods?"
    res_hybrid_synonym = retriever.retrieve(
        query=query_synonym,
        domain="E-commerce",
        top_k=5,
        distinct_documents=True,
        hybrid=True,
    )
    res_dense_synonym = retriever.retrieve(
        query=query_synonym,
        domain="E-commerce",
        top_k=5,
        distinct_documents=True,
        hybrid=False,
    )
    # The expected documents for refunds/returns in E-commerce are E001 (Return) or E002 (Refund)
    hybrid_doc_ids = [r["document_id"] for r in res_hybrid_synonym]
    dense_doc_ids = [r["document_id"] for r in res_dense_synonym]
    assert ("E001" in hybrid_doc_ids or "E002" in hybrid_doc_ids), (
        f"Refund documents missing from hybrid retrieval for 'restitution': {hybrid_doc_ids}"
    )
    assert ("E001" in dense_doc_ids or "E002" in dense_doc_ids), (
        f"Refund documents missing from dense retrieval for 'restitution': {dense_doc_ids}"
    )

    # Case 2: 'penalties' versus 'cancellation'
    query_penalty = "What are the financial penalties for cancelling an order?"
    res_hybrid_penalty = retriever.retrieve(
        query=query_penalty,
        domain="E-commerce",
        top_k=5,
        distinct_documents=True,
        hybrid=True,
    )
    penalty_doc_ids = [r["document_id"] for r in res_hybrid_penalty]
    # E003 is Order Cancellation Policy
    assert "E003" in penalty_doc_ids, f"Order Cancellation document E003 missing for penalty query: {penalty_doc_ids}"


def test_8_hybrid_result_schema_conformance():
    """Verify all 11 required schema fields plus hybrid diagnostics are present and typed."""
    retriever = get_retriever()
    results = retriever.retrieve(
        query="invoice download help",
        domain="Small Business",
        top_k=3,
        distinct_documents=True,
        hybrid=True,
    )

    required_keys = {
        "chunk_id",
        "document_id",
        "domain",
        "title",
        "authority",
        "freshness_status",
        "document_type",
        "retrieval_priority",
        "chunk_text",
        "cosine_distance",
        "similarity",
    }

    assert len(results) > 0
    for item in results:
        # All 11 baseline required keys must be present
        assert required_keys.issubset(set(item.keys()))
        assert isinstance(item["similarity"], float)
        assert isinstance(item["cosine_distance"], float)
        assert isinstance(item["rrf_score"], float)
        assert isinstance(item["bm25_score"], float)
        assert isinstance(item["chunk_text"], str)


def test_9_deterministic_ranking_stability_in_hybrid():
    """Verify that repeated hybrid retrieval calls yield bit-identical rankings and scores."""
    retriever = get_retriever()
    query = "How to request attendance shortage condonation"

    res1 = retriever.retrieve(query, domain="Education", top_k=4, distinct_documents=True, hybrid=True)
    res2 = retriever.retrieve(query, domain="Education", top_k=4, distinct_documents=True, hybrid=True)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1["chunk_id"] == r2["chunk_id"]
        assert r1["document_id"] == r2["document_id"]
        assert r1["rrf_score"] == pytest.approx(r2["rrf_score"], abs=1e-8)
        assert r1["similarity"] == pytest.approx(r2["similarity"], abs=1e-8)
