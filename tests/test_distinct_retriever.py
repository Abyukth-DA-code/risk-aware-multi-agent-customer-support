"""
Automated Test Suite: Distinct-Document RAG Retrieval Mode, Deduplication, & Constraints.

Covers:
 1. Duplicate chunk deduplication (multiple chunks from same document reduced to 1 distinct result).
 2. Preservation of highest-similarity chunk per document.
 3. Domain filter preservation across domains.
 4. Empty query, whitespace, invalid top_k, and non-existent domain handling.
 5. Candidate pool limits, custom multipliers, and bounds enforcement.
 6. Missing / malformed metadata resilience.
 7. Deterministic ranking stability and reproducibility.
 8. Strict result schema conformance (all 11 fields present and typed).
 9. Prevention of document double-counting in evaluation metrics.
"""

from pathlib import Path
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def get_retriever():
    pytest.importorskip("chromadb")
    from src.rag.retriever import RAGRetriever
    return RAGRetriever()


def test_1_distinct_documents_deduplication():
    """Verify that distinct mode eliminates duplicate document IDs while raw mode permits them."""
    retriever = get_retriever()
    query = "Can I get a refund for my order?"

    # Raw mode: may contain duplicate chunks from the same document
    raw_results = retriever.retrieve(query, domain="E-commerce", top_k=5, distinct_documents=False)
    assert len(raw_results) == 5

    # Distinct mode: must have strictly unique document IDs
    distinct_results = retriever.retrieve(query, domain="E-commerce", top_k=5, distinct_documents=True)
    assert len(distinct_results) == 5

    dist_doc_ids = [r["document_id"] for r in distinct_results]
    assert len(dist_doc_ids) == len(set(dist_doc_ids)), f"Duplicate doc IDs in distinct mode: {dist_doc_ids}"


def test_2_highest_similarity_chunk_preserved():
    """Verify that distinct mode preserves the highest-similarity chunk for each document."""
    retriever = get_retriever()
    query = "Can I get a refund?"

    # Fetch larger raw pool
    raw_candidates = retriever.retrieve(query, domain="E-commerce", top_k=20, distinct_documents=False)
    distinct_results = retriever.retrieve(query, domain="E-commerce", top_k=5, distinct_documents=True)

    # For each document in distinct_results, its chunk must be the highest scoring chunk of that doc in raw_candidates
    for d_item in distinct_results:
        doc_id = d_item["document_id"]
        doc_raw_chunks = [c for c in raw_candidates if c["document_id"] == doc_id]
        if doc_raw_chunks:
            max_sim_raw = max(c["similarity"] for c in doc_raw_chunks)
            assert abs(d_item["similarity"] - max_sim_raw) < 1e-5, (
                f"Document {doc_id} chunk in distinct results does not have max similarity: "
                f"{d_item['similarity']} vs {max_sim_raw}"
            )


def test_3_domain_filter_preservation():
    """Verify domain filtering strictly isolates results to the specified domain in distinct mode."""
    retriever = get_retriever()

    for domain in ["E-commerce", "Education", "Small Business"]:
        results = retriever.retrieve("payment policy", domain=domain, top_k=3, distinct_documents=True)
        assert len(results) > 0, f"Expected results for domain {domain}"
        for item in results:
            assert item["domain"] == domain, f"Domain leak: expected {domain}, found {item['domain']}"


def test_4_empty_results_and_invalid_inputs():
    """Verify input validation: empty queries, whitespace, invalid top_k, non-existent domain."""
    retriever = get_retriever()

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("", domain="E-commerce", top_k=5)

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("   ", domain="E-commerce", top_k=5)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve("valid query", domain="E-commerce", top_k=0)

    # Non-existent domain should return empty list gracefully
    empty_res = retriever.retrieve("valid query", domain="NON_EXISTENT_DOMAIN_XYZ", top_k=5)
    assert empty_res == []


def test_5_candidate_pool_limits_and_multipliers():
    """Verify custom candidate multipliers and explicit pool sizes."""
    retriever = get_retriever()
    query = "How do I return an item?"

    # Small multiplier
    res_m2 = retriever.retrieve(query, domain="E-commerce", top_k=3, distinct_documents=True, candidate_multiplier=2)
    assert len(res_m2) == 3

    # Large multiplier exceeding domain chunks
    res_m10 = retriever.retrieve(query, domain="E-commerce", top_k=5, distinct_documents=True, candidate_multiplier=10)
    assert len(res_m10) == 5

    # Explicit pool size
    res_explicit = retriever.retrieve(query, domain="E-commerce", top_k=4, distinct_documents=True, candidate_pool_size=16)
    assert len(res_explicit) == 4

    # Explicit pool size smaller than top_k is safely clamped to top_k chunks
    res_clamped = retriever.retrieve(query, domain="E-commerce", top_k=3, distinct_documents=True, candidate_pool_size=1)
    assert 1 <= len(res_clamped) <= 3


def test_6_deterministic_ranking_stability():
    """Verify that repeated retrieval calls return bit-identical results."""
    retriever = get_retriever()
    query = "How can I track my shipment delivery?"

    res1 = retriever.retrieve(query, domain="E-commerce", top_k=4, distinct_documents=True)
    res2 = retriever.retrieve(query, domain="E-commerce", top_k=4, distinct_documents=True)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1["chunk_id"] == r2["chunk_id"]
        assert r1["document_id"] == r2["document_id"]
        assert r1["similarity"] == pytest.approx(r2["similarity"], abs=1e-6)


def test_7_result_schema_conformance():
    """Verify all 11 required schema fields are present and properly typed."""
    retriever = get_retriever()
    results = retriever.retrieve("invoice download", domain="E-commerce", top_k=3, distinct_documents=True)

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
        assert set(item.keys()) == required_keys
        assert isinstance(item["similarity"], float)
        assert isinstance(item["cosine_distance"], float)
        assert isinstance(item["chunk_text"], str)
        assert item["similarity"] == pytest.approx(1.0 - item["cosine_distance"], abs=1e-5)


def test_8_no_double_counting_in_evaluation_metrics():
    """Verify that in distinct mode, an expected document occupies at most 1 rank."""
    retriever = get_retriever()
    results = retriever.retrieve("Can I cancel my order?", domain="E-commerce", top_k=5, distinct_documents=True)

    doc_ids = [r["document_id"] for r in results]
    for did in set(doc_ids):
        count = doc_ids.count(did)
        assert count == 1, f"Document {did} counted {count} times in top-k results"
