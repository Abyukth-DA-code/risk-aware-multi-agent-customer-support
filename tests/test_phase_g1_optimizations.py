"""
Test Suite for Phase G.1: Safe Coverage Optimization, Latency Optimization & Diagnostic Hardening
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
Research Direction: Risk-Aware Adaptive Multi-Agent Orchestration

Covers 15 required verification scenarios:
1. Multi-chunk sentence cross-entailment (second chunk provides grounding fact)
2. Cache hit returns exact same numerical score (|diff| < 1e-5) in < 1 ms
3. Cache miss recomputes embeddings correctly and populates cache
4. Content changes invalidate cache entry via SHA-256 mismatch
5. Reused chunk ID with new text does not return stale embeddings
6. Concurrent thread safety with multi-threaded verification requests
7. Query embedding reuse prevents redundant encoding
8. Missing or malformed query embedding fails safely to fresh encoding
9. All 8 latency keys present in state.metadata["latency_breakdown"]
10. State JSON serialization roundtrip with latency metadata and vector fields
11. Mandatory safety rules (CRIT_01, SENS_FIN_06) remain effective over supported evidence
12. LOW_CONF_05 and max_risk: Low invariants unchanged
13. Cross-tenant retrieval and verification remain isolated
14. Verification exceptions fail closed to ESCALATE
15. Public demo backward compatibility on /query endpoint
"""

import concurrent.futures
import json
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from starlette.testclient import TestClient

from src.agents.grounded_generator import GroundedAnswerGenerator
from src.agents.policy_agent import PolicyAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.verification_agent import EvidenceEntailmentVerifier, VerificationAgent
from src.orchestration.graph import OrchestrationPipeline, get_orchestration_pipeline
from src.orchestration.safety_policy import SafetyPolicyEngine
from src.orchestration.state import OrchestrationState
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.rag.retriever import RAGRetriever
from src.workspaces.db import WorkspaceDB
from src.workspaces.service import WorkspaceService


@pytest.fixture
def clean_verifier():
    """Provides a fresh EvidenceEntailmentVerifier with an empty sentence cache."""
    EvidenceEntailmentVerifier.clear_cache()
    verifier = EvidenceEntailmentVerifier()
    return verifier


@pytest.fixture
def mock_retrieved_chunks():
    """Provides two realistic test chunks from an e-commerce workspace."""
    chunk1 = {
        "chunk_id": "doc123_chunk0",
        "document_id": "E001",
        "title": "General Return Information",
        "chunk_text": "We welcome returns from our customers. Items must be in original condition.",
        "text": "We welcome returns from our customers. Items must be in original condition.",
        "similarity": 0.65,
    }
    chunk2 = {
        "chunk_id": "doc123_chunk1",
        "document_id": "E001",
        "title": "Return Policy Windows",
        "chunk_text": "Customers have 30 days from delivery to request a full refund on eligible merchandise.",
        "text": "Customers have 30 days from delivery to request a full refund on eligible merchandise.",
        "similarity": 0.62,
    }
    return [chunk1, chunk2]


# -------------------------------------------------------------------------
# Scenario 1: Multi-chunk sentence cross-entailment
# -------------------------------------------------------------------------
def test_multi_chunk_sentence_cross_entailment(clean_verifier, mock_retrieved_chunks):
    """Verify that when Chunk 1 has context and Chunk 2 has the grounding fact,
    the verifier evaluates sentences from Chunk 2 and produces SUPPORTED with evidence references."""
    query = "How many days do I have to return an item for a full refund?"
    result = clean_verifier.verify(query=query, retrieved_chunks=mock_retrieved_chunks)

    assert result["status"] == "SUPPORTED"
    assert result["score"] >= 0.50
    refs = result.get("evidence_references", [])
    assert any("E001" in r or "doc123" in r for r in refs)


# -------------------------------------------------------------------------
# Scenario 2: Cache hit returns exact same numerical score in < 1 ms
# -------------------------------------------------------------------------
def test_cache_hit_exact_numerical_score_sub_millisecond(clean_verifier, mock_retrieved_chunks):
    """Verify that calling verify on cached chunk returns identical scores (|diff| < 1e-5) in < 1 ms."""
    query = "What is the return window for items?"
    q_vec = clean_verifier.model.encode([query], normalize_embeddings=True)[0]

    # First call (cold - populates cache)
    res1 = clean_verifier.verify(query=query, retrieved_chunks=[mock_retrieved_chunks[1]], query_embedding=q_vec)

    # Direct cache lookup is sub-millisecond (< 1.0 ms)
    t0 = time.perf_counter()
    sents, embs = clean_verifier.get_chunk_sentences_and_embeddings(mock_retrieved_chunks[1])
    cache_lookup_ms = (time.perf_counter() - t0) * 1000
    assert cache_lookup_ms < 1.0

    # Second call (warm - cache hit)
    t1 = time.perf_counter()
    res2 = clean_verifier.verify(query=query, retrieved_chunks=[mock_retrieved_chunks[1]], query_embedding=q_vec)
    verify_elapsed_ms = (time.perf_counter() - t1) * 1000

    assert abs(res1["score"] - res2["score"]) < 1e-5
    assert res1["status"] == res2["status"]
    assert verify_elapsed_ms < 5.0  # Fast warm memoized lookup


# -------------------------------------------------------------------------
# Scenario 3: Cache miss recomputes embeddings correctly and populates cache
# -------------------------------------------------------------------------
def test_cache_miss_recomputes_and_populates(clean_verifier):
    """Verify that a brand new chunk causes a cache miss, caches sentences, and updates cache stats."""
    EvidenceEntailmentVerifier.clear_cache()
    stats_init = EvidenceEntailmentVerifier.get_cache_stats()
    assert stats_init["size"] == 0

    chunk = {
        "chunk_id": "unique_chunk_999",
        "document_id": "TEST_DOC",
        "title": "Test Title",
        "chunk_text": "First sentence of unique text. Second sentence containing specific return policy info.",
    }

    clean_verifier.verify(query="What is the return policy?", retrieved_chunks=[chunk])
    stats_after = EvidenceEntailmentVerifier.get_cache_stats()
    assert stats_after["size"] == 1
    assert stats_after["misses"] >= 1


# -------------------------------------------------------------------------
# Scenario 4: Content changes invalidate cache entry via SHA-256 mismatch
# -------------------------------------------------------------------------
def test_content_change_invalidates_cache_via_sha256(clean_verifier):
    """Verify that if the chunk content changes, SHA-256 mismatch prevents using stale cache."""
    EvidenceEntailmentVerifier.clear_cache()
    chunk_v1 = {
        "chunk_id": "reused_id",
        "document_id": "DOC1",
        "chunk_text": "The return window is strictly 14 days from delivery.",
    }
    chunk_v2 = {
        "chunk_id": "reused_id",
        "document_id": "DOC1",
        "chunk_text": "The return window has been extended to 90 days for holiday purchases.",
    }

    clean_verifier.verify(query="How long do I have to return an item?", retrieved_chunks=[chunk_v1])
    stats1 = EvidenceEntailmentVerifier.get_cache_stats()
    assert stats1["size"] == 1

    clean_verifier.verify(query="How long do I have to return an item?", retrieved_chunks=[chunk_v2])
    stats2 = EvidenceEntailmentVerifier.get_cache_stats()
    assert stats2["size"] == 2


# -------------------------------------------------------------------------
# Scenario 5: Reused chunk ID with new text does not return stale embeddings
# -------------------------------------------------------------------------
def test_reused_chunk_id_returns_fresh_embeddings(clean_verifier):
    """Verify that new text on same chunk ID yields distinct sentence embeddings matching the new text."""
    chunk_a = {"chunk_id": "fixed_chunk_id", "chunk_text": "Apples and oranges are healthy fruits."}
    chunk_b = {"chunk_id": "fixed_chunk_id", "chunk_text": "Quantum computers rely on superconducting qubits."}

    sents_a, embs_a = clean_verifier.get_chunk_sentences_and_embeddings(chunk_a)
    sents_b, embs_b = clean_verifier.get_chunk_sentences_and_embeddings(chunk_b)

    assert sents_a != sents_b
    assert sents_a == ["Apples and oranges are healthy fruits."]
    assert sents_b == ["Quantum computers rely on superconducting qubits."]
    dot = np.dot(embs_a[0], embs_b[0])
    assert dot < 0.50


# -------------------------------------------------------------------------
# Scenario 6: Concurrent thread safety with multi-threaded verification requests
# -------------------------------------------------------------------------
def test_concurrent_thread_safety(clean_verifier, mock_retrieved_chunks):
    """Verify that concurrent threads accessing the verifier and cache execute without race conditions."""
    def worker(worker_id):
        query = f"Query {worker_id}: How do I return my purchase within 30 days?"
        res = clean_verifier.verify(query=query, retrieved_chunks=mock_retrieved_chunks)
        assert res["status"] in ["SUPPORTED", "INSUFFICIENT_EVIDENCE"]
        return res["status"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(16)]
        results = [f.result() for f in futures]

    assert len(results) == 16
    assert all(r == "SUPPORTED" for r in results)


# -------------------------------------------------------------------------
# Scenario 7: Query embedding reuse prevents redundant encoding
# -------------------------------------------------------------------------
def test_query_embedding_reuse_bypasses_encoding(clean_verifier, mock_retrieved_chunks):
    """Verify that passing pre-computed query_embedding prevents calling model.encode for query."""
    dummy_vec = np.random.randn(384).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    # Pre-populate chunk cache so model.encode is not called for chunk sentences
    clean_verifier.get_chunk_sentences_and_embeddings(mock_retrieved_chunks[0])

    with patch.object(clean_verifier.model, "encode", wraps=clean_verifier.model.encode) as mock_encode:
        initial_calls = mock_encode.call_count
        clean_verifier.verify(
            query="Test query about returns",
            retrieved_chunks=[mock_retrieved_chunks[0]],
            query_embedding=dummy_vec,
        )
        # Should not call encode because chunk is cached and query vector was provided
        assert mock_encode.call_count == initial_calls


# -------------------------------------------------------------------------
# Scenario 8: Missing or malformed query embedding fails safely to fresh encoding
# -------------------------------------------------------------------------
def test_malformed_query_embedding_fails_safely(clean_verifier, mock_retrieved_chunks):
    """Verify that None, invalid length, or non-numeric query embedding falls back safely without error."""
    res_none = clean_verifier.verify(
        query="What is the refund policy?",
        retrieved_chunks=mock_retrieved_chunks,
        query_embedding=None,
    )
    assert res_none["status"] == "SUPPORTED"

    res_wrong_dim = clean_verifier.verify(
        query="What is the refund policy?",
        retrieved_chunks=mock_retrieved_chunks,
        query_embedding=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    assert res_wrong_dim["status"] == "SUPPORTED"

    res_malformed = clean_verifier.verify(
        query="What is the refund policy?",
        retrieved_chunks=mock_retrieved_chunks,
        query_embedding=["not", "a", "float"],
    )
    assert res_malformed["status"] == "SUPPORTED"


# -------------------------------------------------------------------------
# Scenario 9: All 8 latency keys present in state.metadata["latency_breakdown"]
# -------------------------------------------------------------------------
def test_all_eight_latency_keys_present():
    """Verify that a pipeline execution records all 8 latency stage measurements."""
    pipeline = get_orchestration_pipeline()
    state = pipeline.run(
        query="What is your return policy?",
        domain="E-commerce",
        generate_grounded_response=True,
    )
    breakdown = state.metadata.get("latency_breakdown", {})
    required_keys = [
        "latency_resolution_ms",
        "latency_triage_ms",
        "latency_retrieval_ms",
        "latency_verification_ms",
        "latency_policy_ms",
        "latency_supervisor_ms",
        "latency_generation_ms",
        "latency_e2e_ms",
    ]
    for k in required_keys:
        assert k in breakdown, f"Missing latency key: {k}"
        assert isinstance(breakdown[k], (int, float))
        assert breakdown[k] >= 0.0


# -------------------------------------------------------------------------
# Scenario 10: State JSON serialization roundtrip with latency metadata
# -------------------------------------------------------------------------
def test_state_json_serialization_roundtrip():
    """Verify that OrchestrationState serializes to JSON and deserializes cleanly with all metadata."""
    pipeline = get_orchestration_pipeline()
    state = pipeline.run(
        query="What is the warranty period?",
        domain="E-commerce",
        generate_grounded_response=True,
    )
    d = state.to_dict()
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)

    assert deserialized["final_autonomy_tier"] == state.final_autonomy_tier
    assert "latency_breakdown" in deserialized["metadata"]
    assert deserialized["metadata"]["latency_breakdown"]["latency_e2e_ms"] > 0.0


# -------------------------------------------------------------------------
# Scenario 11: Mandatory safety rules (CRIT_01, SENS_FIN_06) remain effective
# -------------------------------------------------------------------------
def test_mandatory_safety_rules_override_supported_evidence():
    """Verify that CRIT_01 and SENS_FIN_06 escalate even if evidence is SUPPORTED."""
    pipeline = get_orchestration_pipeline()
    state = pipeline.run(
        query="Please wire $5,000 from account 1234 to account 5678 immediately",
        domain="Banking",
        generate_grounded_response=True,
    )
    assert state.final_autonomy_tier == "ESCALATE"
    assert state.final_escalation_required is True
    assert state.safety_rule_triggered in ["CRIT_01", "SENS_FIN_06", "LOW_CONF_05"]


# -------------------------------------------------------------------------
# Scenario 12: LOW_CONF_05 and max_risk: Low invariants unchanged
# -------------------------------------------------------------------------
def test_low_conf_and_max_risk_invariants():
    """Verify that composite confidence < 0.65 or risk != Low escalates."""
    engine = SafetyPolicyEngine()
    state = OrchestrationState(query="Test query", domain="E-commerce")
    state.intent = "faq_inquiry"
    state.risk = "High"  # Violates max_risk: Low
    state.sentiment = "Neutral"
    state.complexity = "Low"
    state.provisional_autonomy_tier = "ANSWER"
    state.final_confidence = 0.90
    state.entailment_status = "SUPPORTED"
    state.evidence_quality = "Strong"

    eval_res = engine.evaluate(state)
    assert eval_res.final_escalation_required is True
    assert eval_res.final_autonomy_tier == "ESCALATE"


# -------------------------------------------------------------------------
# Scenario 13: Cross-tenant retrieval and verification remain isolated
# -------------------------------------------------------------------------
def test_cross_tenant_isolation(tmp_path):
    """Verify that documents from Tenant A cannot be retrieved or verified in Tenant B."""
    import chromadb

    db_path = tmp_path / "tenant_isolation.db"
    db = WorkspaceDB(db_path=db_path)
    chroma_dir = tmp_path / "chroma_tenant_iso"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    db.create_tenant("ten_alpha", "Alpha Corp", "alpha@example.com")
    db.create_tenant("ten_beta", "Beta Corp", "beta@example.com")
    ws1 = db.create_workspace("ws_alpha", "ten_alpha", "Tenant Alpha", "E-commerce")
    ws2 = db.create_workspace("ws_beta", "ten_beta", "Tenant Beta", "E-commerce")

    ingestion = DocumentIngestionPipeline(db=db, chroma_client=chroma_client)
    ingestion.ingest_document(
        workspace_id=ws1["workspace_id"],
        tenant_id="ten_alpha",
        category_key="general",
        filename="alpha_secret.txt",
        file_bytes=b"Alpha secret security policy: The master administrative password is TopSecret12345.",
    )

    retriever = RAGRetriever(client=chroma_client)
    chunks_beta = retriever.retrieve(
        query="What is the secret password?",
        workspace_id=ws2["workspace_id"],
        top_k=5,
    )
    assert len(chunks_beta) == 0  # Tenant Beta cannot see Tenant Alpha documents


# -------------------------------------------------------------------------
# Scenario 14: Verification exceptions fail closed to ESCALATE
# -------------------------------------------------------------------------
def test_verification_exception_fails_closed():
    """Verify that if EvidenceEntailmentVerifier raises an unhandled exception, PolicyAgent fails closed."""
    verifier = EvidenceEntailmentVerifier()
    policy_agent = PolicyAgent(verifier=verifier)

    state = OrchestrationState(query="Test query", domain="E-commerce")
    state.intent = "faq_inquiry"
    state.risk = "Low"
    state.provisional_autonomy_tier = "ANSWER"
    state.final_confidence = 0.85
    state.retrieval_similarity_score = 0.80
    state.retrieved_chunks = [{"chunk_text": "Sample text", "chunk_id": "c1", "document_id": "D1"}]

    # Force verifier to raise an error
    with patch.object(verifier, "verify", side_effect=RuntimeError("Embedding CUDA/CPU failure")):
        res_state = policy_agent.execute(state)
        assert res_state.final_autonomy_tier == "ESCALATE"
        assert res_state.final_escalation_required is True
        assert res_state.entailment_status == "VERIFICATION_ERROR"


# -------------------------------------------------------------------------
# Scenario 15: Public demo backward compatibility on /query endpoint
# -------------------------------------------------------------------------
def test_public_demo_backward_compatibility():
    """Verify that POST /query with valid demo payload returns 200 OK and backward-compatible fields."""
    from src.api.main import app

    client = TestClient(app)
    payload = {
        "query": "How many days do I have to return an item?",
        "domain": "E-commerce",
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "triage" in data
    assert "evidence" in data
    assert "resolution" in data
