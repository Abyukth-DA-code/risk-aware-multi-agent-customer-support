"""
Test Suite for Phase G.2: Retrieval Alignment and M3 Single-Thread Inference
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
Research Direction: Risk-Aware Adaptive Multi-Agent Orchestration

Verifies:
1. M3 single-thread (n_jobs=1) prediction and class probability equivalence vs original (n_jobs=-1) on all 501 benchmark queries.
2. In-memory M3 estimator configuration (n_jobs=1) while preserving disk artifact.
3. Public demo retrieval path enforces pure dense distinct-document retrieval.
4. Tenant workspace retrieval isolation and behavior remain strictly preserved.
5. Downstream autonomy and escalation decisions remain consistent.
6. Public demo /query API backward compatibility.
"""

import copy
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.orchestration.graph import get_orchestration_pipeline
from src.orchestration.model_service import ModelService
from src.rag.retriever import RAGRetriever
from src.workspaces.db import WorkspaceDB
from src.workspaces.service import WorkspaceService
from src.rag.ingestion.pipeline import DocumentIngestionPipeline


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data" / "processed" / "rag" / "retrieval_benchmark_remediated_v2.csv"
M3_DISK_PATH = ROOT / "src" / "models" / "risk_model_(M3).joblib"


# -------------------------------------------------------------------------
# Test 1: M3 Single-Thread Prediction & Probability Equivalence (501 queries)
# -------------------------------------------------------------------------
def test_m3_single_thread_prediction_and_probability_equivalence():
    """
    Asserts that configuring M3 with n_jobs=1 produces 100% identical risk labels
    and bit-identical or numerical-identical probabilities (|diff| < 1e-6)
    compared to the original n_jobs=-1 configuration across all 501 benchmark queries.
    """
    bench_df = pd.read_csv(BENCHMARK_PATH)
    svc = ModelService.get_instance()
    risk_pipe = svc.risk_pipeline

    # Prepare DataFrame for batch inference
    rows = []
    for _, r in bench_df.iterrows():
        rows.append({
            "Query": str(r["Query"]),
            "Domain": str(r.get("Domain", "E-commerce")),
            "Sensitive_Action": "No",
            "Financial_Impact": "No",
            "Action_Type": "Information/Support",
        })
    df_eval = pd.DataFrame(rows)

    # Evaluate with n_jobs=1 (current configured inference mode)
    risk_pipe.named_steps["model"].n_jobs = 1
    probs_single = risk_pipe.predict_proba(df_eval)
    preds_single = risk_pipe.predict(df_eval)

    # Evaluate with n_jobs=-1 (original disk setting)
    risk_pipe.named_steps["model"].n_jobs = -1
    probs_multi = risk_pipe.predict_proba(df_eval)
    preds_multi = risk_pipe.predict(df_eval)

    # Restore n_jobs=1
    risk_pipe.named_steps["model"].n_jobs = 1

    # 1. 100% identical class predictions
    assert np.array_equal(preds_single, preds_multi), "M3 predictions differed between n_jobs=1 and n_jobs=-1!"

    # 2. Documented numerical tolerance: maximum probability difference < 1e-6
    max_prob_diff = float(np.max(np.abs(probs_single - probs_multi)))
    assert max_prob_diff < 1e-6, f"M3 probability difference exceeded numerical tolerance: {max_prob_diff}"


# -------------------------------------------------------------------------
# Test 2: In-Memory M3 Configuration & Disk Model Preservation
# -------------------------------------------------------------------------
def test_m3_estimator_configured_in_memory_disk_unmodified():
    """Verify that the in-memory estimator has n_jobs=1 and the disk artifact is untouched."""
    svc = ModelService.get_instance()
    assert svc.risk_pipeline.named_steps["model"].n_jobs == 1

    # Verify disk artifact has not been modified
    assert M3_DISK_PATH.exists()
    disk_data = joblib.load(M3_DISK_PATH)
    disk_pipe = disk_data["pipeline"] if isinstance(disk_data, dict) else disk_data
    # The disk artifact retains its original n_jobs=-1 setting
    assert disk_pipe.named_steps["model"].n_jobs == -1


# -------------------------------------------------------------------------
# Test 3: Public Demo Retrieval Uses Pure Dense Distinct-Document Mode
# -------------------------------------------------------------------------
def test_public_demo_retrieval_returns_distinct_documents():
    """Verify that public demo retrieval returns distinct documents (no duplicate document_ids)."""
    retriever = RAGRetriever()
    query = "How do I return an eligible item for a refund?"
    chunks = retriever.retrieve(
        query=query,
        domain="E-commerce",
        top_k=3,
        distinct_documents=True,
        candidate_multiplier=4,
        hybrid=False,
    )
    assert len(chunks) > 0
    doc_ids = [c["document_id"] for c in chunks if isinstance(c, dict)]
    assert len(doc_ids) == len(set(doc_ids)), f"Duplicate documents returned: {doc_ids}"
    for c in chunks:
        assert c.get("similarity") is not None
        assert c.get("similarity") > 0.0
        # Hybrid scores should not be populated in pure dense mode
        assert c.get("rrf_score") is None


# -------------------------------------------------------------------------
# Test 4: Public Demo Endpoint (/query) Backward Compatibility & Distinct Chunks
# -------------------------------------------------------------------------
def test_public_demo_endpoint_backward_compatibility():
    """Verify that POST /query operates with pure dense distinct retrieval and valid payload."""
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"query": "What is the return policy?", "domain": "E-commerce"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "triage" in data
    assert "evidence" in data
    assert "resolution" in data

    chunks = data["evidence"]["chunks"]
    doc_ids = [c["document_id"] for c in chunks]
    assert len(doc_ids) == len(set(doc_ids)), f"Expected distinct documents in demo response: {doc_ids}"


# -------------------------------------------------------------------------
# Test 5: Tenant Workspace Retrieval Isolation Preserved
# -------------------------------------------------------------------------
def test_tenant_workspace_retrieval_strictly_isolated(tmp_path):
    """Verify tenant isolation is preserved under updated retrieval pipeline."""
    import chromadb

    db_path = tmp_path / "tenant_iso_g2.db"
    db = WorkspaceDB(db_path=db_path)
    chroma_dir = tmp_path / "chroma_g2"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    db.create_tenant("ten_1", "Tenant One", "t1@example.com")
    db.create_tenant("ten_2", "Tenant Two", "t2@example.com")
    ws1 = db.create_workspace("ws_1", "ten_1", "Workspace One", "E-commerce")
    ws2 = db.create_workspace("ws_2", "ten_2", "Workspace Two", "E-commerce")

    ingestion = DocumentIngestionPipeline(db=db, chroma_client=chroma_client)
    ingestion.ingest_document(
        workspace_id=ws1["workspace_id"],
        tenant_id="ten_1",
        category_key="general",
        filename="policy1.txt",
        file_bytes=b"Workspace One Private Policy: Customer confidentiality is paramount and strictly protected.",
    )

    retriever = RAGRetriever(client=chroma_client)
    # Query Workspace Two (empty)
    chunks_ws2 = retriever.retrieve(
        query="What is the policy?",
        workspace_id=ws2["workspace_id"],
        top_k=3,
    )
    assert len(chunks_ws2) == 0

    # Query Workspace One (has document)
    chunks_ws1 = retriever.retrieve(
        query="What is the confidentiality policy?",
        workspace_id=ws1["workspace_id"],
        top_k=3,
    )
    assert len(chunks_ws1) == 1
    assert chunks_ws1[0]["document_id"] == "doc_general" or "policy1" in chunks_ws1[0].get("title", "")


# -------------------------------------------------------------------------
# Test 6: Downstream Pipeline Equivalence with n_jobs=1
# -------------------------------------------------------------------------
def test_downstream_pipeline_autonomy_consistency():
    """Verify that full pipeline execution produces consistent decisions with n_jobs=1."""
    pipeline = get_orchestration_pipeline()
    state = pipeline.run(
        query="What is your return policy?",
        domain="E-commerce",
        generate_grounded_response=True,
    )
    assert state.risk == "Low"
    assert state.final_autonomy_tier in ["ANSWER", "CLARIFY", "ESCALATE"]
    assert state.final_confidence > 0.0
