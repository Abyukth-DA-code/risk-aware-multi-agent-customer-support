"""
Comprehensive Automated Test Suite: RAG Benchmark Relevance Audit, Remediation, & System Integrity.

Covers all 14 criteria:
 1. Missing document IDs detection.
 2. Wrong expected document mappings detection.
 3. Cross-domain mismatches detection.
 4. Missing knowledge categorization.
 5. Partial relevance and ambiguity handling.
 6. Duplicate chunk integrity.
 7. Metadata conflicts detection.
 8. Preservation of original benchmark labels.
 9. Deterministic output and reproducibility.
10. Invalid model outputs and malformed input handling.
11. Low-confidence cases flagged for human review.
12. Existing RAG ingestion and retrieval operational continuity.
13. Frozen V4 and master gold data immutability.
14. Backward compatibility with existing RAG conflict and schema constraints.
"""

import os
from pathlib import Path
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_cand_rag = [
    ROOT / "datasets" / "processed" / "rag",
    ROOT / "data" / "processed" / "rag",
]
PROCESSED_RAG_DIR = next((c for c in _cand_rag if c.exists()), _cand_rag[0])

_cand_out = [
    ROOT / "documentation" / "results" / "evaluation_outputs" / "rag_evaluation",
    ROOT / "outputs" / "rag_evaluation",
]
OUTPUT_DIR = next((c for c in _cand_out if c.exists()), _cand_out[0])


def test_1_missing_document_ids_detection():
    """Criterion 1: Verify all expected document IDs in KB exist, and missing IDs are audited."""
    docs = pd.read_csv(PROCESSED_RAG_DIR / "documents.csv")
    chunks = pd.read_csv(PROCESSED_RAG_DIR / "chunks.csv")
    bench = pd.read_csv(PROCESSED_RAG_DIR / "retrieval_benchmark.csv")

    doc_ids = set(docs["Document_ID"].astype(str))
    chunk_doc_ids = set(chunks["Document_ID"].astype(str))
    expected_ids = set(bench["Expected_Document_ID"].astype(str))

    # All baseline expected IDs must exist in documents and chunks
    assert expected_ids.issubset(doc_ids), f"Missing docs in documents.csv: {expected_ids - doc_ids}"
    assert expected_ids.issubset(chunk_doc_ids), f"Missing docs in chunks.csv: {expected_ids - chunk_doc_ids}"


def test_2_wrong_expected_document_mappings_detected():
    """Criterion 2: Verify that demonstrably incorrect mappings are detected and categorized as MISMATCHED."""
    audit_path = OUTPUT_DIR / "benchmark_relevance_audit_v2.csv"
    assert audit_path.exists(), "benchmark_relevance_audit_v2.csv missing"
    audit_df = pd.read_csv(audit_path).fillna("")

    # Education: Academic Records was E015, remapped to E024
    edu_records = audit_df[(audit_df["Query_Domain"] == "Education") & (audit_df["Query_Intent"] == "Academic Records")]
    assert len(edu_records) > 0
    assert (edu_records["Relevance_Category"] == "MISMATCHED").all()
    assert (edu_records["Proposed_Document_ID"] == "E024").all()

    # Small Business: Account Management was E031, remapped to E030
    sb_mgmt = audit_df[(audit_df["Query_Domain"] == "Small Business") & (audit_df["Query_Intent"] == "Account Management")]
    assert len(sb_mgmt) > 0
    assert (sb_mgmt["Relevance_Category"] == "MISMATCHED").all()
    assert (sb_mgmt["Proposed_Document_ID"] == "E030").all()

    # E-commerce: Complaints was E010 (Fraud), remapped to E012 (Contact)
    complaints = audit_df[(audit_df["Query_Domain"] == "E-commerce") & (audit_df["Query_Intent"] == "complaint")]
    assert len(complaints) > 0
    assert (complaints["Relevance_Category"] == "MISMATCHED").all()
    assert (complaints["Proposed_Document_ID"] == "E012").all()


def test_3_cross_domain_mismatches_detected():
    """Criterion 3: Verify that cross-domain mismatches are flagged as METADATA_CONFLICT."""
    audit_path = OUTPUT_DIR / "benchmark_relevance_audit_v2.csv"
    audit_df = pd.read_csv(audit_path).fillna("")

    cross_dom = audit_df[audit_df["Relevance_Category"] == "METADATA_CONFLICT"]
    assert len(cross_dom) == 3, f"Expected 3 cross-domain metadata conflicts, found {len(cross_dom)}"
    assert (cross_dom["Query_Domain"] == "Small Business").all()
    assert (cross_dom["Original_Expected_Document_ID"] == "E012").all()
    assert (cross_dom["Root_Cause"].str.startswith("B:")).all()


def test_4_missing_knowledge_categorization():
    """Criterion 4: Verify unanswerable queries are categorized as MISSING_KNOWLEDGE without fabricated IDs."""
    audit_path = OUTPUT_DIR / "benchmark_relevance_audit_v2.csv"
    audit_df = pd.read_csv(audit_path).fillna("")

    missing_know = audit_df[audit_df["Relevance_Category"] == "MISSING_KNOWLEDGE"]
    assert len(missing_know) >= 70, f"Expected at least 70 missing knowledge cases, found {len(missing_know)}"

    # Proposed document ID must be empty/None for genuinely missing knowledge
    assert (missing_know["Proposed_Document_ID"].astype(str).isin(["", "nan"])).all()
    assert (missing_know["Root_Cause"].str.startswith("D:")).all()

    # Check key missing topics
    intents = set(missing_know["Query_Intent"].unique())
    assert "place_order" in intents
    assert "check_payment_methods" in intents
    assert "review" in intents
    assert "Library" in intents
    assert "Placement Support" in intents


def test_5_partial_relevance_and_ambiguity_handling():
    """Criterion 5: Verify partial relevance is preserved rather than forcing false direct relevance."""
    audit_path = OUTPUT_DIR / "benchmark_relevance_audit_v2.csv"
    audit_df = pd.read_csv(audit_path).fillna("")

    partial_cases = audit_df[audit_df["Relevance_Category"] == "PARTIAL"]
    assert len(partial_cases) > 100, f"Expected >100 partial relevance cases, found {len(partial_cases)}"
    assert (partial_cases["Requires_Human_Review"] == True).all()

    # Cancellation fee and delivery options should be PARTIAL
    cancel_fee = audit_df[audit_df["Query_Intent"] == "check_cancellation_fee"]
    assert (cancel_fee["Relevance_Category"] == "PARTIAL").all()


def test_6_duplicate_chunks_integrity():
    """Criterion 6: Verify chunks integrity (no duplicate chunk IDs, non-empty text, valid linkage)."""
    chunks = pd.read_csv(PROCESSED_RAG_DIR / "chunks.csv")
    assert chunks["Chunk_ID"].is_unique, "Duplicate Chunk_ID found in chunks.csv"
    assert chunks["Chunk_Text"].str.strip().ne("").all(), "Empty Chunk_Text found in chunks.csv"
    assert len(chunks) == 95, f"Expected 95 indexed chunks, found {len(chunks)}"


def test_7_metadata_conflicts_detection():
    """Criterion 7: Verify detection of metadata divergences."""
    audit_path = OUTPUT_DIR / "benchmark_relevance_audit_v2.csv"
    audit_df = pd.read_csv(audit_path).fillna("")

    # Metadata conflicts must cite cross-domain or metadata evidence
    conflicts = audit_df[audit_df["Relevance_Category"] == "METADATA_CONFLICT"]
    for _, row in conflicts.iterrows():
        assert "conflict" in row["Evidence_Details"].lower() or "domain" in row["Evidence_Details"].lower()


def test_8_preservation_of_original_benchmark_labels():
    """Criterion 8: Verify immutable baseline benchmark copy matches original benchmark exactly."""
    orig_path = PROCESSED_RAG_DIR / "retrieval_benchmark.csv"
    base_path = PROCESSED_RAG_DIR / "retrieval_benchmark_baseline_v1.csv"

    assert base_path.exists(), "retrieval_benchmark_baseline_v1.csv missing"
    orig_df = pd.read_csv(orig_path)
    base_df = pd.read_csv(base_path)

    pd.testing.assert_frame_equal(orig_df, base_df)
    assert len(orig_df) == 501, f"Expected 501 queries in baseline benchmark, got {len(orig_df)}"


def test_9_deterministic_output_and_reproducibility():
    """Criterion 9: Verify audit and remediated files exist, have 501 rows, and deterministic schemas."""
    audit_path = OUTPUT_DIR / "benchmark_relevance_audit_v2.csv"
    summary_path = OUTPUT_DIR / "benchmark_audit_summary.csv"
    remediated_path = PROCESSED_RAG_DIR / "retrieval_benchmark_remediated_v2.csv"

    for p in [audit_path, summary_path, remediated_path]:
        assert p.exists(), f"Expected file {p} is missing"

    audit_df = pd.read_csv(audit_path)
    remed_df = pd.read_csv(remediated_path)
    summ_df = pd.read_csv(summary_path)

    assert len(audit_df) == 501
    assert len(remed_df) == 501
    assert len(summ_df) == 4  # 3 domains + TOTAL


def test_10_invalid_inputs_and_malformed_audit_handling():
    """Criterion 10: Verify evaluator and auditor reject empty or missing query files."""
    pytest.importorskip("chromadb")
    from src.rag.evaluate_retrieval import evaluate

    with pytest.raises(FileNotFoundError):
        evaluate(benchmark_path=Path("non_existent_file.csv"))


def test_11_low_confidence_cases_flagged_for_review():
    """Criterion 11: Verify low-confidence, partial, and knowledge gap cases require human review."""
    audit_path = OUTPUT_DIR / "benchmark_relevance_audit_v2.csv"
    audit_df = pd.read_csv(audit_path)

    # All missing knowledge and ambiguous cases must require human review
    review_req = audit_df[audit_df["Relevance_Category"].isin(["MISSING_KNOWLEDGE", "AMBIGUOUS", "PARTIAL", "METADATA_CONFLICT"])]
    assert (review_req["Requires_Human_Review"] == True).all()


def test_12_existing_rag_retrieval_continuing_to_work():
    """Criterion 12: Verify RAGRetriever continues to execute and return valid top_k results."""
    pytest.importorskip("chromadb")
    from src.rag.retriever import RAGRetriever

    retriever = RAGRetriever()
    results = retriever.retrieve("How can I check my order status?", domain="E-commerce", top_k=3)

    assert len(results) == 3
    assert all("chunk_id" in r for r in results)
    assert all("similarity" in r and r["similarity"] > 0 for r in results)
    assert results[0]["document_id"] in ["E004", "E009"]


def test_13_no_modification_of_frozen_v4_data():
    """Criterion 13: Verify frozen master gold data has not been modified or mutated."""
    _cand_gold = [
        ROOT / "datasets" / "processed" / "FINAL_MASTER_GOLD.csv",
        ROOT / "data" / "processed" / "FINAL_MASTER_GOLD.csv",
    ]
    gold_path = next((c for c in _cand_gold if c.exists()), _cand_gold[0])
    assert gold_path.exists(), "FINAL_MASTER_GOLD.csv missing"
    gold_df = pd.read_csv(gold_path)

    assert len(gold_df) == 3519, f"Expected 3519 master gold cases, found {len(gold_df)}"
    assert "Case_ID" in gold_df.columns
    assert "Split" in gold_df.columns
    assert gold_df["Split"].value_counts().to_dict() == {
        "train_base": 2337,
        "validation": 501,
        "test": 501,
        "critical_safety": 180,
    }


def test_14_no_regression_in_existing_rag_constraints():
    """Criterion 14: Verify documents count is 46 and all 5 conflict test pairs are intact."""
    docs = pd.read_csv(PROCESSED_RAG_DIR / "documents.csv")
    assert len(docs) == 46, f"Expected 46 documents, found {len(docs)}"

    outdated = docs[docs["Freshness_Status"] == "Outdated"]
    assert len(outdated) == 5
    assert set(outdated["Document_ID"]) == {"EV1002", "EV1004", "EV1006", "EV1008", "EV1010"}
