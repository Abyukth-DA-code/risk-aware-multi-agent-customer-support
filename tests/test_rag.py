"""
Automated Test Suite: RAG Knowledge Base Integrity & Conflict Preservation Constraints
"""

import os
import pytest
import pandas as pd

from pathlib import Path
_cand_proc = Path(__file__).resolve().parents[1] / "datasets" / "processed"
PROCESSED_DIR = str(_cand_proc) if _cand_proc.exists() else os.path.join("data", "processed")

def test_rag_documents_integrity():
    p = os.path.join(PROCESSED_DIR, "rag", "documents.csv")
    assert os.path.exists(p), "documents.csv missing"
    df = pd.read_csv(p)
    assert len(df) == 46, f"Expected 46 documents, found {len(df)}"

    required_cols = [
        'Document_ID', 'Domain', 'Document_Type', 'Title', 'Source',
        'Authority', 'Version', 'Effective_Date', 'Content', 'Source_Type',
        'Freshness_Status', 'Retrieval_Priority'
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing column {c} in documents.csv"
        # Content must not be empty
        assert df['Content'].str.strip().ne('').all(), "Empty content in documents.csv"

def test_rag_conflict_pairs_preserved():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "rag", "documents.csv"))
    
    # 5 intentionally outdated conflict documents must be present
    outdated = df[df['Freshness_Status'] == 'Outdated']
    assert len(outdated) == 5, f"Expected 5 outdated documents for conflict testing, got {len(outdated)}"
    
    expected_outdated_ids = {'EV1002', 'EV1004', 'EV1006', 'EV1008', 'EV1010'}
    assert set(outdated['Document_ID']) == expected_outdated_ids, "Outdated conflict test documents altered or missing"

def test_rag_chunks_linkage():
    df_docs = pd.read_csv(os.path.join(PROCESSED_DIR, "rag", "documents.csv"))
    df_chunks = pd.read_csv(os.path.join(PROCESSED_DIR, "rag", "chunks.csv"))

    assert len(df_chunks) >= len(df_docs), "Chunks count less than document count"
    assert df_chunks['Document_ID'].isin(df_docs['Document_ID']).all(), "Orphan chunk found in chunks.csv"
    assert df_chunks['Chunk_Text'].str.strip().ne('').all(), "Empty chunk text found"
