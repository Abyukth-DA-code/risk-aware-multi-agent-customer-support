"""
Automated Test Suite: Duplicate and Primary Key Uniqueness Constraints
"""

import os
import pytest
import pandas as pd

from pathlib import Path
_cand_proc = Path(__file__).resolve().parents[1] / "datasets" / "processed"
PROCESSED_DIR = str(_cand_proc) if _cand_proc.exists() else os.path.join("data", "processed")

def test_primary_key_uniqueness():
    # 1. Master Gold
    df_master = pd.read_csv(os.path.join(PROCESSED_DIR, "FINAL_MASTER_GOLD.csv"))
    assert df_master['Case_ID'].duplicated().sum() == 0, "Duplicate Case_ID found in FINAL_MASTER_GOLD.csv"
    assert df_master.duplicated().sum() == 0, "Duplicate row found in FINAL_MASTER_GOLD.csv"

    # 2. Operational Primary Keys
    ops = {
        "customers.csv": "Customer_ID",
        "orders.csv": "Order_ID",
        "transactions.csv": "Transaction_ID",
        "support_tickets.csv": "Ticket_ID",
        "education_records.csv": "Student_ID",
        "business_records.csv": "Business_Record_ID"
    }
    for fname, pk in ops.items():
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "database", fname))
        assert df[pk].duplicated().sum() == 0, f"Duplicate PK {pk} in {fname}"

    # 3. RAG Documents
    df_rag = pd.read_csv(os.path.join(PROCESSED_DIR, "rag", "documents.csv"))
    assert df_rag['Document_ID'].duplicated().sum() == 0, "Duplicate Document_ID in documents.csv"

    # 4. RAG Chunks
    df_chunks = pd.read_csv(os.path.join(PROCESSED_DIR, "rag", "chunks.csv"))
    assert df_chunks['Chunk_ID'].duplicated().sum() == 0, "Duplicate Chunk_ID in chunks.csv"

    # 5. Memory
    df_mem = pd.read_csv(os.path.join(PROCESSED_DIR, "memory", "memory.csv"))
    assert df_mem['Memory_ID'].duplicated().sum() == 0, "Duplicate Memory_ID in memory.csv"

def test_evaluation_splits_no_internal_duplicates():
    # Validation and Test sets must contain ZERO duplicate queries
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "intent", "validation.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "intent", "test.csv"))
    crit_df = pd.read_csv(os.path.join(PROCESSED_DIR, "safety", "critical_safety.csv"))

    assert val_df['Case_ID'].duplicated().sum() == 0, "Duplicate Case_ID in Validation set"
    assert val_df['Query'].duplicated().sum() == 0, "Duplicate Query in Validation set"

    assert test_df['Case_ID'].duplicated().sum() == 0, "Duplicate Case_ID in Test set"
    assert test_df['Query'].duplicated().sum() == 0, "Duplicate Query in Test set"

    assert crit_df['Case_ID'].duplicated().sum() == 0, "Duplicate Case_ID in Critical Safety set"
    assert crit_df['Query'].duplicated().sum() == 0, "Duplicate Query in Critical Safety set"
