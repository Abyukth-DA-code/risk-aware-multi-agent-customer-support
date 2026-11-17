"""
Automated Test Suite: Dataset Invariants, Canonical Split Counts & Taxonomy Integrity
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation
File: tests/test_data_invariants.py
"""

from pathlib import Path
import pandas as pd
import pytest

_cand_data = Path(__file__).resolve().parents[1] / "datasets" / "processed"
DATA_DIR = _cand_data if _cand_data.exists() else (Path(__file__).resolve().parents[1] / "data" / "processed")


def test_final_master_gold_invariants():
    """Verifies that FINAL_MASTER_GOLD.csv contains exact canonical invariants."""
    gold_path = DATA_DIR / "FINAL_MASTER_GOLD.csv"
    assert gold_path.is_file(), f"Missing canonical file: {gold_path}"

    df = pd.read_csv(gold_path)
    # Exact row and column count
    assert len(df) == 3519, f"Expected 3,519 rows, got {len(df)}"
    assert len(df.columns) == 31, f"Expected 31 columns, got {len(df.columns)}"

    # Zero missing values across entire gold dataset
    null_counts = df.isnull().sum()
    assert null_counts.sum() == 0, f"Found missing values in columns: {null_counts[null_counts > 0].to_dict()}"

    # Split distributions
    split_counts = df["Split"].value_counts().to_dict()
    assert split_counts == {
        "train_base": 2337,
        "validation": 501,
        "test": 501,
        "critical_safety": 180,
    }

    # Taxonomy domain counts
    domain_counts = df["Domain"].value_counts().to_dict()
    assert domain_counts == {
        "E-commerce": 3050,
        "Small Business": 250,
        "Education": 219,
    }


def test_task_test_splits_counts():
    """Verifies held-out evaluation test splits for all modeling tasks."""
    tasks = {
        "intent": 501,
        "risk": 501,
        "complexity": 501,
        "domain": 501,
        "autonomy": 501,
        "escalation": 501,
        "sentiment": 563,  # 501 base + 62 positive augmented
    }

    for task, expected_count in tasks.items():
        test_path = DATA_DIR / task / "test.csv"
        assert test_path.is_file(), f"Missing test split for {task} at {test_path}"
        df = pd.read_csv(test_path)
        assert len(df) == expected_count, f"Task {task} expected {expected_count} rows, got {len(df)}"


def test_rag_knowledge_base_counts():
    """Verifies exact document, chunk, and benchmark counts."""
    docs = pd.read_csv(DATA_DIR / "rag" / "documents.csv")
    chunks = pd.read_csv(DATA_DIR / "rag" / "chunks.csv")
    bench = pd.read_csv(DATA_DIR / "rag" / "retrieval_benchmark_remediated_v2.csv")

    assert len(docs) == 46, f"Expected 46 documents, got {len(docs)}"
    assert len(chunks) == 95, f"Expected 95 chunks, got {len(chunks)}"
    assert len(bench) == 501, f"Expected 501 benchmark queries, got {len(bench)}"

    # Check answerable vs gap separation
    is_ans = bench["Expected_Document_ID"].fillna("").astype(str).str.strip().apply(
        lambda x: x.lower() not in ("", "nan", "none")
    )
    assert is_ans.sum() == 428, f"Expected 428 answerable queries, got {is_ans.sum()}"
    assert (~is_ans).sum() == 73, f"Expected 73 unanswerable gap queries, got {(~is_ans).sum()}"


def test_operational_database_counts():
    """Verifies row counts of all 6 operational database tables."""
    tables = {
        "customers.csv": 1000,
        "orders.csv": 1500,
        "transactions.csv": 1803,
        "support_tickets.csv": 1000,
        "education_records.csv": 500,
        "business_records.csv": 500,
    }

    db_dir = DATA_DIR / "database"
    for filename, expected_rows in tables.items():
        p = db_dir / filename
        assert p.is_file(), f"Missing operational table {filename}"
        df = pd.read_csv(p)
        assert len(df) == expected_rows, f"Table {filename} expected {expected_rows} rows, got {len(df)}"
