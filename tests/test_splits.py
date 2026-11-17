"""
Automated Test Suite: Split Isolation, Zero-Overlap & Protection Constraints
"""

import os
import re
import pytest
import pandas as pd

from pathlib import Path
_cand_proc = Path(__file__).resolve().parents[1] / "datasets" / "processed"
PROCESSED_DIR = str(_cand_proc) if _cand_proc.exists() else os.path.join("data", "processed")

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def super_normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w]', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def test_split_disjointness():
    # Test intent splits
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "intent", "train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "intent", "validation.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "intent", "test.csv"))
    crit = pd.read_csv(os.path.join(PROCESSED_DIR, "safety", "critical_safety.csv"))

    # 1. Case ID Disjointness
    assert len(set(train['Case_ID']).intersection(set(val['Case_ID']))) == 0, "Train and Val share Case_IDs"
    assert len(set(train['Case_ID']).intersection(set(test['Case_ID']))) == 0, "Train and Test share Case_IDs"
    assert len(set(val['Case_ID']).intersection(set(test['Case_ID']))) == 0, "Val and Test share Case_IDs"
    assert len(set(crit['Case_ID']).intersection(set(train['Case_ID']))) == 0, "Critical Safety and Train share Case_IDs"
    assert len(set(crit['Case_ID']).intersection(set(test['Case_ID']))) == 0, "Critical Safety and Test share Case_IDs"

    # 2. Exact Query Disjointness
    assert len(set(train['Query']).intersection(set(val['Query']))) == 0, "Train and Val share exact Queries"
    assert len(set(train['Query']).intersection(set(test['Query']))) == 0, "Train and Test share exact Queries"
    assert len(set(val['Query']).intersection(set(test['Query']))) == 0, "Val and Test share exact Queries"
    assert len(set(crit['Query']).intersection(set(train['Query']))) == 0, "Critical Safety and Train share exact Queries"
    assert len(set(crit['Query']).intersection(set(test['Query']))) == 0, "Critical Safety and Test share exact Queries"

    # 3. Normalized Query Disjointness
    train_norm = set(train['Query'].apply(normalize_text))
    val_norm = set(val['Query'].apply(normalize_text))
    test_norm = set(test['Query'].apply(normalize_text))
    crit_norm = set(crit['Query'].apply(normalize_text))

    assert len(train_norm.intersection(val_norm)) == 0, "Normalized query leakage between Train and Val"
    assert len(train_norm.intersection(test_norm)) == 0, "Normalized query leakage between Train and Test"
    assert len(val_norm.intersection(test_norm)) == 0, "Normalized query leakage between Val and Test"
    assert len(crit_norm.intersection(train_norm)) == 0, "Normalized query leakage between Critical Safety and Train"
    assert len(crit_norm.intersection(test_norm)) == 0, "Normalized query leakage between Critical Safety and Test"

    # 4. Super-Normalized Disjointness (whitespace-collapsed & punctuation-stripped)
    train_snorm = set(train['Query'].apply(super_normalize))
    val_snorm = set(val['Query'].apply(super_normalize))
    test_snorm = set(test['Query'].apply(super_normalize))
    crit_snorm = set(crit['Query'].apply(super_normalize))

    assert len(train_snorm.intersection(test_snorm)) == 0, "Super-normalized query leakage between Train and Test"
    assert len(train_snorm.intersection(val_snorm)) == 0, "Super-normalized query leakage between Train and Val"
    assert len(val_snorm.intersection(test_snorm)) == 0, "Super-normalized query leakage between Val and Test"
    assert len(crit_snorm.intersection(train_snorm)) == 0, "Super-normalized query leakage between Critical Safety and Train"
    assert len(crit_snorm.intersection(test_snorm)) == 0, "Super-normalized query leakage between Critical Safety and Test"

def test_split_counts():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "intent", "train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "intent", "validation.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "intent", "test.csv"))
    crit = pd.read_csv(os.path.join(PROCESSED_DIR, "safety", "critical_safety.csv"))

    assert len(train) == 5635, f"Expected 5,635 Train cases (after near-duplicate quarantines), found {len(train)}"
    assert len(val) == 501, f"Expected 501 Validation cases, found {len(val)}"
    assert len(test) == 501, f"Expected 501 Test cases, found {len(test)}"
    assert len(crit) == 180, f"Expected 180 Critical Safety cases, found {len(crit)}"
