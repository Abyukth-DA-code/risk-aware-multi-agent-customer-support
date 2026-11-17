"""
Automated Test Suite: Canonical Taxonomy & Label Conformance Constraints
"""

import os
import pytest
import pandas as pd

from pathlib import Path
_cand_proc = Path(__file__).resolve().parents[1] / "datasets" / "processed"
PROCESSED_DIR = str(_cand_proc) if _cand_proc.exists() else os.path.join("data", "processed")

def test_taxonomy_conformance():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "FINAL_MASTER_GOLD.csv"))

    allowed_domains = {'E-commerce', 'Education', 'Small Business'}
    allowed_risks = {'Low', 'Medium', 'High', 'Critical'}
    allowed_complexities = {'Low', 'Medium', 'High'}
    allowed_sentiments = {'Negative', 'Neutral', 'Positive'}
    allowed_autonomy = {'ANSWER', 'ORCHESTRATE', 'ESCALATE'}
    allowed_escalation = {'Yes', 'No'}
    allowed_evidence = {'Strong', 'Insufficient', 'Conflicting'}
    allowed_origins = {'PUBLIC_SOURCE', 'SYNTHETIC', 'DERIVED'}

    assert set(df['Domain']).issubset(allowed_domains), f"Invalid domains: {set(df['Domain']) - allowed_domains}"
    assert set(df['Risk']).issubset(allowed_risks), f"Invalid risks: {set(df['Risk']) - allowed_risks}"
    assert set(df['Complexity']).issubset(allowed_complexities), f"Invalid complexities: {set(df['Complexity']) - allowed_complexities}"
    assert set(df['Sentiment']).issubset(allowed_sentiments), f"Invalid sentiments: {set(df['Sentiment']) - allowed_sentiments}"
    assert set(df['Autonomy_Decision']).issubset(allowed_autonomy), f"Invalid autonomy: {set(df['Autonomy_Decision']) - allowed_autonomy}"
    assert set(df['Escalation_Required']).issubset(allowed_escalation), f"Invalid escalation: {set(df['Escalation_Required']) - allowed_escalation}"
    assert set(df['Evidence_Quality']).issubset(allowed_evidence), f"Invalid evidence: {set(df['Evidence_Quality']) - allowed_evidence}"
    assert set(df['Data_Origin']).issubset(allowed_origins), f"Invalid origin: {set(df['Data_Origin']) - allowed_origins}"

def test_domain_intent_isolation():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "FINAL_MASTER_GOLD.csv"))
    
    ecom_intents = set(df[df['Domain'] == 'E-commerce']['Intent'])
    edu_intents = set(df[df['Domain'] == 'Education']['Intent'])
    sbiz_intents = set(df[df['Domain'] == 'Small Business']['Intent'])

    assert len(ecom_intents) == 27, f"Expected 27 E-commerce intents, got {len(ecom_intents)}"
    assert len(edu_intents) == 10, f"Expected 10 Education intents, got {len(edu_intents)}"
    assert len(sbiz_intents) == 10, f"Expected 10 Small Business intents, got {len(sbiz_intents)}"

    assert len(ecom_intents.intersection(edu_intents)) == 0, "Overlapping intents between E-commerce and Education"
    assert len(ecom_intents.intersection(sbiz_intents)) == 0, "Overlapping intents between E-commerce and Small Business"
    assert len(edu_intents.intersection(sbiz_intents)) == 0, "Overlapping intents between Education and Small Business"
