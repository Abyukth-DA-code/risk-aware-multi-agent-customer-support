"""
Automated Test Suite: Target Leakage and Circularity Constraints
"""

import os
import pytest
import pandas as pd

from pathlib import Path
_cand_proc = Path(__file__).resolve().parents[1] / "datasets" / "processed"
PROCESSED_DIR = str(_cand_proc) if _cand_proc.exists() else os.path.join("data", "processed")

def test_intent_task_leakage_absence():
    for split in ['train.csv', 'validation.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "intent", split))
        forbidden = [
            'Autonomy_Decision', 'Escalation_Required', 'Escalation_Reason',
            'Required_Agents', 'Required_Tools', 'Action_Type', 'Risk_Reason',
            'Complexity_Reason', 'Final_Confidence_Gold', 'AI_Resolution',
            'Human_Decision', 'Correction', 'Reviewer_Notes'
        ]
        for col in forbidden:
            assert col not in df.columns, f"Forbidden leakage column {col} found in intent/{split}"

def test_sentiment_task_leakage_absence():
    for split in ['train.csv', 'validation.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "sentiment", split))
        forbidden = [
            'Autonomy_Decision', 'Escalation_Required', 'Escalation_Reason',
            'Human_Decision', 'Reviewer_Notes', 'Risk', 'Complexity'
        ]
        for col in forbidden:
            assert col not in df.columns, f"Forbidden leakage column {col} found in sentiment/{split}"

def test_risk_task_leakage_absence():
    for split in ['train.csv', 'validation.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "risk", split))
        forbidden = [
            'Risk_Reason', 'Autonomy_Decision', 'Escalation_Required',
            'Escalation_Reason', 'Final_Confidence_Gold', 'Human_Decision'
        ]
        for col in forbidden:
            assert col not in df.columns, f"Forbidden leakage column {col} found in risk/{split}"

def test_complexity_task_leakage_absence():
    for split in ['train.csv', 'validation.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "complexity", split))
        forbidden = [
            'Complexity_Reason', 'Required_Agents', 'Required_Tools',
            'Required_Capabilities', 'Required_Data_Sources', 'Autonomy_Decision',
            'Escalation_Required'
        ]
        for col in forbidden:
            assert col not in df.columns, f"Forbidden circularity column {col} found in complexity/{split}"

def test_escalation_task_leakage_absence():
    for split in ['train.csv', 'validation.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "escalation", split))
        forbidden = [
            'Escalation_Reason', 'Autonomy_Decision', 'AI_Resolution',
            'Human_Decision', 'Correction', 'Reviewer_Notes'
        ]
        for col in forbidden:
            assert col not in df.columns, f"Forbidden leakage column {col} found in escalation/{split}"

def test_autonomy_task_leakage_absence():
    for split in ['train.csv', 'validation.csv', 'test.csv']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "autonomy", split))
        forbidden = [
            'Escalation_Required', 'Escalation_Reason', 'AI_Resolution',
            'Human_Decision', 'Correction', 'Reviewer_Notes'
        ]
        for col in forbidden:
            assert col not in df.columns, f"Forbidden shortcut column {col} found in autonomy/{split}"
