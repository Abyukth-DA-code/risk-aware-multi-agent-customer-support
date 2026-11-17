"""
Automated Test Suite: Schema Integrity & Non-Nullity Constraints
"""

import os
import pytest
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")

def test_final_master_gold_schema():
    path = os.path.join(PROCESSED_DIR, "FINAL_MASTER_GOLD.csv")
    assert os.path.exists(path), "FINAL_MASTER_GOLD.csv does not exist"
    df = pd.read_csv(path)
    assert len(df) == 3519, f"Expected 3,519 rows in FINAL_MASTER_GOLD.csv, got {len(df)}"

    expected_cols = [
        'Case_ID', 'Query', 'Domain', 'Intent', 'Sentiment', 'Risk', 'Risk_Reason',
        'Complexity', 'Complexity_Reason', 'Required_Capabilities', 'Required_Agents',
        'Required_Tools', 'Required_Data_Sources', 'Action_Type', 'Sensitive_Action',
        'Financial_Impact', 'Intent_Confidence_Gold', 'Domain_Confidence_Gold',
        'Evidence_Quality', 'Retrieval_Confidence_Gold', 'Evidence_Agreement_Gold',
        'Verification_Required', 'Verification_Score_Gold', 'Final_Confidence_Gold',
        'Escalation_Required', 'Escalation_Reason', 'Autonomy_Decision',
        'Data_Origin', 'Source_Type', 'Split', 'Quality_Status'
    ]
    for c in expected_cols:
        assert c in df.columns, f"Missing expected column {c} in FINAL_MASTER_GOLD.csv"

    # Assert zero missing values in primary operational columns
    critical_cols = ['Case_ID', 'Query', 'Domain', 'Intent', 'Sentiment', 'Risk',
                     'Complexity', 'Escalation_Required', 'Autonomy_Decision', 'Split']
    nulls = df[critical_cols].isnull().sum()
    for col, count in nulls.items():
        assert count == 0, f"Column {col} has {count} nulls in FINAL_MASTER_GOLD.csv"

def test_operational_tables_schema():
    ops_dir = os.path.join(PROCESSED_DIR, "database")
    tables = {
        "customers.csv": ['Customer_ID', 'Domain', 'Customer_Type', 'Account_Status', 'Verification_Status'],
        "orders.csv": ['Order_ID', 'Customer_ID', 'Order_Status', 'Order_Amount_INR', 'Payment_Status', 'Delivery_Status', 'Return_Status'],
        "transactions.csv": ['Transaction_ID', 'Customer_ID', 'Order_ID', 'Transaction_Type', 'Amount_INR', 'Payment_Method', 'Transaction_Status', 'Fraud_Flag'],
        "support_tickets.csv": ['Ticket_ID', 'Customer_ID', 'Query_Type', 'Customer_Query', 'Ticket_Status', 'Domain'],
        "education_records.csv": ['Student_ID', 'Customer_ID', 'Program', 'Year', 'Attendance_Percent', 'Fee_Status', 'Exam_Status', 'Scholarship_Status'],
        "business_records.csv": ['Business_Record_ID', 'Customer_ID', 'Record_Type', 'Reference_ID', 'Amount_INR', 'Record_Status', 'Approval_Status']
    }
    for filename, cols in tables.items():
        p = os.path.join(ops_dir, filename)
        assert os.path.exists(p), f"Missing operational table {filename}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Operational table {filename} is empty"
        for c in cols:
            assert c in df.columns, f"Missing column {c} in {filename}"
            assert df[c].isnull().sum() == 0, f"Column {c} in {filename} contains nulls"
