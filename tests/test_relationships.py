"""
Automated Test Suite: Relational Foreign Key & Domain Consistency Constraints
"""

import os
import pytest
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")

def test_operational_foreign_keys():
    db_dir = os.path.join(PROCESSED_DIR, "database")
    cust = pd.read_csv(os.path.join(db_dir, "customers.csv"))
    ordr = pd.read_csv(os.path.join(db_dir, "orders.csv"))
    tx = pd.read_csv(os.path.join(db_dir, "transactions.csv"))
    tkt = pd.read_csv(os.path.join(db_dir, "support_tickets.csv"))
    stud = pd.read_csv(os.path.join(db_dir, "education_records.csv"))
    biz = pd.read_csv(os.path.join(db_dir, "business_records.csv"))

    # 1. Orders -> Customers
    assert ordr['Customer_ID'].isin(cust['Customer_ID']).all(), "Orphan customer in Orders"

    # 2. Transactions -> Customers & Orders
    assert tx['Customer_ID'].isin(cust['Customer_ID']).all(), "Orphan customer in Transactions"
    assert tx['Order_ID'].isin(ordr['Order_ID']).all(), "Orphan order in Transactions"

    # Check Transaction customer matches Order customer
    tx_ord_merge = tx.merge(ordr[['Order_ID', 'Customer_ID']], on='Order_ID', suffixes=('_tx', '_ord'))
    assert (tx_ord_merge['Customer_ID_tx'] == tx_ord_merge['Customer_ID_ord']).all(), "Transaction customer mismatch with order customer"

    # 3. Tickets -> Customers
    assert tkt['Customer_ID'].isin(cust['Customer_ID']).all(), "Orphan customer in Support_Tickets"

    # 4. Students -> Customers
    assert stud['Customer_ID'].isin(cust['Customer_ID']).all(), "Orphan customer in Education_Records"

    # 5. Business -> Customers
    assert biz['Customer_ID'].isin(cust['Customer_ID']).all(), "Orphan customer in Business_Records"

def test_domain_consistency():
    db_dir = os.path.join(PROCESSED_DIR, "database")
    cust = pd.read_csv(os.path.join(db_dir, "customers.csv")).set_index('Customer_ID')
    ordr = pd.read_csv(os.path.join(db_dir, "orders.csv"))
    tx = pd.read_csv(os.path.join(db_dir, "transactions.csv"))
    tkt = pd.read_csv(os.path.join(db_dir, "support_tickets.csv"))
    stud = pd.read_csv(os.path.join(db_dir, "education_records.csv"))
    biz = pd.read_csv(os.path.join(db_dir, "business_records.csv"))

    # Domain consistency checks
    assert (ordr['Customer_ID'].map(cust['Domain']) == 'E-commerce').all(), "Non-Ecommerce customer in Orders"
    assert (tx['Customer_ID'].map(cust['Domain']) == 'E-commerce').all(), "Non-Ecommerce customer in Transactions"
    assert (stud['Customer_ID'].map(cust['Domain']) == 'Education').all(), "Non-Education customer in Students"
    assert (biz['Customer_ID'].map(cust['Domain']) == 'Small Business').all(), "Non-SmallBusiness customer in Business_Records"
    assert (tkt['Customer_ID'].map(cust['Domain']) == tkt['Domain']).all(), "Ticket domain does not match customer domain"

def test_memory_foreign_keys():
    cust = pd.read_csv(os.path.join(PROCESSED_DIR, "database", "customers.csv"))
    tkt = pd.read_csv(os.path.join(PROCESSED_DIR, "database", "support_tickets.csv"))
    mem = pd.read_csv(os.path.join(PROCESSED_DIR, "memory", "memory.csv"))

    assert mem['Customer_ID'].isin(cust['Customer_ID']).all(), "Orphan customer in Memory"
    assert mem['Related_Ticket_ID'].isin(tkt['Ticket_ID']).all(), "Orphan ticket in Memory"
