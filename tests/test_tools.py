"""
Automated Test Suite: Operational Database Tools & Customer/Order Validation
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation
File: tests/test_tools.py
"""

import pytest
from tools.database import OperationalDatabase
from tools.customer_tools import CustomerTools
from tools.order_tools import OrderTools


@pytest.fixture(scope="module")
def tools_fixture():
    db = OperationalDatabase.get_instance()
    cust_tools = CustomerTools(db=db)
    order_tools = OrderTools(db=db)
    return {
        "db": db,
        "customer": cust_tools,
        "order": order_tools,
    }


# ===========================================================================
# 1. Operational Database Integrity Tests
# ===========================================================================

def test_operational_database_tables_loaded(tools_fixture):
    """Verifies that all operational tables are properly loaded into memory."""
    db = tools_fixture["db"]
    assert len(db.customers_df) == 1000
    assert len(db.orders_df) == 1500
    assert len(db.transactions_df) == 1803
    assert len(db.tickets_df) == 1000
    assert len(db.education_df) == 500
    assert len(db.business_df) == 500


def test_database_lookups_existing_and_nonexistent(tools_fixture):
    """Verifies exact lookup behavior for known and missing keys."""
    db = tools_fixture["db"]

    # Known existing customer
    cust = db.get_customer("CUST0001")
    assert cust is not None
    assert cust["Domain"] == "E-commerce"

    # Non-existent customer
    assert db.get_customer("CUST_NONEXISTENT_9999") is None

    # Known existing order
    sample_order = db.orders_df.iloc[0]["Order_ID"]
    order = db.get_order(sample_order)
    assert order is not None
    assert order["Order_ID"] == sample_order

    # Non-existent order
    assert db.get_order("ORD_DOES_NOT_EXIST") is None


# ===========================================================================
# 2. Customer Tools & Authorization Tests
# ===========================================================================

def test_customer_profile_and_verification_success(tools_fixture):
    """Verifies successful customer profile retrieval and identity verification."""
    cust_tools = tools_fixture["customer"]

    res = cust_tools.get_customer_profile("CUST0001")
    assert res["status"] == "SUCCESS"
    assert res["domain"] == "E-commerce"
    assert "account_status" in res

    verif = cust_tools.verify_customer_identity("CUST0001", expected_domain="E-commerce")
    assert verif["verified"] is True


def test_customer_verification_domain_mismatch(tools_fixture):
    """Verifies that domain mismatch fails verification safely."""
    cust_tools = tools_fixture["customer"]

    # CUST0001 belongs to E-commerce, test with Education
    verif = cust_tools.verify_customer_identity("CUST0001", expected_domain="Education")
    assert verif["verified"] is False
    assert "Domain mismatch" in verif["reason"]


def test_customer_ticket_history(tools_fixture):
    """Verifies ticket history retrieval."""
    cust_tools = tools_fixture["customer"]
    db = tools_fixture["db"]

    # Find a customer who has tickets
    ticket_cust_id = db.tickets_df.iloc[0]["Customer_ID"]
    tickets = cust_tools.get_ticket_history(ticket_cust_id)
    assert len(tickets) >= 1
    assert all(t["Customer_ID"] == ticket_cust_id for t in tickets)


# ===========================================================================
# 3. Order Tools & Customer Authorization Isolation
# ===========================================================================

def test_order_customer_isolation_enforcement(tools_fixture):
    """
    Verifies that Customer A is forbidden from viewing Customer B's order.
    """
    order_tools = tools_fixture["order"]
    db = tools_fixture["db"]

    sample_order = db.orders_df.iloc[0]
    actual_cust_id = sample_order["Customer_ID"]
    other_cust_id = "CUST_IMPOSTOR_9999"

    # Authorized access
    auth_res = order_tools.get_order_details(sample_order["Order_ID"], customer_id=actual_cust_id)
    assert auth_res["status"] == "SUCCESS"
    assert auth_res["order_id"] == sample_order["Order_ID"]

    # Unauthorized access by different customer
    unauth_res = order_tools.get_order_details(sample_order["Order_ID"], customer_id=other_cust_id)
    assert unauth_res["status"] == "UNAUTHORIZED"
    assert "not authorized" in unauth_res["message"]


def test_order_return_eligibility_logic(tools_fixture):
    """Verifies return eligibility logic against delivery and return statuses."""
    order_tools = tools_fixture["order"]
    db = tools_fixture["db"]

    # Find delivered, returnable order
    delivered_orders = db.orders_df[
        (db.orders_df["Delivery_Status"] == "Delivered") &
        (db.orders_df["Return_Status"] != "Returned") &
        (db.orders_df["Return_Status"] != "Not_Eligible")
    ]
    if not delivered_orders.empty:
        sample = delivered_orders.iloc[0]
        check = order_tools.check_return_eligibility(sample["Order_ID"], customer_id=sample["Customer_ID"])
        assert check["eligible"] is True

    # Find non-delivered order
    undelivered = db.orders_df[db.orders_df["Delivery_Status"] != "Delivered"]
    if not undelivered.empty:
        sample = undelivered.iloc[0]
        check = order_tools.check_return_eligibility(sample["Order_ID"], customer_id=sample["Customer_ID"])
        assert check["eligible"] is False
        assert "not been delivered" in check["reason"]


def test_order_audit_trail_and_fraud_detection(tools_fixture):
    """Verifies audit trail retrieval and fraud flag detection."""
    order_tools = tools_fixture["order"]
    db = tools_fixture["db"]

    sample_order_id = db.orders_df.iloc[0]["Order_ID"]
    audit = order_tools.get_order_audit_trail(sample_order_id)
    assert audit["status"] == "SUCCESS"
    assert "has_fraud_flag" in audit
    assert isinstance(audit["transactions"], list)


# ===========================================================================
# 4. Education & Business Records Tests
# ===========================================================================

def test_education_and_business_domain_records(tools_fixture):
    """Verifies retrieval of Education and Small Business domain records."""
    db = tools_fixture["db"]

    sample_student = db.education_df.iloc[0]["Student_ID"]
    student_rec = db.get_student_record(sample_student)
    assert student_rec is not None
    assert student_rec["Student_ID"] == sample_student

    sample_biz = db.business_df.iloc[0]["Business_Record_ID"]
    biz_rec = db.get_business_record(sample_biz)
    assert biz_rec is not None
    assert biz_rec["Business_Record_ID"] == sample_biz


def test_tools_adversarial_input_resilience(tools_fixture):
    """Verifies resilience against SQL injection strings and malformed inputs."""
    order_tools = tools_fixture["order"]
    cust_tools = tools_fixture["customer"]

    malicious_inputs = [
        "' OR '1'='1",
        "'; DROP TABLE orders; --",
        "<script>alert('xss')</script>",
        "",
        "   ",
        "\x00nullbyte",
    ]

    for evil in malicious_inputs:
        order_res = order_tools.get_order_details(evil)
        assert order_res["status"] == "NOT_FOUND"

        cust_res = cust_tools.get_customer_profile(evil)
        assert cust_res["status"] == "NOT_FOUND"
