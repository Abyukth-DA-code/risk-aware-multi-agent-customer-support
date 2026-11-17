"""
Order and Transaction Domain Operational Tools
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation
File: tools/order_tools.py
"""

from typing import Any, Dict, List, Optional
from tools.database import OperationalDatabase


class OrderTools:
    """Operational toolset for e-commerce order verification, transaction auditing, and return checks."""

    def __init__(self, db: Optional[OperationalDatabase] = None):
        self.db = db or OperationalDatabase.get_instance()

    def get_order_details(self, order_id: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves detailed order information.
        Enforces customer authorization if customer_id is provided.
        """
        order = self.db.get_order(order_id)
        if not order:
            return {"status": "NOT_FOUND", "message": f"Order ID '{order_id}' not found."}

        # Customer isolation check: Customer A cannot view Customer B's order
        if customer_id and order.get("Customer_ID") != customer_id:
            return {
                "status": "UNAUTHORIZED",
                "message": f"Customer '{customer_id}' is not authorized to access Order '{order_id}'."
            }

        return {
            "status": "SUCCESS",
            "order_id": order_id,
            "customer_id": order.get("Customer_ID"),
            "order_status": order.get("Order_Status"),
            "amount_inr": order.get("Order_Amount_INR"),
            "payment_status": order.get("Payment_Status"),
            "delivery_status": order.get("Delivery_Status"),
            "return_status": order.get("Return_Status"),
            "domain": order.get("Domain"),
        }

    def check_return_eligibility(self, order_id: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluates operational return eligibility based on order status and return flags.
        """
        details = self.get_order_details(order_id, customer_id=customer_id)
        if details["status"] != "SUCCESS":
            return {"eligible": False, "reason": details["message"]}

        if details["delivery_status"] != "Delivered":
            return {
                "eligible": False,
                "reason": f"Order has not been delivered yet (Current delivery status: {details['delivery_status']})."
            }

        if details["return_status"] == "Returned":
            return {"eligible": False, "reason": "Order has already been returned and refunded."}

        if details["return_status"] == "Not_Eligible":
            return {"eligible": False, "reason": "Item is designated as non-returnable under category policy."}

        return {
            "eligible": True,
            "order_id": order_id,
            "amount_inr": details["amount_inr"],
            "return_status": details["return_status"],
        }

    def get_order_audit_trail(self, order_id: str) -> Dict[str, Any]:
        """
        Audits order transactions and detects potential fraud flags.
        """
        order = self.db.get_order(order_id)
        if not order:
            return {"status": "NOT_FOUND", "message": f"Order ID '{order_id}' not found."}

        transactions = self.db.get_order_transactions(order_id)
        has_fraud = any(t.get("Fraud_Flag") == "Yes" for t in transactions)

        return {
            "status": "SUCCESS",
            "order_id": order_id,
            "order_status": order.get("Order_Status"),
            "transaction_count": len(transactions),
            "transactions": transactions,
            "has_fraud_flag": has_fraud,
        }
