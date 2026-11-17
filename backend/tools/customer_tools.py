"""
Customer Domain Operational Tools
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation
File: tools/customer_tools.py
"""

from typing import Any, Dict, List, Optional
from tools.database import OperationalDatabase


class CustomerTools:
    """Operational toolset for customer verification and profile inquiries."""

    def __init__(self, db: Optional[OperationalDatabase] = None):
        self.db = db or OperationalDatabase.get_instance()

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Retrieves verified customer profile with domain and account status.
        Returns error dict if customer is not found.
        """
        cust = self.db.get_customer(customer_id)
        if not cust:
            return {"status": "NOT_FOUND", "message": f"Customer ID '{customer_id}' not found."}
        return {
            "status": "SUCCESS",
            "customer_id": customer_id,
            "domain": cust.get("Domain"),
            "customer_type": cust.get("Customer_Type"),
            "account_status": cust.get("Account_Status"),
            "verification_status": cust.get("Verification_Status"),
        }

    def verify_customer_identity(self, customer_id: str, expected_domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Verifies customer identity and domain alignment.
        """
        profile = self.get_customer_profile(customer_id)
        if profile["status"] != "SUCCESS":
            return {"verified": False, "reason": "Customer record does not exist."}

        if profile["account_status"] != "Active":
            return {"verified": False, "reason": f"Account is {profile['account_status']}."}

        if expected_domain and profile["domain"] != expected_domain:
            return {
                "verified": False,
                "reason": f"Domain mismatch: Customer belongs to {profile['domain']}, not {expected_domain}."
            }

        return {
            "verified": True,
            "customer_id": customer_id,
            "domain": profile["domain"],
            "verification_status": profile["verification_status"],
        }

    def get_ticket_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Retrieves support ticket history for a customer."""
        return self.db.get_customer_tickets(customer_id)
