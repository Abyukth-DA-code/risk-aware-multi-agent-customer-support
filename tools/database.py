"""
Operational Database Engine for Multi-Domain Capstone System
Project: Autonomous Multi-Agent Customer Support Resolution System with HITL Escalation
File: tools/database.py

Provides safe, in-memory indexed query access to synthetic operational records:
- Customers (1,000 records)
- Orders (1,500 records)
- Transactions (1,803 records)
- Support Tickets (1,000 records)
- Education Records (500 records)
- Business Records (500 records)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger("tools.database")


class OperationalDatabase:
    """Manages indexed operational records across synthetic enterprise domains."""

    _instance: Optional["OperationalDatabase"] = None

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "database"
        else:
            self.data_dir = Path(data_dir)

        self._load_tables()

    @classmethod
    def get_instance(cls, data_dir: Optional[Path] = None) -> "OperationalDatabase":
        """Singleton accessor for operational database."""
        if cls._instance is None:
            cls._instance = cls(data_dir=data_dir)
        return cls._instance

    def _load_tables(self) -> None:
        """Loads and indexes all operational CSV datasets."""
        self.customers_df = pd.read_csv(self.data_dir / "customers.csv")
        self.orders_df = pd.read_csv(self.data_dir / "orders.csv")
        self.transactions_df = pd.read_csv(self.data_dir / "transactions.csv")
        self.tickets_df = pd.read_csv(self.data_dir / "support_tickets.csv")
        self.education_df = pd.read_csv(self.data_dir / "education_records.csv")
        self.business_df = pd.read_csv(self.data_dir / "business_records.csv")

        # Create quick lookup dicts
        self._customers_by_id = self.customers_df.set_index("Customer_ID").to_dict(orient="index")
        self._orders_by_id = self.orders_df.set_index("Order_ID").to_dict(orient="index")
        self._transactions_by_id = self.transactions_df.set_index("Transaction_ID").to_dict(orient="index")

    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Look up customer by Customer_ID."""
        row = self._customers_by_id.get(customer_id)
        if row:
            res = dict(row)
            res["Customer_ID"] = customer_id
            return res
        return None

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Look up order by Order_ID."""
        row = self._orders_by_id.get(order_id)
        if row:
            res = dict(row)
            res["Order_ID"] = order_id
            return res
        return None

    def get_customer_orders(self, customer_id: str) -> List[Dict[str, Any]]:
        """Retrieve all orders placed by a specific customer."""
        matches = self.orders_df[self.orders_df["Customer_ID"] == customer_id]
        return matches.to_dict(orient="records")

    def get_order_transactions(self, order_id: str) -> List[Dict[str, Any]]:
        """Retrieve all transactions associated with an order."""
        matches = self.transactions_df[self.transactions_df["Order_ID"] == order_id]
        return matches.to_dict(orient="records")

    def get_customer_tickets(self, customer_id: str) -> List[Dict[str, Any]]:
        """Retrieve all prior support tickets for a customer."""
        matches = self.tickets_df[self.tickets_df["Customer_ID"] == customer_id]
        return matches.to_dict(orient="records")

    def get_student_record(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Look up education record by Student_ID."""
        matches = self.education_df[self.education_df["Student_ID"] == student_id]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None

    def get_business_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Look up small business record by Business_Record_ID."""
        matches = self.business_df[self.business_df["Business_Record_ID"] == record_id]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None
