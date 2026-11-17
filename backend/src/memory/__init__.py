"""
Multi-Tenant Memory & Reference Resolution Package
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
"""

from src.memory.customer_memory import CustomerMemoryManager
from src.memory.resolver import ReferenceResolver
from src.memory.session_memory import SessionMemoryManager

__all__ = [
    "ReferenceResolver",
    "SessionMemoryManager",
    "CustomerMemoryManager",
]
