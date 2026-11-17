"""
Long-Term Customer Memory Manager (TTL Retention & GDPR Deletion)
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Maintains persistent customer profile and operational context keyed by (workspace_id, customer_id).
Features:
- Configurable TTL retention window.
- Read-time expiration filtering.
- Technical Right-to-be-Forgotten (irrevocable deletion).
- Architectural safety boundary: memory is NEVER treated as authoritative policy evidence.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from src.workspaces.db import WorkspaceDB


class CustomerMemoryManager:
    """Manages long-term customer context with privacy and retention controls."""

    def __init__(self, db: Optional[WorkspaceDB] = None):
        self.db = db or WorkspaceDB()

    def add_memory(
        self,
        workspace_id: str,
        tenant_id: str,
        customer_id: str,
        category: str,
        content: str,
        confidence: float = 1.0,
        ttl_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Creates a customer memory entry with optional retention expiration."""
        expires_at = None
        if ttl_days is not None and ttl_days > 0:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

        return self.db.create_customer_memory(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            customer_id=customer_id,
            category=category,
            content=content,
            confidence=confidence,
            expires_at=expires_at,
        )

    def get_memories(
        self,
        workspace_id: str,
        customer_id: str,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieves active customer memories, excluding expired entries."""
        if not workspace_id or not customer_id:
            return []
        return self.db.list_customer_memories(
            workspace_id=workspace_id,
            customer_id=customer_id,
            include_expired=include_expired,
        )

    def delete_memory(self, workspace_id: str, memory_id: str) -> bool:
        """Deletes a specific memory entry."""
        return self.db.delete_customer_memory(workspace_id=workspace_id, memory_id=memory_id)

    def delete_customer_data(self, workspace_id: str, customer_id: str) -> int:
        """
        Technical GDPR right-to-be-forgotten deletion.
        Permanently purges all memory records for the customer in the workspace.
        """
        return self.db.delete_all_customer_memories(workspace_id=workspace_id, customer_id=customer_id)

    def cleanup_expired(self, workspace_id: Optional[str] = None) -> int:
        """Prunes expired records past their TTL."""
        return self.db.prune_expired_customer_memories(workspace_id=workspace_id)
