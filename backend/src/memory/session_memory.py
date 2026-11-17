"""
Short-Term Multi-Turn Session Memory Manager
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Maintains chronological conversation state keyed strictly by (workspace_id, session_id).
Guarantees session and tenant isolation.
"""

from typing import Any, Dict, List, Optional
from src.workspaces.db import WorkspaceDB


class SessionMemoryManager:
    """Encapsulates short-term conversation turn history."""

    def __init__(self, db: Optional[WorkspaceDB] = None):
        self.db = db or WorkspaceDB()

    def get_history(
        self,
        workspace_id: str,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieves past turns strictly for the given workspace and session."""
        if not workspace_id or not session_id:
            return []
        return self.db.get_session_turns(workspace_id=workspace_id, session_id=session_id, limit=limit)

    def save_turn(
        self,
        workspace_id: str,
        tenant_id: str,
        session_id: str,
        user_query: str,
        resolved_query: Optional[str],
        bot_response: Optional[str],
        final_autonomy_tier: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Appends a new verified turn into the session log."""
        return self.db.save_session_turn(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            session_id=session_id,
            user_query=user_query,
            resolved_query=resolved_query,
            bot_response=bot_response,
            final_autonomy_tier=final_autonomy_tier,
            citations=citations,
        )

    def clear_session(self, workspace_id: str, session_id: str) -> int:
        """Wipes session history."""
        return self.db.clear_session_turns(workspace_id=workspace_id, session_id=session_id)
