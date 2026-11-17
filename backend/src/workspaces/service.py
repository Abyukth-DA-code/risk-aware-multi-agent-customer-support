"""
Workspace Management & Tenant Authorization Service
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Enforces strict tenant isolation, authenticated credentials, and dynamic checklist management.
"""

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import secrets
from typing import Any, Dict, List, Optional, Tuple
import uuid

from src.workspaces.checklist_engine import ChecklistEngine
from src.workspaces.db import WorkspaceDB
from src.workspaces.models import (
    BusinessDomain,
    ChecklistItemResponse,
    ChecklistItemStatus,
    CustomCategoryRequest,
    CustomerMemoryResponse,
    EscalationListItem,
    EscalationResponse,
    EscalationReviewRequest,
    EscalationStatus,
    ReadinessReport,
    TenantRegisterRequest,
    TenantResponse,
    WidgetConfigResponse,
    WidgetTokenResponse,
    WorkspaceCreateRequest,
    WorkspaceResponse,
)


class WorkspaceService:
    """Manages tenant onboarding, workspace lifecycle, authentication, and dynamic checklists."""

    def __init__(self, db: Optional[WorkspaceDB] = None):
        self.db = db or WorkspaceDB()

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        """Computes SHA-256 hash for secure storage and constant-time comparison."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    # -----------------------------------------------------------------------
    # Authentication & Authorization
    # -----------------------------------------------------------------------
    def authenticate_api_key(self, raw_api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validates raw API key and returns authenticated tenant and workspace permissions.
        Returns None if key is invalid, missing, or revoked.
        """
        if not raw_api_key or not isinstance(raw_api_key, str):
            return None

        hashed = self._hash_key(raw_api_key.strip())
        record = self.db.get_api_key_record(hashed)
        if not record:
            return None

        # Verify tenant still exists and is active
        tenant = self.db.get_tenant(record["tenant_id"])
        if not tenant:
            return None

        return {
            "tenant_id": record["tenant_id"],
            "workspace_id": record["workspace_id"],
            "key_id": record["key_id"],
            "tenant_name": tenant["name"],
        }

    def verify_workspace_access(self, authenticated_tenant_id: str, workspace_id: str) -> Dict[str, Any]:
        """
        Enforces tenant authorization over workspace.
        Never trusts client-supplied tenant_id. Rejects cross-tenant access with PermissionError.
        """
        ws = self.db.get_workspace(workspace_id)
        if not ws:
            raise LookupError(f"Workspace '{workspace_id}' not found.")

        if ws["tenant_id"] != authenticated_tenant_id:
            # Physical isolation check: caller's authenticated tenant does NOT own this workspace!
            raise PermissionError(
                f"Unauthorized: Tenant '{authenticated_tenant_id}' cannot access workspace '{workspace_id}'."
            )

        return ws

    # -----------------------------------------------------------------------
    # Tenant Lifecycle
    # -----------------------------------------------------------------------
    def register_tenant(self, req: TenantRegisterRequest) -> TenantResponse:
        """Registers a new tenant and generates a secure root API key."""
        tenant_id = f"ten_{uuid.uuid4().hex[:12]}"
        created = self.db.create_tenant(
            tenant_id=tenant_id,
            name=req.name.strip(),
            email=req.email.strip().lower(),
        )

        # Generate secure API key
        raw_token = f"cap_{secrets.token_urlsafe(32)}"
        hashed_token = self._hash_key(raw_token)
        key_id = f"key_{uuid.uuid4().hex[:8]}"

        self.db.create_api_key(
            key_id=key_id,
            tenant_id=tenant_id,
            hashed_key=hashed_token,
            name="Primary Business Root Key",
        )

        return TenantResponse(
            tenant_id=created["tenant_id"],
            name=created["name"],
            email=created["email"],
            created_at=created["created_at"],
            api_key=raw_token,
        )

    # -----------------------------------------------------------------------
    # Workspace Lifecycle
    # -----------------------------------------------------------------------
    def create_workspace(
        self,
        authenticated_tenant_id: str,
        req: WorkspaceCreateRequest,
    ) -> WorkspaceResponse:
        """
        Creates a new business workspace bound to the authenticated tenant
        and instantiates its domain-specific knowledge checklist.
        """
        workspace_id = f"ws_{uuid.uuid4().hex[:12]}"
        ws_dict = self.db.create_workspace(
            workspace_id=workspace_id,
            tenant_id=authenticated_tenant_id,
            name=req.name.strip(),
            domain=req.domain.value,
            description=req.description.strip() if req.description else None,
        )

        # Automatically instantiate the domain checklist
        checklist_items = ChecklistEngine.get_default_checklist_items(
            domain=req.domain.value,
            workspace_id=workspace_id,
        )
        self.db.insert_checklist_items(checklist_items)

        return WorkspaceResponse(**ws_dict)

    def get_workspace(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceResponse:
        """Retrieves workspace metadata with strict tenant authorization check."""
        ws = self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        return WorkspaceResponse(**ws)

    def list_workspaces(self, authenticated_tenant_id: str) -> List[WorkspaceResponse]:
        """Lists only the workspaces owned by the authenticated tenant."""
        workspaces = self.db.list_workspaces_for_tenant(authenticated_tenant_id)
        return [WorkspaceResponse(**w) for w in workspaces]

    # -----------------------------------------------------------------------
    # Checklist & Readiness Operations
    # -----------------------------------------------------------------------
    def get_checklist(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
    ) -> List[ChecklistItemResponse]:
        """Retrieves the dynamic checklist for the authorized workspace."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        items = self.db.get_checklist_items(workspace_id)
        return [ChecklistItemResponse(**item) for item in items]

    def add_custom_category(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
        req: CustomCategoryRequest,
    ) -> ChecklistItemResponse:
        """Allows workspace owners to add custom categories beyond default checklist."""
        ws = self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        cat_key = req.category_name.strip().lower().replace(" ", "_")
        now = datetime.now(timezone.utc).isoformat()

        item_dict = {
            "item_id": f"chk_{uuid.uuid4().hex[:12]}",
            "workspace_id": workspace_id,
            "domain": ws["domain"],
            "category_key": cat_key,
            "category_name": req.category_name.strip(),
            "is_required": req.is_required,
            "status": ChecklistItemStatus.NOT_UPLOADED.value,
            "extracted_records_count": 0,
            "last_updated": now,
        }
        self.db.insert_checklist_items([item_dict])
        return ChecklistItemResponse(**item_dict)

    def get_readiness_report(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
    ) -> ReadinessReport:
        """Computes true operational readiness from validated checklist items."""
        ws = self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        items = self.db.get_checklist_items(workspace_id)
        return ChecklistEngine.calculate_readiness(
            checklist_items=items,
            domain=ws["domain"],
            workspace_id=workspace_id,
        )

    def update_item_status(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
        category_key: str,
        status: ChecklistItemStatus,
        document_id: Optional[str] = None,
        filename: Optional[str] = None,
        extracted_records_count: int = 0,
        validation_errors: Optional[str] = None,
    ) -> bool:
        """
        Updates category ingestion status.
        Enforces tenant authorization over workspace before performing modification.
        """
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        return self.db.update_checklist_item_status(
            workspace_id=workspace_id,
            category_key=category_key,
            status=status.value,
            document_id=document_id,
            filename=filename,
            extracted_records_count=extracted_records_count,
            validation_errors=validation_errors,
        )

    # -----------------------------------------------------------------------
    # Escalation Queue Operations
    # -----------------------------------------------------------------------
    def list_escalations(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
        status: Optional[str] = None,
    ) -> List[EscalationListItem]:
        """Lists escalation records strictly for the authorized workspace."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        records = self.db.list_escalations(workspace_id=workspace_id, status=status)
        return [EscalationListItem(**r) for r in records]

    def get_escalation(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
        escalation_id: str,
    ) -> EscalationResponse:
        """Retrieves single escalation dossier with strict workspace ownership check."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        record = self.db.get_escalation(workspace_id=workspace_id, escalation_id=escalation_id)
        if not record:
            raise LookupError(f"Escalation dossier '{escalation_id}' not found in workspace '{workspace_id}'.")
        return EscalationResponse(**record)

    def review_escalation(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
        escalation_id: str,
        req: EscalationReviewRequest,
    ) -> EscalationResponse:
        """Submits human supervisor review decision with state machine validation."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        updated = self.db.update_escalation_review(
            workspace_id=workspace_id,
            escalation_id=escalation_id,
            status=req.status.value,
            human_decision=req.human_decision,
            reviewer_notes=req.reviewer_notes,
            resolved_by=req.resolved_by,
        )
        if not updated:
            raise LookupError(f"Escalation dossier '{escalation_id}' not found in workspace '{workspace_id}'.")
        return EscalationResponse(**updated)

    # -----------------------------------------------------------------------
    # Widget Token Operations (Public Scoped Auth)
    # -----------------------------------------------------------------------
    def get_widget_token(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
    ) -> WidgetTokenResponse:
        """Retrieves or creates scoped widget token for public website embedding."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        token = self.db.get_or_create_widget_token(workspace_id)
        return WidgetTokenResponse(
            workspace_id=workspace_id,
            widget_token=token,
            revoked=False,
            message="Public widget token active. Safe for embedding in website HTML.",
        )

    def rotate_widget_token(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
    ) -> WidgetTokenResponse:
        """Rotates widget token immediately, invalidating previous browser embeds."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        new_token = self.db.rotate_widget_token(workspace_id)
        return WidgetTokenResponse(
            workspace_id=workspace_id,
            widget_token=new_token,
            revoked=False,
            message="Widget token rotated successfully. Update your website embed snippet.",
        )

    def revoke_widget_token(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
    ) -> WidgetTokenResponse:
        """Revokes widget token, immediately disabling external chat widget queries."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        self.db.revoke_widget_token(workspace_id)
        return WidgetTokenResponse(
            workspace_id=workspace_id,
            widget_token="",
            revoked=True,
            message="Widget token revoked. External widget chat queries will be rejected.",
        )

    def verify_widget_token(
        self,
        workspace_id: str,
        widget_token: str,
    ) -> Dict[str, Any]:
        """
        Validates widget token for public chat queries.
        Rejects non-widget tokens, revoked tokens, and mismatched workspaces.
        """
        if not widget_token or not widget_token.startswith("wtk_"):
            raise PermissionError("Invalid widget token format. Widget token must begin with 'wtk_'.")

        ws = self.db.get_workspace_by_widget_token(widget_token)
        if not ws:
            raise PermissionError("Widget token is invalid or has been revoked.")

        if ws["workspace_id"] != workspace_id:
            raise PermissionError("Widget token does not grant access to the requested workspace.")

        if not ws["is_active"]:
            raise PermissionError("Workspace is currently inactive.")

        return ws

    # -----------------------------------------------------------------------
    # Multi-Turn Session Memory Operations
    # -----------------------------------------------------------------------
    def save_session_turn(
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
        """Persists a single turn into session memory."""
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

    def get_session_history(
        self,
        workspace_id: str,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieves chronological turn history strictly for the designated workspace and session."""
        return self.db.get_session_turns(workspace_id=workspace_id, session_id=session_id, limit=limit)

    def clear_session_history(
        self,
        workspace_id: str,
        session_id: str,
    ) -> int:
        """Clears all turns for a specific session."""
        return self.db.clear_session_turns(workspace_id=workspace_id, session_id=session_id)

    get_session_turns = get_session_history
    clear_session_turns = clear_session_history

    # -----------------------------------------------------------------------
    # Long-Term Customer Memory Operations (GDPR & TTL Lifecycle)
    # -----------------------------------------------------------------------
    def create_customer_memory(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
        customer_id: str,
        category: str,
        content: str,
        confidence: float = 1.0,
        ttl_days: Optional[int] = None,
    ) -> CustomerMemoryResponse:
        """Creates long-term customer context with optional configurable retention."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        expires_at = None
        if ttl_days is not None and ttl_days > 0:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

        record = self.db.create_customer_memory(
            workspace_id=workspace_id,
            tenant_id=authenticated_tenant_id,
            customer_id=customer_id,
            category=category,
            content=content,
            confidence=confidence,
            expires_at=expires_at,
        )
        return CustomerMemoryResponse(**record)

    def list_customer_memories(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
        customer_id: str,
        include_expired: bool = False,
    ) -> List[CustomerMemoryResponse]:
        """Lists active customer memories with automatic read-time TTL expiration filtering."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        records = self.db.list_customer_memories(
            workspace_id=workspace_id,
            customer_id=customer_id,
            include_expired=include_expired,
        )
        return [CustomerMemoryResponse(**r) for r in records]

    def delete_customer_memory(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
        memory_id: str,
    ) -> bool:
        """Deletes a specific customer memory record."""
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        return self.db.delete_customer_memory(workspace_id=workspace_id, memory_id=memory_id)

    def delete_all_customer_memories(
        self,
        authenticated_tenant_id: str,
        workspace_id: str,
        customer_id: str,
    ) -> int:
        """
        Executes technical right-to-be-forgotten / GDPR memory wipe.
        Irrevocably deletes all customer memory entries for the specified customer in this workspace.
        """
        self.verify_workspace_access(authenticated_tenant_id, workspace_id)
        return self.db.delete_all_customer_memories(workspace_id=workspace_id, customer_id=customer_id)

