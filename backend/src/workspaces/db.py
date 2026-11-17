"""
SQLite Relational Repository for Multi-Tenant Workspaces & Dynamic Checklists
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import uuid

ROOT = Path(__file__).resolve().parents[2]
_cand_db = [
    ROOT / "database" / "tenants.db",
    Path(__file__).resolve().parents[3] / "backend" / "database" / "tenants.db",
    Path(__file__).resolve().parents[3] / "database" / "tenants.db",
    Path("backend/database/tenants.db"),
    Path("database/tenants.db"),
]
DEFAULT_DB_PATH = next((c for c in _cand_db if c.exists()), _cand_db[0])


class WorkspaceDB:
    """Thread-safe SQLite repository with foreign-key enforcement and parameterized queries."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self.get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                name TEXT NOT NULL,
                domain TEXT NOT NULL CHECK (domain IN ('E-commerce', 'Education', 'Small Business')),
                description TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                workspace_id TEXT,
                hashed_key TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                revoked_at TEXT,
                FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS checklist_items (
                item_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                category_key TEXT NOT NULL,
                category_name TEXT NOT NULL,
                is_required INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'Not Uploaded',
                document_id TEXT,
                filename TEXT,
                extracted_records_count INTEGER NOT NULL DEFAULT 0,
                validation_errors TEXT,
                last_updated TEXT NOT NULL,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                UNIQUE (workspace_id, category_key)
            );

            CREATE TABLE IF NOT EXISTS workspace_documents (
                document_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                category_key TEXT NOT NULL,
                filename TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS escalations (
                escalation_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                session_id TEXT,
                customer_id TEXT,
                query TEXT NOT NULL,
                domain TEXT NOT NULL,
                intent TEXT,
                intent_confidence REAL,
                risk TEXT,
                risk_confidence REAL,
                complexity TEXT,
                provisional_autonomy_tier TEXT,
                final_autonomy_tier TEXT NOT NULL,
                evidence_quality TEXT,
                retrieval_similarity_score REAL,
                retrieved_chunks_json TEXT,
                safety_rule_triggered TEXT,
                escalation_reason TEXT NOT NULL,
                recommended_action TEXT,
                status TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'IN_REVIEW', 'RESOLVED', 'REJECTED')),
                human_decision TEXT,
                reviewer_notes TEXT,
                resolved_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS session_turns (
                turn_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                user_query TEXT NOT NULL,
                resolved_query TEXT,
                bot_response TEXT,
                final_autonomy_tier TEXT NOT NULL,
                citations_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS customer_memories (
                memory_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                memory_category TEXT NOT NULL,
                memory_content TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                status TEXT NOT NULL DEFAULT 'Active' CHECK (status IN ('Active', 'Deleted', 'Expired')),
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_workspaces_tenant ON workspaces(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_checklist_workspace ON checklist_items(workspace_id);
            CREATE INDEX IF NOT EXISTS idx_apikeys_hash ON api_keys(hashed_key);
            CREATE INDEX IF NOT EXISTS idx_escalations_workspace ON escalations(workspace_id);
            CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(workspace_id, status);
            CREATE INDEX IF NOT EXISTS idx_session_turns_lookup ON session_turns(workspace_id, session_id, turn_index);
            CREATE INDEX IF NOT EXISTS idx_cust_mem_lookup ON customer_memories(workspace_id, customer_id, status);
            CREATE INDEX IF NOT EXISTS idx_cust_mem_expires ON customer_memories(expires_at);
            """)

            # Backward-compatible migrations for existing databases
            cur = conn.execute("PRAGMA table_info(workspaces)")
            cols = {row["name"] for row in cur.fetchall()}
            if "widget_token" not in cols:
                conn.execute("ALTER TABLE workspaces ADD COLUMN widget_token TEXT")
            if "widget_token_revoked_at" not in cols:
                conn.execute("ALTER TABLE workspaces ADD COLUMN widget_token_revoked_at TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_workspaces_widget_token ON workspaces(widget_token) WHERE widget_token IS NOT NULL")

    # -----------------------------------------------------------------------
    # Tenant Operations
    # -----------------------------------------------------------------------
    def create_tenant(self, tenant_id: str, name: str, email: str) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO tenants (tenant_id, name, email, created_at) VALUES (?, ?, ?, ?)",
                (tenant_id, name, email, now),
            )
        return {"tenant_id": tenant_id, "name": name, "email": email, "created_at": now}

    def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT tenant_id, name, email, created_at FROM tenants WHERE tenant_id = ?",
                (tenant_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # -----------------------------------------------------------------------
    # API Key Operations
    # -----------------------------------------------------------------------
    def create_api_key(
        self,
        key_id: str,
        tenant_id: str,
        hashed_key: str,
        name: str = "Root Key",
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (key_id, tenant_id, workspace_id, hashed_key, name, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key_id, tenant_id, workspace_id, hashed_key, name, now),
            )
        return {
            "key_id": key_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "name": name,
            "created_at": now,
        }

    def get_api_key_record(self, hashed_key: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT key_id, tenant_id, workspace_id, hashed_key, name, created_at, revoked_at
                FROM api_keys
                WHERE hashed_key = ? AND revoked_at IS NULL
                """,
                (hashed_key,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # -----------------------------------------------------------------------
    # Workspace Operations
    # -----------------------------------------------------------------------
    def create_workspace(
        self,
        workspace_id: str,
        tenant_id: str,
        name: str,
        domain: str,
        description: Optional[str] = None,
        widget_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        token = widget_token or f"wtk_{secrets.token_urlsafe(24)}"
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO workspaces (workspace_id, tenant_id, name, domain, description, is_active, widget_token, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (workspace_id, tenant_id, name, domain, description, token, now, now),
            )
        return {
            "workspace_id": workspace_id,
            "tenant_id": tenant_id,
            "name": name,
            "domain": domain,
            "description": description,
            "is_active": True,
            "widget_token": token,
            "widget_token_revoked_at": None,
            "created_at": now,
            "updated_at": now,
        }

    def get_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT workspace_id, tenant_id, name, domain, description, is_active, widget_token, widget_token_revoked_at, created_at, updated_at
                FROM workspaces
                WHERE workspace_id = ?
                """,
                (workspace_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            res["is_active"] = bool(res["is_active"])
            return res

    def list_workspaces_for_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT workspace_id, tenant_id, name, domain, description, is_active, widget_token, widget_token_revoked_at, created_at, updated_at
                FROM workspaces
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                """,
                (tenant_id,),
            )
            rows = cur.fetchall()
            out = []
            for r in rows:
                item = dict(r)
                item["is_active"] = bool(item["is_active"])
                out.append(item)
            return out

    # -----------------------------------------------------------------------
    # Checklist Operations
    # -----------------------------------------------------------------------
    def insert_checklist_items(self, items: List[Dict[str, Any]]) -> None:
        with self.get_connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO checklist_items (
                    item_id, workspace_id, domain, category_key, category_name,
                    is_required, status, extracted_records_count, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["item_id"],
                        item["workspace_id"],
                        item["domain"],
                        item["category_key"],
                        item["category_name"],
                        1 if item["is_required"] else 0,
                        item.get("status", "Not Uploaded"),
                        item.get("extracted_records_count", 0),
                        item["last_updated"],
                    )
                    for item in items
                ],
            )

    def get_checklist_items(self, workspace_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT item_id, workspace_id, domain, category_key, category_name,
                       is_required, status, document_id, filename, extracted_records_count,
                       validation_errors, last_updated
                FROM checklist_items
                WHERE workspace_id = ?
                ORDER BY is_required DESC, category_name ASC
                """,
                (workspace_id,),
            )
            rows = cur.fetchall()
            out = []
            for r in rows:
                item = dict(r)
                item["is_required"] = bool(item["is_required"])
                out.append(item)
            return out

    def update_checklist_item_status(
        self,
        workspace_id: str,
        category_key: str,
        status: str,
        document_id: Optional[str] = None,
        filename: Optional[str] = None,
        extracted_records_count: int = 0,
        validation_errors: Optional[str] = None,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE checklist_items
                SET status = ?, document_id = ?, filename = ?,
                    extracted_records_count = ?, validation_errors = ?, last_updated = ?
                WHERE workspace_id = ? AND category_key = ?
                """,
                (
                    status,
                    document_id,
                    filename,
                    extracted_records_count,
                    validation_errors,
                    now,
                    workspace_id,
                    category_key,
                ),
            )
            return cur.rowcount > 0

    # -----------------------------------------------------------------------
    # Document Operations
    # -----------------------------------------------------------------------
    def insert_workspace_document(
        self,
        document_id: str,
        workspace_id: str,
        tenant_id: str,
        category_key: str,
        filename: str,
        content_hash: str,
        file_size_bytes: int,
        status: str = "Validated",
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO workspace_documents (
                    document_id, workspace_id, tenant_id, category_key,
                    filename, content_hash, file_size_bytes, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    workspace_id,
                    tenant_id,
                    category_key,
                    filename,
                    content_hash,
                    file_size_bytes,
                    status,
                    now,
                ),
            )
        return {
            "document_id": document_id,
            "workspace_id": workspace_id,
            "tenant_id": tenant_id,
            "category_key": category_key,
            "filename": filename,
            "content_hash": content_hash,
            "file_size_bytes": file_size_bytes,
            "status": status,
            "created_at": now,
        }

    def get_workspace_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT document_id, workspace_id, tenant_id, category_key,
                       filename, content_hash, file_size_bytes, status, created_at
                FROM workspace_documents
                WHERE document_id = ?
                """,
                (document_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def find_document_by_hash(self, workspace_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT document_id, workspace_id, tenant_id, category_key,
                       filename, content_hash, file_size_bytes, status, created_at
                FROM workspace_documents
                WHERE workspace_id = ? AND content_hash = ? AND status != 'Deleted'
                """,
                (workspace_id, content_hash),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_workspace_documents(self, workspace_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT document_id, workspace_id, tenant_id, category_key,
                       filename, content_hash, file_size_bytes, status, created_at
                FROM workspace_documents
                WHERE workspace_id = ? AND status != 'Deleted'
                ORDER BY created_at DESC
                """,
                (workspace_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def delete_workspace_document(self, document_id: str) -> bool:
        with self.get_connection() as conn:
            cur = conn.execute(
                "UPDATE workspace_documents SET status = 'Deleted' WHERE document_id = ?",
                (document_id,),
            )
            return cur.rowcount > 0

    # -----------------------------------------------------------------------
    # Escalation Queue Operations
    # -----------------------------------------------------------------------
    def create_escalation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Inserts a new traceable escalation record into the workspace queue."""
        now = datetime.now(timezone.utc).isoformat()
        item = dict(data)
        item.setdefault("status", "PENDING")
        item.setdefault("created_at", now)
        item.setdefault("updated_at", now)
        item.setdefault("resolved_at", None)
        item.setdefault("human_decision", None)
        item.setdefault("reviewer_notes", None)
        item.setdefault("resolved_by", None)

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO escalations (
                    escalation_id, workspace_id, tenant_id, session_id, customer_id,
                    query, domain, intent, intent_confidence, risk, risk_confidence,
                    complexity, provisional_autonomy_tier, final_autonomy_tier,
                    evidence_quality, retrieval_similarity_score, retrieved_chunks_json,
                    safety_rule_triggered, escalation_reason, recommended_action,
                    status, human_decision, reviewer_notes, resolved_by,
                    created_at, updated_at, resolved_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    item["escalation_id"],
                    item["workspace_id"],
                    item["tenant_id"],
                    item.get("session_id"),
                    item.get("customer_id"),
                    item["query"],
                    item["domain"],
                    item.get("intent"),
                    item.get("intent_confidence"),
                    item.get("risk"),
                    item.get("risk_confidence"),
                    item.get("complexity"),
                    item.get("provisional_autonomy_tier"),
                    item["final_autonomy_tier"],
                    item.get("evidence_quality"),
                    item.get("retrieval_similarity_score"),
                    item.get("retrieved_chunks_json"),
                    item.get("safety_rule_triggered"),
                    item["escalation_reason"],
                    item.get("recommended_action"),
                    item["status"],
                    item.get("human_decision"),
                    item.get("reviewer_notes"),
                    item.get("resolved_by"),
                    item["created_at"],
                    item["updated_at"],
                    item.get("resolved_at"),
                ),
            )
        return item

    def get_escalation(self, workspace_id: str, escalation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single escalation record ensuring workspace ownership."""
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM escalations
                WHERE workspace_id = ? AND escalation_id = ?
                """,
                (workspace_id, escalation_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_escalations(
        self, workspace_id: str, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Lists escalation records for a workspace with optional status filter."""
        with self.get_connection() as conn:
            if status:
                cur = conn.execute(
                    """
                    SELECT * FROM escalations
                    WHERE workspace_id = ? AND status = ?
                    ORDER BY created_at DESC
                    """,
                    (workspace_id, status),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT * FROM escalations
                    WHERE workspace_id = ?
                    ORDER BY created_at DESC
                    """,
                    (workspace_id,),
                )
            return [dict(r) for r in cur.fetchall()]

    def update_escalation_review(
        self,
        workspace_id: str,
        escalation_id: str,
        status: str,
        human_decision: Optional[str] = None,
        reviewer_notes: Optional[str] = None,
        resolved_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Applies a validated state transition and review decision to an escalation.
        Enforces lifecycle constraints:
        - PENDING -> IN_REVIEW, RESOLVED, REJECTED
        - IN_REVIEW -> RESOLVED, REJECTED
        - Terminal states (RESOLVED, REJECTED) cannot be reopened.
        """
        valid_statuses = {"PENDING", "IN_REVIEW", "RESOLVED", "REJECTED"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid escalation status '{status}'. Must be one of {valid_statuses}")

        current = self.get_escalation(workspace_id, escalation_id)
        if not current:
            return None

        curr_status = current["status"]
        if curr_status in ("RESOLVED", "REJECTED"):
            raise ValueError(f"Cannot modify escalation in terminal state '{curr_status}'.")

        if curr_status == "IN_REVIEW" and status == "PENDING":
            raise ValueError("Cannot transition escalation from 'IN_REVIEW' back to 'PENDING'.")

        now = datetime.now(timezone.utc).isoformat()
        resolved_at = now if status in ("RESOLVED", "REJECTED") else current.get("resolved_at")

        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE escalations
                SET status = ?,
                    human_decision = COALESCE(?, human_decision),
                    reviewer_notes = COALESCE(?, reviewer_notes),
                    resolved_by = COALESCE(?, resolved_by),
                    updated_at = ?,
                    resolved_at = ?
                WHERE workspace_id = ? AND escalation_id = ?
                """,
                (
                    status,
                    human_decision,
                    reviewer_notes,
                    resolved_by,
                    now,
                    resolved_at,
                    workspace_id,
                    escalation_id,
                ),
            )
        return self.get_escalation(workspace_id, escalation_id)

    # -----------------------------------------------------------------------
    # Public Widget Token Operations
    # -----------------------------------------------------------------------
    def get_or_create_widget_token(self, workspace_id: str) -> str:
        ws = self.get_workspace(workspace_id)
        if not ws:
            raise LookupError(f"Workspace '{workspace_id}' not found.")
        if ws.get("widget_token") and not ws.get("widget_token_revoked_at"):
            return ws["widget_token"]
        new_token = f"wtk_{secrets.token_urlsafe(24)}"
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE workspaces SET widget_token = ?, widget_token_revoked_at = NULL, updated_at = ? WHERE workspace_id = ?",
                (new_token, now, workspace_id),
            )
        return new_token

    def rotate_widget_token(self, workspace_id: str) -> str:
        ws = self.get_workspace(workspace_id)
        if not ws:
            raise LookupError(f"Workspace '{workspace_id}' not found.")
        new_token = f"wtk_{secrets.token_urlsafe(24)}"
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE workspaces SET widget_token = ?, widget_token_revoked_at = NULL, updated_at = ? WHERE workspace_id = ?",
                (new_token, now, workspace_id),
            )
        return new_token

    def revoke_widget_token(self, workspace_id: str) -> bool:
        ws = self.get_workspace(workspace_id)
        if not ws:
            raise LookupError(f"Workspace '{workspace_id}' not found.")
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE workspaces SET widget_token_revoked_at = ?, updated_at = ? WHERE workspace_id = ?",
                (now, now, workspace_id),
            )
        return True

    def get_workspace_by_widget_token(self, widget_token: str) -> Optional[Dict[str, Any]]:
        if not widget_token or not isinstance(widget_token, str):
            return None
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT workspace_id, tenant_id, name, domain, description, is_active, widget_token, widget_token_revoked_at, created_at, updated_at
                FROM workspaces
                WHERE widget_token = ? AND widget_token_revoked_at IS NULL
                """,
                (widget_token.strip(),),
            )
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            res["is_active"] = bool(res["is_active"])
            return res

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
        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        cit_json = json.dumps(citations or [], default=str)
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) + 1 AS next_idx FROM session_turns WHERE workspace_id = ? AND session_id = ?",
                (workspace_id, session_id),
            )
            turn_index = cur.fetchone()["next_idx"]
            conn.execute(
                """
                INSERT INTO session_turns (
                    turn_id, workspace_id, tenant_id, session_id, turn_index,
                    user_query, resolved_query, bot_response, final_autonomy_tier,
                    citations_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    workspace_id,
                    tenant_id,
                    session_id,
                    turn_index,
                    user_query,
                    resolved_query,
                    bot_response,
                    final_autonomy_tier,
                    cit_json,
                    now,
                ),
            )
        return {
            "turn_id": turn_id,
            "workspace_id": workspace_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "turn_index": turn_index,
            "user_query": user_query,
            "resolved_query": resolved_query,
            "bot_response": bot_response,
            "final_autonomy_tier": final_autonomy_tier,
            "citations_json": cit_json,
            "created_at": now,
        }

    def get_session_turns(
        self,
        workspace_id: str,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT turn_id, workspace_id, tenant_id, session_id, turn_index,
                       user_query, resolved_query, bot_response, final_autonomy_tier,
                       citations_json, created_at
                FROM session_turns
                WHERE workspace_id = ? AND session_id = ?
                ORDER BY turn_index ASC
                LIMIT ?
                """,
                (workspace_id, session_id, limit),
            )
            turns = []
            for row in cur.fetchall():
                d = dict(row)
                try:
                    d["citations"] = json.loads(d["citations_json"]) if d["citations_json"] else []
                except Exception:
                    d["citations"] = []
                turns.append(d)
            return turns

    def clear_session_turns(self, workspace_id: str, session_id: str) -> int:
        with self.get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM session_turns WHERE workspace_id = ? AND session_id = ?",
                (workspace_id, session_id),
            )
            return cur.rowcount

    # -----------------------------------------------------------------------
    # Long-Term Customer Memory Operations (GDPR & TTL Lifecycle)
    # -----------------------------------------------------------------------
    def create_customer_memory(
        self,
        workspace_id: str,
        tenant_id: str,
        customer_id: str,
        category: str,
        content: str,
        confidence: float = 1.0,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        memory_id = f"cmem_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO customer_memories (
                    memory_id, workspace_id, tenant_id, customer_id,
                    memory_category, memory_content, confidence, created_at,
                    expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Active')
                """,
                (
                    memory_id,
                    workspace_id,
                    tenant_id,
                    customer_id,
                    category,
                    content,
                    confidence,
                    now,
                    expires_at,
                ),
            )
        return {
            "memory_id": memory_id,
            "workspace_id": workspace_id,
            "tenant_id": tenant_id,
            "customer_id": customer_id,
            "memory_category": category,
            "memory_content": content,
            "confidence": confidence,
            "created_at": now,
            "expires_at": expires_at,
            "status": "Active",
        }

    def get_customer_memory(self, workspace_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT memory_id, workspace_id, tenant_id, customer_id,
                       memory_category, memory_content, confidence, created_at,
                       expires_at, status
                FROM customer_memories
                WHERE workspace_id = ? AND memory_id = ?
                  AND status = 'Active'
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (workspace_id, memory_id, now),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_customer_memories(
        self,
        workspace_id: str,
        customer_id: str,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        query = """
            SELECT memory_id, workspace_id, tenant_id, customer_id,
                   memory_category, memory_content, confidence, created_at,
                   expires_at, status
            FROM customer_memories
            WHERE workspace_id = ? AND customer_id = ?
        """
        params = [workspace_id, customer_id]
        if not include_expired:
            query += " AND status = 'Active' AND (expires_at IS NULL OR expires_at > ?)"
            params.append(now)
        query += " ORDER BY created_at DESC"

        with self.get_connection() as conn:
            cur = conn.execute(query, tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def delete_customer_memory(self, workspace_id: str, memory_id: str) -> bool:
        with self.get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM customer_memories WHERE workspace_id = ? AND memory_id = ?",
                (workspace_id, memory_id),
            )
            return cur.rowcount > 0

    def delete_all_customer_memories(self, workspace_id: str, customer_id: str) -> int:
        with self.get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM customer_memories WHERE workspace_id = ? AND customer_id = ?",
                (workspace_id, customer_id),
            )
            return cur.rowcount

    def prune_expired_customer_memories(self, workspace_id: Optional[str] = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            if workspace_id:
                cur = conn.execute(
                    "DELETE FROM customer_memories WHERE workspace_id = ? AND expires_at IS NOT NULL AND expires_at <= ?",
                    (workspace_id, now),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM customer_memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )
            return cur.rowcount


