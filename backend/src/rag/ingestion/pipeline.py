"""
Document Ingestion Pipeline for Multi-Tenant Workspaces
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Orchestrates secure file storage, multi-format text extraction, semantic chunking,
embedding generation, and tenant-isolated ChromaDB indexing.
"""

from datetime import datetime, timezone
import hashlib
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from src.rag.ingestion.chunker import SemanticChunker
from src.rag.ingestion.extractors import ExtractedDocument, TextExtractionError, get_extractor
from src.workspaces.db import WorkspaceDB
from src.workspaces.models import ChecklistItemStatus

ROOT = Path(__file__).resolve().parents[3]
_cand_vector = [
    ROOT / "vector_store" / "rag_chroma",
    Path(__file__).resolve().parents[4] / "backend" / "vector_store" / "rag_chroma",
    Path(__file__).resolve().parents[4] / "vector_store" / "rag_chroma",
    Path("backend/vector_store/rag_chroma"),
    Path("vector_store/rag_chroma"),
]
VECTOR_DIR = next((c for c in _cand_vector if c.exists()), _cand_vector[0])

_cand_tenants = [
    Path(__file__).resolve().parents[4] / "datasets" / "tenants",
    ROOT / "datasets" / "tenants",
    Path(__file__).resolve().parents[4] / "data" / "tenants",
    ROOT / "data" / "tenants",
    Path("datasets/tenants"),
    Path("data/tenants"),
]
DATA_TENANTS_DIR = next((c for c in _cand_tenants if c.exists()), _cand_tenants[0])
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


class DocumentIngestionPipeline:
    """End-to-end ingestion pipeline with strict tenant isolation and error recovery."""

    _shared_model = None

    def __init__(self, db: Optional[WorkspaceDB] = None, chroma_client: Optional[chromadb.PersistentClient] = None):
        self.db = db or WorkspaceDB()
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chroma_client or chromadb.PersistentClient(path=str(VECTOR_DIR))

        if DocumentIngestionPipeline._shared_model is None:
            try:
                DocumentIngestionPipeline._shared_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME, local_files_only=True
                )
            except Exception:
                DocumentIngestionPipeline._shared_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embed_model = DocumentIngestionPipeline._shared_model
        self.chunker = SemanticChunker()

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get_workspace_collection(self, workspace_id: str) -> chromadb.Collection:
        """Returns or creates the isolated ChromaDB vector collection for this workspace."""
        coll_name = f"workspace_{workspace_id}"
        return self.chroma_client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine", "workspace_id": workspace_id},
        )

    def ingest_document(
        self,
        workspace_id: str,
        tenant_id: str,
        category_key: str,
        filename: str,
        file_bytes: bytes,
        replace_document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes and indexes a document into the workspace's private knowledge base:
        1. File validation (size, type).
        2. Duplicate detection via SHA-256.
        3. Secure filesystem storage.
        4. Text extraction with quality checks.
        5. Semantic chunking.
        6. SentenceTransformer embeddings.
        7. Isolated ChromaDB collection upsert.
        8. Checklist item transition to 'Validated'.
        """
        # 1. Size Validation
        if len(file_bytes) == 0:
            raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            mb = len(file_bytes) / (1024 * 1024)
            raise ValueError(
                f"File size ({mb:.2f} MB) exceeds maximum allowed upload limit of 10 MB."
            )

        # 2. Duplicate Detection
        content_hash = self._compute_hash(file_bytes)
        existing_doc = self.db.find_document_by_hash(workspace_id, content_hash)
        if existing_doc and existing_doc["document_id"] != replace_document_id:
            raise ValueError(
                f"Duplicate document detected: exact same content was already uploaded as "
                f"'{existing_doc['filename']}' (Document ID: {existing_doc['document_id']})."
            )

        # Verify workspace exists and retrieve primary domain
        ws = self.db.get_workspace(workspace_id)
        if not ws:
            raise LookupError(f"Workspace '{workspace_id}' not found.")
        domain = ws["domain"]

        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        # 3. Secure Filesystem Storage
        storage_dir = DATA_TENANTS_DIR / tenant_id / "workspaces" / workspace_id / "documents"
        storage_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = f"{document_id}_{Path(filename).name}"
        saved_file_path = storage_dir / safe_filename

        try:
            with open(saved_file_path, "wb") as f:
                f.write(file_bytes)

            # 4. Text Extraction & Quality Checks
            extractor = get_extractor(filename)
            extracted_doc = extractor.extract(file_bytes=file_bytes, filename=filename)

            # 5. Semantic Chunking
            chunks = self.chunker.chunk_document(extracted_doc, document_id)
            if not chunks:
                raise TextExtractionError(
                    f"No valid text chunks could be extracted from '{filename}'."
                )

            # 6. Generate Embeddings
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embed_model.encode(
                chunk_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            # 7. Workspace Collection Indexing (with multi-tenant metadata guardrails)
            collection = self.get_workspace_collection(workspace_id)

            # Handle replacement of existing document: purge old chunks first
            if replace_document_id:
                try:
                    collection.delete(where={"document_id": replace_document_id})
                    self.db.delete_workspace_document(replace_document_id)
                except Exception:
                    pass

            chunk_ids = [c.chunk_id for c in chunks]
            metadatas = []
            for c in chunks:
                metadatas.append({
                    "workspace_id": workspace_id,
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "domain": domain,
                    "category_key": category_key,
                    "filename": filename,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "char_length": c.char_length,
                    "authority": "High",
                    "freshness_status": "Current",
                    "ingested_at": now,
                })

            collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas,
            )

            # 8. Record in Database
            doc_record = self.db.insert_workspace_document(
                document_id=document_id,
                workspace_id=workspace_id,
                tenant_id=tenant_id,
                category_key=category_key,
                filename=filename,
                content_hash=content_hash,
                file_size_bytes=len(file_bytes),
                status="Validated",
            )

            # 9. Mark checklist category as VALIDATED
            self.db.update_checklist_item_status(
                workspace_id=workspace_id,
                category_key=category_key,
                status=ChecklistItemStatus.VALIDATED.value,
                document_id=document_id,
                filename=filename,
                extracted_records_count=len(chunks),
                validation_errors=None,
            )

            return {
                "document_id": document_id,
                "workspace_id": workspace_id,
                "category_key": category_key,
                "filename": filename,
                "chunks_count": len(chunks),
                "chunks_indexed": len(chunks),
                "total_characters": extracted_doc.char_count,
                "character_count": extracted_doc.char_count,
                "status": "Validated",
                "indexed_at": now,
            }

        except Exception as exc:
            # Mark checklist item as 'Needs Attention' with clear error
            err_msg = str(exc)
            self.db.update_checklist_item_status(
                workspace_id=workspace_id,
                category_key=category_key,
                status=ChecklistItemStatus.NEEDS_ATTENTION.value,
                document_id=None,
                filename=filename,
                extracted_records_count=0,
                validation_errors=err_msg,
            )
            # Clean up raw file if it was created
            if saved_file_path.exists():
                try:
                    saved_file_path.unlink()
                except OSError:
                    pass
            raise

    def delete_document(self, workspace_id: str, tenant_id: str, document_id: str) -> bool:
        """
        Securely removes a document, purges vector chunks from ChromaDB,
        and resets the associated checklist category back to 'Not Uploaded'.
        """
        doc = self.db.get_workspace_document(document_id)
        if not doc or doc["workspace_id"] != workspace_id or doc["tenant_id"] != tenant_id:
            raise LookupError(f"Document '{document_id}' not found in workspace '{workspace_id}'.")

        category_key = doc["category_key"]

        # 1. Delete vector chunks from ChromaDB
        try:
            collection = self.get_workspace_collection(workspace_id)
            collection.delete(where={"document_id": document_id})
        except Exception:
            pass

        # 2. Mark document as deleted in DB
        self.db.delete_workspace_document(document_id)

        # 3. Reset checklist item back to Not Uploaded
        self.db.update_checklist_item_status(
            workspace_id=workspace_id,
            category_key=category_key,
            status=ChecklistItemStatus.NOT_UPLOADED.value,
            document_id=None,
            filename=None,
            extracted_records_count=0,
            validation_errors=None,
        )

        # 4. Remove physical file from disk
        storage_dir = DATA_TENANTS_DIR / tenant_id / "workspaces" / workspace_id / "documents"
        safe_filename = f"{document_id}_{Path(doc['filename']).name}"
        stored_path = storage_dir / safe_filename
        if stored_path.exists():
            try:
                stored_path.unlink()
            except OSError:
                pass

        return True
