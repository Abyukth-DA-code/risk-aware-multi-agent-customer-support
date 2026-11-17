"""
Comprehensive Test Suite for Phase C: Multi-Format Document Ingestion & Tenant Vector Indexing
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
"""

import io
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.rag.ingestion.chunker import SemanticChunker
from src.rag.ingestion.extractors import (
    CSVExtractor,
    DocxExtractor,
    ExtractedDocument,
    PDFExtractor,
    TextExtractionError,
    TextExtractor,
    get_extractor,
)
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.workspaces.db import WorkspaceDB
from src.workspaces.models import BusinessDomain, ChecklistItemStatus, WorkspaceCreateRequest, TenantRegisterRequest
from src.workspaces.service import WorkspaceService


@pytest.fixture
def temp_workspace_env(tmp_path):
    """Provides isolated DB and Chroma client for ingestion testing."""
    import chromadb
    db_file = tmp_path / "test_tenants_ingest.db"
    db = WorkspaceDB(db_path=db_file)
    chroma_dir = tmp_path / "chroma_test"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    service = WorkspaceService(db=db)
    pipeline = DocumentIngestionPipeline(db=db, chroma_client=chroma_client)
    return {
        "db": db,
        "chroma_client": chroma_client,
        "service": service,
        "pipeline": pipeline,
    }


# ===========================================================================
# 1. Multi-Format Text Extractor Unit Tests
# ===========================================================================

def test_text_extractor_valid_and_invalid():
    """Verify TXT extraction quality checks."""
    extractor = TextExtractor()

    # Valid substantive policy text
    valid_text = (
        "Eligible products can be returned within 30 days of delivery. "
        "The customer must provide the original receipt and ensure items are unworn and in original packaging. "
        "Refunds are processed within 5 business days after inspection."
    )
    doc = extractor.extract(valid_text.encode("utf-8"), "return_policy.txt")
    assert doc.word_count > 20
    assert "Eligible products" in doc.full_text

    # Empty file error
    with pytest.raises(TextExtractionError, match="empty"):
        extractor.extract(b"   \n\t  ", "empty.txt")

    # Too short text error (<50 characters)
    with pytest.raises(TextExtractionError, match="too short"):
        extractor.extract(b"Too short policy", "short.txt")

    # Mostly non-alphanumeric noise
    with pytest.raises(TextExtractionError, match="insufficient substantive text"):
        extractor.extract(b"--------------------------------------------------------------------------------", "symbols.txt")


def test_csv_extractor_valid_and_invalid():
    """Verify CSV extraction parses tabular policy rows into structured text."""
    extractor = CSVExtractor()

    valid_csv = (
        "Policy_Category,Condition,Timeframe,Fee\n"
        "Electronics,Unopened,14 days,Zero\n"
        "Apparel,Tags attached,30 days,Zero\n"
        "Clearance Items,Non-returnable,N/A,Full\n"
    )
    doc = extractor.extract(valid_csv.encode("utf-8"), "returns.csv")
    assert "Policy_Category: Electronics" in doc.full_text
    assert "Condition: Tags attached" in doc.full_text

    # Empty or header-only CSV
    with pytest.raises(TextExtractionError, match="at least a header and 1 data row"):
        extractor.extract(b"Header1,Header2\n", "single_row.csv")


def test_docx_extractor_valid():
    """Verify Word (.docx) document extraction."""
    import docx
    doc_io = io.BytesIO()
    doc_obj = docx.Document()
    doc_obj.add_heading("Customer Support Attendance Policy", level=1)
    doc_obj.add_paragraph(
        "Students are required to maintain a minimum of 75% attendance in all enrolled courses. "
        "Failure to satisfy this requirement will result in being barred from the semester-end examination."
    )
    doc_obj.save(doc_io)

    extractor = DocxExtractor()
    extracted = extractor.extract(doc_io.getvalue(), "attendance_policy.docx")
    assert "minimum of 75% attendance" in extracted.full_text


def test_pdf_extractor_valid_and_malformed():
    """Verify PDF extraction and malformed byte detection."""
    extractor = PDFExtractor()

    # Valid synthetic PDF
    valid_pdf_bytes = b"""%PDF-1.4
1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj
2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj
3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources <</Font <</F1 4 0 R>>>> /Contents 5 0 R>> endobj
4 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj
5 0 obj <</Length 145>> stream
BT
/F1 12 Tf
72 712 Td
(Eligible items can be returned within 30 days of delivery with original proof of purchase. Refunds are issued after validation.) Tj
ET
endstream endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000227 00000 n
0000000294 00000 n
trailer <</Size 6 /Root 1 0 R>>
startxref
490
%%EOF"""
    doc = extractor.extract(valid_pdf_bytes, "policy.pdf")
    assert "Eligible items can be returned" in doc.full_text

    # Corrupt/malformed PDF
    with pytest.raises(TextExtractionError, match="Failed to parse PDF"):
        extractor.extract(b"Not a valid pdf binary stream at all!", "corrupt.pdf")


def test_unsupported_file_extension():
    """Verify unsupported file types are rejected immediately."""
    with pytest.raises(TextExtractionError, match="Unsupported file format"):
        get_extractor("malicious_script.exe")


# ===========================================================================
# 2. Semantic Chunker Unit Tests
# ===========================================================================

def test_semantic_chunker():
    """Verify chunking splits text with boundaries and metadata."""
    chunker = SemanticChunker(target_chars=200, overlap_chars=40)
    text = (
        "Section 1: General Product Return Rules.\n"
        "Products must be returned within the active return window. "
        "All tags must remain attached and the product must not show signs of wear.\n\n"
        "Section 2: Return Shipping and Handling.\n"
        "Customers are provided a prepaid return label. "
        "Drop-off can be completed at any authorized postal station."
    )
    doc = ExtractedDocument("policy.txt", text, [{"page_number": 1, "text": text}])
    chunks = chunker.chunk_document(doc, "DOC001")

    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_id == "DOC001"
        assert c.chunk_id.startswith("DOC001_C")
        assert len(c.text) >= 40


# ===========================================================================
# 3. Document Ingestion Pipeline & ChromaDB Indexing Tests
# ===========================================================================

def test_pipeline_ingest_duplicate_and_delete(temp_workspace_env):
    """
    Test complete lifecycle:
    1. Ingest document -> verified in Chroma collection and checklist.
    2. Duplicate ingestion -> rejected with ValueError.
    3. Document deletion -> purges vector store and resets checklist.
    """
    service = temp_workspace_env["service"]
    pipeline = temp_workspace_env["pipeline"]
    chroma_client = temp_workspace_env["chroma_client"]

    # Setup tenant and workspace
    t = service.register_tenant(TenantRegisterRequest(name="ShopCo", email="shop@test.com"))
    w = service.create_workspace(t.tenant_id, WorkspaceCreateRequest(name="Shop", domain=BusinessDomain.ECOMMERCE))

    policy_content = (
        "Return requests for eligible products must be initiated within the applicable return window "
        "and comply with condition requirements. Items must be undamaged with original packaging intact. "
        "Refunds are credited to the original payment method within 5 business days."
    ).encode("utf-8")

    # 1. Ingest document
    result = pipeline.ingest_document(
        workspace_id=w.workspace_id,
        tenant_id=t.tenant_id,
        category_key="return_policy",
        filename="product_returns.txt",
        file_bytes=policy_content,
    )
    assert result["status"] == "Validated"
    assert result["chunks_count"] >= 1
    doc_id = result["document_id"]

    # Verify chunks exist in isolated ChromaDB collection
    coll = chroma_client.get_collection(f"workspace_{w.workspace_id}")
    assert coll.count() == result["chunks_count"]

    # Verify checklist item updated to Validated
    checklist = service.get_checklist(t.tenant_id, w.workspace_id)
    ret_item = next(i for i in checklist if i.category_key == "return_policy")
    assert ret_item.status == ChecklistItemStatus.VALIDATED
    assert ret_item.document_id == doc_id
    assert ret_item.extracted_records_count == result["chunks_count"]

    # 2. Duplicate Detection
    with pytest.raises(ValueError, match="Duplicate document detected"):
        pipeline.ingest_document(
            workspace_id=w.workspace_id,
            tenant_id=t.tenant_id,
            category_key="return_policy",
            filename="copy_of_returns.txt",
            file_bytes=policy_content,
        )

    # 3. Delete Document
    deleted = pipeline.delete_document(w.workspace_id, t.tenant_id, doc_id)
    assert deleted is True

    # Verify vector chunks were purged from ChromaDB
    assert coll.count() == 0

    # Verify checklist item reset to Not Uploaded
    checklist_after = service.get_checklist(t.tenant_id, w.workspace_id)
    ret_item_after = next(i for i in checklist_after if i.category_key == "return_policy")
    assert ret_item_after.status == ChecklistItemStatus.NOT_UPLOADED
    assert ret_item_after.document_id is None
    assert ret_item_after.extracted_records_count == 0


# ===========================================================================
# 4. Cross-Tenant Security Invariant Tests
# ===========================================================================

def test_cross_tenant_document_upload_and_delete_blocking(temp_workspace_env, monkeypatch):
    """
    Critical Security Invariant:
    Tenant B must never be able to upload, list, or delete documents in Tenant A's workspace.
    """
    service = temp_workspace_env["service"]
    pipeline = temp_workspace_env["pipeline"]

    from src.api import workspace_routes
    monkeypatch.setattr(workspace_routes, "_workspace_service", service)
    monkeypatch.setattr(workspace_routes, "_ingestion_pipeline", pipeline)
    client = TestClient(app)

    # Register Tenant A & Workspace A
    t_a = client.post("/api/v1/tenants/register", json={"name": "Tenant A", "email": "ta@test.com"}).json()
    w_a = client.post(
        "/api/v1/workspaces",
        json={"name": "Workspace A", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {t_a['api_key']}"},
    ).json()

    # Register Tenant B & Workspace B
    t_b = client.post("/api/v1/tenants/register", json={"name": "Tenant B", "email": "tb@test.com"}).json()

    headers_a = {"Authorization": f"Bearer {t_a['api_key']}"}
    headers_b = {"Authorization": f"Bearer {t_b['api_key']}"}

    # 1. Tenant A uploads valid policy to Workspace A
    sample_file = (
        "Eligible products may be returned within 30 days of purchase with valid proof. "
        "Defective items will be replaced or refunded promptly upon receipt and review."
    ).encode("utf-8")

    upload_res = client.post(
        f"/api/v1/workspaces/{w_a['workspace_id']}/documents/upload",
        data={"category_key": "return_policy"},
        files={"file": ("returns.txt", sample_file, "text/plain")},
        headers=headers_a,
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 2. CROSS-TENANT ATTACK: Tenant B attempts to upload document to Workspace A -> 403 Forbidden!
    attack_upload = client.post(
        f"/api/v1/workspaces/{w_a['workspace_id']}/documents/upload",
        data={"category_key": "refund_policy"},
        files={"file": ("hack.txt", sample_file, "text/plain")},
        headers=headers_b,
    )
    assert attack_upload.status_code == 403

    # 3. CROSS-TENANT ATTACK: Tenant B attempts to list Workspace A documents -> 403 Forbidden!
    attack_list = client.get(
        f"/api/v1/workspaces/{w_a['workspace_id']}/documents",
        headers=headers_b,
    )
    assert attack_list.status_code == 403

    # 4. CROSS-TENANT ATTACK: Tenant B attempts to delete Tenant A's document -> 403 Forbidden!
    attack_delete = client.delete(
        f"/api/v1/workspaces/{w_a['workspace_id']}/documents/{doc_id}",
        headers=headers_b,
    )
    assert attack_delete.status_code == 403

    # 5. Tenant A deletes own document: 200 OK
    delete_res = client.delete(
        f"/api/v1/workspaces/{w_a['workspace_id']}/documents/{doc_id}",
        headers=headers_a,
    )
    assert delete_res.status_code == 200
