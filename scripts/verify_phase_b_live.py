"""
Live verification script for Phase B endpoints, security invariants, and regressions.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starlette.testclient import TestClient
from src.api.main import app
from src.rag.retriever import RAGRetriever
from src.orchestration.model_service import ModelService

def main():
    print("=== LIVE VERIFICATION OF PHASE B & EXISTING SUBSYSTEMS ===")
    client = TestClient(app)
    import uuid
    uid = uuid.uuid4().hex[:6]

    # 1. Test Tenant A Registration
    r_a = client.post("/api/v1/tenants/register", json={"name": "Alpha Corp", "email": f"alpha_{uid}@test.com"})
    assert r_a.status_code == 201, f"Failed registration: {r_a.text}"
    data_a = r_a.json()
    tenant_a_id = data_a["tenant_id"]
    key_a = data_a["api_key"]
    print(f"[OK] Registered Tenant A: {tenant_a_id}")

    # 2. Test Tenant B Registration
    r_b = client.post("/api/v1/tenants/register", json={"name": "Beta University", "email": f"beta_{uid}@test.com"})
    assert r_b.status_code == 201, f"Failed registration: {r_b.text}"
    data_b = r_b.json()
    tenant_b_id = data_b["tenant_id"]
    key_b = data_b["api_key"]
    print(f"[OK] Registered Tenant B: {tenant_b_id}")

    # 3. Tenant A creates E-commerce Workspace
    ws_res_a = client.post(
        "/api/v1/workspaces",
        json={"name": "Alpha Retail", "domain": "E-commerce"},
        headers={"Authorization": f"Bearer {key_a}"},
    )
    assert ws_res_a.status_code == 201
    ws_a_id = ws_res_a.json()["workspace_id"]
    print(f"[OK] Created Workspace A ({ws_a_id}) under Tenant A")

    # 4. Tenant B creates Education Workspace
    ws_res_b = client.post(
        "/api/v1/workspaces",
        json={"name": "Beta College", "domain": "Education"},
        headers={"Authorization": f"Bearer {key_b}"},
    )
    assert ws_res_b.status_code == 201
    ws_b_id = ws_res_b.json()["workspace_id"]
    print(f"[OK] Created Workspace B ({ws_b_id}) under Tenant B")

    # 5. Security Test: Tenant B attempts to read Workspace A metadata
    cross_read = client.get(
        f"/api/v1/workspaces/{ws_a_id}",
        headers={"Authorization": f"Bearer {key_b}"},
    )
    assert cross_read.status_code == 403, f"Expected 403 Forbidden, got {cross_read.status_code}"
    print(f"[OK] Cross-tenant access blocked (Tenant B -> Workspace A): HTTP {cross_read.status_code}")

    # 6. Security Test: Tenant B attempts to spoof tenant_id header/body
    spoof_attempt = client.get(
        f"/api/v1/workspaces/{ws_a_id}?tenant_id={tenant_a_id}",
        headers={"Authorization": f"Bearer {key_b}", "X-Tenant-Id": tenant_a_id},
    )
    assert spoof_attempt.status_code == 403, f"Expected 403, got {spoof_attempt.status_code}"
    print(f"[OK] Spoofed tenant_id query/header blocked: HTTP {spoof_attempt.status_code}")

    # 7. Security Test: Tenant B attempts to read Workspace A checklist
    cross_chk = client.get(
        f"/api/v1/workspaces/{ws_a_id}/checklist",
        headers={"Authorization": f"Bearer {key_b}"},
    )
    assert cross_chk.status_code == 403
    print(f"[OK] Cross-tenant checklist access blocked: HTTP {cross_chk.status_code}")

    # 8. Check dynamic checklist domain conformance
    chk_a = client.get(f"/api/v1/workspaces/{ws_a_id}/checklist", headers={"Authorization": f"Bearer {key_a}"}).json()
    chk_b = client.get(f"/api/v1/workspaces/{ws_b_id}/checklist", headers={"Authorization": f"Bearer {key_b}"}).json()
    cats_a = [x["category_name"] for x in chk_a]
    cats_b = [x["category_name"] for x in chk_b]
    assert "Return Policy" in cats_a and "Attendance Policy" not in cats_a
    assert "Attendance Policy" in cats_b and "Return Policy" not in cats_b
    print(f"[OK] Dynamic checklists are strictly domain-specific (Ecomm vs Education)")

    # 9. Verify Readiness Report
    ready_a = client.get(f"/api/v1/workspaces/{ws_a_id}/readiness", headers={"Authorization": f"Bearer {key_a}"}).json()
    assert ready_a["is_ready_for_testing"] is False
    assert ready_a["readiness_percentage"] == 0.0
    assert len(ready_a["unanswerable_query_warnings"]) == 6
    print(f"[OK] Readiness is derived from backend checks (Initial readiness = 0.0%)")

    # 10. Verify Existing Demo Knowledge Base & RAG Retriever
    retriever = RAGRetriever()
    results = retriever.retrieve("How can I return an eligible product?", domain="E-commerce", top_k=2)
    assert len(results) > 0
    assert results[0]["document_id"] == "E001"
    print(f"[OK] Demonstration RAG collection 'capstone_knowledge_v1' operational (Doc E001, Sim: {results[0]['similarity']:.4f})")

    # 11. Verify Model Service (M1-M6)
    model_svc = ModelService.get_instance()
    analysis = model_svc.analyze_query("I want to cancel my order", domain="E-commerce")
    assert analysis["domain"]["domain"] == "E-commerce"
    assert "cancel_order" in analysis["intent"]["intent"].lower() or "order" in analysis["intent"]["intent"].lower()
    print(f"[OK] Models M1-M6 operational (Intent: {analysis['intent']['intent']}, Risk: {analysis['risk']['risk']})")

    print("=== ALL STEP 1 VERIFICATIONS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
