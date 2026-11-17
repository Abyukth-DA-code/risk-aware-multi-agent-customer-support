# Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation

**Research Track:** Risk-Aware Adaptive Multi-Agent Orchestration
**Academic Domain:** Final-Year B.Tech CSE (AI & ML) Capstone Project
**Current Milestone:** Phase 5 Certified (273/273 Automated Tests Passing & Frozen Model Hash Integrity Validated)
**System Status:** 100% Operational Local Production Architecture (Cloud Deployment Planned as Future Roadmap)

---

## 1. Executive Summary & Research Motivation

Contemporary AI customer support systems face a critical dilemma:
1. **Unconstrained Large Language Models (LLMs)** are non-deterministic, prone to hallucination (e.g., fabricating unauthorized return exceptions or cash refunds), vulnerable to adversarial prompt injection, and transmit sensitive customer Personally Identifiable Information (PII) to third-party APIs.
2. **Traditional Rule-Based Chatbots** are rigid, incapable of understanding conversational context (such as pronouns or follow-up ellipses), and fail when inquiries deviate from exact keyword triggers.

### The Proposed Solution
This repository implements **Risk-Aware Adaptive Multi-Agent Orchestration**, a multi-tiered architecture that combines:
- **Specialized Small Machine Learning Models (M1–M6)** for fast, deterministic intent classification, sentiment analysis, risk scoring, task complexity, and provisional autonomy routing.
- **Context-Aware Follow-Up Reference Resolution** that resolves pronouns (*"Does it apply to damaged goods?"*) and ellipses using session-isolated conversation history without cross-session data leakage.
- **Tenant-Isolated Dense Retrieval-Augmented Generation (RAG)** in ChromaDB (HNSW index, cosine distance, `all-MiniLM-L6-v2`), generating answers strictly derived from authenticated policy documents with verified source citations.
- **Deterministic Safety Policy Engine** enforcing strict guardrails (`CRIT_01` to `SENS_FIN_06`) that automatically disqualify high-risk, legal, or financial inquiries from autonomous execution.
- **Human-in-the-Loop (HITL) Supervisor Desk** providing real-time case dossiers, full conversational context, diagnostic model confidences, and terminal-state immutability.
- **Customer Memory Management** providing persistent profile and preference retention with configurable TTL expiration and GDPR Right-to-be-Forgotten technical deletion.

---

## 2. End-to-End System Architecture

```
                                  Customer Inquiry
                                         │
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │    Context-Aware Follow-Up Reference Resolver   │
                 │   (Resolves Pronouns & Ellipses via Turn Memory) │
                 └───────────────────────┬─────────────────────────┘
                                         │ Disambiguated Query
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │       Fast-Path Model Cascade (Triage M1–M6)    │
                 │  • M1: Intent (47 classes)   • M2: Sentiment     │
                 │  • M3: Operational Risk      • M4: Complexity    │
                 │  • M5: NLI Entailment        • M6: Autonomy Tier │
                 └───────────────────────┬─────────────────────────┘
                                         │
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │      Tenant-Isolated Vector Retrieval (M4 RAG)  │
                 │   ChromaDB Cosine HNSW (Strictly by Workspace)  │
                 └───────────────────────┬─────────────────────────┘
                                         │
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │       Deterministic Safety Policy Engine        │
                 │   (Calculates Observable Confidence & Rules)    │
                 └───────────────────────┬─────────────────────────┘
                                         │
                 ┌───────────────────────┼─────────────────────────┐
                 ▼                       ▼                         ▼
          [ANSWER Tier]          [CLARIFY Tier]             [ESCALATE Tier]
                 │                       │                         │
     Grounded Extractive         Targeted Prompt            Structured Dossier
      Answer Synthesizer           to Customer              Persisted to SQLite
     + Source Citations                  │                         │
                 │                       │                         ▼
                 ▼                       ▼                ┌─────────────────┐
           Customer Chat           Customer Chat          │ Supervisor Desk │
             Response                Response             │  (/supervisor)  │
                                                          └─────────────────┘
```

---

## 3. Four Integrated User-Facing Web Surfaces

The platform provides four independent, professional web interfaces:

| Surface | URL Path | Target Audience | Purpose |
| :--- | :--- | :--- | :--- |
| 🛒 **Client Storefront** | `/widget/demo` or `http://127.0.0.1:8501` | End Customers | Live e-commerce shopping experience with embedded multi-agent AI concierge. |
| 🏢 **Business Admin Console** | `/admin` | Business Administrators | Workspace management, document policy upload, chunking, readiness checks, and GDPR memory controls. |
| 🛡️ **Human Supervisor Desk** | `/supervisor` | Customer Support Leads | Reviewing escalated dossiers, inspecting model confidences, and executing authorized terminal resolutions. |
| 📖 **Interactive API Documentation** | `/docs` | Developers & Integrators | OpenAPI / Swagger interactive specification for all 25+ REST endpoints. |

---

## 4. Quick Start: Running the Project Locally (2 Minutes)

### Prerequisites
- Python 3.11 or 3.12 (64-bit)
- PowerShell (Windows) or Bash (macOS/Linux)
- Git

### Step 1: Clone and Set Up Virtual Environment
```powershell
# Clone the repository
git clone https://github.com/your-username/customer-support-ai.git
cd customer-support-ai

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start the FastAPI Platform Backend
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
*The backend is now live at `http://127.0.0.1:8000`.*

### Step 3: Launch the Streamlit E-Commerce Storefront
In a second terminal:
```powershell
streamlit run hemanth_store_app.py
```
*The customer storefront is now live at `http://localhost:8501`.*

---

## 5. Pre-Configured Test Credentials & Workspace

For instant testing and viva demonstration, the repository includes a pre-seeded, fully operational workspace:

| Parameter | Pre-Configured Value | Notes |
| :--- | :--- | :--- |
| **Workspace Name** | `Hemanth Online store` | Active E-Commerce Workspace |
| **Workspace ID** | `ws_26d7520d00c0` | Dedicated Chroma collection & SQLite records |
| **Public Scoped Widget Token** | `wtk_gSIAy-ZvQ_S0i_rLTozSwRkyvBtL31ob` | Rate-limited public token for storefront embedding |
| **Master Admin API Key** | `cap_demo_admin_key` | Full administrative authorization for `/admin` & `/supervisor` |
| **Operational Readiness** | **100.0% Complete** | 10 validated policy documents, 66 chunks indexed |
| **Active Customer ID** | `cust_hemanth_01` | Pre-seeded with profile, preferences, and order history |

---

## 6. How to Verify All Core Capabilities (Demonstration Script)

### Scenario A: Grounded Policy Q&A with Authentic Citations
1. Open the Storefront at `http://localhost:8501` (or `http://127.0.0.1:8000/widget/demo`).
2. Ask: *"What is your return policy and how many days do I have to return an item?"*
3. **Verified Output:** AI quotes the official 30-day return policy and outputs verified citation: `[doc_f922b681da26] return_policy.txt (Authority: High, Status: Current)`.

### Scenario B: Multi-Turn Contextual Pronoun Resolution
1. Immediately in the same conversation, ask: *"Does it apply to damaged goods?"*
2. **Verified Output:**
   - Reference Resolver detects anaphora (*"it"* &rarr; return policy).
   - Generates resolved search query: `"What is your return policy — Regarding follow-up: Does it apply to damaged goods?"`.
   - AI outputs the 48-hour damaged transit claim procedure with citations.

### Scenario C: Authenticated Customer Account & Order Lookup
1. Ask: *"Can you check my orders?"*
2. **Verified Output:** The assistant queries authenticated Customer Memory (`database/tenants.db`), lists Order `#ORD-99482` (Sony Headphones, delivered via FedEx), and details return eligibility.

### Scenario D: High-Risk Deterministic Safety Escalation
1. Ask: *"I demand an immediate cash refund of $850 on order #ORD-99482 right now or I will file a lawsuit!"*
2. **Verified Output:**
   - Safety rule triggers on high financial claim and legal threat.
   - The bot safely refuses unilateral financial commitment and creates an escalation ticket: `esc_...`.
3. Open the **Supervisor Desk** at `http://127.0.0.1:8000/supervisor` with key `cap_demo_admin_key`.
4. Inspect the **Case Dossier**, view the audit trail, and resolve the ticket.

---

## 7. Automated Test Suite & Quality Assurance

The repository includes a comprehensive automated test suite with **273 test cases across 32 modules**:

```powershell
# Execute complete regression test suite
python -m pytest tests/ -q
```
**Latest Audit Result:** `273 passed in 50.36s (100% pass rate, 0 failures)`.

### Cryptographic Asset Freeze Verification
All machine learning model weights and canonical evaluation datasets are frozen and cryptographically audited:
- Intent, Sentiment, Risk, Complexity, Escalation, and Autonomy models (`src/models/*.joblib`) match canonical SHA-256 baseline hashes.
- Zero data leakage between Train, Validation, and Test splits.
- Master provenance log: `data/processed/data_provenance.csv`.

---

## 8. Repository Structure

```
Final_Year_AIML_Capstron_Project/
├── configs/                               # Autonomy rules and safety policy thresholds
│   └── autonomy_rules.yaml                # Deterministic guardrails (CRIT_01 to SENS_FIN_06)
├── database/                              # Relational database & vector storage
│   ├── tenants.db                         # SQLite operational database (tenants, memory, tickets)
│   └── chroma_db/                         # Local ChromaDB persistent collections
├── vector_store/
│   └── rag_chroma/                        # Multi-tenant partitioned vector index (66 chunks/ws)
├── data/
│   ├── raw/V4/                            # Frozen canonical raw datasets
│   └── processed/                         # Split data (train, val, test) with provenance
├── sample_documents/
│   └── ecommerce_hemanth/                 # 10 verified policy documents for live store
├── src/
│   ├── agents/                            # Multi-agent implementations
│   │   ├── grounded_generator.py          # Extractive citation-backed answer generator
│   │   ├── policy_agent.py                # Safety policy evaluation agent
│   │   ├── retrieval_agent.py             # Workspace-isolated vector retriever
│   │   ├── supervisor_agent.py            # Adaptive resolution & escalation queuing
│   │   └── triage_agent.py                # Fast-path model cascade
│   ├── api/                               # FastAPI backend application
│   │   ├── main.py                        # Server entry point & CORS configuration
│   │   ├── workspace_routes.py            # 25+ REST endpoints (chat, documents, memory)
│   │   └── supervisor_routes.py           # Human reviewer queue & dossier endpoints
│   ├── memory/                            # Conversation & customer context
│   │   ├── customer_memory.py             # Persistent long-term memory (TTL & GDPR RTBF)
│   │   ├── resolver.py                    # Multi-turn pronoun & ellipsis reference resolver
│   │   └── session_memory.py              # Ephemeral session-turn store
│   ├── orchestration/                     # Multi-agent coordination engine
│   │   ├── graph.py                       # Master orchestration pipeline
│   │   ├── safety_policy.py               # Deterministic rule engine
│   │   └── state.py                       # Shared typed state container
│   ├── static/                            # Web user interface assets
│   │   ├── admin.html                     # Business Admin Console UI
│   │   ├── demo.html                      # Storefront demo UI
│   │   ├── supervisor.html                # Human Supervisor Desk UI
│   │   └── widget.js                      # Standalone vanilla JS embeddable widget
│   └── workspaces/                        # Multi-tenant isolation & database access
│       ├── db.py                          # Relational SQLite data access layer
│       ├── models.py                      # Pydantic schemas for multi-tenancy
│       └── service.py                     # Tenant authorization & checklist engine
├── tests/                                 # 32 test modules (273 automated tests)
├── hemanth_store_app.py                   # Streamlit e-commerce storefront web app
├── requirements.txt                       # Python dependencies
├── PHASE_5_ACCEPTANCE_REPORT.md           # Authoritative forensic audit report
└── README.md                              # Master project documentation (this file)
```

---

## 9. Academic Research Contributions (For Paper & Journal Writing)

This repository serves as the empirical foundation for academic research publications in Multi-Agent AI and Trustworthy NLP.

### Key Research Contributions:
1. **Adaptive Autonomy Calibration:** Replaces static rule trees with observable multi-factor confidence evaluation combining semantic similarity, evidence quality, intent certainty, and risk scoring.
2. **Context-Aware Reference Resolution without Leakage:** Demonstrates deterministic resolution of pronouns and conversational ellipsis within strict session boundaries, eliminating cross-session and cross-tenant leakage.
3. **Guaranteed Zero-Hallucination Extractive Generation:** Solves enterprise financial liability by restricting autonomous answers to verified policy citations while escalating ungrounded or disputed claims.
4. **Tenant Isolation with Cryptographic Integrity:** Enforces workspace-level vector collection partitioning, public scoped widget tokens, and verifiable terminal-state audit logs.

*For detailed paper structure, mathematical formulations, and LaTeX tables, refer to [`RESEARCH_PAPER_NOTES.md`](RESEARCH_PAPER_NOTES.md).*

---

## 10. Future Work: Cloud Transition Roadmap

As part of the ongoing project evolution and future scope:
- **Containerization:** Production `Dockerfile` and `docker-compose.yml` are created and ready for multi-container deployment.
- **AWS Infrastructure Blueprint:** Architecture designed for **AWS ECS Fargate** (serverless compute), **Amazon Aurora Serverless v2 PostgreSQL with `pgvector`** (high-concurrency partitioned vector storage), and **Amazon S3 with KMS encryption** (policy documents).
- **Enterprise Integrations:** Webhook connectors planned for **Zoho Desk** and **Salesforce Service Cloud** to bridge custom escalation dossiers with legacy corporate helpdesks.
- **Small-LLM Abstractive Synthesis:** Exploring local quantized models (e.g. Llama-3-8B-Instruct via vLLM) strictly within the ANSWER tier for natural conversational tone while preserving deterministic safety gates.

---

## 11. License & Academic Attribution

This project is developed as an academic capstone project for the B.Tech Degree in Computer Science and Engineering (Artificial Intelligence and Machine Learning).
All rights reserved © 2026.
