# Autonomous Multi-Agent Customer Support Resolution System with Risk-Aware Adaptive Orchestration and Human-in-the-Loop Escalation

**Academic Domain:** Final-Year B.Tech CSE (AI & ML) Capstone Major Project  
**Base Paper:** STELLAR (*ACL 2024*)  
**Evaluation Scope:** Strict Two-Policy Empirical Comparison (Policy A vs. Policy B)  
**Quality Assurance:** 274/274 Automated Regression Tests Passing (100% Pass Rate)  
**Architecture Status:** Production-Ready Modular Architecture (Frontend, Backend, Datasets, Config, Documentation, Tests, Scripts)

---

## 1. Executive Summary & Research Motivation

Modern automated customer support systems navigate a fundamental tradeoff between **operational autonomy** and **liability safety**:
1. **Unconstrained Generative LLMs:** Non-deterministic, hallucination-prone on policy terms, vulnerable to prompt injection, and transmit confidential customer Personally Identifiable Information (PII) to external cloud endpoints.
2. **Fixed Static Rule Trees & Pre-Retrieval Classifiers (Policy A):** Rigid, incapable of adapting when retrieval returns weak or contradictory evidence, routing complex multi-turn inquiries with static argmax thresholds that cause unsafe autonomous answers or high escalation bottlenecks.

### The Proposed Paradigm (Policy B: Risk-Aware Adaptive Autonomy)
This project introduces a **Multi-Signal Adaptive Multi-Agent Architecture**:
- **Multi-Signal Fast Triage Cascade (M1–M6):** Evaluates Intent (M1, 47 classes), Sentiment (M2), Operational Risk (M3), Complexity (M4), Entailment/Domain (M5), and Provisional Escalation Need (M6).
- **Hybrid Knowledge Retrieval:** Integrates BM25 lexical search with dense vector embeddings in ChromaDB (HNSW, Cosine similarity, `sentence-transformers/all-MiniLM-L6-v2`) under strict multi-tenant workspace isolation.
- **Evidence Verification & Confidence Arbitration:** Dynamic entailment verification between customer assertions and retrieved policy chunks, computing multi-factor observable confidence:
  $$\text{Confidence} = 0.35 \cdot S_{\text{sem}} + 0.30 \cdot C_{\text{intent}} + 0.15 \cdot F_{\text{doc}} + 0.10 \cdot A_{\text{evid}} + 0.10 \cdot C_{\text{db}}$$
- **Deterministic Safety Policy Engine:** Six non-negotiable guardrails (`CRIT_01` through `SENS_FIN_06`) that deterministically override provisional model autonomy when high risk, legal threats, or unverified financial claims appear.
- **Human-in-the-Loop (HITL) Supervisor Desk:** Structured case dossiers with immutable audit trails, turn-level memory snapshots, and supervisory resolution mechanisms.

---

## 2. Research Policy Comparison (Benchmark Results)

The system was evaluated across the frozen 501-case benchmark under rigorous empirical controls:

| Evaluation Metric | Policy A: Fixed M6 Autonomy | Policy B1: Proposed Adaptive (Original) | Policy B2: Risk-Calibrated Adaptive (Optimized) |
| :--- | :---: | :---: | :---: |
| **Routing Accuracy** | 24.35% (122/501) | **25.55%** (128/501) | **29.54%** (148/501) |
| **Macro-F1 Score** | 0.2785 | **0.2882** | **0.3142** |
| **Unsafe Autonomous Answers** | 6.39% (32/501) | **6.16%** (31/501) | **4.99%** (25/501) |
| **High / Critical Safety Breaches** | 0 / 30 (0.0%) | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** |
| **Knowledge-Gap Breaches** | 32 / 32 (100% fail) | **0 / 32 (0.0% breach)** | **0 / 32 (0.0% breach)** |
| **Escalation Recall** | 94.61% | **94.86%** | **95.23%** |
| **Escalation Precision** | 81.20% | **83.13%** | **84.43%** |
| **Escalation Workload** | 412 cases (82.2%) | **409 cases (81.6%)** | **379 cases (75.6%)** |
| **End-to-End Latency (p50 / p95)** | 54.3 ms / 112.5 ms | 218.4 ms / 384.2 ms | 224.1 ms / 391.0 ms |
| **Statistical Significance (McNemar)** | Baseline Reference | $p = 0.0412$ (vs Policy A) | $p = 0.00049$ (vs Policy A) |

> **Key Research Finding:** Policy B eliminates 100% of knowledge-gap autonomous hallucinations (32/32 prevented) and maintains zero High/Critical safety breaches, while Policy B2 reduces supervisor escalation workload by 33 cases (75.6% vs 82.2%) while increasing routing accuracy to 29.54%.

---

## 3. Reorganized Repository Structure

The repository is organized into a clean, professional, and modular layout ready for GitHub:

```
Final_Year_AIML_Capstron_Project/
├── backend/                               # Core backend service package
│   ├── database/                          # Persistent SQLite database & DDL
│   │   ├── schema.sql                     # Relational schema DDL
│   │   ├── tenants.db                     # Multi-tenant SQLite repository (tenants, workspaces, memory, dossiers)
│   │   └── chroma_db/                     # Local Chroma client state
│   ├── src/                               # Application source code
│   │   ├── agents/                        # Multi-agent implementations (Triage, Retrieval, Policy, Verification, Supervisor)
│   │   ├── api/                           # FastAPI REST routes (main.py, schemas.py, workspace_routes.py)
│   │   ├── memory/                        # Session memory, pronoun/ellipsis resolver, customer memory
│   │   ├── models/                        # Pre-trained scikit-learn models (M1-M6, encoders)
│   │   ├── models_train_code/             # Training scripts and evaluation code for classifiers
│   │   ├── orchestration/                 # LangGraph state machine, model service, safety policy engine
│   │   ├── rag/                           # BM25Okapi, ChromaDB retriever, ingestion pipeline, chunker
│   │   ├── static/                        # Web interfaces (demo.html, admin.html, supervisor.html, widget.js)
│   │   └── workspaces/                    # Multi-tenant isolation service, checklist engine, models
│   ├── tools/                             # Operational tool interfaces (customer_tools.py, order_tools.py, database.py)
│   └── vector_store/
│       └── rag_chroma/                    # High-density ChromaDB HNSW vector index (32 collections, 66 chunks/ws)
│
├── frontend/                              # Frontend client application
│   ├── hemanth_store_app.py               # Streamlit e-commerce storefront web application
│   └── static/                            # Standalone static assets (widget.js, demo, admin, supervisor UIs)
│
├── config/                                # System configuration & declarative safety rules
│   ├── autonomy_rules.yaml                # Deterministic safety tiers & rule criteria (CRIT_01 to SENS_FIN_06)
│   ├── domains.yaml                       # Domain-specific checklist requirements (E-commerce, Education, Small Business)
│   └── model_config.yaml                  # Model inference hyperparameters & weights
│
├── datasets/                              # All empirical data artifacts & provenances
│   ├── chromadb/                          # Canonical ChromaDB sqlite artifact
│   ├── powerbi/                           # PowerBI operational reporting extracts (10 CSVs)
│   ├── processed/                         # Split data with zero cross-split leakage
│   │   ├── autonomy/                      # train.csv, validation.csv, test.csv (501 cases)
│   │   ├── complexity/                    # train.csv, validation.csv, test.csv
│   │   ├── database/                      # Operational records (customers, orders, transactions, tickets)
│   │   ├── domain/                        # train.csv, validation.csv, test.csv
│   │   ├── escalation/                    # train.csv, validation.csv, test.csv
│   │   ├── human_feedback/               # Human feedback audit records
│   │   ├── intent/                        # train.csv, validation.csv, test.csv
│   │   ├── memory/                        # Synthetic customer conversational memory
│   │   ├── rag/                           # Chunks, documents, retrieval benchmarks
│   │   ├── risk/                          # train.csv, validation.csv, test.csv
│   │   ├── safety/                        # Critical safety benchmark queries
│   │   ├── sentiment/                     # Baseline and refined sentiment splits
│   │   ├── data_provenance.csv            # Cryptographic SHA-256 data split provenance log
│   │   └── FINAL_MASTER_GOLD.csv          # 3,519-row golden benchmark master dataset
│   ├── raw/V4/                            # Frozen canonical raw Excel master files (7 workbooks)
│   └── sample_documents/                  # Domain knowledge base documents (e.g., return policies, warranties)
│
├── documentation/                         # Comprehensive research documentation & deliverables
│   ├── figures/                           # Visual figures and architectural diagrams
│   │   ├── exploratory/                   # 8 dataset distribution and exploratory analysis plots
│   │   ├── frontend/                      # User interface audit screenshots
│   │   ├── results/                       # 6 high-resolution publication-quality evaluation graphs
│   │   └── Final_Research_Architecture.png# Master 300-DPI system architecture diagram
│   ├── reports/                           # Academic, forensic, and integrity audit reports
│   │   ├── data_audit/                    # Operational integrity reports, data dictionary, label audits
│   │   ├── model_cards/                   # Detailed model cards for M1–M6 classifiers
│   │   ├── DISCREPANCY_RESOLUTION_REPORT.md # Forensic resolution of evaluation discrepancies
│   │   ├── FINAL_RESEARCH_CHANGELOG.md    # Changelog documenting two-policy restructuring
│   │   ├── FINAL_RESULTS_RECONCILIATION.md# Final data reconciliation log
│   │   ├── IEEE_RESEARCH_PAPER_MASTER_COMPENDIUM.md # Formatted IEEE paper draft
│   │   ├── PHASE_5_ACCEPTANCE_REPORT.md   # System milestone acceptance report
│   │   ├── RESEARCH_PAPER_NOTES.md        # Technical research notes and LaTeX formulas
│   │   ├── TWO_POLICY_RESTRUCTURING_CHANGELOG.md # Restructuring log removing Policy C
│   │   ├── Verification_and_Gaps_Report.md# Empirical gap audit report
│   │   └── AWS_DEPLOYMENT_GUIDE.md        # Cloud architecture roadmap guide
│   └── results/                           # Primary deliverables and evaluation outputs
│       ├── evaluation_outputs/            # Row-level CSV/JSON logs across evaluation experiments
│       ├── Final_Major_Project_Research_Evaluation.docx # Master 30-chapter research compendium (Word)
│       └── Final_Results_Comparison.xlsx  # Professional 6-sheet evaluation workbook (Excel)
│
├── scripts/                               # Evaluation benchmarks, profiling, and generator scripts
│   ├── audit_full_resolution.py           # Verification script for dataset integrity and McNemar test
│   ├── benchmark_fixed_vs_adaptive_orchestration.py # Master benchmark runner comparing Policy A and B
│   ├── build_docx.py                      # Automated generator for master Word compendium
│   ├── generate_excel.py                  # Automated generator for multi-tab Excel results
│   ├── generate_figures.py                # Publication graph generator for all evaluation plots
│   ├── profile_pipeline_latencies.py      # Latency profiler for pipeline stages
│   └── run_m6_controlled_experiment.py    # Controlled experiment runner for M6 autonomy
│
├── tests/                                 # 32 automated test modules (274 test cases)
│   ├── conftest.py                        # Root pytest fixtures, path resolvers, and offline environment flags
│   ├── test_admin_phase_2.py              # Business Admin console & workspace test suite
│   ├── test_agent_orchestration_phase_e.py# Multi-agent graph orchestration tests
│   ├── test_api.py                        # Core FastAPI endpoint tests
│   ├── test_data_invariants.py            # Data invariant verification suite
│   ├── test_hybrid_retriever.py           # BM25 + Chroma hybrid retrieval tests
│   ├── test_safety_policy.py              # Safety engine and deterministic guardrail tests
│   └── ... (25 additional test modules)
│
├── .dockerignore                          # Docker build exclusion list
├── .gitignore                             # Clean repository exclusion rules
├── conftest.py                            # Top-level pytest configuration
├── docker-compose.yml                     # Multi-container orchestration specification
├── Dockerfile                             # Container build file for FastAPI backend
├── pytest.ini                             # Pytest discovery and isolation configuration
├── README.md                              # Repository overview and documentation (this file)
└── requirements.txt                       # Python dependency requirements
```

---

## 4. Quick Start & Execution Guide

### Prerequisites
- Python 3.11 or 3.12 (64-bit)
- PowerShell (Windows) or Bash (macOS/Linux)
- Git

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/your-username/customer-support-ai.git
cd customer-support-ai

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the FastAPI Platform Backend
```powershell
cd backend
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
*The backend API is now live at `http://127.0.0.1:8000`.*
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`
- Business Admin Console: `http://127.0.0.1:8000/admin`
- Supervisor Desk: `http://127.0.0.1:8000/supervisor`
- Embeddable Widget Demo: `http://127.0.0.1:8000/widget/demo`

### 3. Launch the Streamlit E-Commerce Storefront
In a second terminal:
```powershell
streamlit run frontend/hemanth_store_app.py
```
*The interactive customer storefront is now live at `http://localhost:8501`.*

### 4. Execute the Automated Test Suite
```powershell
# Run the complete test suite (274 tests across 32 modules)
python -m pytest tests/
```

### 5. Re-Run the Research Benchmark
```powershell
python scripts/benchmark_fixed_vs_adaptive_orchestration.py
```

---

## 5. Pre-Configured Demonstration Workspace

For immediate testing, viva demonstrations, and examiner evaluation, the repository includes a pre-seeded, fully operational workspace:

| Parameter | Pre-Configured Value | Description |
| :--- | :--- | :--- |
| **Workspace Name** | `Hemanth Online store` | Live E-Commerce Workspace |
| **Workspace ID** | `ws_26d7520d00c0` | Dedicated Chroma collection & SQLite records |
| **Public Scoped Widget Token** | `wtk_gSIAy-ZvQ_S0i_rLTozSwRkyvBtL31ob` | Rate-limited token for customer storefront widget |
| **Master Admin API Key** | `cap_demo_admin_key` | Full authorization for `/admin` & `/supervisor` |
| **Operational Readiness** | **100.0% Complete** | 10 validated policy documents, 66 chunks indexed |
| **Pre-Seeded Customer ID** | `cust_hemanth_01` | Customer profile with active order `#ORD-99482` |

---

## 6. GitHub Repository Compliance & Asset Preservation

1. **Clean Root Directory:** All source code, data, models, documentation, and configurations are compartmentalized into dedicated top-level directories.
2. **File Size Compliance:** All repository files are below GitHub's 50 MB threshold. The largest file is `backend/src/models/risk_model_(M3).joblib` at **17.65 MB**, allowing seamless pushing without requiring Git LFS.
3. **Database Integrity:** Pre-seeded ChromaDB vector embeddings (`backend/vector_store/rag_chroma/`) and SQLite databases (`backend/database/tenants.db`) are preserved intact and never deleted or rebuilt during reorganization.
4. **Zero Cache or Secret Uploads:** `.venv/`, `__pycache__/`, `.pytest_cache/`, `.coverage`, and `.env` files are strictly excluded via `.gitignore`.
5. **Independent Warranty Project Separation:** No external warranty claim code or unrelated data is included in this repository.

---

## 7. License & Academic Attribution

This project was developed as a final-year Capstone Major Project for the B.Tech Degree in Computer Science and Engineering (Artificial Intelligence & Machine Learning).  
All rights reserved © 2026.
