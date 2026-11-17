import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "documentation" / "figures"
GRAPHS_DIR = OUT_DIR / "results"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

# ===========================================================================
# 1. Architecture Diagram (Two-Policy Paradigms: Fixed vs Adaptive)
# ===========================================================================
def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 10.5), dpi=300)
    ax.axis("off")

    def draw_box(x, y, w, h, text, title=None, color="#e9f1f7", edge_color="#1f4e79", text_color="#111111", bold_title=True):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", facecolor=color, edgecolor=edge_color, linewidth=2, zorder=2)
        ax.add_patch(rect)
        if title:
            ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=10.5, fontweight="bold" if bold_title else "normal", color=edge_color, zorder=3)
            ax.text(x + w / 2, y + (h - 0.3) / 2, text, ha="center", va="center", fontsize=9, color=text_color, zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_color, zorder=3)

    def draw_arrow(x1, y1, x2, y2, label=None, label_pos=(0.5, 0.5), color="#1f4e79", style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=2, mutation_scale=15), zorder=1)
        if label:
            lx = x1 + (x2 - x1) * label_pos[0]
            ly = y1 + (y2 - y1) * label_pos[1]
            ax.text(lx, ly, label, ha="center", va="center", fontsize=8.5, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=4)

    # 1. Customer Query Ingestion
    draw_box(0.5, 9.2, 13.0, 0.8, "Raw User Input & Multi-Turn Context (Conversational Ellipses & Anaphoric Pronoun Resolution)", "1. INCOMING CUSTOMER QUERY & SESSION CONTEXT", color="#dbe5f1", edge_color="#1f4e79")

    draw_arrow(7.0, 9.2, 7.0, 8.6, label="Disambiguated Ingestion")

    # 2. Fast-Path Cascaded Triage Pipeline (M1 to M6)
    rect_cascade = plt.Rectangle((0.3, 5.2), 13.4, 3.2, facecolor="#fbfbfd", edgecolor="#385723", linewidth=2.5, linestyle="--", zorder=1)
    ax.add_patch(rect_cascade)
    ax.text(7.0, 8.15, "2. FAST-PATH CASCADED TRIAGE PIPELINE (M1–M6) — Sequential Inference on CPU", ha="center", va="top", fontsize=11, fontweight="bold", color="#385723", zorder=3)

    m_w, m_h, m_y = 1.95, 2.2, 5.5
    draw_box(0.6, m_y, m_w, m_h, "LinearSVC\nTF-IDF n-grams\n47 Classes\n94.3% Acc", "M1: Intent", color="#ffffff", edge_color="#385723")
    draw_box(2.8, m_y, m_w, m_h, "Logistic Reg\nTF-IDF n-grams\n3 Classes\nF1: 0.884", "M2: Sentiment", color="#ffffff", edge_color="#385723")
    draw_box(5.0, m_y, m_w, m_h, "Random Forest\n200 Trees, D=20\nLow, Med, High\n98.1% Prec", "M3: Risk", color="#ffffff", edge_color="#385723")
    draw_box(7.2, m_y, m_w, m_h, "Logistic Reg\nTF-IDF + Domain\nLow, Med, High\nF1: 0.862", "M4: Complexity", color="#ffffff", edge_color="#385723")
    draw_box(9.4, m_y, m_w, m_h, "LinearSVC\nTF-IDF 5k feats\n3 Domains\n99.2% Acc", "M5: Domain", color="#ffffff", edge_color="#385723")
    draw_box(11.6, m_y, m_w, m_h, "Logistic Reg\nCascaded Feats\nProvisional Triage\n88.62% Acc", "M6: Autonomy", color="#fff2cc", edge_color="#d6604d")

    # Cascading arrows
    draw_arrow(2.55, 6.6, 2.8, 6.6)
    draw_arrow(4.75, 6.6, 5.0, 6.6)
    draw_arrow(6.95, 6.6, 7.2, 6.6)
    draw_arrow(9.15, 6.6, 9.4, 6.6)
    draw_arrow(11.35, 6.6, 11.6, 6.6)

    # BRANCH 1: POLICY A (FIXED BASELINE)
    draw_arrow(12.57, 5.5, 12.57, 4.4, color="#c00000")
    draw_box(10.5, 3.4, 3.2, 0.9, "Policy A: Fixed Baseline\nRoutes directly via argmax P(y|x)\n(No Retrieval / No Risk Overrides)", "POLICY A: FIXED ROUTE", color="#fce4d6", edge_color="#c00000")

    # BRANCH 2: POLICY B (ADAPTIVE ORCHESTRATION)
    draw_arrow(6.0, 5.2, 4.5, 4.4, color="#1f4e79")
    draw_arrow(11.6, 5.2, 8.5, 4.4, color="#1f4e79")

    # 3. Tenant-Isolated Retrieval
    draw_box(0.5, 3.3, 4.6, 1.0, "ChromaDB (Cosine HNSW, all-MiniLM-L6-v2)\n+ In-Memory BM25Okapi Lexical Ranker\nDistinct-Doc Deduplication (m=4)", "3. TENANT-ISOLATED RETRIEVAL", color="#e9f1f7", edge_color="#2e75b6")

    # 4. Evidence Entailment & Composite Scoring
    draw_box(5.4, 3.3, 4.7, 1.0, "Computes Composite Confidence C in [0, 1]\n(Similarity 0.25, Reranker 0.25,\nAuthority 0.15, Freshness 0.15, Agree 0.10)", "4. EVIDENCE VERIFICATION", color="#e9f1f7", edge_color="#2e75b6")

    draw_arrow(5.1, 3.8, 5.4, 3.8)

    # 5. Deterministic Safety Policy Engine (Policy B)
    draw_box(1.0, 1.8, 11.0, 1.0, "Policy B Deterministic Invariants & Qualification Gates\nCRIT_01 (High/Critical Risk) -> ESCALATE | EVID_INSUF_04 (Knowledge Gap) -> ESCALATE\nAutonomous Qualification: Composite Confidence C >= 0.70", "5. DETERMINISTIC SAFETY POLICY ENGINE (POLICY B)", color="#fbe5d6", edge_color="#c00000")

    draw_arrow(7.7, 3.3, 7.0, 2.8)

    # Final Operational Tiers
    draw_arrow(2.5, 1.8, 1.8, 1.1)
    draw_arrow(6.5, 1.8, 6.5, 1.1)
    draw_arrow(10.5, 1.8, 11.2, 1.1)

    draw_box(0.5, 0.4, 3.2, 0.65, "Grounded Direct Answer\n+ Verified Policy Citations", "TIER 1: ANSWER", color="#e2efda", edge_color="#385723")
    draw_box(4.9, 0.4, 3.2, 0.65, "Specialized Domain Dispatch\n(E-com / Edu / SMB Agents)", "TIER 2: ORCHESTRATE", color="#deebf7", edge_color="#2e75b6")
    draw_box(9.3, 0.4, 3.5, 0.65, "Human Supervisor Desk\n(Audit Dossier & Telemetry)", "TIER 3: ESCALATE (HITL)", color="#fce4d6", edge_color="#c65911")

    ax.set_xlim(-0.2, 14.2)
    ax.set_ylim(0.0, 10.2)
    plt.tight_layout()
    arch_path = OUT_DIR / "Final_Research_Architecture.png"
    plt.savefig(arch_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved two-policy architecture diagram to:", arch_path)

# ===========================================================================
# 2. Safety Violation Comparison Bar Chart
# ===========================================================================
def generate_safety_comparison():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    categories = [
        "Unsafe Cohort Answer Rate\n(N=341, Lower is Better)",
        "High/Critical Risk Answers\n(N=30, Lower is Better)",
        "Knowledge Gap Answers\n(N=73, Lower is Better)",
        "Unnecessary Escalations on Safe\n(N=94, Lower is Better)"
    ]
    
    policy_a_vals = [79.77, 6.67, 100.0, 0.0]
    policy_b_vals = [6.16, 0.0, 0.0, 51.06]

    x = np.arange(len(categories))
    width = 0.32

    rects1 = ax.bar(x - width/2, policy_a_vals, width, label="Policy A: Fixed Baseline (M6 argmax)", color="#d6604d", edgecolor="#b2182b", linewidth=1.2)
    rects2 = ax.bar(x + width/2, policy_b_vals, width, label="Policy B: Risk-Aware Adaptive Autonomy", color="#4393c3", edgecolor="#2166ac", linewidth=1.2)

    ax.set_ylabel("Observed Violation / Escalation Rate (%)", fontweight="bold")
    ax.set_title("Safety Violation & Escalation Comparison: Policy A vs. Policy B (N=501)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.legend(frameon=True, facecolor="#fbfbfd", edgecolor="#d9d9d9", loc="upper right")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    out_path = GRAPHS_DIR / "safety_violation_comparison.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved two-policy safety comparison to:", out_path)

# ===========================================================================
# 3. Routing Accuracy & Macro-F1 Bar Chart
# ===========================================================================
def generate_accuracy_f1_comparison():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    metrics = [
        "Overall 3-Class Accuracy\n(Static Pre-Retrieval Labels)",
        "Macro-Averaged F1\n(3 Routing Classes)",
        "Answer Class F1\n(Direct Autonomous Resolution)",
        "Orchestrate Class F1\n(Multi-Agent Delegation)"
    ]

    policy_a_vals = [88.62, 63.16, 94.44, 87.50]
    policy_b_vals = [25.55, 28.82, 29.16, 38.10]

    x = np.arange(len(metrics))
    width = 0.32

    rects1 = ax.bar(x - width/2, policy_a_vals, width, label="Policy A: Fixed Baseline (M6 argmax)", color="#f4a582", edgecolor="#d6604d", linewidth=1.2)
    rects2 = ax.bar(x + width/2, policy_b_vals, width, label="Policy B: Risk-Aware Adaptive Autonomy", color="#2166ac", edgecolor="#053061", linewidth=1.2)

    ax.set_ylabel("Score (%) / F1 (x100)", fontweight="bold")
    ax.set_title("Routing Accuracy and Class F1 Scores: Policy A vs. Policy B", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5)
    ax.legend(frameon=True, facecolor="#fbfbfd", edgecolor="#d9d9d9", loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    out_path = GRAPHS_DIR / "routing_accuracy_macro_f1.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved two-policy accuracy & F1 comparison to:", out_path)

# ===========================================================================
# 4. Escalation Recall and Workload Comparison
# ===========================================================================
def generate_escalation_workload_comparison():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    metrics = [
        "Escalation Recall (%)\n(N=49 GT Escalations, Higher is Better)",
        "Escalation Precision (%)\n(Precision on Escalation Queue)",
        "Total Escalation Workload (%)\n(Total Inquiries Routed to Humans)",
        "Unnecessary Escalations on Safe (%)\n(N=94 Safe Queries, Lower is Better)"
    ]

    policy_a_vals = [4.08, 50.00, 0.80, 0.00]
    policy_b_vals = [89.80, 10.76, 81.64, 51.06]

    x = np.arange(len(metrics))
    width = 0.32

    rects1 = ax.bar(x - width/2, policy_a_vals, width, label="Policy A: Fixed Baseline", color="#fdbb84", edgecolor="#e34a33", linewidth=1.2)
    rects2 = ax.bar(x + width/2, policy_b_vals, width, label="Policy B: Adaptive Autonomy", color="#74a9cf", edgecolor="#0570b0", linewidth=1.2)

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Human-in-the-Loop Escalation Dynamics & Workload: Policy A vs. Policy B", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5)
    ax.legend(frameon=True, facecolor="#fbfbfd", edgecolor="#d9d9d9", loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    out_path = GRAPHS_DIR / "escalation_workload_comparison.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved escalation workload comparison to:", out_path)

# ===========================================================================
# 5. Latency Percentiles Comparison
# ===========================================================================
def generate_latency_comparison():
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    percentiles = ["Min (P0)", "P25", "Median (P50)", "P75", "P90", "P95", "P99", "Max (P100)", "Mean"]
    
    lat_a = [24.47, 37.98, 39.76, 46.08, 98.31, 163.19, 182.54, 410.29, 55.17]
    lat_b = [37.74, 66.94, 87.23, 115.48, 219.57, 287.70, 444.16, 506.08, 112.77]

    x = np.arange(len(percentiles))
    width = 0.35

    rects1 = ax.bar(x - width/2, lat_a, width, label="Policy A: Fixed Baseline", color="#92c5de", edgecolor="#4393c3", linewidth=1.2)
    rects2 = ax.bar(x + width/2, lat_b, width, label="Policy B: Adaptive Autonomy", color="#d6604d", edgecolor="#b2182b", linewidth=1.2)

    ax.set_ylabel("Latency (milliseconds, CPU Monotonic)", fontweight="bold")
    ax.set_title("Operational Latency Distribution: Policy A vs. Policy B (N=501)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(percentiles, fontsize=9.5)
    ax.legend(frameon=True, facecolor="#fbfbfd", edgecolor="#d9d9d9")
    ax.set_ylim(0, 560)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{int(round(height))}ms",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out_path = GRAPHS_DIR / "latency_percentiles_comparison.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved two-policy latency comparison to:", out_path)

# ===========================================================================
# 6. Confusion Matrices Heatmap (Two Subplots: Policy A vs Policy B)
# ===========================================================================
def generate_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), dpi=300)

    cm_a = np.array([
        [365, 6, 2],
        [2, 77, 0],
        [33, 14, 2]
    ])

    cm_b = np.array([
        [64, 3, 306],
        [0, 20, 59],
        [2, 3, 44]
    ])

    classes = ["ANSWER", "ORCHESTRATE", "ESCALATE"]
    titles = [
        "Policy A: Fixed Baseline\n(Acc: 88.62%, Macro-F1: 0.6316)",
        "Policy B: Adaptive Autonomy\n(Acc: 25.55%, Macro-F1: 0.2882)"
    ]
    matrices = [cm_a, cm_b]

    for idx, ax in enumerate(axes):
        cm = matrices[idx]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=375)
        ax.set_title(titles[idx], fontweight="bold", fontsize=11, pad=12)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=9.5)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=9.5)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                color = "white" if val > 180 else "black"
                ax.text(j, i, format(val, "d"),
                        ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

        ax.set_ylabel("Ground Truth Class", fontweight="bold" if idx == 0 else "normal")
        ax.set_xlabel("Predicted Route", fontweight="bold")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.22, 0.02, 0.58])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out_path = GRAPHS_DIR / "confusion_matrices_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved two-policy confusion matrices heatmap to:", out_path)

# ===========================================================================
# 7. Retrieval Strategy Evaluation
# ===========================================================================
def generate_retrieval_comparison():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    strategies = [
        "Pure Dense\n(Raw Chunks)",
        "Pure Dense Distinct\n(m=4) [RECOMMENDED]",
        "Hybrid Distinct\n(k=60, m=2)",
        "Hybrid Distinct\n(k=60, m=4)"
    ]
    
    recall_at_1 = [61.92, 61.92, 52.10, 52.34]
    recall_at_5 = [82.24, 87.38, 87.38, 83.88]
    mrr_x100 = [69.77, 71.92, 64.96, 63.64]

    x = np.arange(len(strategies))
    width = 0.25

    rects1 = ax.bar(x - width, recall_at_1, width, label="Ans Recall@1 (%)", color="#a6bddb", edgecolor="#2b8cbe")
    rects2 = ax.bar(x, recall_at_5, width, label="Ans Recall@5 (%)", color="#2b8cbe", edgecolor="#08589e")
    rects3 = ax.bar(x + width, mrr_x100, width, label="Ans MRR (x100)", color="#ece7f2", edgecolor="#2b8cbe")

    ax.set_ylabel("Score (%) / Rank Multiplier", fontweight="bold")
    ax.set_title("Tenant-Isolated Retrieval Strategy Evaluation (N=428 Answerable)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9.5)
    ax.legend(frameon=True, facecolor="#fbfbfd", edgecolor="#d9d9d9", loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out_path = GRAPHS_DIR / "retrieval_strategy_recall_mrr.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved retrieval comparison to:", out_path)

if __name__ == "__main__":
    generate_architecture_diagram()
    generate_safety_comparison()
    generate_accuracy_f1_comparison()
    generate_escalation_workload_comparison()
    generate_latency_comparison()
    generate_confusion_matrices()
    generate_retrieval_comparison()
    print("All 7 figures successfully generated in FINAL_RESEARCH_DOCUMENTATION!")
