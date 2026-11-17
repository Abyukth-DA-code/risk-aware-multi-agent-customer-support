import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "documentation" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUT_DIR / "Final_Results_Comparison.xlsx"

wb = openpyxl.Workbook()
wb.remove(wb.active)

# Styling definitions
font_title = Font(name="Calibri", size=14, bold=True, color="1F4E79")
font_subtitle = Font(name="Calibri", size=10, italic=True, color="595959")
font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
font_bold = Font(name="Calibri", size=10, bold=True, color="000000")
font_regular = Font(name="Calibri", size=10, color="000000")
font_green = Font(name="Calibri", size=10, bold=True, color="276A3C")
font_red = Font(name="Calibri", size=10, bold=True, color="C00000")

fill_header = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
fill_sub_header = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
fill_zebra = PatternFill(start_color="F2F5F9", end_color="F2F5F9", fill_type="solid")
fill_highlight_green = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
fill_highlight_yellow = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
fill_highlight_red = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

thin_border = Border(
    left=Side(style='thin', color='D9D9D9'),
    right=Side(style='thin', color='D9D9D9'),
    top=Side(style='thin', color='D9D9D9'),
    bottom=Side(style='thin', color='D9D9D9')
)
header_border = Border(
    left=Side(style='thin', color='FFFFFF'),
    right=Side(style='thin', color='FFFFFF'),
    top=Side(style='medium', color='1F4E79'),
    bottom=Side(style='medium', color='1F4E79')
)

align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)
align_left = Alignment(horizontal="left", vertical="center", wrap_text=True)
align_right = Alignment(horizontal="right", vertical="center")

def autofit_columns(ws, min_width=12, max_width=48):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.row in [1, 2]:
                continue
            val_str = str(cell.value or "")
            if "\n" in val_str:
                val_str = max(val_str.split("\n"), key=len)
            max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = max(min(max_len + 3, max_width), min_width)

# ==============================================================================
# TAB 1: Policy_Comparison_Summary
# ==============================================================================
ws1 = wb.create_sheet(title="Policy_Comparison_Summary")
ws1.views.sheetView[0].showGridLines = True

ws1["A1"] = "Autonomous Multi-Agent Customer Support System - Research Benchmark Freeze"
ws1["A1"].font = font_title
ws1["A2"] = "Head-to-Head Comparison: Policy A (Fixed Baseline) vs. Original Policy B vs. Optimized Policy B | N=501 Held-Out Cases"
ws1["A2"].font = font_subtitle

headers_tab1 = [
    "Evaluation Dimension",
    "Metric / Evaluated Cohort",
    "Policy A: Fixed Baseline",
    "Original Policy B (Uncalibrated)",
    "Optimized Policy B (Validation-Calibrated)",
    "Original B vs. A Delta",
    "Optimized B vs. A Delta",
    "Statistical Significance (p-value)",
    "Empirical Verdict & Practical Significance"
]

ws1.append([])
ws1.append(headers_tab1)
for col_idx in range(1, len(headers_tab1) + 1):
    cell = ws1.cell(row=4, column=col_idx)
    cell.font = font_header
    cell.fill = fill_header
    cell.alignment = align_center
    cell.border = header_border

summary_rows = [
    # Routing Performance
    ("Routing Performance", "Total Correct Predictions (Accuracy)", "88.62% (444/501)", "25.55% (128/501)", "29.54% (148/501)", "-63.07%", "-59.08%", "p < 0.001 (McNemar chi2=248.06)", "Policy A matches pre-retrieval static labels by guessing ANSWER; rejected as safe policy"),
    ("Routing Performance", "Accuracy 95% Wilson CI", "[85.54%, 91.11%]", "[21.93%, 29.54%]", "[25.71%, 33.68%]", "--", "--", "--", "Wilson score intervals show non-overlapping accuracy profiles"),
    ("Routing Performance", "Macro-Averaged F1-Score", "0.6316", "0.2882", "0.3142", "-0.3434", "-0.3174", "--", "Optimized Policy B improves Macro-F1 over Original B (+0.0260)"),
    ("Routing Performance", "Correct Autonomous Answers (TP ANS)", "365 / 373 (97.86%)", "64 / 373 (17.16%)", "89 / 373 (23.86%)", "-301 answers", "-276 answers", "p < 0.001", "Optimized B unlocks 25 additional safe autonomous answers (+38.8% answer yield over B)"),
    ("Routing Performance", "Escalation Class Recall (N=49)", "4.08% (2/49)", "89.80% (44/49)", "79.59% (39/49)", "+85.72%", "+75.51%", "p < 0.001 (McNemar)", "Policy A missed 95.9% of true escalations; Policy B captures human reviews"),
    ("Routing Performance", "Escalation Class Precision", "50.00% (2/4)", "10.76% (44/409)", "10.29% (39/379)", "-39.24%", "-39.71%", "--", "Trade-off: strict fail-safe posture routes uncertain queries to humans"),
    
    # Safety & Risk Control
    ("Safety & Risk Control", "Unsafe Cohort Answer Rate (N=341)", "79.77% (272/341)", "6.16% (21/341)", "8.21% (28/341)", "-73.61% (-92.3% rel)", "-71.56% (-89.7% rel)", "p < 0.001 (McNemar chi2=249.00)", "PRIMARY SAFETY HYPOTHESIS CONFIRMED: 92.28% relative reduction in unsafe answering"),
    ("Safety & Risk Control", "Unsafe Answer Rate 95% Wilson CI", "[75.18%, 83.69%]", "[4.06%, 9.23%]", "[5.74%, 11.61%]", "--", "--", "--", "Dramatic separation: baseline answers unsafe cases; adaptive policy intercepts"),
    ("Safety & Risk Control", "High/Critical Risk Breaches (N=30)", "6.67% (2/30)", "0.00% (0/30)", "0.00% (0/30)", "-6.67% (-100% rel)", "-6.67% (-100% rel)", "Deterministic Invariant", "Zero breaches observed on High/Critical test cases under invariant CRIT_01"),
    ("Safety & Risk Control", "Knowledge-Gap Breaches (N=73)", "100.00% (73/73)", "0.00% (0/73)", "0.00% (0/73)", "-100.00% (-100% rel)", "-100.00% (-100% rel)", "Deterministic Invariant", "Zero breaches observed on unanswerable queries under invariant EVID_INSUF_04"),
    ("Safety & Risk Control", "Unnecessary Escalations on Safe (N=94)", "0.00% (0/94)", "51.06% (48/94)", "51.06% (48/94)", "+51.06%", "+51.06%", "p < 0.001", "Over-conservatism: conservative evidence threshold routes safe queries to humans"),
    ("Safety & Risk Control", "Total Human Escalation Workload", "4 (0.8%)", "409 (81.6%)", "379 (75.6%)", "+405 escalations", "+375 escalations", "--", "Optimized B recovers 30 queries from human queue back to autonomous answer"),
    
    # Operational Latency
    ("Operational Efficiency", "Mean Latency (ms)", "55.17 ms", "112.77 ms", "112.78 ms", "+57.60 ms", "+57.61 ms", "p < 0.001 (t = -24.89)", "Adaptive overhead is 57.6 ms (due to ChromaDB vector search + entailment scoring)"),
    ("Operational Efficiency", "Median Latency (P50, ms)", "39.76 ms", "87.23 ms", "87.23 ms", "+47.47 ms", "+47.47 ms", "p < 0.001 (Wilcoxon)", "P50 latency under 90 ms; imperceptible in interactive user sessions"),
    ("Operational Efficiency", "90th Percentile Latency (P90, ms)", "98.31 ms", "219.57 ms", "219.58 ms", "+121.26 ms", "+121.27 ms", "--", "P90 within 220 ms"),
    ("Operational Efficiency", "95th Percentile Latency (P95, ms)", "163.19 ms", "287.70 ms", "287.71 ms", "+124.51 ms", "+124.52 ms", "--", "P95 latency operates well below standard interactive web thresholds (1.0-2.0s)"),
    ("Operational Efficiency", "99th Percentile Latency (P99, ms)", "182.54 ms", "444.16 ms", "444.16 ms", "+261.62 ms", "+261.62 ms", "--", "Tail latency bounded under 445 ms across 99% of evaluated queries")
]

for row_idx, r in enumerate(summary_rows, start=5):
    ws1.append(list(r))
    for col_idx in range(1, len(r) + 1):
        cell = ws1.cell(row=row_idx, column=col_idx)
        cell.font = font_regular
        cell.border = thin_border
        if col_idx in [1, 2, 9]:
            cell.alignment = align_left
        else:
            cell.alignment = align_center
        
        if row_idx % 2 == 1:
            cell.fill = fill_zebra
        
        if col_idx in [4, 5] and "0.00%" in str(cell.value):
            cell.fill = fill_highlight_green
            cell.font = font_green
        elif col_idx == 3 and ("79.77%" in str(cell.value) or "100.00%" in str(cell.value)):
            cell.fill = fill_highlight_red
            cell.font = font_red

autofit_columns(ws1)

# ==============================================================================
# TAB 2: Per_Class_Performance
# ==============================================================================
ws2 = wb.create_sheet(title="Per_Class_Performance")
ws2.views.sheetView[0].showGridLines = True

ws2["A1"] = "Per-Class Performance Breakdown: Policy A vs. Original B vs. Optimized B"
ws2["A1"].font = font_title
ws2["A2"] = "Evaluated on N=501 Held-Out Test Queries | Ground Truth: ANSWER (373), ORCHESTRATE (79), ESCALATE (49)"
ws2["A2"].font = font_subtitle

headers_tab2 = [
    "Policy Configuration",
    "Routing Class",
    "Class Support (GT Cases)",
    "Predicted Count",
    "True Positives (TP)",
    "Precision",
    "Recall",
    "F1-Score",
    "Operational Routing Mechanics & Denominator Notes"
]

ws2.append([])
ws2.append(headers_tab2)
for col_idx in range(1, len(headers_tab2) + 1):
    cell = ws2.cell(row=4, column=col_idx)
    cell.font = font_header
    cell.fill = fill_header
    cell.alignment = align_center
    cell.border = header_border

per_class_data = [
    # Policy A
    ("Policy A (Fixed Baseline)", "ANSWER", 373, 400, 365, "91.25%", "97.86%", "0.9444", "High recall because M6 defaults to ANSWER; causes 272 unsafe answers"),
    ("Policy A (Fixed Baseline)", "ORCHESTRATE", 79, 97, 77, "79.38%", "97.47%", "0.8750", "High recall on multi-step workflows; ignores evidence verification"),
    ("Policy A (Fixed Baseline)", "ESCALATE", 49, 4, 2, "50.00%", "4.08%", "0.0755", "Catastrophic failure: misses 47 of 49 true human escalation cases"),
    ("Policy A (Fixed Baseline)", "3-CLASS TOTAL", 501, 501, 444, "73.54%", "66.47%", "0.6316", "Overall accuracy 88.62% (444/501) is inflated by ignoring downstream risk"),
    
    # Original Policy B
    ("Original Policy B (Uncalibrated)", "ANSWER", 373, 66, 64, "96.97%", "17.16%", "0.2916", "Correct autonomous answers: 64/373 (17.16% recall, 96.97% precision)"),
    ("Original Policy B (Uncalibrated)", "ORCHESTRATE", 79, 26, 20, "76.92%", "25.32%", "0.3810", "Routes 59 multi-agent workflows to escalation due to medium-risk penalties"),
    ("Original Policy B (Uncalibrated)", "ESCALATE", 49, 409, 44, "10.76%", "89.80%", "0.1921", "Captures 44/49 escalations (89.8% recall); high human review burden"),
    ("Original Policy B (Uncalibrated)", "3-CLASS TOTAL", 501, 501, 128, "61.55%", "44.09%", "0.2882", "Total correct predictions across 3 classes: 128/501 = 25.55% accuracy"),
    
    # Optimized Policy B
    ("Optimized Policy B (Calibrated Gate)", "ANSWER", 373, 96, 89, "92.71%", "23.86%", "0.3795", "Correct autonomous answers: 89/373 (23.86% recall, +25 answers recovered)"),
    ("Optimized Policy B (Calibrated Gate)", "ORCHESTRATE", 79, 26, 20, "76.92%", "25.32%", "0.3810", "Maintains conservative agent orchestration threshold"),
    ("Optimized Policy B (Calibrated Gate)", "ESCALATE", 49, 379, 39, "10.29%", "79.59%", "0.1822", "Reduces human queue from 409 to 379 while preserving 79.6% escalation recall"),
    ("Optimized Policy B (Calibrated Gate)", "3-CLASS TOTAL", 501, 501, 148, "59.97%", "42.92%", "0.3142", "Total correct predictions across 3 classes: 148/501 = 29.54% accuracy")
]

for row_idx, r in enumerate(per_class_data, start=5):
    ws2.append(list(r))
    for col_idx in range(1, len(r) + 1):
        cell = ws2.cell(row=row_idx, column=col_idx)
        cell.font = font_regular
        cell.border = thin_border
        if col_idx in [1, 2, 9]:
            cell.alignment = align_left
        else:
            cell.alignment = align_center
        
        if "3-CLASS TOTAL" in str(r[1]):
            cell.font = font_bold
            cell.fill = fill_highlight_yellow
        elif row_idx % 2 == 1:
            cell.fill = fill_zebra

autofit_columns(ws2)

# ==============================================================================
# TAB 3: Confusion_Matrices
# ==============================================================================
ws3 = wb.create_sheet(title="Confusion_Matrices")
ws3.views.sheetView[0].showGridLines = True

ws3["A1"] = "Contingency and Confusion Matrices: Policy A vs. Original B vs. Optimized B"
ws3["A1"].font = font_title
ws3["A2"] = "Rows = Ground Truth (Actual Class) | Columns = Predicted Route | Total N = 501"
ws3["A2"].font = font_subtitle

def build_cm_block(ws, start_row, policy_title, cm_data, notes):
    ws.cell(row=start_row, column=1, value=policy_title).font = Font(name="Calibri", size=11, bold=True, color="1F4E79")
    
    headers = ["Ground Truth Class", "Pred: ANSWER", "Pred: ORCHESTRATE", "Pred: ESCALATE", "Actual Total (Row Sum)", "Recall %", "Key Failure / Safe Mode"]
    for c_idx, h in enumerate(headers, start=1):
        c = ws.cell(row=start_row + 1, column=c_idx, value=h)
        c.font = font_header
        c.fill = fill_sub_header
        c.alignment = align_center
        c.border = header_border
    
    classes = ["ANSWER", "ORCHESTRATE", "ESCALATE"]
    col_totals = [0, 0, 0]
    
    for r_idx, cls_name in enumerate(classes):
        row_num = start_row + 2 + r_idx
        row_vals = cm_data[cls_name]
        row_sum = sum(row_vals)
        recall = (row_vals[r_idx] / row_sum * 100) if row_sum > 0 else 0
        note = notes[cls_name]
        
        ws.cell(row=row_num, column=1, value=cls_name).font = font_bold
        ws.cell(row=row_num, column=1).alignment = align_left
        ws.cell(row=row_num, column=1).border = thin_border
        
        for p_idx, pred_val in enumerate(row_vals):
            c = ws.cell(row=row_num, column=2 + p_idx, value=pred_val)
            c.font = font_regular
            c.alignment = align_center
            c.border = thin_border
            col_totals[p_idx] += pred_val
            if p_idx == r_idx:
                c.fill = fill_highlight_green
                c.font = font_green
            elif pred_val > 50:
                c.fill = fill_highlight_red
                c.font = font_red
        
        c_tot = ws.cell(row=row_num, column=5, value=row_sum)
        c_tot.font = font_bold
        c_tot.alignment = align_center
        c_tot.border = thin_border
        
        c_rec = ws.cell(row=row_num, column=6, value=f"{recall:.2f}%")
        c_rec.font = font_bold
        c_rec.alignment = align_center
        c_rec.border = thin_border
        
        c_note = ws.cell(row=row_num, column=7, value=note)
        c_note.font = font_regular
        c_note.alignment = align_left
        c_note.border = thin_border
        
    tot_row = start_row + 5
    ws.cell(row=tot_row, column=1, value="Predicted Total (Col Sum)").font = font_bold
    ws.cell(row=tot_row, column=1).border = thin_border
    
    for p_idx, col_sum in enumerate(col_totals):
        c = ws.cell(row=tot_row, column=2 + p_idx, value=col_sum)
        c.font = font_bold
        c.alignment = align_center
        c.border = thin_border
        
    ws.cell(row=tot_row, column=5, value=sum(col_totals)).font = font_bold
    ws.cell(row=tot_row, column=5).alignment = align_center
    ws.cell(row=tot_row, column=5).border = thin_border
    
    prec_row = start_row + 6
    ws.cell(row=prec_row, column=1, value="Precision %").font = font_bold
    ws.cell(row=prec_row, column=1).border = thin_border
    for p_idx, col_sum in enumerate(col_totals):
        diag_val = cm_data[classes[p_idx]][p_idx]
        prec = (diag_val / col_sum * 100) if col_sum > 0 else 0
        c = ws.cell(row=prec_row, column=2 + p_idx, value=f"{prec:.2f}%")
        c.font = font_bold
        c.alignment = align_center
        c.border = thin_border
    ws.cell(row=prec_row, column=5, value="--").alignment = align_center
    ws.cell(row=prec_row, column=5).border = thin_border

cm_a = {
    "ANSWER": [365, 6, 2],
    "ORCHESTRATE": [2, 77, 0],
    "ESCALATE": [33, 14, 2]
}
notes_a = {
    "ANSWER": "Correct on safe queries, but blindly answers 73 unanswerable queries",
    "ORCHESTRATE": "High recall on multi-agent intents, zero safety checks on tools",
    "ESCALATE": "CRITICAL RISK: 33 escalation cases erroneously answered directly"
}
build_cm_block(ws3, 4, "1. Policy A: Fixed Baseline (M6 argmax without safety engine)", cm_a, notes_a)

cm_b = {
    "ANSWER": [64, 3, 306],
    "ORCHESTRATE": [0, 20, 59],
    "ESCALATE": [2, 3, 44]
}
notes_b = {
    "ANSWER": "306 queries diverted to ESCALATE due to strict evidence threshold C >= 0.75",
    "ORCHESTRATE": "59 queries diverted to ESCALATE due to medium risk / agent complexity",
    "ESCALATE": "SAFE POSTURE: 44 of 49 true escalations captured (89.80% recall)"
}
build_cm_block(ws3, 13, "2. Original Policy B (Uncalibrated Baseline Adaptive Policy)", cm_b, notes_b)

cm_c = {
    "ANSWER": [89, 3, 281],
    "ORCHESTRATE": [0, 20, 59],
    "ESCALATE": [7, 3, 39]
}
notes_c = {
    "ANSWER": "25 additional safe queries routed to ANSWER (C >= 0.65 threshold)",
    "ORCHESTRATE": "Identical orchestration behavior to Original Policy B",
    "ESCALATE": "Maintains 79.59% escalation recall while reducing human queue"
}
build_cm_block(ws3, 22, "3. Optimized Policy B (Validation-Calibrated Gate)", cm_c, notes_c)

autofit_columns(ws3)

# ==============================================================================
# TAB 4: Latency_Percentiles
# ==============================================================================
ws4 = wb.create_sheet(title="Latency_Percentiles")
ws4.views.sheetView[0].showGridLines = True

ws4["A1"] = "Inference and Operational Latency Distribution: Policy A vs. Policy B"
ws4["A1"].font = font_title
ws4["A2"] = "Measured across N=501 Sequential Passes on CPU (AMD Ryzen 5, Python 3.11, ChromaDB HNSW)"
ws4["A2"].font = font_subtitle

headers_tab4 = [
    "Percentile / Statistic",
    "Policy A (ms)",
    "Policy B (ms)",
    "Policy B vs. A Overhead (ms)",
    "Policy B vs. A Overhead (%)",
    "Architectural Source of Latency Overhead"
]

ws4.append([])
ws4.append(headers_tab4)
for col_idx in range(1, len(headers_tab4) + 1):
    cell = ws4.cell(row=4, column=col_idx)
    cell.font = font_header
    cell.fill = fill_header
    cell.alignment = align_center
    cell.border = header_border

latency_rows = [
    ("Minimum (P0)", 24.47, 37.74, "+13.27 ms", "+54.2%", "Fast-path M1-M6 classifier cascade overhead alone"),
    ("10th Percentile (P10)", 33.12, 54.21, "+21.09 ms", "+63.7%", "Cached vector lookup or fast retrieval candidate"),
    ("25th Percentile (P25)", 37.98, 66.94, "+28.96 ms", "+76.3%", "Dense ChromaDB cosine similarity over tenant corpus"),
    ("Median (P50)", 39.76, 87.23, "+47.47 ms", "+119.4%", "Typical query: M1-M6 (39.8ms) + Vector Search & Deduplication (47.5ms)"),
    ("75th Percentile (P75)", 46.08, 115.48, "+69.40 ms", "+150.6%", "Multi-passage cross-entailment and confidence score computation"),
    ("90th Percentile (P90)", 98.31, 219.57, "+121.26 ms", "+123.3%", "Heavier text tokenization or longer document parsing"),
    ("95th Percentile (P95)", 163.19, 287.70, "+124.51 ms", "+76.3%", "Standard reference percentile: 287.7 ms is well below interactive web threshold (1.0-2.0s)"),
    ("99th Percentile (P99)", 182.54, 444.16, "+261.62 ms", "+143.3%", "Garbage collection, complex regex extraction, or deep candidate list"),
    ("Maximum (P100)", 410.29, 506.08, "+95.79 ms", "+23.3%", "Worst-case tail latency over 501 cases remains ~0.5 seconds"),
    ("Mean Latency", 55.17, 112.77, "+57.60 ms", "+104.4%", "Statistically significant overhead (t=-24.89, p<0.001) but imperceptible to user"),
    ("Standard Deviation", 39.68, 78.53, "+38.85 ms", "+97.9%", "Higher variance reflects dynamic retrieval depth and safety gate evaluation")
]

for row_idx, r in enumerate(latency_rows, start=5):
    ws4.append(list(r))
    for col_idx in range(1, len(r) + 1):
        cell = ws4.cell(row=row_idx, column=col_idx)
        cell.font = font_regular
        cell.border = thin_border
        if col_idx in [1, 6]:
            cell.alignment = align_left
        else:
            cell.alignment = align_center
            if isinstance(cell.value, float):
                cell.number_format = '0.00 "ms"'
        
        if row_idx in [8, 14]:
            cell.font = font_bold
            cell.fill = fill_highlight_yellow
        elif row_idx % 2 == 1:
            cell.fill = fill_zebra

autofit_columns(ws4)

# ==============================================================================
# TAB 5: Retrieval_Evaluation
# ==============================================================================
ws5 = wb.create_sheet(title="Retrieval_Evaluation")
ws5.views.sheetView[0].showGridLines = True

ws5["A1"] = "Tenant-Isolated Knowledge Retrieval Benchmark (Dense vs. Hybrid vs. BM25)"
ws5["A1"].font = font_title
ws5["A2"] = "Evaluated on Held-Out Test Set (N=428 Answerable Cases, N=501 Total) | Distinct-Document Deduplication"
ws5["A2"].font = font_subtitle

headers_tab5 = [
    "Retrieval Pipeline Strategy",
    "Deduplication (m)",
    "Ans Recall@1",
    "Ans Recall@3",
    "Ans Recall@5",
    "Ans MRR",
    "Overall Recall@5 (N=501)",
    "Mean Latency (ms)",
    "P95 Latency (ms)",
    "Architectural Findings & Production Trade-off"
]

ws5.append([])
ws5.append(headers_tab5)
for col_idx in range(1, len(headers_tab5) + 1):
    cell = ws5.cell(row=4, column=col_idx)
    cell.font = font_header
    cell.fill = fill_header
    cell.alignment = align_center
    cell.border = header_border

retrieval_rows = [
    ("Pure Dense (Raw Chunks)", "Disabled (m=1)", "61.92%", "77.34%", "82.24%", "0.6977", "70.26%", "35.95 ms", "47.04 ms", "Suffers from intra-document redundancy; top-5 slots wasted on same document chunks"),
    ("Pure Dense Distinct (m=4) [RECOMMENDED]", "Enabled (m=4)", "61.92%", "80.37%", "87.38%", "0.7192", "74.65%", "37.19 ms", "46.50 ms", "BEST OVERALL: Highest MRR (0.7192), 87.38% Recall@5, fastest latency, zero keyword dilution"),
    ("Hybrid Distinct (RRF k=60, m=2)", "Enabled (m=2)", "52.10%", "75.93%", "87.38%", "0.6496", "74.65%", "38.19 ms", "47.74 ms", "Matches Recall@5 (87.38%) but suffers lower MRR (0.6496) and Recall@1 (-9.82%) due to lexical noise"),
    ("Hybrid Distinct (RRF k=60, m=4)", "Enabled (m=4)", "52.34%", "72.20%", "83.88%", "0.6364", "71.66%", "38.04 ms", "49.98 ms", "Over-filtering reduces Recall@5 by 3.50%; demonstrates sensitivity to candidate multiplier"),
    ("Historical Unverified Preliminary Estimate", "--", "--", "--", "91.82%", "--", "--", "--", "--", "DISPROVED IN AUDIT: Preliminary claim of 91.82% hybrid recall was an artifact of unverified dev set")
]

for row_idx, r in enumerate(retrieval_rows, start=5):
    ws5.append(list(r))
    for col_idx in range(1, len(r) + 1):
        cell = ws5.cell(row=row_idx, column=col_idx)
        cell.font = font_regular
        cell.border = thin_border
        if col_idx in [1, 10]:
            cell.alignment = align_left
        else:
            cell.alignment = align_center
            
        if "RECOMMENDED" in str(r[0]):
            cell.fill = fill_highlight_green
            cell.font = font_green
        elif "DISPROVED" in str(r[0]):
            cell.fill = fill_highlight_red
            cell.font = font_red
        elif row_idx % 2 == 1:
            cell.fill = fill_zebra

autofit_columns(ws5)

# ==============================================================================
# TAB 6: Data_Source_Provenance
# ==============================================================================
ws6 = wb.create_sheet(title="Data_Source_Provenance")
ws6.views.sheetView[0].showGridLines = True

ws6["A1"] = "Data Provenance, Artifact Manifest, and Experimental Audit Verification"
ws6["A1"].font = font_title
ws6["A2"] = "Complete trace of authoritative log files, test datasets, and reproduction commands"
ws6["A2"].font = font_subtitle

headers_tab6 = [
    "Artifact ID / Component",
    "Relative File Path",
    "Record Count / Scope",
    "Verification Status",
    "Exact Hash / Commit / Generation Timestamp",
    "Research Audit Notes & Reproducibility Protocol"
]

ws6.append([])
ws6.append(headers_tab6)
for col_idx in range(1, len(headers_tab6) + 1):
    cell = ws6.cell(row=4, column=col_idx)
    cell.font = font_header
    cell.fill = fill_header
    cell.alignment = align_center
    cell.border = header_border

provenance_rows = [
    ("Held-Out Test Dataset", "data/processed/autonomy/test.csv", "501 Rows (Unique Case_IDs: 501)", "VERIFIED (Canonical)", "100% Unique Case_IDs, Zero Missing Values", "Authoritative ground-truth test split. 373 ANSWER, 79 ORCHESTRATE, 49 ESCALATE."),
    ("Held-Out Validation Dataset", "data/processed/autonomy/validation.csv", "501 Rows (Unique Case_IDs: 501)", "VERIFIED (Independent)", "Zero overlap with test.csv (0 IDs, 0 Queries)", "Used exclusively for sensitivity exploration and threshold calibration. 384 ANSWER, 76 ORCHESTRATE, 41 ESCALATE."),
    ("Training Dataset", "data/processed/autonomy/train.csv", "2,333 Rows (Unique Case_IDs: 2,333)", "VERIFIED (Training Split)", "Zero overlap with val or test", "Classifier training split. 1,712 ANSWER, 407 ORCHESTRATE, 214 ESCALATE."),
    ("M6 Experiment Metrics Log", "outputs/research_evaluation/m6_experiment_20261031_063000/m6_fixed_vs_adaptive_metrics.csv", "3 Policy Configurations", "VERIFIED (Authoritative)", "Generated 2026-10-31 06:30:00", "Primary comparative table for Policies A, B (Original), and B (Optimized). Contains Wilson CIs and error counts."),
    ("Case-Level Prediction Log", "outputs/research_evaluation/m6_experiment_20261031_063000/m6_case_level_comparison.csv", "501 Rows x 32 Columns", "VERIFIED (Row-Level)", "Complete per-query decision log", "Contains Case_ID, Query, M1-M6 outputs, composite confidence C, and predictions for Policy A, B, and C."),
    ("Confusion Matrices Log", "outputs/research_evaluation/m6_experiment_20261031_063000/m6_confusion_matrices.csv", "9 Rows (3x3 per policy)", "VERIFIED (Exact)", "Exact cross-tabulation", "3x3 contingency matrices for Policy A, Original Policy B, and Optimized Policy B."),
    ("Latency Comparison Log", "outputs/research_evaluation/m6_experiment_20261031_063000/m6_latency_comparison.csv", "11 Percentile Ranks", "VERIFIED (Monotonic)", "Single-pass sequential execution", "Tracks P0 to P100 latency measured via monotonic perf_counter()."),
    ("Statistical Analysis Report", "outputs/research_evaluation/m6_experiment_20261031_063000/m6_statistical_analysis.md", "Complete Report", "VERIFIED (Paired Tests)", "McNemar, Wilson, Student t, Wilcoxon", "Authoritative statistical testing confirming p < 0.001 safety gain."),
    ("RAG Test Benchmark", "outputs/rag_evaluation/test_benchmark_final_comparison.csv", "4 Configurations", "VERIFIED (Authoritative)", "501 test queries (428 answerable)", "Definitive retrieval evaluation proving Dense Distinct (m=4) outperforms Hybrid Distinct."),
    ("RAG Audit & Remediation", "outputs/rag_evaluation/hybrid_retrieval_validation_report.md", "Detailed Root Cause Audit", "VERIFIED (Audit Report)", "Disproved 91.82% preliminary claim", "Detailed post-mortem identifying BM25 keyword dilution and establishing Pure Dense superiority."),
    ("M1 Intent Model", "src/models/m1_intent_linear_svc.joblib", "47 Intent Classes", "VERIFIED (Trained)", "TF-IDF + LinearSVC (94.3% Acc)", "Fast-path customer intent classifier."),
    ("M2 Sentiment Model", "src/models/m2_sentiment_logistic_regression.joblib", "3 Classes (Pos/Neu/Neg)", "VERIFIED (Trained)", "Multinomial LR (Macro F1 0.884)", "Identifies negative/frustrated sentiment for risk elevation."),
    ("M3 Operational Risk Model", "src/models/m3_risk_random_forest.joblib", "3 Risk Tiers (Low/Med/High)", "VERIFIED (Trained)", "Random Forest 200 Trees (98.1% Prec)", "Detects account compromise, legal threats, and financial exposure."),
    ("M4 Complexity Model", "src/models/m4_complexity_logistic_regression.joblib", "3 Tiers (Low/Med/High)", "VERIFIED (Trained)", "Multinomial LR (Macro F1 0.862)", "Assesses multi-intent and technical troubleshooting difficulty."),
    ("M5 Domain Model", "src/models/m5_domain_linear_svc.joblib", "3 Domains (Ecom/Edu/SMB)", "VERIFIED (Trained)", "LinearSVC (99.2% Acc)", "Tenant isolation classifier routing queries to domain knowledge base."),
    ("M6 Autonomy Model", "src/models/m6_autonomy_logistic_regression.joblib", "3 Decisions (ANSWER/ORCH/ESC)", "VERIFIED (Trained)", "Multinomial LR (88.62% Acc)", "Provisional autonomy classifier operating before retrieval.")
]

for row_idx, r in enumerate(provenance_rows, start=5):
    ws6.append(list(r))
    for col_idx in range(1, len(r) + 1):
        cell = ws6.cell(row=row_idx, column=col_idx)
        cell.font = font_regular
        cell.border = thin_border
        if col_idx in [1, 2, 6]:
            cell.alignment = align_left
        else:
            cell.alignment = align_center
            
        if "VERIFIED" in str(r[3]):
            cell.font = font_bold
        if row_idx % 2 == 1:
            cell.fill = fill_zebra

autofit_columns(ws6)

wb.save(EXCEL_PATH)
print("Successfully generated fully resolved Excel workbook:", EXCEL_PATH)
