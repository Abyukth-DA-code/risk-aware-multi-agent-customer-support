"""
Build Development Retrieval Benchmark from the validation split of FINAL_MASTER_GOLD.csv.

This creates a held-out development dataset (501 queries) completely disjoint from the
test benchmark, allowing hyperparameter tuning (e.g. candidate pool multiplier)
without repeatedly evaluating or overfitting on the final test benchmark.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "processed" / "FINAL_MASTER_GOLD.csv"
DOCS_PATH = ROOT / "data" / "processed" / "rag" / "documents.csv"
DEV_BENCHMARK_PATH = ROOT / "data" / "processed" / "rag" / "retrieval_dev_benchmark.csv"

# Verified intent-to-document mappings established during the audit
INTENT_TO_DOC_MAPPING = {
    # E-commerce
    ("E-commerce", "cancel_order"): ("E003", "Order Cancellation Policy", "High", "Current"),
    ("E-commerce", "change_order"): ("E003", "Order Cancellation Policy", "High", "Current"),
    ("E-commerce", "change_shipping_address"): ("E007", "Shipping Address Policy", "High", "Current"),
    ("E-commerce", "check_cancellation_fee"): ("E003", "Order Cancellation Policy", "High", "Current"),
    ("E-commerce", "check_invoice"): ("E009", "Invoice and Receipt Policy", "High", "Current"),
    ("E-commerce", "check_payment_methods"): ("", "", "", ""),  # Gap: no payment methods policy
    ("E-commerce", "check_refund_policy"): ("E002", "Refund Processing Policy", "High", "Current"),
    ("E-commerce", "complaint"): ("E012", "Customer Support Contact Policy", "High", "Current"),
    ("E-commerce", "contact_customer_service"): ("E012", "Customer Support Contact Policy", "High", "Current"),
    ("E-commerce", "contact_human_agent"): ("E012", "Customer Support Contact Policy", "High", "Current"),
    ("E-commerce", "create_account"): ("E006", "Account Security Policy", "High", "Current"),
    ("E-commerce", "delete_account"): ("E006", "Account Security Policy", "High", "Current"),
    ("E-commerce", "delivery_options"): ("E004", "Delivery Delay Policy", "High", "Current"),
    ("E-commerce", "delivery_period"): ("E004", "Delivery Delay Policy", "High", "Current"),
    ("E-commerce", "edit_account"): ("E006", "Account Security Policy", "High", "Current"),
    ("E-commerce", "get_invoice"): ("E009", "Invoice and Receipt Policy", "High", "Current"),
    ("E-commerce", "get_refund"): ("E002", "Refund Processing Policy", "High", "Current"),
    ("E-commerce", "newsletter_subscription"): ("E008", "Subscription Cancellation Policy", "High", "Current"),
    ("E-commerce", "payment_issue"): ("E005", "Duplicate Payment Policy", "High", "Current"),
    ("E-commerce", "place_order"): ("", "", "", ""),  # Gap: no shopping catalog policy
    ("E-commerce", "recover_password"): ("E006", "Account Security Policy", "High", "Current"),
    ("E-commerce", "registration_problems"): ("E006", "Account Security Policy", "High", "Current"),
    ("E-commerce", "review"): ("", "", "", ""),  # Gap: no review policy
    ("E-commerce", "set_up_shipping_address"): ("E007", "Shipping Address Policy", "High", "Current"),
    ("E-commerce", "switch_account"): ("E006", "Account Security Policy", "High", "Current"),
    ("E-commerce", "track_order"): ("E004", "Delivery Delay Policy", "High", "Current"),
    ("E-commerce", "track_refund"): ("E002", "Refund Processing Policy", "High", "Current"),

    # Education
    ("Education", "Academic Records"): ("E024", "Document Request Policy", "High", "Current"),
    ("Education", "Attendance Policy"): ("E013", "Student Attendance Policy", "High", "Current"),
    ("Education", "Course Registration"): ("E019", "Course Registration Policy", "High", "Current"),
    ("Education", "Exam Registration"): ("E014", "Examination Rules", "High", "Current"),
    ("Education", "Exam Schedule"): ("E014", "Examination Rules", "High", "Current"),
    ("Education", "Fees and Payment"): ("E016", "Tuition Fee Payment Policy", "High", "Current"),
    ("Education", "Library"): ("", "", "", ""),  # Gap: no library policy
    ("Education", "Placement Support"): ("", "", "", ""),  # Gap: no placement policy
    ("Education", "Scholarship"): ("E021", "Scholarship Eligibility Policy", "High", "Current"),
    ("Education", "Student ID"): ("E022", "Student Portal Access Policy", "High", "Current"),

    # Small Business
    ("Small Business", "Account Management"): ("E030", "Customer Information Policy", "High", "Current"),
    ("Small Business", "Business Payment"): ("E026", "Customer Payment Policy", "High", "Current"),
    ("Small Business", "Customer Support"): ("E030", "Customer Information Policy", "High", "Current"),
    ("Small Business", "Delivery Inquiry"): ("", "", "", ""),  # Gap: no delivery address policy in SB
    ("Small Business", "Invoice Inquiry"): ("E025", "Business Invoice Policy", "High", "Current"),
    ("Small Business", "Order Management"): ("E028", "Order Cancellation Policy", "High", "Current"),
    ("Small Business", "Product Information"): ("", "", "", ""),  # Gap: no product catalog policy
    ("Small Business", "Return and Refund"): ("E027", "Customer Refund Policy", "High", "Current"),
    ("Small Business", "Subscription Management"): ("", "", "", ""),  # Gap: no subscription policy in SB
    ("Small Business", "Vendor Support"): ("", "", "", ""),  # Gap: no vendor contact policy in SB
}


def build_dev_benchmark():
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold master dataset not found: {GOLD_PATH}")

    gold = pd.read_csv(GOLD_PATH)
    val = gold[gold["Split"] == "validation"].copy()

    print(f"Loaded {len(val)} validation queries from FINAL_MASTER_GOLD.csv")

    rows = []
    for _, r in val.iterrows():
        domain = str(r["Domain"]).strip()
        intent = str(r["Intent"]).strip()

        mapping = INTENT_TO_DOC_MAPPING.get((domain, intent), ("", "", "", ""))
        exp_id, exp_title, exp_auth, exp_status = mapping

        rows.append({
            "Case_ID": r["Case_ID"],
            "Query": r["Query"],
            "Domain": domain,
            "Intent": intent,
            "Expected_Document_ID": exp_id,
            "Expected_Title": exp_title,
            "Expected_Authority": exp_auth,
            "Expected_Status": exp_status,
            "Validation_Method": "HeldOut_Validation_Split",
            "Evaluation_Split": "Validation",
        })

    dev_df = pd.DataFrame(rows)
    DEV_BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    dev_df.to_csv(DEV_BENCHMARK_PATH, index=False)

    print(f"Created dev benchmark: {DEV_BENCHMARK_PATH}")
    print(f"Total dev queries: {len(dev_df)}")
    print(f"Answerable queries: {(dev_df['Expected_Document_ID'].str.strip() != '').sum()}")
    print(f"Knowledge gap queries: {(dev_df['Expected_Document_ID'].str.strip() == '').sum()}")
    return dev_df


if __name__ == "__main__":
    build_dev_benchmark()
