"""
Dynamic Knowledge-Base Checklist Engine
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Generates domain-specific required and recommended knowledge-base checklists,
tracks category ingestion status, and computes true backend readiness.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from src.workspaces.models import BusinessDomain, ChecklistItemStatus, ReadinessReport

# ---------------------------------------------------------------------------
# Canonical Domain Checklist Specifications
# ---------------------------------------------------------------------------

DOMAIN_CHECKLIST_SPECS: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    BusinessDomain.ECOMMERCE.value: {
        "required": [
            {
                "key": "return_policy",
                "name": "Return Policy",
                "warning": "Customer inquiries regarding product returns, return windows, and item conditions cannot be answered.",
            },
            {
                "key": "refund_policy",
                "name": "Refund Policy",
                "warning": "Queries regarding refund approval timelines, deductions, and payment reversal cannot be answered.",
            },
            {
                "key": "order_cancellation",
                "name": "Order Cancellation Policy",
                "warning": "Inquiries regarding order cancellation windows and dispatch restrictions cannot be answered.",
            },
            {
                "key": "delivery_shipping",
                "name": "Delivery and Shipping Policy",
                "warning": "Shipping timeline, carrier, tracking, and delivery delay questions cannot be answered.",
            },
            {
                "key": "payment_methods",
                "name": "Payment Methods and Billing Policy",
                "warning": "Payment gateway, duplicate charges, and billing error questions cannot be answered.",
            },
            {
                "key": "customer_support_escalation",
                "name": "Customer Support and Escalation Policy",
                "warning": "Queries requiring formal escalation, supervisor review, or human contact cannot be routed properly.",
            },
        ],
        "recommended": [
            {"key": "product_warranty", "name": "Product Warranty Policy"},
            {"key": "faq", "name": "Frequently Asked Questions (FAQ)"},
            {"key": "product_service_info", "name": "Product and Service Information"},
            {"key": "order_fulfilment_guidance", "name": "Order and Fulfilment Guidance"},
        ],
    },
    BusinessDomain.EDUCATION.value: {
        "required": [
            {
                "key": "attendance_policy",
                "name": "Attendance Policy",
                "warning": "Student queries regarding mandatory attendance percentage, shortage condonation, and medical leave cannot be answered.",
            },
            {
                "key": "examination_grading",
                "name": "Examination and Grading Policy",
                "warning": "Exam registration, grading rubrics, revaluation, and hall ticket eligibility queries cannot be answered.",
            },
            {
                "key": "course_registration_records",
                "name": "Course Registration and Academic Records",
                "warning": "Course enrollment prerequisites, grade corrections, and transcript queries cannot be answered.",
            },
            {
                "key": "fees_payment",
                "name": "Fees and Payment Policy",
                "warning": "Tuition deadlines, installment plans, overdue holds, and fee refund queries cannot be answered.",
            },
            {
                "key": "student_support_escalation",
                "name": "Student Support and Escalation Policy",
                "warning": "Academic grievances, disciplinary appeals, and dean escalation processes cannot be routed.",
            },
        ],
        "recommended": [
            {"key": "placement_policy", "name": "Placement Policy"},
            {"key": "library_policy", "name": "Library Policy"},
            {"key": "admission_eligibility", "name": "Admission and Eligibility Rules"},
            {"key": "academic_calendar", "name": "Academic Calendar"},
            {"key": "faq", "name": "Frequently Asked Questions (FAQ)"},
        ],
    },
    BusinessDomain.SMALL_BUSINESS.value: {
        "required": [
            {
                "key": "products_services_info",
                "name": "Products and Services Information",
                "warning": "Customer questions regarding service specifications, deliverables, and capabilities cannot be answered.",
            },
            {
                "key": "pricing_invoices_billing",
                "name": "Pricing, Invoices, and Billing Policy",
                "warning": "Pricing schedules, GST/tax invoicing, and credit term inquiries cannot be answered.",
            },
            {
                "key": "subscription_cancellation",
                "name": "Subscription and Cancellation Policy",
                "warning": "Recurring subscription renewals, notice periods, and termination terms cannot be answered.",
            },
            {
                "key": "orders_inventory_fulfillment",
                "name": "Orders, Inventory, or Fulfilment Guidance",
                "warning": "Stock availability, lead times, order tracking, and fulfilment procedures cannot be answered.",
            },
            {
                "key": "customer_support_escalation",
                "name": "Customer Support and Escalation Policy",
                "warning": "Dispute handling, manager review, and formal escalation procedures cannot be routed.",
            },
        ],
        "recommended": [
            {"key": "vendor_supplier_policies", "name": "Vendor and Supplier Policies"},
            {"key": "returns_warranty", "name": "Returns and Warranty Information"},
            {"key": "faq", "name": "Frequently Asked Questions (FAQ)"},
            {"key": "business_operating_procedures", "name": "Business Operating Procedures"},
        ],
    },
}


class ChecklistEngine:
    """Manages domain-specific checklists and evaluates true operational readiness."""

    @staticmethod
    def get_default_checklist_items(domain: str, workspace_id: str) -> List[Dict[str, Any]]:
        """Generates the initial unpopulated checklist for a new workspace."""
        spec = DOMAIN_CHECKLIST_SPECS.get(domain)
        if not spec:
            raise ValueError(
                f"Unknown domain '{domain}'. Must be one of: "
                f"{[d.value for d in BusinessDomain]}"
            )

        now = datetime.now(timezone.utc).isoformat()
        items = []

        # Required items
        for entry in spec["required"]:
            items.append({
                "item_id": f"chk_{uuid.uuid4().hex[:12]}",
                "workspace_id": workspace_id,
                "domain": domain,
                "category_key": entry["key"],
                "category_name": entry["name"],
                "is_required": True,
                "status": ChecklistItemStatus.NOT_UPLOADED.value,
                "extracted_records_count": 0,
                "last_updated": now,
            })

        # Recommended items
        for entry in spec["recommended"]:
            items.append({
                "item_id": f"chk_{uuid.uuid4().hex[:12]}",
                "workspace_id": workspace_id,
                "domain": domain,
                "category_key": entry["key"],
                "category_name": entry["name"],
                "is_required": False,
                "status": ChecklistItemStatus.NOT_UPLOADED.value,
                "extracted_records_count": 0,
                "last_updated": now,
            })

        return items

    @staticmethod
    def calculate_readiness(
        checklist_items: List[Dict[str, Any]],
        domain: str,
        workspace_id: str,
    ) -> ReadinessReport:
        """
        Derives operational readiness from verified checklist items.
        A category is ONLY counted as complete if its status is 'Validated'.
        """
        spec = DOMAIN_CHECKLIST_SPECS.get(domain, {"required": [], "recommended": []})
        warning_map = {item["key"]: item.get("warning", "") for item in spec.get("required", [])}

        total_categories = len(checklist_items)
        required_items = [item for item in checklist_items if item["is_required"]]
        recommended_items = [item for item in checklist_items if not item["is_required"]]

        # Count completed (must be explicitly 'Validated')
        required_completed = sum(1 for item in required_items if item["status"] == ChecklistItemStatus.VALIDATED.value)
        recommended_completed = sum(1 for item in recommended_items if item["status"] == ChecklistItemStatus.VALIDATED.value)

        # Missing required items & associated unanswerable query warnings
        missing_required = []
        warnings = []
        for item in required_items:
            if item["status"] != ChecklistItemStatus.VALIDATED.value:
                cat_name = item["category_name"]
                cat_key = item["category_key"]
                missing_required.append(cat_name)
                w = warning_map.get(cat_key, f"Questions regarding {cat_name} cannot be reliably answered.")
                warnings.append(w)

        # Status summary counts
        status_summary: Dict[str, int] = {}
        for item in checklist_items:
            s = item["status"]
            status_summary[s] = status_summary.get(s, 0) + 1

        total_required = len(required_items)
        readiness_percentage = round(
            (required_completed / total_required * 100.0) if total_required > 0 else 0.0,
            2,
        )

        # Ready for testing ONLY when 100% of required categories are validated
        is_ready = (required_completed == total_required) and (total_required > 0)

        return ReadinessReport(
            workspace_id=workspace_id,
            domain=domain,
            total_categories=total_categories,
            required_categories=total_required,
            required_completed=required_completed,
            recommended_completed=recommended_completed,
            is_ready_for_testing=is_ready,
            readiness_percentage=readiness_percentage,
            missing_required_categories=missing_required,
            unanswerable_query_warnings=warnings,
            status_summary=status_summary,
        )
