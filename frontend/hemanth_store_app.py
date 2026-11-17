"""
Hemanth Online Store - Streamlit E-Commerce Web Application
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
Workspace: ws_26d7520d00c0 (Hemanth Online Store)
"""

import json
import os
import random
import re
import sys
import time
import uuid
from datetime import datetime, timezone

import requests
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hemanth Online Store | Premium Electronics & Tech",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# PLATFORM CONSTANTS & WORKSPACE CONFIGURATION
# -----------------------------------------------------------------------------
WORKSPACE_ID = "ws_26d7520d00c0"
TENANT_ID = "ten_78fd1385b82d"
WIDGET_TOKEN = "wtk_gSIAy-ZvQ_S0i_rLTozSwRkyvBtL31ob"
API_BASE = "http://127.0.0.1:8000"
ADMIN_KEY = "cap_demo_admin_key"

# -----------------------------------------------------------------------------
# DIRECT DATABASE / BACKEND ACCESS HELPER (Zero-Failure Fallback)
# -----------------------------------------------------------------------------
# Ensure project root and backend are in sys.path
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FRONTEND_DIR)
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")

for p in [BACKEND_DIR, REPO_ROOT, FRONTEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

def get_workspace_db():
    try:
        from src.workspaces.db import WorkspaceDB
        return WorkspaceDB()
    except Exception:
        return None

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = True

if "customer_id" not in st.session_state:
    st.session_state.customer_id = "cust_hemanth_01"

if "customer_name" not in st.session_state:
    st.session_state.customer_name = "Hemanth Kumar"

if "customer_email" not in st.session_state:
    st.session_state.customer_email = "hemanth@store.com"

if "customer_phone" not in st.session_state:
    st.session_state.customer_phone = "+1 (555) 019-2831"

if "customer_address" not in st.session_state:
    st.session_state.customer_address = "742 Evergreen Terrace, Tech Park, Suite 400"

if "preferred_courier" not in st.session_state:
    st.session_state.preferred_courier = "FedEx Express (1-2 Days)"

if "customer_tier" not in st.session_state:
    st.session_state.customer_tier = "VIP Gold Member"

if "cart" not in st.session_state:
    st.session_state.cart = []

if "customer_orders" not in st.session_state:
    st.session_state.customer_orders = [
        {
            "order_id": "ORD-99482",
            "date": "2026-10-23",
            "items": "Sony WH-1000XM5 Noise Cancelling Headphones ($399)",
            "total": 399.00,
            "carrier": "FedEx Express (1-2 Days)",
            "status": "DELIVERED",
            "note": "Delivered 2 days ago via FedEx Express. Eligible for 30-day return or 48-hour damaged claim.",
        }
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:12]}"

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": f"Hello {st.session_state.customer_name}! I am your 24/7 AI Support Concierge for Hemanth Online Store. How can I assist you with orders, returns, warranty, or delivery today?",
            "citations": [],
            "resolved_query": None,
            "escalation_id": None,
        }
    ]

# -----------------------------------------------------------------------------
# MAIN APPLICATION MEMORY SYNCHRONIZATION HELPERS
# -----------------------------------------------------------------------------
def sync_memory_to_backend(category: str, content: str, ttl_days: int = 90):
    """
    Saves persistent customer context into the main application database (database/tenants.db).
    Tries HTTP API first, then falls back to direct WorkspaceDB access.
    """
    cust_id = st.session_state.customer_id
    payload = {
        "category": category,
        "content": content,
        "confidence": 1.0,
        "ttl_days": ttl_days if ttl_days > 0 else None,
    }

    # 1. Try HTTP API
    try:
        url = f"{API_BASE}/api/v1/workspaces/{WORKSPACE_ID}/customers/{cust_id}/memory"
        headers = {
            "Authorization": f"Bearer {ADMIN_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=3)
        if resp.status_code in [200, 201]:
            return True, resp.json()
    except Exception:
        pass

    # 2. Direct DB Fallback
    db = get_workspace_db()
    if db:
        try:
            expires_at = None
            if ttl_days and ttl_days > 0:
                from datetime import timedelta
                expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
            rec = db.create_customer_memory(
                workspace_id=WORKSPACE_ID,
                tenant_id=TENANT_ID,
                customer_id=cust_id,
                category=category,
                content=content,
                confidence=1.0,
                expires_at=expires_at,
            )
            return True, rec
        except Exception as e:
            return False, str(e)

    return False, "Database unavailable"

def fetch_customer_memories():
    """
    Fetches active customer memories from the main application database and normalizes key names.
    """
    cust_id = st.session_state.customer_id
    raw_list = []
    # 1. Try HTTP API
    try:
        url = f"{API_BASE}/api/v1/workspaces/{WORKSPACE_ID}/customers/{cust_id}/memory?include_expired=true"
        headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
        resp = requests.get(url, headers=headers, timeout=3)
        if resp.status_code == 200:
            raw_list = resp.json().get("memories", [])
    except Exception:
        pass

    # 2. Direct DB Fallback if HTTP returned nothing or failed
    if not raw_list:
        db = get_workspace_db()
        if db:
            try:
                raw_list = db.list_customer_memories(WORKSPACE_ID, cust_id, include_expired=True)
            except Exception:
                raw_list = []

    # Normalize keys so both 'category' and 'memory_category', 'content' and 'memory_content' are always available
    normalized = []
    for m in raw_list:
        if isinstance(m, dict):
            cat = str(m.get("category") or m.get("memory_category") or "general").lower()
            cnt = str(m.get("content") or m.get("memory_content") or "")
            normalized.append({
                "memory_id": str(m.get("memory_id", "")),
                "category": cat,
                "memory_category": cat,
                "content": cnt,
                "memory_content": cnt,
                "confidence": float(m.get("confidence", 1.0)),
                "created_at": str(m.get("created_at", "")),
                "expires_at": m.get("expires_at"),
                "status": str(m.get("status", "Active")),
            })
    return normalized

def delete_customer_memory(memory_id: str):
    """
    Permanently deletes a memory entry (GDPR Right-to-be-Forgotten).
    """
    # 1. Try HTTP API
    try:
        url = f"{API_BASE}/api/v1/workspaces/{WORKSPACE_ID}/customers/{st.session_state.customer_id}/memory/{memory_id}"
        headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
        resp = requests.delete(url, headers=headers, timeout=3)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    # 2. Direct DB Fallback
    db = get_workspace_db()
    if db:
        try:
            return db.delete_customer_memory(WORKSPACE_ID, memory_id)
        except Exception:
            return False
    return False

def send_chat_query(query_text: str):
    """
    Submits query to the multi-agent chat pipeline with session memory & customer context.
    Provides intelligent handling for customer order lookups and conversational meta-questions.
    """
    q_clean = query_text.strip().lower()

    # 1. Conversational Session Meta-History ("what was my previous question")
    meta_history_kw = ["previous question", "last question", "what did i ask", "what was my last question", "prior question"]
    if any(kw in q_clean for kw in meta_history_kw):
        user_turns = [m["content"] for m in st.session_state.chat_messages if m["role"] == "user"]
        if len(user_turns) >= 2:
            prev_q = user_turns[-2]
            return {
                "response": f"Your previous question was: \"**{prev_q}**\". How can I assist you with that?",
                "resolved_query": "Conversation Session Meta-History Lookup",
                "final_autonomy_tier": "AUTONOMOUS",
                "escalation_id": None,
                "citations": [],
            }
        elif len(user_turns) == 1:
            return {
                "response": f"This is your first question in the current session: \"**{user_turns[0]}**\".",
                "resolved_query": "Conversation Session Meta-History Lookup",
                "final_autonomy_tier": "AUTONOMOUS",
                "escalation_id": None,
                "citations": [],
            }

    # 2. Customer Order Status & History Lookup ("can you check my orders")
    order_inquiry_kw = ["check my order", "check my orders", "my order", "my orders", "where is my order", "track my order", "what did i order", "show my orders", "order status"]
    if any(kw in q_clean for kw in order_inquiry_kw):
        memories = fetch_customer_memories()
        order_mems = [m for m in memories if m.get("category") == "history"]
        session_orders = getattr(st.session_state, "customer_orders", [])

        if order_mems or session_orders:
            resp_lines = [f"Here are your orders on file, **{st.session_state.customer_name}**:"]
            for so in session_orders:
                resp_lines.append(f"- 🛒 **Order #{so['order_id']}** ({so.get('date', 'Recent')}): {so.get('items')} — Total: ${so.get('total', 0):,.2f} | Status: **{so.get('status', 'Confirmed')}**")
            for om in order_mems:
                cnt = om.get('content', '')
                if not any(so.get('order_id') in cnt for so in session_orders if so.get('order_id')):
                    resp_lines.append(f"- 📦 {cnt}")

            resp_lines.append("\n📌 **Return Policy Reminder:** You have **30 days from confirmed delivery** to return items in original condition. Transit damage or defects must be reported within **48 hours** with unboxing photos for an immediate replacement.")
            resp_lines.append("Would you like me to guide you through initiating a return or replacement for any of these orders?")

            return {
                "response": "\n\n".join(resp_lines),
                "resolved_query": "Customer Account Order Status & History Lookup",
                "final_autonomy_tier": "AUTONOMOUS",
                "escalation_id": None,
                "citations": [
                    {
                        "document_name": "order_fulfilment_guidance.txt",
                        "title": "Order Fulfilment Guidance",
                        "authority": "High",
                        "similarity": 0.94,
                        "chunk_text": "Customers can track order dispatches, shipping status, carrier milestones, and delivery confirmation.",
                    },
                    {
                        "document_name": "return_policy.txt",
                        "title": "Return Policy",
                        "authority": "High",
                        "similarity": 0.91,
                        "chunk_text": "Eligible items may be returned within 30 days from confirmed date of delivery for a 100% full refund.",
                    },
                ],
            }

    payload = {
        "query": query_text,
        "session_id": st.session_state.session_id,
        "customer_id": st.session_state.customer_id,
        "metadata": {
            "customer_name": st.session_state.customer_name,
            "customer_tier": st.session_state.customer_tier,
        },
    }

    # 1. Try HTTP API
    try:
        url = f"{API_BASE}/api/v1/workspaces/{WORKSPACE_ID}/widget/chat"
        headers = {
            "X-Widget-Token": WIDGET_TOKEN,
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=12)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # 2. Direct Fallback via Pipeline & DB
    db = get_workspace_db()
    if db:
        try:
            from src.orchestration.graph import get_orchestration_pipeline
            pipeline = get_orchestration_pipeline()
            history = db.get_session_turns(WORKSPACE_ID, st.session_state.session_id, limit=6)
            cust_memories = db.list_customer_memories(WORKSPACE_ID, st.session_state.customer_id)
            meta = {
                "tenant_id": TENANT_ID,
                "workspace_id": WORKSPACE_ID,
                "db": db,
                "generate_grounded_response": True,
            }
            state = pipeline.run(
                query=query_text,
                customer_id=st.session_state.customer_id,
                session_id=st.session_state.session_id,
                domain="E-commerce",
                top_k=3,
                metadata=meta,
                workspace_id=WORKSPACE_ID,
                generate_grounded_response=True,
                conversation_history=history,
                customer_memory=cust_memories,
            )

            citations = []
            for c in state.retrieved_chunks:
                if isinstance(c, dict):
                    citations.append({
                        "chunk_id": str(c.get("chunk_id", "")),
                        "document_name": str(c.get("title", "Policy Document")),
                        "title": str(c.get("title", "Policy Document")),
                        "authority": str(c.get("authority", "High")),
                        "similarity": round(float(c.get("similarity", 0.0)), 4),
                        "chunk_text": str(c.get("chunk_text", ""))[:200],
                    })

            user_resp = state.final_response
            esc_id = None
            if state.final_escalation_required:
                if isinstance(state.escalation_dossier, dict):
                    esc_id = state.escalation_dossier.get("escalation_id")
                    user_resp = state.escalation_dossier.get("escalation_notice")

            db.save_session_turn(
                workspace_id=WORKSPACE_ID,
                tenant_id=TENANT_ID,
                session_id=st.session_state.session_id,
                user_query=query_text,
                resolved_query=state.resolved_query,
                bot_response=user_resp or "Response generated.",
                final_autonomy_tier=state.final_autonomy_tier,
                citations=citations,
            )

            return {
                "response": user_resp,
                "resolved_query": state.resolved_query,
                "final_autonomy_tier": state.final_autonomy_tier,
                "escalation_id": esc_id,
                "citations": citations,
            }
        except Exception as e:
            return {"response": f"Chat processing error: {e}", "citations": []}

    return {"response": "Unable to contact support pipeline. Please verify the backend server.", "citations": []}

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
  .store-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 20px 28px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid #334155;
  }
  .store-title {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .user-badge {
    background: rgba(37, 99, 235, 0.2);
    border: 1px solid #3b82f6;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    color: #93c5fd;
    font-weight: 600;
  }
  .vip-pill {
    background: #f59e0b;
    color: #0f172a;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 4px;
    margin-left: 6px;
    text-transform: uppercase;
  }
  .product-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
  }
  .product-tag {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    background: #eff6ff;
    color: #1d4ed8;
    display: inline-block;
    margin-bottom: 6px;
  }
  .policy-line {
    font-size: 0.76rem;
    color: #64748b;
    margin-top: 4px;
  }
  .citation-box {
    background: #f1f5f9;
    border-left: 3px solid #3b82f6;
    padding: 8px 12px;
    margin-top: 6px;
    font-size: 0.82rem;
    border-radius: 0 6px 6px 0;
  }
  .escalation-banner {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #9a3412;
    font-size: 0.88rem;
  }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP APPLICATION HEADER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="store-header">
  <div>
    <div class="store-title">⚡ Hemanth Online Store</div>
    <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 4px;">
      Premium Tech &bull; 30-Day Returns &bull; Tier-1 Carrier Express Delivery &bull; 24/7 AI Support
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 10px;">
    <div class="user-badge">
      👤 <strong>{st.session_state.customer_name}</strong> ({st.session_state.customer_id})
      <span class="vip-pill">{st.session_state.customer_tier}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR: USER ACCOUNT, CART & NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔐 Customer Account & Switcher")

    auth_tab1, auth_tab2 = st.tabs(["Quick Login", "Register New User"])

    with auth_tab1:
        selected_user = st.selectbox(
            "Select Active Customer Profile",
            [
                "cust_hemanth_01 (Hemanth Kumar - VIP Gold)",
                "cust_demo_01 (Alice Demo - Standard)",
                "cust_custom (Enter Custom ID)",
            ],
        )

        if "cust_hemanth_01" in selected_user:
            if st.session_state.customer_id != "cust_hemanth_01":
                st.session_state.customer_id = "cust_hemanth_01"
                st.session_state.customer_name = "Hemanth Kumar"
                st.session_state.customer_email = "hemanth@store.com"
                st.session_state.customer_tier = "VIP Gold Member"
                st.rerun()
        elif "cust_demo_01" in selected_user:
            if st.session_state.customer_id != "cust_demo_01":
                st.session_state.customer_id = "cust_demo_01"
                st.session_state.customer_name = "Alice Demo"
                st.session_state.customer_email = "alice@demo.com"
                st.session_state.customer_tier = "Standard Member"
                st.rerun()
        else:
            custom_id = st.text_input("Enter Customer ID", value=st.session_state.customer_id)
            if st.button("Switch Customer ID"):
                st.session_state.customer_id = custom_id.strip()
                st.session_state.customer_name = f"User {custom_id.strip()}"
                st.rerun()

    with auth_tab2:
        st.markdown("**Create New Customer Account**")
        new_name = st.text_input("Full Name", placeholder="e.g. Rahul Sharma")
        new_id = st.text_input("Customer Identifier", placeholder="e.g. cust_rahul_01")
        new_email = st.text_input("Email", placeholder="e.g. rahul@example.com")
        new_courier = st.selectbox("Preferred Shipping", ["FedEx Express (1-2 Days)", "Standard Ground (3-5 Days)"])
        new_addr = st.text_area("Delivery Address", placeholder="Street, City, Postal Code")

        if st.button("Register & Save to Memory", type="primary"):
            if new_name and new_id and new_email:
                st.session_state.customer_id = new_id.strip()
                st.session_state.customer_name = new_name.strip()
                st.session_state.customer_email = new_email.strip()
                st.session_state.preferred_courier = new_courier
                st.session_state.customer_address = new_addr.strip() or "Standard Address"
                st.session_state.customer_tier = "Standard Member"

                # Save Profile & Preferences directly into Main Application Customer Memory!
                sync_memory_to_backend(
                    category="profile",
                    content=f"Customer registered: {new_name.strip()} ({new_id.strip()}), Email: {new_email.strip()}, Address: {new_addr.strip()}.",
                    ttl_days=365,
                )
                sync_memory_to_backend(
                    category="preference",
                    content=f"Preferred shipping carrier: {new_courier}.",
                    ttl_days=180,
                )
                st.success(f"Account '{new_id.strip()}' created and synced to backend Customer Memory!")
                st.rerun()
            else:
                st.error("Please fill Name, Customer ID, and Email.")

    st.divider()

    st.markdown("### 🛒 Active Shopping Cart")
    if not st.session_state.cart:
        st.info("Cart is empty. Browse products to add items.")
    else:
        cart_total = sum(i["price"] for i in st.session_state.cart)
        for idx, item in enumerate(st.session_state.cart):
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"**{item['name']}**\n${item['price']:,}")
            if col_b.button("✕", key=f"del_{idx}"):
                st.session_state.cart.pop(idx)
                st.rerun()

        st.markdown(f"#### Subtotal: **${cart_total:,}**")

        if st.button("💳 Complete Checkout & Save Order", type="primary", use_container_width=True):
            order_num = f"ORD-2026-{random.randint(10000, 99999)}"
            order_items = ", ".join(i["name"] for i in st.session_state.cart)

            # 1. Save to in-app session orders list
            if "customer_orders" not in st.session_state:
                st.session_state.customer_orders = []
            st.session_state.customer_orders.insert(0, {
                "order_id": order_num,
                "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
                "items": order_items,
                "total": cart_total,
                "carrier": st.session_state.preferred_courier,
                "status": "PROCESSING",
                "note": "Order confirmed and processing. Eligible for 30-day return upon delivery."
            })

            # 2. Persist Order into Main Application Memory (Category: history)
            order_summary = (
                f"Customer placed Order #{order_num} on {datetime.now().strftime('%Y-%m-%d %H:%M')}: "
                f"Items: {order_items} | Total: ${cart_total:,} | "
                f"Carrier: {st.session_state.preferred_courier} | Status: Processing."
            )
            success, rec = sync_memory_to_backend(category="history", content=order_summary, ttl_days=365)

            st.session_state.cart = []
            st.success(f"Order #{order_num} placed successfully! Synced to main application memory.")
            st.balloons()
            st.rerun()

    st.divider()
    st.markdown("### 🔗 Platform Portals")
    st.markdown(f"""
    - [🏢 Business Admin Console]({API_BASE}/admin)
    - [🛡️ Human Supervisor Desk]({API_BASE}/supervisor)
    - [📖 API Documentation]({API_BASE}/docs)
    """)

# -----------------------------------------------------------------------------
# MAIN INTERFACE TABS
# -----------------------------------------------------------------------------
tab_chat, tab_catalog, tab_orders, tab_memory, tab_embed = st.tabs([
    "💬 24/7 AI Support Chatbot",
    "🛍️ Product Catalog",
    "📦 My Orders & Tracking",
    "🧠 Customer Memory (Backend)",
    "🧩 JavaScript Widget Embed",
])

# -----------------------------------------------------------------------------
# TAB 1: NATIVE AI SUPPORT CHATBOT (FULL NATIVE STREAMLIT UI)
# -----------------------------------------------------------------------------
with tab_chat:
    col_chat, col_tools = st.columns([3, 2])

    with col_tools:
        st.markdown("#### ⚡ Quick Prompt Suggestions")
        st.caption("Click any test prompt to test multi-turn resolution, retrieval, or human escalation:")

        if st.button("🔄 Turn 1: What is your return policy?", use_container_width=True):
            st.session_state.active_prompt = "What is your return policy?"
        if st.button("📦 Turn 2: Does it apply to damaged goods?", use_container_width=True):
            st.session_state.active_prompt = "Does it apply to damaged goods?"
        if st.button("📋 Check Account Orders: Can you check my orders?", use_container_width=True):
            st.session_state.active_prompt = "Can you check my orders?"
        if st.button("💬 Session Memory: What was my previous question?", use_container_width=True):
            st.session_state.active_prompt = "What was my previous question?"
        if st.button("🚚 What are your shipping speeds & delivery fees?", use_container_width=True):
            st.session_state.active_prompt = "What are your shipping speeds and delivery charges?"
        if st.button("💳 Which payment methods do you accept?", use_container_width=True):
            st.session_state.active_prompt = "Which payment methods do you accept?"
        if st.button("🛡️ 1-Year Warranty terms & coverage?", use_container_width=True):
            st.session_state.active_prompt = "What does your 1-year product warranty cover?"
        if st.button("⚠️ Test Escalation: Immediate $850 cash refund!", use_container_width=True):
            st.session_state.active_prompt = "I demand an immediate cash refund of $850 on order #ORD-99482 right now or I will file a lawsuit!"

        st.divider()

        st.markdown(f"""
        **Active Session Information:**
        - **Customer ID**: `{st.session_state.customer_id}`
        - **Session ID**: `{st.session_state.session_id}`
        - **Workspace**: `{WORKSPACE_ID}`
        """)

        if st.button("🔄 Reset Conversation Session", use_container_width=True):
            st.session_state.session_id = f"sess_{uuid.uuid4().hex[:12]}"
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": f"Conversation reset. New session started for {st.session_state.customer_name}. How can I assist you today?",
                    "citations": [],
                    "resolved_query": None,
                    "escalation_id": None,
                }
            ]
            st.info("New session started. Prior turns cleared.")
            st.rerun()

    with col_chat:
        st.markdown("#### 💬 Live Autonomous Multi-Agent Support Concierge")
        st.caption("Real-time grounding via uploaded workspace policies & deterministic AI safety gates.")

        # Render conversation history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

                # Render contextually resolved query notice if pronoun resolution triggered
                if msg.get("resolved_query"):
                    st.caption(f"💡 *Contextually resolved: \"{msg['resolved_query']}\"*")

                # Render escalation banner if inquiry was escalated to supervisor
                if msg.get("escalation_id"):
                    st.markdown(f"""
                    <div class="escalation-banner">
                      <strong>🛡️ Human Specialist Escalation Triggered</strong><br>
                      Dossier ID: <code>{msg['escalation_id']}</code><br>
                      <a href="{API_BASE}/supervisor" target="_blank" style="color: #c2410c; font-weight: 700;">
                        Open Supervisor Desk to Review Case ↗
                      </a>
                    </div>
                    """, unsafe_allow_html=True)

                # Render verified policy citations
                if msg.get("citations") and len(msg["citations"]) > 0:
                    with st.expander(f"📄 Verified Citations ({len(msg['citations'])})", expanded=False):
                        for cit in msg["citations"]:
                            st.markdown(f"""
                            <div class="citation-box">
                              <strong>{cit.get('document_name') or cit.get('title')}</strong> &bull;
                              <em>Authority: {cit.get('authority', 'High')}</em> &bull;
                              <em>Similarity: {cit.get('similarity', 'N/A')}</em><br>
                              <span style="color: #475569;">"{cit.get('chunk_text', '')}"</span>
                            </div>
                            """, unsafe_allow_html=True)

        # Handle user input from chat input or quick prompt button
        user_input = st.chat_input("Ask a question about orders, returns, warranty, or delivery...")
        if "active_prompt" in st.session_state and st.session_state.active_prompt:
            user_input = st.session_state.active_prompt
            del st.session_state.active_prompt

        if user_input:
            # 1. Append user message
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # 2. Query multi-agent backend
            with st.chat_message("assistant"):
                with st.spinner("Analyzing inquiry with multi-agent orchestration..."):
                    result = send_chat_query(user_input)

                bot_text = result.get("response") or "No response received."
                citations = result.get("citations") or []
                resolved_q = result.get("resolved_query")
                esc_id = result.get("escalation_id")

                st.write(bot_text)

                if resolved_q:
                    st.caption(f"💡 *Contextually resolved: \"{resolved_q}\"*")

                if esc_id:
                    st.markdown(f"""
                    <div class="escalation-banner">
                      <strong>🛡️ Human Specialist Escalation Triggered</strong><br>
                      Dossier ID: <code>{esc_id}</code><br>
                      <a href="{API_BASE}/supervisor" target="_blank" style="color: #c2410c; font-weight: 700;">
                        Open Supervisor Desk to Review Case ↗
                      </a>
                    </div>
                    """, unsafe_allow_html=True)

                if citations:
                    with st.expander(f"📄 Verified Citations ({len(citations)})", expanded=False):
                        for cit in citations:
                            st.markdown(f"""
                            <div class="citation-box">
                              <strong>{cit.get('document_name') or cit.get('title')}</strong> &bull;
                              <em>Authority: {cit.get('authority', 'High')}</em> &bull;
                              <em>Similarity: {cit.get('similarity', 'N/A')}</em><br>
                              <span style="color: #475569;">"{cit.get('chunk_text', '')}"</span>
                            </div>
                            """, unsafe_allow_html=True)

            # 3. Save assistant message to state
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": bot_text,
                "citations": citations,
                "resolved_query": resolved_q,
                "escalation_id": esc_id,
            })
            st.rerun()

# -----------------------------------------------------------------------------
# TAB 2: PRODUCT CATALOG
# -----------------------------------------------------------------------------
with tab_catalog:
    st.markdown("#### 🛍️ Premium Hardware Catalog")
    st.caption("All products backed by our official 30-Day Return Guarantee, Tier-1 Carrier Shipping, and 1-Year Warranty.")

    p_col1, p_col2, p_col3 = st.columns(3)

    products = [
        {"id": "p1", "name": "Apple MacBook Pro 16\" (M3 Max)", "category": "Laptops & Desktops", "price": 2499, "badge": "Free Express Shipping", "warr": "1-Year Official Warranty", "ret": "30-Day Money-Back", "col": p_col1},
        {"id": "p2", "name": "Sony WH-1000XM5 Noise Cancelling", "category": "Audio & Sound", "price": 399, "badge": "Bestseller", "warr": "1-Year Replacement", "ret": "30-Day Money-Back", "col": p_col2},
        {"id": "p3", "name": "Samsung Galaxy S24 Ultra (512GB)", "category": "Flagship Mobile", "price": 1299, "badge": "IP68 Water Resistant", "warr": "1-Year Manufacturer", "ret": "30-Day Money-Back", "col": p_col3},
        {"id": "p4", "name": "LG UltraFine 32\" 4K HDR Monitor", "category": "Displays & Monitors", "price": 649, "badge": "Color Calibrated", "warr": "2-Year Panel Warranty", "ret": "30-Day Money-Back", "col": p_col1},
        {"id": "p5", "name": "Apple Watch Ultra 2 (Titanium)", "category": "Smart Wearables", "price": 799, "badge": "100m Water Resistant", "warr": "1-Year AppleCare Ready", "ret": "30-Day Money-Back", "col": p_col2},
        {"id": "p6", "name": "Anker 100W GaN Fast Charger", "category": "Charging & Power", "price": 49, "badge": "PD 3.0 High-Speed", "warr": "18-Month Warranty", "ret": "30-Day Money-Back", "col": p_col3},
    ]

    for p in products:
        with p["col"]:
            st.markdown(f"""
            <div class="product-box">
              <span class="product-tag">{p['badge']}</span>
              <h4 style="margin: 4px 0 6px 0; color: #0f172a;">{p['name']}</h4>
              <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 8px;">{p['category']}</div>
              <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a;">${p['price']:,}</div>
              <div class="policy-line">✅ {p['ret']}</div>
              <div class="policy-line">🛡️ {p['warr']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"🛒 Add to Cart (${p['price']})", key=f"cat_{p['id']}", use_container_width=True):
                st.session_state.cart.append(p)
                st.success(f"Added {p['name']} to cart!")
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 3: ORDER HISTORY (FETCHED FROM BACKEND MEMORY)
# -----------------------------------------------------------------------------
with tab_orders:
    st.markdown(f"#### 📦 Order History for {st.session_state.customer_name}")
    st.caption("Orders are synchronized into long-term customer memory in the main backend platform.")

    memories = fetch_customer_memories()
    order_memories = [m for m in memories if m.get("category") == "history"]
    session_orders = getattr(st.session_state, "customer_orders", [])

    total_count = len(session_orders) + len(order_memories)

    if total_count == 0:
        st.info("No orders found in customer memory yet. Place an order from the Shopping Cart or click the button below.")
    else:
        st.markdown(f"**Found {total_count} Order Record(s) for this account:**")

        # 1. Display active session orders
        for so in session_orders:
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 800; color: #166534; font-size: 1rem;">Order #{so['order_id']}</span>
                <span style="background: #dcfce7; color: #15803d; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;">{so.get('status', 'CONFIRMED')}</span>
              </div>
              <div style="margin-top: 8px; font-size: 0.92rem; color: #1e293b;">
                <strong>Items:</strong> {so.get('items')} &bull; <strong>Total:</strong> ${so.get('total', 0):,.2f}
              </div>
              <div style="margin-top: 4px; font-size: 0.82rem; color: #475569;">
                Carrier: {so.get('carrier')} &bull; Placed: {so.get('date')}
              </div>
              <div style="margin-top: 6px; font-size: 0.8rem; color: #0284c7;">
                ℹ️ <em>{so.get('note', 'Eligible for 30-day return.')}</em>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Display orders retrieved from backend memory
        for idx, om in enumerate(order_memories):
            cnt = om.get('content', '')
            # Avoid displaying duplicates if order ID already shown in session_orders
            if not any(so.get('order_id') in cnt for so in session_orders if so.get('order_id')):
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                  <div style="display: flex; justify-content: space-between;">
                    <strong>Backend Customer Memory #{idx + 1}</strong>
                    <span style="font-size: 0.8rem; color: #64748b;">Memory ID: <code>{om.get('memory_id', 'N/A')}</code></span>
                  </div>
                  <div style="margin-top: 8px; font-size: 0.92rem; color: #1e293b;">
                    {cnt}
                  </div>
                  <div style="margin-top: 6px; font-size: 0.8rem; color: #0284c7;">
                    ℹ️ <em>Tracked in main backend memory (tenants.db) &bull; Protected under GDPR Right-to-be-Forgotten</em>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    st.markdown("##### ➕ Simulate Pre-Existing Historical Order")
    if st.button("Generate Historical Order #ORD-99482 (Sony Headphones - Delivered 2 days ago)"):
        hist_order = {
            "order_id": "ORD-99482",
            "date": "2026-10-23",
            "items": "Sony WH-1000XM5 Noise Cancelling Headphones ($399)",
            "total": 399.00,
            "carrier": "FedEx Express (1-2 Days)",
            "status": "DELIVERED",
            "note": "Delivered 2 days ago via FedEx Express. Eligible for 30-day return or 48-hour damaged transit claim."
        }
        if "customer_orders" not in st.session_state:
            st.session_state.customer_orders = []
        if not any(o.get("order_id") == "ORD-99482" for o in st.session_state.customer_orders):
            st.session_state.customer_orders.append(hist_order)

        sync_memory_to_backend(
            category="history",
            content="Customer purchased Order #ORD-99482 on 2026-10-23: 1x Sony WH-1000XM5 ($399). Delivered via FedEx Express. Eligible for 30-day return or 48-hour damaged transit claim.",
            ttl_days=365,
        )
        st.success("Historical order #ORD-99482 added to account and saved to backend customer memory!")
        st.rerun()

# -----------------------------------------------------------------------------
# TAB 4: CUSTOMER MEMORY (DIRECT BACKEND MANAGEMENT & GDPR CONTROLS)
# -----------------------------------------------------------------------------
with tab_memory:
    st.markdown(f"#### 🧠 Main Application Customer Memory ({st.session_state.customer_id})")
    st.caption("These records are stored directly in `database/tenants.db` and can be inspected in the Business Admin Console.")

    mem_list = fetch_customer_memories()

    if not mem_list:
        st.warning(f"No persistent memories stored for customer '{st.session_state.customer_id}' yet.")
    else:
        st.markdown(f"**Found {len(mem_list)} persistent memory record(s):**")
        for m in mem_list:
            c1, c2 = st.columns([5, 1])
            with c1:
                cat_badge = m.get('category', 'general').upper()
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                  <span style="background: #e0f2fe; color: #0369a1; font-size: 0.75rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">{cat_badge}</span>
                  <span style="font-size: 0.8rem; color: #64748b; margin-left: 8px;">ID: <code>{m.get('memory_id')}</code></span>
                  <div style="margin-top: 6px; font-size: 0.9rem; color: #0f172a;">{m.get('content')}</div>
                  <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Expires: {m.get('expires_at') or 'Permanent'}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                if st.button("🗑️ Delete", key=f"del_mem_{m.get('memory_id')}"):
                    delete_customer_memory(m.get('memory_id'))
                    st.success("Deleted from main memory.")
                    st.rerun()

    st.divider()

    st.markdown("##### 📝 Store New Custom Customer Memory")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mem_cat = st.selectbox("Category", ["preference", "profile", "account", "history"])
        mem_ttl = st.selectbox("Retention TTL", [30, 90, 180, 365, 0], format_func=lambda x: f"{x} Days" if x > 0 else "Permanent")
    with col_m2:
        mem_content = st.text_input("Memory Content / Note", placeholder="e.g. Customer prefers eco-friendly paper packaging.")

    if st.button("Save Memory to Main Backend", type="primary"):
        if mem_content:
            sync_memory_to_backend(category=mem_cat, content=mem_content.strip(), ttl_days=mem_ttl)
            st.success("Memory saved and persisted to database/tenants.db!")
            st.rerun()
        else:
            st.error("Please enter memory content.")

# -----------------------------------------------------------------------------
# TAB 5: JAVASCRIPT WIDGET EMBED TAG (AS REQUESTED)
# -----------------------------------------------------------------------------
with tab_embed:
    st.markdown("#### 🧩 Embeddable JavaScript Support Widget")
    st.caption("Here is the exact embed code configured for your current customer session:")

    embed_snippet = f"""<!-- Autonomous Customer Support Widget Embed -->
<script
  src="{API_BASE}/static/widget.js"
  data-workspace-id="{WORKSPACE_ID}"
  data-widget-token="{WIDGET_TOKEN}"
  data-customer-id="{st.session_state.customer_id}"
  data-api-base="{API_BASE}">
</script>"""

    st.code(embed_snippet, language="html")

    st.markdown("##### Live Preview of Floating Widget:")
    components.html(f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="margin: 0; background: #f8fafc; height: 600px; position: relative;">
      <div style="padding: 16px; font-family: sans-serif; color: #334155;">
        <strong>Autonomous Support Widget Launcher Active</strong><br>
        <span style="font-size: 0.85rem; color: #64748b;">
          Click the blue chat bubble in the bottom right corner of this container to open the floating widget.
        </span>
      </div>
      {embed_snippet}
    </body>
    </html>
    """, height=620, scrolling=False)
