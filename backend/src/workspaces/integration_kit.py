"""
Integration Kit Generator for Reusable Domain-Adaptive Multi-Agent Customer Support
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Dynamically packages an in-memory ZIP integration kit containing:
1. widget.js (vanilla client embeddable chat widget)
2. embed_example.html (ready-to-run HTML integration preview)
3. INTEGRATION_GUIDE.md (step-by-step developer integration manual)
4. api_client_example.py (Python direct REST client example)
5. api_client_example.js (Node.js/JavaScript fetch client example)

Security Guarantees:
- Never includes tenant master API keys (cap_...).
- Never includes private business documents or other tenants' data.
- Scoped strictly to public widget tokens (wtk_...) for customer chat endpoints.
"""

import io
from pathlib import Path
from typing import Optional
import zipfile

ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT_DIR / "src" / "static"


def _get_widget_js_content() -> str:
    """Reads the authentic widget.js from the static directory."""
    widget_path = STATIC_DIR / "widget.js"
    if widget_path.exists():
        return widget_path.read_text(encoding="utf-8")
    # Fallback to direct path if running from alternative working directory
    alt_path = Path("src/static/widget.js")
    if alt_path.exists():
        return alt_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"widget.js not found at {widget_path} or {alt_path}")


def _build_embed_example_html(
    workspace_id: str,
    widget_token: str,
    workspace_name: str,
    domain: str,
    api_base: str,
    is_customized: bool,
) -> str:
    """Constructs a clean, ready-to-run embed_example.html."""
    title = f"{workspace_name} - Customer Support Widget Preview" if is_customized else "Autonomous Support Widget - Client Integration Example"
    heading = f"{workspace_name} Support Integration" if is_customized else "Autonomous Multi-Agent Support Widget"
    token_display = widget_token if is_customized else "YOUR_WIDGET_TOKEN (Obtain from /admin console)"
    ws_display = workspace_id if is_customized else "YOUR_WORKSPACE_ID"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --code-bg: #0f172a;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 40px 20px;
      line-height: 1.6;
    }}
    .container {{
      max-width: 820px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 36px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }}
    .badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 9999px;
      background: #eff6ff;
      color: var(--primary);
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 16px;
    }}
    h1 {{
      margin-top: 0;
      color: var(--text);
      font-size: 1.75rem;
      letter-spacing: -0.02em;
    }}
    .config-card {{
      background: #f1f5f9;
      border-left: 4px solid var(--primary);
      padding: 16px 20px;
      border-radius: 0 8px 8px 0;
      margin: 24px 0;
      font-size: 0.95rem;
    }}
    .config-item {{
      margin: 6px 0;
    }}
    .config-item strong {{
      color: #334155;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      background: #e2e8f0;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.88rem;
    }}
    pre {{
      background: var(--code-bg);
      color: #f8fafc;
      padding: 18px;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 0.88rem;
      line-height: 1.5;
    }}
    pre code {{
      background: transparent;
      padding: 0;
      color: inherit;
    }}
    .security-note {{
      background: #fefce8;
      border: 1px solid #fef08a;
      color: #713f12;
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 0.9rem;
      margin-top: 20px;
    }}
    .footer-note {{
      margin-top: 28px;
      color: var(--muted);
      font-size: 0.9rem;
      border-top: 1px solid var(--border);
      padding-top: 20px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <span class="badge">Integration Kit Preview</span>
    <h1>{heading}</h1>
    <p>This page demonstrates the <strong>Autonomous Multi-Agent Customer Support Widget</strong> embedded into an external web application or storefront.</p>

    <div class="config-card">
      <div class="config-item"><strong>Target Domain:</strong> <code>{domain}</code></div>
      <div class="config-item"><strong>Workspace ID:</strong> <code>{ws_display}</code></div>
      <div class="config-item"><strong>Public Widget Token:</strong> <code>{token_display}</code></div>
      <div class="config-item"><strong>Backend API Base:</strong> <code>{api_base}</code></div>
    </div>

    <h2>Try It Out</h2>
    <p>Look at the floating <strong>💬 Support</strong> button in the bottom-right corner of this page. Click it to open the conversational interface, ask questions about your policies, and test citation verification.</p>

    <h2>How to Embed in Your Website</h2>
    <p>Add the following script tag just before the closing <code>&lt;/body&gt;</code> tag on any HTML page:</p>

    <pre><code>&lt;!-- Autonomous Customer Support Widget Embed --&gt;
&lt;script
  src="widget.js"
  data-workspace-id="{workspace_id}"
  data-widget-token="{widget_token}"
  data-api-base="{api_base}"&gt;
&lt;/script&gt;</code></pre>

    <div class="security-note">
      <strong>Security Guarantee:</strong> This embed uses a low-privilege public widget token (<code>wtk_...</code>) scoped strictly to customer queries on this workspace. The master business API key is never exposed to browser clients.
    </div>

    <div class="footer-note">
      For React, Vue, WordPress, Shopify, or server-to-server integrations, please refer to <code>INTEGRATION_GUIDE.md</code>.
    </div>
  </div>

  <!-- Embed Widget Client -->
  <script
    src="widget.js"
    data-workspace-id="{workspace_id}"
    data-widget-token="{widget_token}"
    data-api-base="{api_base}">
  </script>
</body>
</html>
"""


def _build_integration_guide_md(
    workspace_id: str,
    widget_token: str,
    workspace_name: str,
    domain: str,
    api_base: str,
    is_customized: bool,
) -> str:
    """Constructs the comprehensive INTEGRATION_GUIDE.md."""
    token_str = widget_token if is_customized else "YOUR_WIDGET_TOKEN"
    ws_str = workspace_id if is_customized else "YOUR_WORKSPACE_ID"

    return f"""# Autonomous Multi-Agent Customer Support — Framework Integration Guide

Welcome to the **Autonomous Multi-Agent Customer Support System** client integration kit.

This guide provides instructions for embedding the customer support chat widget into your website or connecting external services directly via our REST API.

---

## 1. Architecture & Security Invariants

The platform implements **Risk-Aware Adaptive Multi-Agent Orchestration** with strict tenant isolation:

```
+-------------------------------------------------------------------------+
| Customer Browser (HTML / React / Vue)                                  |
|   └── widget.js [Public Token: wtk_...]                                |
+-------------------------------------------------------------------------+
                                    │
                  HTTPS POST /api/v1/workspaces/{ws_str}/widget/chat
                                    ▼
+-------------------------------------------------------------------------+
| Fast-Path Model Cascade (M1-M6) & Adaptive Autonomy Engine              |
|   ├── ChromaDB Workspace Vector Store (Cosine Similarity)              |
|   ├── Evidence Verification & Entailment Gate                           |
|   └── Human Supervisor Escalation Queue (HITL)                          |
+-------------------------------------------------------------------------+
```

### Security Invariants:
1. **Public Widget Token (`wtk_...`)**:
   - Scoped strictly to customer chat queries for your designated workspace.
   - Safe to embed in public HTML and frontend bundles.
   - Cannot access administrative functions (e.g., document ingestion, tenant creation, escalation reviews).
2. **Tenant Master API Key (`cap_...`)**:
   - Holds administrative authority over your workspaces.
   - **NEVER** expose your master API key in client-side code, git repositories, or browser environments.

---

## 2. Quick Start: HTML Embed

Add `widget.js` to your website's static assets directory, then paste this snippet before `</body>`:

```html
<!-- Autonomous Support Widget Embed -->
<script
  src="widget.js"
  data-workspace-id="{ws_str}"
  data-widget-token="{token_str}"
  data-api-base="{api_base}">
</script>
```

### Programmatic Initialization
If you prefer to initialize the widget programmatically (e.g., after user login):

```html
<script src="widget.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {{
    window.AutonomousSupportWidget.init({{
      workspaceId: "{ws_str}",
      widgetToken: "{token_str}",
      apiBase: "{api_base}",
      customerId: "cust_12345", // Optional customer identifier
      title: "{workspace_name} Support",
      primaryColor: "#2563eb"
    }});
  }});
</script>
```

---

## 3. Single-Page Application (SPA) Integration

### React / Next.js
Create a reusable React component (e.g., `SupportWidget.tsx`):

```tsx
import React, {{ useEffect }} from 'react';

declare global {{
  interface Window {{
    AutonomousSupportWidget?: {{
      init: (config: Record<string, any>) => void;
    }};
  }}
}}

export const SupportWidget: React.FC = () => {{
  useEffect(() => {{
    const script = document.createElement('script');
    script.src = '/widget.js';
    script.async = true;
    script.onload = () => {{
      if (window.AutonomousSupportWidget) {{
        window.AutonomousSupportWidget.init({{
          workspaceId: '{ws_str}',
          widgetToken: '{token_str}',
          apiBase: '{api_base}',
          title: '{workspace_name} Support'
        }});
      }}
    }};
    document.body.appendChild(script);

    return () => {{
      document.body.removeChild(script);
    }};
  }}, []);

  return null;
}};
```

### Vue.js (Vue 3 / Nuxt)
```vue
<template>
  <div></div>
</template>

<script setup>
import {{ onMounted }} from 'vue';

onMounted(() => {{
  const script = document.createElement('script');
  script.src = '/widget.js';
  script.onload = () => {{
    window.AutonomousSupportWidget?.init({{
      workspaceId: '{ws_str}',
      widgetToken: '{token_str}',
      apiBase: '{api_base}',
      title: '{workspace_name} Support'
    }});
  }};
  document.body.appendChild(script);
}});
</script>
```

---

## 4. Direct REST API Integration (Backend / Mobile Apps)

If you are developing a mobile app or custom backend integration, you can communicate directly with the chat API.

### Endpoint:
`POST {api_base}/api/v1/workspaces/{ws_str}/widget/chat`

### Request Headers:
```http
Content-Type: application/json
X-Widget-Token: {token_str}
```

### Request Body:
```json
{{
  "query": "What is the return policy for damaged items?",
  "session_id": "sess_user_abc123",
  "customer_id": "optional_customer_id"
}}
```

### Sample Response:
```json
{{
  "workspace_id": "{ws_str}",
  "session_id": "sess_user_abc123",
  "query": "What is the return policy for damaged items?",
  "resolution": "Damaged or defective items must be reported within 48 hours of delivery for a full refund or replacement.",
  "action_status": "COMPLETED",
  "verification_status": "SUPPORTED",
  "citations": [
    {{
      "document_id": "doc_return_policy",
      "source_document": "store_return_policy.txt",
      "citation_text": "Damaged or defective items must be reported within 48 hours of delivery.",
      "similarity_score": 0.8842
    }}
  ],
  "escalation_required": false,
  "confidence": 0.94
}}
```

---

## 5. Token Management & Rotation

If you suspect your public widget token has been misused or rate-limited:
1. Log into your Admin Console (`/admin`).
2. Navigate to **Widget Credentials** for workspace `{ws_str}`.
3. Click **Rotate Token**. A new `wtk_...` token will be generated, immediately invalidating the previous one.
4. Update the `data-widget-token` attribute in your website HTML.

---

## 6. Included Files in this Integration Kit

| File | Purpose |
| :--- | :--- |
| `widget.js` | Production chat widget bundle (dependency-free vanilla JavaScript). |
| `embed_example.html` | Ready-to-run preview page demonstrating widget integration. |
| `INTEGRATION_GUIDE.md` | This technical guide. |
| `api_client_example.py` | Python REST client example using standard library `urllib`. |
| `api_client_example.js` | Node.js / JavaScript REST client example using native `fetch`. |
"""


def _build_api_client_example_py(
    workspace_id: str,
    widget_token: str,
    api_base: str,
) -> str:
    """Constructs the Python API client example script."""
    return f'''"""
Autonomous Multi-Agent Customer Support — Python API Client Example
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Demonstrates communicating directly with the workspace customer support endpoint
using standard Python libraries (compatible with Python 3.8+).
"""

import json
import sys
import urllib.error
import urllib.request

# Active Workspace Configuration
API_BASE = "{api_base}"
WORKSPACE_ID = "{workspace_id}"
WIDGET_TOKEN = "{widget_token}"


def send_support_query(query: str, session_id: str = "sess_demo_py", customer_id: str = None):
    """
    Sends a customer support query to the workspace-scoped widget endpoint.
    """
    url = f"{{API_BASE}}/api/v1/workspaces/{{WORKSPACE_ID}}/widget/chat"
    headers = {{
        "Content-Type": "application/json",
        "X-Widget-Token": WIDGET_TOKEN,
    }}
    payload = {{
        "query": query,
        "session_id": session_id,
    }}
    if customer_id:
        payload["customer_id"] = customer_id

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    print(f"\\n==================================================")
    print(f"Sending Query: '{{query}}'")
    print(f"Workspace: {{WORKSPACE_ID}} | Session: {{session_id}}")
    print(f"==================================================")

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Status:       {{data.get('action_status')}}")
            print(f"Resolution:   {{data.get('resolution')}}")
            print(f"Verification: {{data.get('verification_status')}}")
            print(f"Escalated:    {{data.get('escalation_required')}}")

            citations = data.get("citations", [])
            if citations:
                print(f"\\nVerified Sources ({{len(citations)}}):")
                for idx, c in enumerate(citations, start=1):
                    doc = c.get("source_document", "Unknown")
                    score = c.get("similarity_score", 0.0)
                    snippet = c.get("citation_text", "").strip()[:100]
                    print(f"  {{idx}}. [{{doc}}] (score: {{score:.3f}}) \\"{{snippet}}...\\"")
            return data
    except urllib.error.HTTPError as exc:
        err_msg = exc.read().decode("utf-8")
        print(f"HTTP Error {{exc.code}}: {{err_msg}}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"Connection failed: {{exc}}", file=sys.stderr)
        return None


if __name__ == "__main__":
    # Test query
    sample_query = "What is the return policy?"
    send_support_query(sample_query)
'''


def _build_api_client_example_js(
    workspace_id: str,
    widget_token: str,
    api_base: str,
) -> str:
    """Constructs the JavaScript / Node.js API client example script."""
    return f'''/**
 * Autonomous Multi-Agent Customer Support — JavaScript API Client Example
 * Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
 * 
 * Demonstrates querying the customer support widget endpoint via native fetch
 * (compatible with Node.js 18+ and modern browsers).
 */

const API_BASE = "{api_base}";
const WORKSPACE_ID = "{workspace_id}";
const WIDGET_TOKEN = "{widget_token}";

async function sendSupportQuery(query, sessionId = "sess_demo_js", customerId = null) {{
  const url = `${{API_BASE}}/api/v1/workspaces/${{WORKSPACE_ID}}/widget/chat`;
  const payload = {{
    query: query,
    session_id: sessionId,
    ...(customerId ? {{ customer_id: customerId }} : {{}}),
  }};

  console.log("\\n==================================================");
  console.log(`Sending Query: "${{query}}"`);
  console.log(`Workspace: ${{WORKSPACE_ID}} | Session: ${{sessionId}}`);
  console.log("==================================================");

  try {{
    const response = await fetch(url, {{
      method: "POST",
      headers: {{
        "Content-Type": "application/json",
        "X-Widget-Token": WIDGET_TOKEN,
      }},
      body: JSON.stringify(payload),
    }});

    if (!response.ok) {{
      const errorText = await response.text();
      console.error(`HTTP Error ${{response.status}}:`, errorText);
      return null;
    }}

    const data = await response.json();
    console.log("Status:      ", data.action_status);
    console.log("Resolution:  ", data.resolution);
    console.log("Verification:", data.verification_status);
    console.log("Escalated:   ", data.escalation_required);

    if (data.citations && data.citations.length > 0) {{
      console.log(`\\nVerified Sources (${{data.citations.length}}):`);
      data.citations.forEach((c, idx) => {{
        const snippet = (c.citation_text || "").trim().substring(0, 100);
        console.log(`  ${{idx + 1}}. [${{c.source_document}}] (score: ${{Number(c.similarity_score || 0).toFixed(3)}}) "${{snippet}}..."`);
      }});
    }}
    return data;
  }} catch (err) {{
    console.error("Network request failed:", err.message);
    return null;
  }}
}}

// Run example
sendSupportQuery("What is the return policy?");
'''


def generate_integration_kit_zip(
    workspace_id: Optional[str] = None,
    widget_token: Optional[str] = None,
    workspace_name: Optional[str] = None,
    domain: Optional[str] = None,
    api_base: str = "http://localhost:8000",
) -> bytes:
    """
    Generates an in-memory ZIP package containing all 5 required integration assets.
    
    If workspace_id and widget_token are provided, the assets are customized for that workspace.
    Otherwise, generic placeholder configuration is used without leaking private credentials.
    """
    is_customized = bool(workspace_id and widget_token)
    ws_id = workspace_id or "YOUR_WORKSPACE_ID"
    token = widget_token or "YOUR_WIDGET_TOKEN"
    name = workspace_name or "Autonomous Customer Support"
    dom = domain or "Configurable (E-commerce / Education / Small Business)"

    widget_js = _get_widget_js_content()
    embed_html = _build_embed_example_html(
        workspace_id=ws_id,
        widget_token=token,
        workspace_name=name,
        domain=dom,
        api_base=api_base,
        is_customized=is_customized,
    )
    guide_md = _build_integration_guide_md(
        workspace_id=ws_id,
        widget_token=token,
        workspace_name=name,
        domain=dom,
        api_base=api_base,
        is_customized=is_customized,
    )
    client_py = _build_api_client_example_py(
        workspace_id=ws_id,
        widget_token=token,
        api_base=api_base,
    )
    client_js = _build_api_client_example_js(
        workspace_id=ws_id,
        widget_token=token,
        api_base=api_base,
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("widget.js", widget_js)
        zf.writestr("embed_example.html", embed_html)
        zf.writestr("INTEGRATION_GUIDE.md", guide_md)
        zf.writestr("api_client_example.py", client_py)
        zf.writestr("api_client_example.js", client_js)

    buffer.seek(0)
    return buffer.getvalue()
