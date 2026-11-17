/**
 * Embeddable Customer Support Chat Widget
 * Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
 * 
 * Secure, dependency-free vanilla JavaScript widget embeddable into any website.
 * Communicates with backend via scoped public widget tokens (wtk_...).
 * Enforces evidence grounding, authentic citations, and safe HITL escalation.
 */

(function () {
  'use strict';

  // Helper: HTML escaping for XSS prevention
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Helper: Generate or retrieve persistent session ID per workspace
  function getOrCreateSessionId(workspaceId) {
    const storageKey = 'ast_session_' + (workspaceId || 'default');
    let sessionId = null;
    try {
      sessionId = window.sessionStorage.getItem(storageKey);
    } catch (e) {
      // sessionStorage might be restricted
    }
    if (!sessionId) {
      sessionId = 'sess_' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
      try {
        window.sessionStorage.setItem(storageKey, sessionId);
      } catch (e) {}
    }
    return sessionId;
  }

  function resetSessionId(workspaceId) {
    const storageKey = 'ast_session_' + (workspaceId || 'default');
    const newSessionId = 'sess_' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
    try {
      window.sessionStorage.setItem(storageKey, newSessionId);
    } catch (e) {}
    return newSessionId;
  }

  class SupportChatWidget {
    constructor(config = {}) {
      this.config = Object.assign(
        {
          apiBase: '',
          workspaceId: '',
          widgetToken: '',
          customerId: null,
          title: 'Customer Support',
          primaryColor: '#2563eb',
        },
        config
      );

      // Normalize apiBase
      if (this.config.apiBase.endsWith('/')) {
        this.config.apiBase = this.config.apiBase.slice(0, -1);
      }

      this.isOpen = false;
      this.isLoading = false;
      this.sessionId = getOrCreateSessionId(this.config.workspaceId);
      this.messages = [];
      this.lastFailedQuery = null;

      this.initDOM();
      this.fetchWorkspaceConfig();
    }

    initDOM() {
      // Inject CSS
      const styleId = 'ast-widget-styles';
      if (!document.getElementById(styleId)) {
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
          .ast-widget-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 14px;
            color: #1e293b;
            box-sizing: border-box;
            z-index: 999999;
          }
          .ast-widget-container * {
            box-sizing: border-box;
          }
          .ast-launcher-btn {
            position: fixed;
            bottom: 24px;
            right: 24px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: ${this.config.primaryColor};
            box-shadow: 0 4px 18px rgba(37, 99, 235, 0.4);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s ease;
            z-index: 1000000;
          }
          .ast-launcher-btn:hover {
            transform: scale(1.08);
            box-shadow: 0 6px 24px rgba(37, 99, 235, 0.55);
          }
          .ast-launcher-btn:active {
            transform: scale(0.96);
          }

          /* Chat Window */
          .ast-chat-window {
            position: fixed;
            bottom: 96px;
            right: 24px;
            width: 390px;
            max-width: calc(100vw - 32px);
            height: 560px;
            max-height: calc(100vh - 120px);
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.22);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid #cbd5e1;
            transition: opacity 0.2s ease, transform 0.2s ease;
            z-index: 1000000;
          }
          .ast-chat-window.ast-hidden {
            display: none;
            opacity: 0;
            transform: translateY(16px);
          }

          /* Mobile Responsive Drawer */
          @media (max-width: 520px) {
            .ast-chat-window {
              width: 100vw !important;
              max-width: 100vw !important;
              height: 100vh !important;
              max-height: 100vh !important;
              bottom: 0 !important;
              right: 0 !important;
              border-radius: 0 !important;
              border: none !important;
            }
            .ast-launcher-btn {
              bottom: 16px;
              right: 16px;
              width: 54px;
              height: 54px;
            }
          }

          /* Header */
          .ast-chat-header {
            background: ${this.config.primaryColor};
            color: white;
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            user-select: none;
          }
          .ast-header-title {
            font-weight: 600;
            font-size: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
            flex: 1;
          }
          .ast-title-text {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          .ast-header-status {
            display: inline-block;
            width: 9px;
            height: 9px;
            background: #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 6px #22c55e;
            flex-shrink: 0;
          }
          .ast-header-actions {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
          }
          .ast-header-btn {
            background: rgba(255, 255, 255, 0.15);
            border: none;
            color: white;
            cursor: pointer;
            width: 30px;
            height: 30px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            transition: background 0.2s;
          }
          .ast-header-btn:hover {
            background: rgba(255, 255, 255, 0.3);
          }
          .ast-btn-close {
            font-size: 17px;
            font-weight: bold;
          }

          /* Messages Area */
          .ast-chat-messages {
            flex: 1;
            padding: 18px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 14px;
            background: #f8fafc;
            scroll-behavior: smooth;
          }
          .ast-message-row {
            display: flex;
            flex-direction: column;
            max-width: 88%;
          }
          .ast-message-user {
            align-self: flex-end;
          }
          .ast-message-bot {
            align-self: flex-start;
          }
          .ast-bubble {
            padding: 11px 15px;
            border-radius: 14px;
            font-size: 13.5px;
            line-height: 1.5;
            word-break: break-word;
          }
          .ast-bubble-user {
            background: ${this.config.primaryColor};
            color: white;
            border-bottom-right-radius: 4px;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
          }
          .ast-bubble-bot {
            background: #ffffff;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-bottom-left-radius: 4px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
          }

          /* Turn metadata indicator */
          .ast-resolved-query {
            font-size: 11px;
            color: #64748b;
            margin-top: 3px;
            font-style: italic;
          }

          /* Clarify Prompt Box */
          .ast-clarify-box {
            background: #fefce8;
            border-left: 3px solid #eab308;
            padding: 10px 14px;
            border-radius: 6px;
            margin-top: 8px;
            font-size: 13px;
            color: #854d0e;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
          }

          /* Escalation Box */
          .ast-escalate-box {
            background: #fef2f2;
            border-left: 3px solid #ef4444;
            padding: 12px 14px;
            border-radius: 6px;
            margin-top: 8px;
            font-size: 13px;
            color: #991b1b;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
          }
          .ast-escalate-badge {
            display: inline-block;
            font-weight: 700;
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: #fee2e2;
            color: #b91c1c;
            padding: 2px 8px;
            border-radius: 4px;
            margin-bottom: 6px;
            border: 1px solid #fca5a5;
          }

          /* Citations Accordion */
          .ast-citations-container {
            margin-top: 8px;
            font-size: 12px;
          }
          .ast-citations-toggle {
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            color: #334155;
            cursor: pointer;
            padding: 4px 10px;
            font-size: 11.5px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            border-radius: 12px;
            transition: all 0.2s;
          }
          .ast-citations-toggle:hover {
            background: #e2e8f0;
            color: #0f172a;
          }
          .ast-citations-list {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 12px;
            margin-top: 6px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.03);
          }
          .ast-citation-item {
            font-size: 11.5px;
            color: #475569;
            border-bottom: 1px solid #f1f5f9;
            padding-bottom: 6px;
          }
          .ast-citation-item:last-child {
            border-bottom: none;
            padding-bottom: 0;
          }
          .ast-citation-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 3px;
          }
          .ast-citation-source {
            font-weight: 600;
            color: #1e293b;
          }
          .ast-citation-chip {
            font-size: 10px;
            padding: 1px 6px;
            border-radius: 4px;
            background: #e0f2fe;
            color: #0369a1;
            font-weight: 600;
          }
          .ast-citation-quote {
            font-style: italic;
            color: #64748b;
            line-height: 1.4;
          }

          /* Typing Indicator */
          .ast-typing-indicator {
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 10px 16px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            width: fit-content;
          }
          .ast-dot {
            width: 7px;
            height: 7px;
            background: #94a3b8;
            border-radius: 50%;
            animation: ast-pulse 1.4s infinite ease-in-out both;
          }
          .ast-dot:nth-child(1) { animation-delay: -0.32s; }
          .ast-dot:nth-child(2) { animation-delay: -0.16s; }
          @keyframes ast-pulse {
            0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
            40% { transform: scale(1); opacity: 1; }
          }

          /* Input Area */
          .ast-chat-input-area {
            border-top: 1px solid #e2e8f0;
            padding: 12px 14px;
            background: #ffffff;
            display: flex;
            gap: 10px;
            align-items: center;
          }
          .ast-chat-input {
            flex: 1;
            padding: 10px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 22px;
            outline: none;
            font-size: 13.5px;
            transition: border-color 0.2s, box-shadow 0.2s;
          }
          .ast-chat-input:focus {
            border-color: ${this.config.primaryColor};
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
          }
          .ast-chat-send-btn {
            background: ${this.config.primaryColor};
            border: none;
            color: white;
            border-radius: 50%;
            width: 38px;
            height: 38px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s, transform 0.1s;
            flex-shrink: 0;
          }
          .ast-chat-send-btn:hover {
            filter: brightness(0.92);
          }
          .ast-chat-send-btn:active {
            transform: scale(0.94);
          }
          .ast-chat-send-btn:disabled {
            background: #cbd5e1;
            cursor: not-allowed;
          }

          /* Error Banner */
          .ast-error-banner {
            background: #fee2e2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 10px 14px;
            font-size: 12.5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 8px;
            margin-top: 6px;
            gap: 8px;
          }
          .ast-retry-btn {
            background: #ef4444;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 11.5px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
          }
          .ast-retry-btn:hover {
            background: #dc2626;
          }
        `;
        document.head.appendChild(style);
      }

      // Create Container
      const container = document.createElement('div');
      container.className = 'ast-widget-container';
      container.innerHTML = `
        <button class="ast-launcher-btn" aria-label="Open support chat" title="Chat with support">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        </button>

        <div class="ast-chat-window ast-hidden" role="dialog" aria-modal="true" aria-label="Support Chat">
          <div class="ast-chat-header">
            <div class="ast-header-title">
              <span class="ast-header-status" title="Active"></span>
              <span class="ast-title-text">${escapeHtml(this.config.title)}</span>
            </div>
            <div class="ast-header-actions">
              <button class="ast-header-btn ast-btn-reset" title="Start new conversation" aria-label="Start new conversation">🔄</button>
              <button class="ast-header-btn ast-btn-close" title="Close chat window" aria-label="Close chat window">&times;</button>
            </div>
          </div>

          <div class="ast-chat-messages" id="ast-messages">
            <div class="ast-message-row ast-message-bot">
              <div class="ast-bubble ast-bubble-bot">
                Hello! How can I assist you today?
              </div>
            </div>
          </div>

          <div class="ast-chat-input-area">
            <input type="text" class="ast-chat-input" placeholder="Type your inquiry here..." maxlength="1000" />
            <button class="ast-chat-send-btn" aria-label="Send message">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </div>
        </div>
      `;

      document.body.appendChild(container);

      this.containerEl = container;
      this.launcherBtn = container.querySelector('.ast-launcher-btn');
      this.chatWindow = container.querySelector('.ast-chat-window');
      this.closeBtn = container.querySelector('.ast-btn-close');
      this.resetBtn = container.querySelector('.ast-btn-reset');
      this.messagesEl = container.querySelector('#ast-messages');
      this.inputEl = container.querySelector('.ast-chat-input');
      this.sendBtn = container.querySelector('.ast-chat-send-btn');
      this.titleEl = container.querySelector('.ast-title-text');

      // Bind events
      this.launcherBtn.addEventListener('click', () => this.toggleWindow());
      this.closeBtn.addEventListener('click', () => this.closeWindow());
      this.resetBtn.addEventListener('click', () => this.resetConversation());
      this.sendBtn.addEventListener('click', () => this.handleSend());
      this.inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.handleSend();
        }
      });
    }

    async fetchWorkspaceConfig() {
      if (!this.config.workspaceId) return;
      try {
        const url = `${this.config.apiBase}/api/v1/workspaces/${encodeURIComponent(this.config.workspaceId)}/widget/config`;
        const headers = { 'Content-Type': 'application/json' };
        if (this.config.widgetToken) {
          headers['X-Widget-Token'] = this.config.widgetToken;
        }
        const res = await fetch(url, { headers });
        if (res.ok) {
          const data = await res.json();
          if (data.name && this.titleEl) {
            this.titleEl.textContent = `${data.name} Support`;
          }
        }
      } catch (err) {
        // Silently preserve default title on config failure
      }
    }

    toggleWindow() {
      if (this.isOpen) {
        this.closeWindow();
      } else {
        this.openWindow();
      }
    }

    openWindow() {
      this.isOpen = true;
      this.chatWindow.classList.remove('ast-hidden');
      this.inputEl.focus();
      this.scrollToBottom();
    }

    closeWindow() {
      this.isOpen = false;
      this.chatWindow.classList.add('ast-hidden');
    }

    resetConversation() {
      this.sessionId = resetSessionId(this.config.workspaceId);
      this.messages = [];
      this.messagesEl.innerHTML = `
        <div class="ast-message-row ast-message-bot">
          <div class="ast-bubble ast-bubble-bot">
            Conversation reset. New session started. How can I assist you today?
          </div>
        </div>
      `;
      this.inputEl.focus();
      this.scrollToBottom();
    }

    scrollToBottom() {
      requestAnimationFrame(() => {
        if (this.messagesEl) {
          this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
        }
      });
    }

    showTyping() {
      this.removeTyping();
      const typingEl = document.createElement('div');
      typingEl.id = 'ast-typing-node';
      typingEl.className = 'ast-message-row ast-message-bot';
      typingEl.innerHTML = `
        <div class="ast-typing-indicator">
          <div class="ast-dot"></div>
          <div class="ast-dot"></div>
          <div class="ast-dot"></div>
        </div>
      `;
      this.messagesEl.appendChild(typingEl);
      this.scrollToBottom();
    }

    removeTyping() {
      const el = document.getElementById('ast-typing-node');
      if (el) el.remove();
    }

    appendUserMessage(text) {
      const row = document.createElement('div');
      row.className = 'ast-message-row ast-message-user';
      row.innerHTML = `<div class="ast-bubble ast-bubble-user">${escapeHtml(text)}</div>`;
      this.messagesEl.appendChild(row);
      this.scrollToBottom();
    }

    appendBotResponse(data) {
      const row = document.createElement('div');
      row.className = 'ast-message-row ast-message-bot';

      let innerHtml = `<div class="ast-bubble ast-bubble-bot">${escapeHtml(data.response || 'No response returned.')}</div>`;

      // Multi-Turn Resolved Reference Notice
      if (data.resolved_query && data.resolved_query.trim() !== (data.query || '').trim()) {
        innerHtml += `<div class="ast-resolved-query">Contextually resolved: "${escapeHtml(data.resolved_query)}"</div>`;
      }

      // Escalation Banner
      if (data.escalation_required) {
        const escId = data.escalation_id ? ` (Dossier: ${escapeHtml(data.escalation_id)})` : '';
        innerHtml += `
          <div class="ast-escalate-box">
            <span class="ast-escalate-badge">🛡️ Human Specialist Escalation</span>
            <div>Your inquiry requires human specialist verification and has been routed to our review queue${escId}. A support agent will assist you shortly.</div>
          </div>
        `;
      }

      // Clarification Prompt
      if (data.autonomy_tier === 'CLARIFY' && data.clarification_prompt && data.clarification_prompt !== data.response) {
        innerHtml += `
          <div class="ast-clarify-box">
            <strong>❓ Please clarify:</strong> ${escapeHtml(data.clarification_prompt)}
          </div>
        `;
      }

      // Verified Authentic Citations Accordion
      if (data.citations && data.citations.length > 0) {
        const count = data.citations.length;
        const listItems = data.citations
          .map((c) => {
            const cat = c.category_key ? `<span class="ast-citation-chip">${escapeHtml(c.category_key)}</span>` : '';
            const doc = c.document_name ? escapeHtml(c.document_name) : 'Official Policy';
            const snip = c.chunk_text ? `<div class="ast-citation-quote">"${escapeHtml(c.chunk_text)}"</div>` : '';
            return `
              <div class="ast-citation-item">
                <div class="ast-citation-header">
                  <span class="ast-citation-source">📄 ${doc}</span>
                  ${cat}
                </div>
                ${snip}
              </div>
            `;
          })
          .join('');

        const uniqueToggleId = 'cits_' + Math.random().toString(36).substring(2, 8);

        innerHtml += `
          <div class="ast-citations-container">
            <button class="ast-citations-toggle" onclick="(function(btn){
              var el = document.getElementById('${uniqueToggleId}');
              if(el.style.display==='none'){ el.style.display='flex'; btn.innerHTML='&#x25BC; Hide Sources (${count})'; }
              else { el.style.display='none'; btn.innerHTML='&#x25B6; Verified Sources (${count})'; }
            })(this)">&#x25B6; Verified Sources (${count})</button>
            <div id="${uniqueToggleId}" class="ast-citations-list" style="display:none;">
              ${listItems}
            </div>
          </div>
        `;
      }

      row.innerHTML = innerHtml;
      this.messagesEl.appendChild(row);
      this.scrollToBottom();
    }

    showErrorBanner(message, query) {
      this.lastFailedQuery = query;
      const banner = document.createElement('div');
      banner.className = 'ast-error-banner';
      banner.innerHTML = `
        <span>⚠️ ${escapeHtml(message)}</span>
        <button class="ast-retry-btn">Retry</button>
      `;
      banner.querySelector('.ast-retry-btn').addEventListener('click', () => {
        banner.remove();
        if (this.lastFailedQuery) {
          this.submitQuery(this.lastFailedQuery);
        }
      });
      this.messagesEl.appendChild(banner);
      this.scrollToBottom();
    }

    async handleSend() {
      if (this.isLoading) return;
      const text = this.inputEl.value.trim();
      if (!text) return;

      this.inputEl.value = '';
      this.appendUserMessage(text);
      await this.submitQuery(text);
    }

    async submitQuery(text) {
      if (!this.config.workspaceId || !this.config.widgetToken) {
        this.showErrorBanner('Configuration error: Missing workspace ID or widget token.', text);
        return;
      }

      this.isLoading = true;
      this.sendBtn.disabled = true;
      this.showTyping();

      const url = `${this.config.apiBase}/api/v1/workspaces/${encodeURIComponent(this.config.workspaceId)}/widget/chat`;
      const payload = {
        query: text,
        session_id: this.sessionId,
        customer_id: this.config.customerId,
      };

      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Widget-Token': this.config.widgetToken,
          },
          body: JSON.stringify(payload),
        });

        this.removeTyping();

        if (response.status === 429) {
          this.showErrorBanner('Rate limit exceeded. Please wait a moment before sending another message.', text);
          return;
        }

        if (response.status === 403) {
          this.showErrorBanner('Access denied. Public widget token is invalid or revoked.', text);
          return;
        }

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          const errMsg = errData.detail || `Server error (${response.status})`;
          this.showErrorBanner(errMsg, text);
          return;
        }

        const data = await response.json();
        this.appendBotResponse(data);
      } catch (err) {
        this.removeTyping();
        this.showErrorBanner('Network error: Unable to reach support servers. Check your connection.', text);
      } finally {
        this.isLoading = false;
        this.sendBtn.disabled = false;
        this.inputEl.focus();
      }
    }
  }

  // Export to global window object
  window.AutonomousSupportWidget = {
    init: function (config) {
      return new SupportChatWidget(config);
    },
  };

  // Auto-init if script tag has attributes
  if (typeof document !== 'undefined') {
    const currentScript =
      document.currentScript ||
      Array.from(document.getElementsByTagName('script')).find((s) => s.src && s.src.includes('widget.js'));

    if (currentScript) {
      const workspaceId = currentScript.getAttribute('data-workspace-id');
      const widgetToken = currentScript.getAttribute('data-widget-token');
      const apiBase = currentScript.getAttribute('data-api-base') || '';
      const customerId = currentScript.getAttribute('data-customer-id') || null;

      if (workspaceId && widgetToken) {
        if (document.readyState === 'loading') {
          document.addEventListener('DOMContentLoaded', () => {
            window.AutonomousSupportWidget.init({ workspaceId, widgetToken, apiBase, customerId });
          });
        } else {
          window.AutonomousSupportWidget.init({ workspaceId, widgetToken, apiBase, customerId });
        }
      }
    }
  }
})();
