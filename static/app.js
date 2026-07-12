(() => {
  "use strict";

  const STAGE_LABELS = {
    rate_limit: "Rate",
    length: "Length",
    injection_pattern: "Pattern",
    topic_relevance: "Topic",
  };
  const STAGE_ORDER = ["rate_limit", "length", "injection_pattern", "topic_relevance"];

  const el = {
    thread: document.getElementById("chat-thread"),
    emptyState: document.getElementById("empty-state"),
    composer: document.getElementById("composer"),
    input: document.getElementById("message-input"),
    sendBtn: document.getElementById("send-btn"),
    sessionList: document.getElementById("session-list"),
    fileInput: document.getElementById("file-input"),
    uploadDrop: document.getElementById("upload-drop"),
    uploadStatus: document.getElementById("upload-status"),
    docName: document.getElementById("doc-name"),
    docSummary: document.getElementById("doc-summary"),
    topbarDoc: document.getElementById("topbar-doc"),
    rail: document.getElementById("rail"),
    docpanel: document.getElementById("docpanel"),
    scrim: document.getElementById("scrim"),
    toggleSessions: document.getElementById("toggle-sessions"),
    toggleDoc: document.getElementById("toggle-doc"),
  };

  let state = {
    sessions: [],
    currentSessionId: null,
  };

  // ---------- API helpers ----------

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    return res.json();
  }

  // ---------- mobile drawers ----------

  function closeDrawers() {
    el.rail.classList.remove("open");
    el.docpanel.classList.remove("open");
    el.scrim.classList.remove("visible");
  }
  el.toggleSessions.addEventListener("click", () => {
    const willOpen = !el.rail.classList.contains("open");
    closeDrawers();
    if (willOpen) {
      el.rail.classList.add("open");
      el.scrim.classList.add("visible");
    }
  });
  el.toggleDoc.addEventListener("click", () => {
    const willOpen = !el.docpanel.classList.contains("open");
    closeDrawers();
    if (willOpen) {
      el.docpanel.classList.add("open");
      el.scrim.classList.add("visible");
    }
  });
  el.scrim.addEventListener("click", closeDrawers);

  // ---------- sessions ----------

  async function loadSessions() {
    state.sessions = await api("/api/sessions");
    renderSessionList();
  }

  function renderSessionList() {
    if (!state.sessions.length) {
      el.sessionList.innerHTML = '<p class="session-empty">No sessions yet.</p>';
      return;
    }
    el.sessionList.innerHTML = "";
    for (const s of state.sessions) {
      const item = document.createElement("div");
      item.className = "session-item" + (s.id === state.currentSessionId ? " active" : "");
      item.innerHTML = `
        <span class="session-item-name">${escapeHtml(s.pdf_name)}</span>
        <button class="session-del" aria-label="Delete session">
          <svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        </button>
      `;
      item.addEventListener("click", (e) => {
        if (e.target.closest(".session-del")) return;
        selectSession(s.id);
        closeDrawers();
      });
      item.querySelector(".session-del").addEventListener("click", async (e) => {
        e.stopPropagation();
        await api(`/api/sessions/${s.id}`, { method: "DELETE" });
        if (state.currentSessionId === s.id) resetToEmpty();
        await loadSessions();
      });
      el.sessionList.appendChild(item);
    }
  }

  async function selectSession(sessionId) {
    const session = await api(`/api/sessions/${sessionId}`);
    state.currentSessionId = sessionId;
    setDocInfo(session.pdf_name, session.summary);
    renderSessionList();

    el.thread.innerHTML = "";
    el.emptyState.remove?.();
    for (const msg of session.history) {
      appendMessage(msg.role, msg.content, null);
    }
    enableComposer(true);
    scrollToBottom();
  }

  function resetToEmpty() {
    state.currentSessionId = null;
    setDocInfo(null, null);
    el.thread.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.id = "empty-state";
    empty.innerHTML = `
      <span class="empty-eyebrow">Start here</span>
      <h1>Upload a PDF to begin</h1>
      <p>Margin reads the document, then answers only from what's in it. Every question you ask passes through four checks before it reaches the model — you'll see each one marked on your message.</p>
    `;
    el.thread.appendChild(empty);
    enableComposer(false);
  }

  function setDocInfo(name, summary) {
    if (!name) {
      el.docName.textContent = "No document loaded";
      el.docSummary.textContent = "Upload a PDF from the sidebar to see its summary here.";
      el.topbarDoc.textContent = "No document loaded";
      return;
    }
    el.docName.textContent = name;
    el.docSummary.textContent = summary || "Summarizing…";
    el.topbarDoc.textContent = name;
  }

  function enableComposer(enabled) {
    el.input.disabled = !enabled;
    el.sendBtn.disabled = !enabled || !el.input.value.trim();
    if (enabled) el.input.focus();
  }

  // ---------- upload ----------

  async function uploadFile(file) {
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      showUploadStatus("Only PDF files are supported.", true);
      return;
    }
    showUploadStatus(`Reading ${file.name}…`, false);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/sessions/upload", { method: "POST", body: formData });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || "Upload failed");
      }
      const data = await res.json();
      showUploadStatus(null);
      await loadSessions();
      await selectSession(data.session_id);
    } catch (err) {
      showUploadStatus(err.message || "Upload failed", true);
    }
  }

  function showUploadStatus(text, isError) {
    if (!text) {
      el.uploadStatus.hidden = true;
      return;
    }
    el.uploadStatus.hidden = false;
    el.uploadStatus.textContent = text;
    el.uploadStatus.classList.toggle("error", !!isError);
  }

  el.fileInput.addEventListener("change", (e) => uploadFile(e.target.files[0]));
  ["dragover", "dragenter"].forEach((evt) =>
    el.uploadDrop.addEventListener(evt, (e) => {
      e.preventDefault();
      el.uploadDrop.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    el.uploadDrop.addEventListener(evt, (e) => {
      e.preventDefault();
      el.uploadDrop.classList.remove("dragover");
    })
  );
  el.uploadDrop.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    uploadFile(file);
  });

  // ---------- messaging ----------

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function appendMessage(role, content, trace) {
    document.getElementById("empty-state")?.remove();

    const isBlocked = role === "assistant" && content.startsWith("Blocked:");
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}` + (isBlocked ? " blocked" : "");

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = content;
    wrap.appendChild(bubble);

    if (role === "user" && trace) {
      wrap.appendChild(renderTrace(trace));
    }

    el.thread.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function renderTrace(trace) {
    const row = document.createElement("div");
    row.className = "trace";
    const byStage = Object.fromEntries(trace.map((t) => [t.stage, t]));
    for (const stage of STAGE_ORDER) {
      const entry = byStage[stage] || { status: "skipped" };
      const pill = document.createElement("span");
      pill.className = `trace-stage ${entry.status}`;
      pill.title = entry.detail || "";
      pill.innerHTML = `<span class="dot"></span>${STAGE_LABELS[stage]}`;
      row.appendChild(pill);
    }
    return row;
  }

  function appendTyping() {
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    wrap.id = "typing-indicator";
    wrap.innerHTML = `<div class="bubble typing"><span></span><span></span><span></span></div>`;
    el.thread.appendChild(wrap);
    scrollToBottom();
  }
  function removeTyping() {
    document.getElementById("typing-indicator")?.remove();
  }

  function scrollToBottom() {
    el.thread.scrollTop = el.thread.scrollHeight;
  }

  async function sendMessage(text) {
    if (!state.currentSessionId || !text.trim()) return;

    el.input.value = "";
    el.input.style.height = "auto";
    enableComposer(true);
    el.sendBtn.disabled = true;

    const userMsgEl = appendMessage("user", text, null);
    appendTyping();

    try {
      const data = await api(`/api/sessions/${state.currentSessionId}/messages`, {
        method: "POST",
        body: JSON.stringify({ message: text }),
      });
      // Attach the trace to the message we already rendered.
      userMsgEl.appendChild(renderTrace(data.trace));
      removeTyping();
      appendMessage("assistant", data.answer, null);
    } catch (err) {
      removeTyping();
      appendMessage("assistant", `Blocked: ${err.message}`, null);
    }
  }

  el.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(el.input.value);
  });

  el.input.addEventListener("input", () => {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 160) + "px";
    el.sendBtn.disabled = el.input.disabled || !el.input.value.trim();
  });

  el.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      el.composer.requestSubmit();
    }
  });

  // ---------- init ----------

  (async function init() {
    enableComposer(false);
    try {
      await loadSessions();
    } catch (err) {
      showUploadStatus("Couldn't reach the server. Is app.py running?", true);
    }
  })();
})();
