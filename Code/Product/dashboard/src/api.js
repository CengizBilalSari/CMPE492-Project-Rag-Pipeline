const PIPELINER_BASE = "";
const EVALUATOR_BASE = "/eval";

const CHAT_ID_KEY = "graphrag_chat_id";
const USERNAME_KEY = "graphrag_username";

export function getChatId() {
  return localStorage.getItem(CHAT_ID_KEY);
}

export function getUsername() {
  return localStorage.getItem(USERNAME_KEY);
}

export async function login(username) {
  const res = await fetch(`${PIPELINER_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  const data = await res.json();
  localStorage.setItem(CHAT_ID_KEY, data.chat_id);
  localStorage.setItem(USERNAME_KEY, data.username);
  return data;
}

export function logout() {
  localStorage.removeItem(CHAT_ID_KEY);
  localStorage.removeItem(USERNAME_KEY);
}

function headers(extra = {}) {
  const chatId = getChatId();
  if (!chatId) throw new Error("Not logged in.");
  return { "X-Chat-Id": chatId, ...extra };
}

// ── Documents ───────────────────────────────────────

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${PIPELINER_BASE}/api/documents/upload`, {
    method: "POST",
    headers: headers(),
    body: form,
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

// ── Pipeline (WebSocket) ────────────────────────────

export function connectPipeline(payload, onMessage, onClose) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/pipeline/run`);

  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        chat_id: getChatId(),
        ...payload,
      })
    );
  };

  ws.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      onMessage(data);
    } catch {
      onMessage({ type: "status", message: evt.data });
    }
  };

  ws.onclose = () => onClose?.();
  ws.onerror = (err) => onMessage({ type: "error", message: "WebSocket error" });

  return ws;
}

// ── Evaluation ──────────────────────────────────────

export async function startEvaluation(searchTypes, questionSource, file, provider, model, docId) {
  const form = new FormData();
  form.append("search_types", searchTypes.join(","));
  form.append("question_source", questionSource);
  form.append("llm_provider", provider || "openai");
  form.append("llm_model", model || "gpt-4o");
  if (file) form.append("file", file);
  if (docId) form.append("doc_id", docId);

  const res = await fetch(`${EVALUATOR_BASE}/evaluate/start`, {
    method: "POST",
    headers: headers(),
    body: form,
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

export async function getEvalStatus(jobId) {
  const res = await fetch(`${EVALUATOR_BASE}/evaluate/status/${jobId}`);
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

export async function getEvalResults(jobId) {
  const res = await fetch(`${EVALUATOR_BASE}/evaluate/results/${jobId}`);
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

// ── History ─────────────────────────────────────────

export async function getDocuments() {
  const res = await fetch(`${PIPELINER_BASE}/api/history/documents`, {
    headers: headers(),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function getPipelineRuns() {
  const res = await fetch(`${PIPELINER_BASE}/api/history/pipeline-runs`, {
    headers: headers(),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function getEvaluationJobs() {
  const res = await fetch(`${PIPELINER_BASE}/api/history/evaluation-jobs`, {
    headers: headers(),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function getEvaluationJobDetails(jobId) {
  const res = await fetch(`${PIPELINER_BASE}/api/history/evaluation-jobs/${jobId}/details`, {
    headers: headers(),
  });
  if (!res.ok) return null;
  return res.json();
}
