const PIPELINER_BASE = "";
const EVALUATOR_BASE = "/eval";

const CHAT_ID_KEY   = "graphrag_chat_id";
const CHAT_NAME_KEY = "graphrag_chat_name";
const USERNAME_KEY  = "graphrag_username";


// ── Auth ─────────────────────────────────────────────────────

/** Step 2a: list all chat bases. */
export async function listChats() {
  const res = await fetch(`${PIPELINER_BASE}/api/auth/chats`);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return await res.json();
}

/** Step 2b: create a new chat base. */
export async function createChat(name, embeddingModel = "all-MiniLM-L6-v2") {
  const res = await fetch(`${PIPELINER_BASE}/api/auth/chats`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, embedding_model: embeddingModel }),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  const data = await res.json();
  selectChat(data);
  return data;
}

/** Select a chat, save to local storage. */
export function selectChat(chat) {
  localStorage.setItem(CHAT_ID_KEY, chat.chat_id);
  localStorage.setItem(CHAT_NAME_KEY, chat.name);
  if (chat.embedding_model) {
    localStorage.setItem("graphrag_embedding_model", chat.embedding_model);
  }
}

/** Get the active chat. */
export function getChatId() {
  return localStorage.getItem(CHAT_ID_KEY);
}

export function getChatName() {
  return localStorage.getItem(CHAT_NAME_KEY);
}

/** Clear active chat. */
export function clearChat() {
  localStorage.removeItem(CHAT_ID_KEY);
  localStorage.removeItem(CHAT_NAME_KEY);
  localStorage.removeItem("graphrag_embedding_model");
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

/** Fetch available pipeline configuration options (embedding models, LLM models, etc.) */
export async function getPipelineConfig() {
  const res = await fetch(`${PIPELINER_BASE}/api/pipeline/config`);
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

// ── Evaluation ──────────────────────────────────────

/** Fetch available LLM providers and their models from the evaluator backend. */
export async function getEvaluateConfig() {
  const res = await fetch(`${EVALUATOR_BASE}/evaluate/config`);
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

export async function generateQuestions(provider, model, docId, numQuestions, onProgress) {
  const form = new FormData();
  form.append("llm_provider", provider || "openai");
  form.append("llm_model", model || "gpt-4o");
  form.append("doc_id", docId);
  form.append("num_questions", numQuestions);

  const res = await fetch(`${EVALUATOR_BASE}/evaluate/generate-questions`, {
    method: "POST",
    headers: headers(),
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === "progress" && onProgress) {
            onProgress(event.message);
          } else if (event.type === "complete") {
            result = event;
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
        } catch (e) {
          if (e.message && !e.message.includes("JSON")) throw e;
        }
      }
    }
  }

  if (!result) throw new Error("Stream ended without completion");
  return result;
}

export async function startEvaluation(searchTypes, questionSource, file, provider, model, docId, qaPairs) {
  const form = new FormData();
  form.append("search_types", searchTypes.join(","));
  form.append("question_source", questionSource);
  form.append("llm_provider", provider || "openai");
  form.append("llm_model", model || "gpt-4o");
  if (file) form.append("file", file);
  if (docId) form.append("doc_id", docId);
  if (qaPairs) form.append("qa_pairs", JSON.stringify(qaPairs));

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

export async function getEvalDetails(jobId) {
  const res = await fetch(`${EVALUATOR_BASE}/evaluate/details/${jobId}`);
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

// ── Ollama Models ───────────────────────────────────────

export async function getOllamaRecommendations() {
  const res = await fetch(`${PIPELINER_BASE}/api/ollama/recommendations`);
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

export async function pullOllamaModel(model, onProgress) {
  const res = await fetch(`${PIPELINER_BASE}/api/ollama/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        if (onProgress) onProgress(event);
      } catch (e) {
        if (e.message && e.message !== "Unexpected end of JSON input") throw e;
      }
    }
  }
}

