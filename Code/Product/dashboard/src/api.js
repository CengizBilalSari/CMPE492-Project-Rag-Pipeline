const PIPELINER_BASE = "";
const EVALUATOR_BASE = "/eval";

function getUserId() {
  let uid = localStorage.getItem("graphrag_user_id");
  if (!uid) {
    uid = crypto.randomUUID();
    localStorage.setItem("graphrag_user_id", uid);
  }
  return uid;
}

function headers(extra = {}) {
  return { "X-User-Id": getUserId(), ...extra };
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
        user_id: getUserId(),
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

export async function startEvaluation(searchTypes, questionSource, file) {
  const form = new FormData();
  form.append("search_types", searchTypes.join(","));
  form.append("question_source", questionSource);
  if (file) form.append("file", file);

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

// ── History (direct Supabase reads via backend proxy would be ideal,
//    but for now we expose simple GET endpoints) ─────

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

export { getUserId };
