import { useState, useEffect, useRef } from "react";
import { connectPipeline, getDocuments, getPipelineConfig } from "../api";

// Fallbacks used until the API response arrives
const DEFAULT_PROVIDERS = ["openai", "lmstudio"];
const DEFAULT_MODELS = {
  openai: ["gpt-4o", "gpt-4o-mini"],
  lmstudio: [
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "llama-3-22b-instruct-v0.1",
    "google/gemma-4-31b",
  ],
};
const DEFAULT_CHUNKERS = [
  "sentence", "token", "character",
  "recursive", "semantic", "propositional",
];
const DEFAULT_EMBEDDING_MODELS = [
  { name: "all-MiniLM-L6-v2", dimensions: 384, description: "Fast, lightweight, classic default" },
];

const PIPELINE_STEPS = [
  { id: "chunk",     label: "Chunking",              keywords: ["chunk"] },
  { id: "extract",   label: "Entity Extraction",      keywords: ["extract"] },
  { id: "resolve",   label: "Entity Resolution",      keywords: ["resolv"] },
  { id: "embed",     label: "Embedding",              keywords: ["embed"] },
  { id: "community", label: "Community Detection",    keywords: ["community"] },
  { id: "summarize", label: "Summarization",          keywords: ["summar"] },
];

// Known pipeline step keywords → classify log line appearance
function classifyLog(text) {
  const t = text.toLowerCase();
  if (t.includes("error") || t.includes("failed") || t.includes("exception")) return "log-error";
  if (t.includes("warn"))                  return "log-warn";
  if (t.includes("step") || t.includes("starting") || t.includes("running") || t.includes("phase")) return "log-step";
  if (t.includes("complete") || t.includes("done") || t.includes("success") || t.includes("finished")) return "log-ok";
  return "";
}

function logPrefix(cls) {
  if (cls === "log-error") return "✖";
  if (cls === "log-warn")  return "⚠";
  if (cls === "log-step")  return "◆";
  if (cls === "log-ok")    return "✔";
  return "›";
}

function now() {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}

export default function Pipeline() {
  const [docs, setDocs] = useState([]);
  const [docId, setDocId] = useState("");
  const [provider, setProvider] = useState("lmstudio");
  const [model, setModel] = useState(DEFAULT_MODELS.lmstudio[0]);
  const [chunker, setChunker] = useState("recursive");
  const [chunkSize, setChunkSize] = useState(512);
  const [overlap, setOverlap] = useState(50);
  const [embeddingModel, setEmbeddingModel] = useState(localStorage.getItem("graphrag_embedding_model") || "all-MiniLM-L6-v2");
  const [useLlmRes, setUseLlmRes] = useState(true);
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [completedSteps, setCompletedSteps] = useState(new Set());
  const wsRef = useRef(null);
  const logEndRef = useRef(null);

  // Dynamic config from backend
  const [llmProviders, setLlmProviders] = useState(DEFAULT_MODELS);
  const [embeddingModels, setEmbeddingModels] = useState(DEFAULT_EMBEDDING_MODELS);
  const [chunkers, setChunkers] = useState(DEFAULT_CHUNKERS);

  useEffect(() => {
    getDocuments().then((d) => {
      setDocs(d);
      if (d.length) setDocId(d[0].id);
    });
    getPipelineConfig()
      .then((cfg) => {
        if (cfg.llm_providers) setLlmProviders(cfg.llm_providers);
        if (cfg.embedding_models) setEmbeddingModels(cfg.embedding_models);
        if (cfg.chunking_strategies) setChunkers(cfg.chunking_strategies);
      })
      .catch(() => {}); // fall back to defaults
  }, []);

  const PROVIDERS = Object.keys(llmProviders);
  const MODELS = llmProviders;

  useEffect(() => {
    if (MODELS[provider]) setModel(MODELS[provider][0]);
  }, [provider, llmProviders]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  function pushLog(text, type = "") {
    const cls = type || classifyLog(text);
    setLogs((prev) => [...prev, { text, cls, time: now() }]);
  }

  function start() {
    if (!docId) return;
    setLogs([]);
    setDone(false);
    setRunning(true);
    setActiveStep(-1);
    setCompletedSteps(new Set());
    pushLog("Connecting to pipeline…", "log-dim");

    const payload = {
      document_id: docId,
      doc_title: docs.find((d) => d.id === docId)?.name || "untitled",
      doc_source: "upload",
      config: {
        llm: { provider, model },
        chunking: { strategy: chunker, chunk_size: chunkSize, overlap },
        embedding: { model: embeddingModel },
        entity_resolution: { use_llm: useLlmRes },
      },
    };

    wsRef.current = connectPipeline(
      payload,
      (msg) => {
        const text = msg.message || JSON.stringify(msg);
        const isError = msg.type === "error";
        const isComplete =
          msg.type === "complete" ||
          text.toLowerCase().includes("pipeline complete") ||
          text.toLowerCase().includes("finished");

        pushLog(text, isError ? "log-error" : "");

        // Update step progress
        const lower = text.toLowerCase();
        PIPELINE_STEPS.forEach((step, idx) => {
          if (step.keywords.some((kw) => lower.includes(kw))) {
            if (lower.includes("complete") || lower.includes("skipped")) {
              setCompletedSteps((prev) => new Set([...prev, idx]));
              setActiveStep((prev) => (prev <= idx ? idx + 1 : prev));
            } else {
              setActiveStep(idx);
            }
          }
        });

        if (isComplete || isError) {
          setRunning(false);
          setDone(!isError);
          if (!isError) setCompletedSteps(new Set(PIPELINE_STEPS.map((_, i) => i)));
        }
      },
      () => {
        setRunning(false);
        pushLog("Connection closed.", "log-dim");
      }
    );
  }

  function stop() {
    wsRef.current?.close();
    setRunning(false);
  }

  const selectedDoc = docs.find((d) => d.id === docId);

  return (
    <>
      <div className="page-header">
        <h2>Graph Generation</h2>
        <p>Configure and run the GraphRAG knowledge graph construction pipeline.</p>
      </div>

      {/* Configuration */}
      <div className="card">
        <div className="card-header">
          <div className="card-icon">⚙️</div>
          <div>
            <h3>Configuration</h3>
            <div className="card-subtitle">Set LLM, chunking strategy, and select a document</div>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Document</label>
            <select value={docId} onChange={(e) => setDocId(e.target.value)}>
              {docs.length === 0 && <option value="">No documents uploaded</option>}
              {docs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>LLM Provider</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              {PROVIDERS.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {(MODELS[provider] || []).map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Chunking Strategy</label>
            <select value={chunker} onChange={(e) => setChunker(e.target.value)}>
              {chunkers.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Embedding Model (Locked to Workspace)</label>
            <div style={{
              padding: "10px 12px",
              backgroundColor: "var(--bg-subtle)",
              borderRadius: "6px",
              border: "1px solid var(--border)",
              marginTop: "4px",
              display: "flex",
              alignItems: "center",
              gap: "8px"
            }}>
              <span style={{ fontSize: "16px" }}>🧬</span>
              <span className="code-text" style={{ fontSize: "14px", fontWeight: 500 }}>{embeddingModel}</span>
            </div>
            <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 8 }}>
              This chat base was created using the <strong>{embeddingModel}</strong> model. It cannot be changed to ensure vector space integrity.
            </div>
          </div>
        </div>

        <div className="form-row" style={{ marginBottom: 20 }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Chunk Size</label>
            <input
              type="number"
              value={chunkSize}
              min={100} max={4096}
              onChange={(e) => setChunkSize(Number(e.target.value))}
            />
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Overlap</label>
            <input
              type="number"
              value={overlap}
              min={0} max={500}
              onChange={(e) => setOverlap(Number(e.target.value))}
            />
          </div>
        </div>

        <div className="form-row" style={{ marginBottom: 20 }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Entity Resolution</label>
            <div className="checkbox-group mt-16" style={{ marginTop: 8 }}>
              <label>
                <input
                  type="checkbox"
                  checked={useLlmRes}
                  onChange={(e) => setUseLlmRes(e.target.checked)}
                />
                🤖 Use LLM Verification
              </label>
            </div>
            <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 8 }}>
              Unchecking this skips the LLM deduplication check (faster, but less accurate).
            </div>
          </div>
        </div>

        {/* Summary pill */}
        {selectedDoc && (
          <div className="stat-row" style={{ marginBottom: 16 }}>
            <div className="stat-pill">📄 <span>{selectedDoc.name}</span></div>
            <div className="stat-pill">🤖 <span>{provider} / {model}</span></div>
            <div className="stat-pill">🧬 <span>{embeddingModel}</span></div>
            <div className="stat-pill">✂️ <span>{chunker} · {chunkSize}t · {overlap}o</span></div>
          </div>
        )}

        <div className="flex gap-8">
          <button
            className="btn btn-primary"
            disabled={running || !docId}
            onClick={start}
          >
            {running ? (
              <>
                <span style={{ display: "inline-block", animation: "spin 1s linear infinite" }}>⚙</span>
                Running…
              </>
            ) : (
              <> ▶ Run Graph Generation</>
            )}
          </button>

          {running && (
            <button className="btn btn-outline" onClick={stop}>
              ⏹ Stop
            </button>
          )}

          {done && !running && (
            <span style={{ color: "var(--green)", fontSize: 13, fontWeight: 600, alignSelf: "center" }}>
              ✔ Graph Generation completed successfully
            </span>
          )}
        </div>
      </div>

      {/* Pipeline Step Progress */}
      {(running || done) && (
        <div className="card" style={{ padding: 20 }}>
          <div className="card-header" style={{ marginBottom: 12 }}>
            <div className="card-icon">📋</div>
            <div>
              <h3>Graph Generation Progress</h3>
              <div className="card-subtitle">
                {done
                  ? "All steps completed"
                  : activeStep >= 0
                  ? `Step ${Math.min(activeStep + 1, PIPELINE_STEPS.length)}/${PIPELINE_STEPS.length}`
                  : "Initializing…"}
              </div>
            </div>
          </div>
          <div className="pipeline-steps">
            {PIPELINE_STEPS.map((step, idx) => {
              const isDone = completedSteps.has(idx);
              const isActive = running && activeStep === idx && !isDone;
              const cls = isDone ? "done" : isActive ? "active" : "";
              return (
                <div key={step.id} className={`step-item ${cls}`}>
                  <div className="step-bullet">
                    {isDone ? "✓" : isActive ? "⚙" : idx + 1}
                  </div>
                  <div className="step-label">{step.label}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Terminal log */}
      {logs.length > 0 && (
        <div className="terminal">
          <div className="terminal-titlebar">
            <span className="terminal-dot red" />
            <span className="terminal-dot yellow" />
            <span className="terminal-dot green" />
            <span className="terminal-title" style={{ marginLeft: 8 }}>
              graphrag-pipeline — {selectedDoc?.name || "run"}
            </span>
            <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-dim)" }}>
              {logs.length} line{logs.length !== 1 ? "s" : ""}
            </span>
          </div>

          <div className="terminal-body">
            {logs.map((l, i) => (
              <div key={i} className={`log-line ${l.cls}`}>
                <span className="log-time">{l.time}</span>
                <span className="log-prefix">{logPrefix(l.cls)}</span>
                <span className="log-msg">{l.text}</span>
              </div>
            ))}
            {running && (
              <div className="log-line">
                <span className="log-time">{now()}</span>
                <span className="log-prefix" style={{ color: "var(--accent)" }}>›</span>
                <span className="log-msg">
                  <span className="log-cursor" />
                </span>
              </div>
            )}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
    </>
  );
}
