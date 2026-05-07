import { useState, useEffect, useRef } from "react";
import { connectPipeline, getDocuments, getPipelineConfig, getOllamaRecommendations, pullOllamaModel } from "../api";

// Fallbacks used until the API response arrives
const DEFAULT_PROVIDERS = ["openai", "lmstudio", "ollama"];
const DEFAULT_MODELS = {
  openai: ["gpt-4o-mini", "gpt-4o"],
  lmstudio: [
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "llama-3-22b-instruct-v0.1",
    "google/gemma-4-31b",
  ],
  ollama: [
    "llama3:latest",
    "mistral:latest",
    "phi3:latest",
    "gemma:latest",
    "qwen2:latest"
  ]
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

const DETAIL_STEP_MAP = {
  chunk: 0, extract: 1, resolve: 2, resolv: 2,
  embed: 3, community: 4, summarize: 5, summar: 5,
};

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
  const [provider, setProvider] = useState("openai");
  const [model, setModel] = useState(DEFAULT_MODELS.openai[0]);
  const [chunker, setChunker] = useState("recursive");
  const [chunkSize, setChunkSize] = useState(512);
  const [overlap, setOverlap] = useState(50);
  const [embeddingModel, setEmbeddingModel] = useState(localStorage.getItem("graphrag_embedding_model") || "all-MiniLM-L6-v2");
  const [useLlmRes, setUseLlmRes] = useState(true);
  const [pipelineType, setPipelineType] = useState("custom");
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [completedSteps, setCompletedSteps] = useState(new Set());
  const [stepDetails, setStepDetails] = useState({});  // { stepIdx: ["detail msg", ...] }
  const [expandedSteps, setExpandedSteps] = useState(new Set());
  const wsRef = useRef(null);
  const logEndRef = useRef(null);

  // Dynamic config from backend
  const [llmProviders, setLlmProviders] = useState(DEFAULT_MODELS);
  const [embeddingModels, setEmbeddingModels] = useState(DEFAULT_EMBEDDING_MODELS);
  const [chunkers, setChunkers] = useState(DEFAULT_CHUNKERS);

  // Ollama Model Manager State
  const [showOllamaModal, setShowOllamaModal] = useState(false);
  const [ollamaRecs, setOllamaRecs] = useState(null);
  const [pullingModel, setPullingModel] = useState(null);
  const [pullProgress, setPullProgress] = useState("");
  const [userRam, setUserRam] = useState(16); // Default selection

  useEffect(() => {
    if (showOllamaModal && !ollamaRecs) {
      getOllamaRecommendations().then(setOllamaRecs).catch(console.error);
    }
  }, [showOllamaModal, ollamaRecs]);

  async function handlePullModel(modelName) {
    if (pullingModel) return;
    setPullingModel(modelName);
    setPullProgress("Starting download...");
    try {
      await pullOllamaModel(modelName, (event) => {
         if (event.status) {
           let p = event.status;
           if (event.total && event.completed) {
              const pct = Math.round((event.completed / event.total) * 100);
              p += ` (${pct}%)`;
           }
           setPullProgress(p);
         }
      });
      setPullProgress("Download complete!");
      // Refresh models list in dropdown
      const cfg = await getPipelineConfig();
      if (cfg.llm_providers) setLlmProviders(cfg.llm_providers);
      setTimeout(() => {
         setPullingModel(null);
      }, 1500);
    } catch(e) {
      setPullProgress("Error: " + e.message);
      setTimeout(() => setPullingModel(null), 3000);
    }
  }

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
    setStepDetails({});
    setExpandedSteps(new Set());
    pushLog("Connecting to pipeline…", "log-dim");

    const payload = {
      document_id: docId,
      doc_title: docs.find((d) => d.id === docId)?.name || "untitled",
      doc_source: "upload",
      pipeline_type: pipelineType,
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

        // Handle detail messages — don't add to main logs
        if (text.startsWith("detail:")) {
          const parts = text.slice(7).split(":", 2); // "chunk:message..."
          const stepKey = parts[0];
          const detailMsg = parts[1] || "";
          const stepIdx = DETAIL_STEP_MAP[stepKey];
          if (stepIdx !== undefined) {
            setStepDetails((prev) => ({
              ...prev,
              [stepIdx]: [...(prev[stepIdx] || []), detailMsg],
            }));
          }
          return;
        }

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

          <div className="form-group">
            <label>Pipeline Type</label>
            <select value={pipelineType} onChange={(e) => setPipelineType(e.target.value)}>
              <option value="custom">Custom (Neo4j)</option>
              <option value="ms-graphrag">Microsoft GraphRAG</option>
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
              <label style={{ marginBottom: 0 }}>Model</label>
              {provider === "ollama" && (
                <button 
                  className="btn btn-outline" 
                  style={{ padding: "2px 8px", fontSize: "11px" }}
                  onClick={() => setShowOllamaModal(true)}
                  disabled={running}
                >
                  📥 Manage Models
                </button>
              )}
            </div>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {(MODELS[provider] || []).map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </div>

          {pipelineType === "custom" && (
            <div className="form-group">
              <label>Chunking Strategy</label>
              <select value={chunker} onChange={(e) => setChunker(e.target.value)}>
                {chunkers.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        {pipelineType === "custom" && (
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
        )}

        {pipelineType === "custom" && (
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
        )}

        {pipelineType === "custom" && (
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
        )}

        {pipelineType === "ms-graphrag" && (
          <div style={{
            padding: "14px 16px",
            backgroundColor: "rgba(99, 102, 241, 0.08)",
            borderRadius: "8px",
            border: "1px solid rgba(99, 102, 241, 0.2)",
            marginBottom: 20,
            fontSize: 13,
            lineHeight: 1.6,
            color: "var(--text-dim)"
          }}>
            <div style={{ fontWeight: 600, marginBottom: 6, color: "var(--text)" }}>
              🔬 Microsoft GraphRAG
            </div>
            The Microsoft GraphRAG library handles chunking, entity extraction, community detection,
            and summarization internally using its own pipeline. Configuration options above
            (chunk size, overlap, entity resolution) are not applicable.
            <br /><br />
            Output will be stored as parquet files and can be queried using
            <strong> ms-graphrag-global</strong> and <strong>ms-graphrag-local</strong> retrievers
            in the Evaluation page.
          </div>
        )}

        {/* Summary pill */}
        {selectedDoc && (
          <div className="stat-row" style={{ marginBottom: 16 }}>
            <div className="stat-pill">📄 <span>{selectedDoc.name}</span></div>
            <div className="stat-pill">🤖 <span>{provider} / {model}</span></div>
            <div className="stat-pill">🔧 <span>{pipelineType === "ms-graphrag" ? "Microsoft GraphRAG" : "Custom (Neo4j)"}</span></div>
            {pipelineType === "custom" && (
              <>
                <div className="stat-pill">🧬 <span>{embeddingModel}</span></div>
                <div className="stat-pill">✂️ <span>{chunker} · {chunkSize}t · {overlap}o</span></div>
              </>
            )}
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
              <> ▶ Run {pipelineType === "ms-graphrag" ? "Microsoft GraphRAG" : "Graph Generation"}</>
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
              const details = stepDetails[idx] || [];
              const isExpanded = expandedSteps.has(idx);
              const hasDetails = details.length > 0;
              return (
                <div key={step.id} className={`step-item ${cls}`}>
                  <div className="step-main-row">
                    <div className="step-bullet">
                      {isDone ? "✓" : isActive ? "⚙" : idx + 1}
                    </div>
                    <div className="step-label">{step.label}</div>
                    {hasDetails && (
                      <button
                        className={`step-details-toggle ${isExpanded ? "expanded" : ""}`}
                        onClick={() => {
                          setExpandedSteps((prev) => {
                            const next = new Set(prev);
                            if (next.has(idx)) next.delete(idx);
                            else next.add(idx);
                            return next;
                          });
                        }}
                      >
                        <span className="step-details-count">{details.length}</span>
                        <span className="step-details-chevron">{isExpanded ? "▾" : "▸"}</span>
                        Details
                      </button>
                    )}
                    {isActive && !hasDetails && (
                      <span className="step-working-indicator">working…</span>
                    )}
                  </div>
                  {isExpanded && hasDetails && (
                    <div className="step-details-panel">
                      {details.map((d, i) => (
                        <div key={i} className="step-detail-line">
                          <span className="step-detail-dot" />
                          {d}
                        </div>
                      ))}
                    </div>
                  )}
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

      {/* Ollama Model Manager Modal */}
      {showOllamaModal && (
        <div className="modal-overlay" style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: "rgba(0,0,0,0.5)", zIndex: 1000,
          display: "flex", alignItems: "center", justifyContent: "center"
        }}>
          <div className="modal-content card" style={{ width: 600, maxWidth: "90%", maxHeight: "90vh", overflowY: "auto", padding: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
              <div>
                <h3 style={{ margin: "0 0 8px 0" }}>📥 Ollama Model Manager</h3>
                <div style={{ fontSize: 13, color: "var(--text-dim)", display: "flex", alignItems: "center", gap: 8 }}>
                  <span>My Mac's Unified Memory:</span>
                  <select 
                    value={userRam} 
                    onChange={(e) => setUserRam(Number(e.target.value))}
                    style={{ padding: "2px 8px", fontSize: 12, borderRadius: 4 }}
                  >
                    <option value={8}>8 GB</option>
                    <option value={16}>16 GB</option>
                    <option value={32}>32 GB+</option>
                  </select>
                </div>
              </div>
              <button className="btn btn-outline" onClick={() => setShowOllamaModal(false)} disabled={!!pullingModel}>✕</button>
            </div>

            {ollamaRecs && ollamaRecs.tiers.map((tier) => {
              const isRecommended = userRam >= tier.min_ram_gb && userRam < tier.max_ram_gb;
              return (
              <div key={tier.id} style={{
                marginBottom: 16,
                padding: 16,
                borderRadius: 8,
                border: isRecommended ? "2px solid var(--accent)" : "1px solid var(--border)",
                backgroundColor: isRecommended ? "rgba(99, 102, 241, 0.05)" : "transparent"
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <h4 style={{ margin: 0 }}>{tier.name}</h4>
                  {isRecommended && <span style={{ fontSize: 11, fontWeight: "bold", color: "var(--accent)", backgroundColor: "rgba(99, 102, 241, 0.1)", padding: "2px 8px", borderRadius: 12 }}>⭐ Recommended for {userRam}GB</span>}
                </div>
                <div style={{ fontSize: 13, color: "var(--text-dim)", marginBottom: 12 }}>{tier.description}</div>
                
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {tier.models.map((m) => {
                    const baseName = m.name.split(":")[0];
                    const isDownloaded = MODELS["ollama"]?.some(installed => installed.split(":")[0] === baseName);
                    const isPulling = pullingModel === m.name;
                    return (
                      <div key={m.name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", backgroundColor: "var(--bg-subtle)", borderRadius: 6 }}>
                        <div>
                          <div style={{ fontWeight: 500, fontSize: 14 }}>{m.name} <span style={{ fontSize: 11, color: "var(--text-dim)", marginLeft: 8 }}>{m.size}</span></div>
                          <div style={{ fontSize: 12, color: "var(--text-dim)" }}>{m.desc}</div>
                        </div>
                        <div style={{ textAlign: "right", minWidth: 120 }}>
                          {isDownloaded ? (
                            <span style={{ fontSize: 12, color: "var(--green)", fontWeight: 500 }}>✔ Downloaded</span>
                          ) : isPulling ? (
                            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}>
                              <span style={{ fontSize: 12, fontWeight: 500, color: "var(--accent)" }}>Downloading...</span>
                              <span style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "monospace", whiteSpace: "nowrap" }}>{pullProgress}</span>
                            </div>
                          ) : (
                            <button 
                              className="btn btn-primary" 
                              style={{ padding: "4px 12px", fontSize: 12 }}
                              disabled={!!pullingModel}
                              onClick={() => handlePullModel(m.name)}
                            >
                              ⬇ Download
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              );
            })}


          </div>
        </div>
      )}
    </>
  );
}
