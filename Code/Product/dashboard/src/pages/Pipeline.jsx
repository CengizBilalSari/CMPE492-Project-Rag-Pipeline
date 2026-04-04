import { useState, useEffect, useRef } from "react";
import { connectPipeline, getDocuments } from "../api";

const PROVIDERS = ["openai", "lmstudio"];
const MODELS = {
  openai: ["gpt-4o", "gpt-4o-mini"],
  lmstudio: [
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "llama-3-22b-instruct-v0.1",
    "google/gemma-4-31b",
  ],
};
const CHUNKERS = [
  "sentence",
  "token",
  "character",
  "recursive",
  "semantic",
  "propositional",
];

export default function Pipeline() {
  const [docs, setDocs] = useState([]);
  const [docId, setDocId] = useState("");
  const [provider, setProvider] = useState("openai");
  const [model, setModel] = useState(MODELS.openai[0]);
  const [chunker, setChunker] = useState("recursive");
  const [chunkSize, setChunkSize] = useState(512);
  const [overlap, setOverlap] = useState(50);
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  const wsRef = useRef(null);
  const logEndRef = useRef(null);

  useEffect(() => {
    getDocuments().then((d) => {
      setDocs(d);
      if (d.length) setDocId(d[0].id);
    });
  }, []);

  useEffect(() => {
    setModel(MODELS[provider][0]);
  }, [provider]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  function start() {
    if (!docId) return;
    setLogs([]);
    setRunning(true);

    const doc = docs.find((d) => d.id === docId);

    const payload = {
      document_id: docId,
      doc_title: doc?.filename || "untitled",
      doc_source: "upload",
      config: {
        llm: { provider, model },
        chunking: { strategy: chunker, chunk_size: chunkSize, overlap },
      },
    };

    wsRef.current = connectPipeline(
      payload,
      (msg) => {
        setLogs((prev) => [
          ...prev,
          {
            text: msg.message || JSON.stringify(msg),
            error: msg.type === "error",
          },
        ]);
        if (msg.type === "complete" || msg.type === "error") {
          setRunning(false);
        }
      },
      () => setRunning(false)
    );
  }

  return (
    <>
      <h2>Pipeline</h2>

      <div className="card">
        <h3>Configuration</h3>

        <div className="form-row">
          <div className="form-group">
            <label>Document</label>
            <select value={docId} onChange={(e) => setDocId(e.target.value)}>
              {docs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Provider</label>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            >
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
              {MODELS[provider].map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Chunking Strategy</label>
            <select value={chunker} onChange={(e) => setChunker(e.target.value)}>
              {CHUNKERS.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Chunk Size</label>
            <input
              type="number"
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
            />
          </div>
          <div className="form-group">
            <label>Overlap</label>
            <input
              type="number"
              value={overlap}
              onChange={(e) => setOverlap(Number(e.target.value))}
            />
          </div>
        </div>

        <button
          className="btn btn-primary mt-16"
          disabled={running || !docId}
          onClick={start}
        >
          {running ? "Running..." : "Run Pipeline"}
        </button>
      </div>

      {logs.length > 0 && (
        <div className="card">
          <h3>Status Log</h3>
          <div className="status-log">
            {logs.map((l, i) => (
              <div key={i} className={`line${l.error ? " error" : ""}`}>
                {l.text}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
    </>
  );
}
