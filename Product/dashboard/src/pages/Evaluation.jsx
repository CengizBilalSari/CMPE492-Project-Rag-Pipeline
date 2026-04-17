import { useState, useRef, useEffect } from "react";
import { startEvaluation, getEvalStatus, getEvalResults, getDocuments, generateQuestions, getEvaluationJobs, getEvaluationJobDetails } from "../api";
import { exportEvalResults } from "../utils/exportCsv";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const SEARCH_TYPES = ["global", "local", "lazy", "ppr", "rag", "no-retriever"];

const PROVIDERS = ["openai", "lmstudio"];
const MODELS = {
  openai: ["gpt-4o", "gpt-4o-mini"],
  lmstudio: [
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "llama-3-22b-instruct-v0.1",
    "google/gemma-4-31b",
  ],
};

const CHART_COLORS = [
  "rgba(139,92,246,0.7)",
  "rgba(34,211,238,0.7)",
  "rgba(16,185,129,0.7)",
  "rgba(245,158,11,0.7)",
  "rgba(239,68,68,0.7)",
  "rgba(99,102,241,0.7)",
];

const CHART_BORDERS = [
  "rgba(139,92,246,1)",
  "rgba(34,211,238,1)",
  "rgba(16,185,129,1)",
  "rgba(245,158,11,1)",
  "rgba(239,68,68,1)",
  "rgba(99,102,241,1)",
];

function classifyLog(text) {
  const t = (text || "").toLowerCase();
  if (t.includes("error") || t.includes("failed") || t.includes("exception")) return "log-error";
  if (t.includes("warn")) return "log-warn";
  if (t.includes("evaluating") || t.includes("generating") || t.includes("parsing") || t.includes("starting")) return "log-step";
  if (t.includes("complete") || t.includes("done") || t.includes("success") || t.includes("finished")) return "log-ok";
  return "";
}

function logPrefix(cls) {
  if (cls === "log-error") return "✖";
  if (cls === "log-warn") return "⚠";
  if (cls === "log-step") return "◆";
  if (cls === "log-ok") return "✔";
  return "›";
}

function now() {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}

function makeChart(title, labels, values) {
  return {
    labels,
    datasets: [
      {
        label: title,
        data: values,
        backgroundColor: CHART_COLORS.slice(0, labels.length),
        borderColor: CHART_BORDERS.slice(0, labels.length),
        borderWidth: 1.5,
        borderRadius: 6,
      },
    ],
  };
}

const chartOpts = (title) => ({
  responsive: true,
  plugins: {
    legend: { display: false },
    title: {
      display: true,
      text: title,
      font: { size: 12, family: "'Inter', sans-serif", weight: "600" },
      color: "#64748b",
      padding: { bottom: 12 },
    },
    tooltip: {
      backgroundColor: "rgba(15,17,32,0.95)",
      borderColor: "rgba(139,92,246,0.3)",
      borderWidth: 1,
      titleColor: "#e2e8f0",
      bodyColor: "#94a3b8",
      padding: 10,
    },
  },
  scales: {
    y: {
      beginAtZero: true,
      grid: { color: "rgba(255,255,255,0.05)" },
      ticks: { color: "#64748b", font: { size: 11 } },
    },
    x: {
      grid: { display: false },
      ticks: { color: "#94a3b8", font: { size: 11 } },
    },
  },
});

export default function Evaluation() {
  const [selected, setSelected] = useState(["global", "local"]);
  const [questionSrc, setQuestionSrc] = useState("auto");
  const [file, setFile] = useState(null);
  const [docs, setDocs] = useState([]);
  const [docId, setDocId] = useState("");
  const [provider, setProvider] = useState("lmstudio");
  const [model, setModel] = useState(MODELS.lmstudio[0]);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [progressMsg, setProgressMsg] = useState("");
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const [numQuestions, setNumQuestions] = useState(10);
  const [generatedQaPairs, setGeneratedQaPairs] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [isApproved, setIsApproved] = useState(false);
  const [reviewExpanded, setReviewExpanded] = useState(true);

  const [allPastJobs, setAllPastJobs] = useState([]);
  const [pastJobId, setPastJobId] = useState("");
  const [loadingPast, setLoadingPast] = useState(false);

  const pollRef = useRef(null);
  const fileRef = useRef();
  const logEndRef = useRef(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    getDocuments().then((d) => {
      setDocs(d);
      if (d.length) setDocId(d[0].id);
    });
    getEvaluationJobs().then(jobs => {
      setAllPastJobs(jobs);
    });
  }, []);

  // Update selected past job when document or jobs change
  useEffect(() => {
    const docJobs = allPastJobs.filter(j => j.document_id === docId);
    if (docJobs.length > 0) {
      setPastJobId(docJobs[0].id);
    } else {
      setPastJobId("");
    }
  }, [docId, allPastJobs]);

  // Reset loaded questions and approval state if they switch datasets/docs
  useEffect(() => {
    setGeneratedQaPairs(null);
    setIsApproved(false);
    setReviewExpanded(true);
  }, [docId, pastJobId, questionSrc]);

  function toggle(t) {
    setSelected((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );
  }

  useEffect(() => { setModel(MODELS[provider][0]); }, [provider]);

  async function start() {
    if (!selected.length) return;
    if ((questionSrc === "auto" || questionSrc === "past") && (!generatedQaPairs || generatedQaPairs.length === 0 || !isApproved)) {
      setError("Please generate/load, review, and approve questions first.");
      return;
    }
    setError("");
    setResults(null);
    setLogs([]);
    setLoading(true);
    try {
      const qaPairsToPass = (questionSrc === "auto" || questionSrc === "past") ? generatedQaPairs : null;
      const actualSource = (questionSrc === "auto" || questionSrc === "past") ? "manual" : questionSrc;

      const res = await startEvaluation(selected, actualSource, file, provider, model, docId, qaPairsToPass);
      setJobId(res.job_id);
      setStatus("pending");
      setProgressMsg("");
      setLogs([{ text: "Initializing evaluation job...", cls: "log-dim", time: now() }]);
      startPolling(res.job_id);
    } catch (e) {
      setError(e.message);
      setLoading(false);
    }
  }

  async function handleGenerateQuestions() {
    if (!docId) return;
    setError("");
    setGenerating(true);
    setGeneratedQaPairs(null);
    setIsApproved(false);
    setReviewExpanded(true);
    try {
      const res = await generateQuestions(provider, model, docId, numQuestions);
      setGeneratedQaPairs(res.questions || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setGenerating(false);
    }
  }

  async function handleLoadPastDataset() {
    if (!pastJobId) return;
    setError("");
    setLoadingPast(true);
    setGeneratedQaPairs(null);
    setIsApproved(false);
    setReviewExpanded(true);
    try {
      const details = await getEvaluationJobDetails(pastJobId);
      if (details?.qa_pairs) {
        setGeneratedQaPairs(details.qa_pairs.map(p => ({
          question: p.question,
          ground_truth_answer: p.ground_truth_answer
        })));
      } else {
        setError("No questions found in this past dataset.");
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingPast(false);
    }
  }

  function startPolling(id) {
    pollRef.current = setInterval(async () => {
      try {
        const s = await getEvalStatus(id);
        setStatus(s.status);
        setProgressMsg(s.progress || "");

        if (s.progress) {
          setLogs(prev => {
            if (prev.length > 0 && prev[prev.length - 1].text === s.progress) return prev;
            return [...prev, { text: s.progress, cls: classifyLog(s.progress), time: now() }];
          });
        }

        if (s.status === "completed" || s.status === "failed") {
          clearInterval(pollRef.current);
          setLoading(false);

          if (s.status === "completed") {
            const r = await getEvalResults(id);
            setResults(r.results);
            setLogs(prev => [...prev, { text: "Evaluation completed successfully.", cls: "log-ok", time: now() }]);
          }
          if (s.status === "failed") {
            setError(s.error || "Evaluation failed.");
            setLogs(prev => [...prev, { text: `Error: ${s.error || "Evaluation failed."}`, cls: "log-error", time: now() }]);
          }
        }
      } catch {
        clearInterval(pollRef.current);
        setLoading(false);
      }
    }, 3000);
  }

  useEffect(() => () => clearInterval(pollRef.current), []);

  const labels = results ? results.map((r) => r.search_type) : [];

  return (
    <>
      <div className="page-header">
        <h2>Evaluation</h2>
        <p>Compare search strategies and visualize performance metrics.</p>
      </div>

      {error && <div className="error-msg">⚠️ {error}</div>}

      {/* Search type selection */}
      <div className="card">
        <div className="card-header">
          <div className="card-icon">🔍</div>
          <div>
            <h3>Search Strategies</h3>
            <div className="card-subtitle">Select one or more strategies to compare</div>
          </div>
        </div>
        <div className="checkbox-group">
          {SEARCH_TYPES.map((t) => (
            <label key={t}>
              <input
                type="checkbox"
                checked={selected.includes(t)}
                onChange={() => toggle(t)}
              />
              {t}
            </label>
          ))}
        </div>
        {selected.length > 0 && (
          <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)" }}>
            {selected.length} strategy selected • comparing: {selected.join(", ")}
          </div>
        )}
      </div>

      {/* Language Model */}
      <div className="card">
        <div className="card-header">
          <div className="card-icon">🧠</div>
          <div>
            <h3>Language Model Configuration</h3>
            <div className="card-subtitle">Select evaluating model to judge the answers</div>
          </div>
        </div>
        <div className="form-row" style={{ marginBottom: 20 }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>LLM Provider</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              {PROVIDERS.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS[provider].map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Question source */}
      <div className="card">
        <div className="card-header">
          <div className="card-icon">❓</div>
          <div>
            <h3>Question Source</h3>
            <div className="card-subtitle">How should evaluation questions be generated?</div>
          </div>
        </div>

        <div className="radio-group mb-20">
          <div
            className={`radio-opt ${questionSrc === "auto" ? "selected" : ""}`}
            onClick={() => setQuestionSrc("auto")}
          >
            <input type="radio" readOnly checked={questionSrc === "auto"} />
            <div className="radio-opt-label">🤖 Auto-generate</div>
            <div className="radio-opt-desc">LLM generates questions from your documents</div>
          </div>
          <div
            className={`radio-opt ${questionSrc === "custom" ? "selected" : ""}`}
            onClick={() => { setQuestionSrc("custom"); setGeneratedQaPairs(null); setIsApproved(false); }}
          >
            <input type="radio" readOnly checked={questionSrc === "custom"} />
            <div className="radio-opt-label">📂 Upload CSV</div>
            <div className="radio-opt-desc">question, ground_truth_answer columns</div>
          </div>
          <div
            className={`radio-opt ${questionSrc === "past" ? "selected" : ""}`}
            onClick={() => { setQuestionSrc("past"); setGeneratedQaPairs(null); setIsApproved(false); }}
          >
            <input type="radio" readOnly checked={questionSrc === "past"} />
            <div className="radio-opt-label">💾 Past Dataset</div>
            <div className="radio-opt-desc">Reuse questions generated in the past</div>
          </div>
        </div>

        {questionSrc === "custom" ? (
          <div
            className="file-drop"
            onClick={() => fileRef.current.click()}
            style={{ marginBottom: 16, padding: "20px 24px" }}
          >
            <span className="file-drop-icon" style={{ fontSize: 24 }}>📊</span>
            <div className="file-drop-text">
              {file ? `✅ ${file.name}` : "Click to select a CSV file"}
            </div>
            <div className="file-drop-sub">Requires: question, ground_truth_answer columns</div>
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0])}
            />
          </div>
        ) : questionSrc === "past" ? (
          <div className="card-body">
            <p style={{ marginBottom: "12px" }}>Load a previously generated dataset for the selected document.</p>
            <div className="form-row">
              <div className="form-group" style={{ marginBottom: 0, flex: 1 }}>
                <label>Target Document</label>
                <select value={docId} onChange={e => setDocId(e.target.value)}>
                  {docs.length === 0 ? <option value="">No documents found</option> : null}
                  {docs.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>
              </div>
              <div className="form-group" style={{ marginBottom: 0, flex: 2 }}>
                <label>Select Past Dataset</label>
                <select value={pastJobId} onChange={e => setPastJobId(e.target.value)}>
                  {allPastJobs.filter(j => j.document_id === docId).length === 0 ? <option value="">No past datasets found for this document</option> : null}
                  {allPastJobs.filter(j => j.document_id === docId).map(j => (
                    <option key={j.id} value={j.id}>{new Date(j.created_at).toLocaleString()} - source: {j.question_source}</option>
                  ))}
                </select>
              </div>
            </div>

            {!generatedQaPairs && (
              <button
                className="btn btn-secondary"
                onClick={handleLoadPastDataset}
                disabled={loadingPast || !pastJobId}
                style={{ marginTop: 16 }}
              >
                {loadingPast ? "Loading..." : "⬇️ Load Selected Dataset"}
              </button>
            )}
          </div>
        ) : (
          <div className="card-body">
            <p style={{ marginBottom: "12px" }}>Auto-generate evaluation questions from your existing documents.</p>
            <div className="form-row">
              <div className="form-group" style={{ marginBottom: 0, flex: 1 }}>
                <label>Target Document</label>
                <select value={docId} onChange={e => setDocId(e.target.value)}>
                  {docs.length === 0 ? <option value="">No documents found</option> : null}
                  {docs.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>
              </div>
              <div className="form-group" style={{ marginBottom: 0, width: "120px" }}>
                <label># Questions</label>
                <input
                  type="number"
                  min={1} max={20}
                  value={numQuestions}
                  onChange={e => setNumQuestions(Number(e.target.value))}
                />
              </div>
            </div>

          </div>
        )}

        {(questionSrc === "auto" || questionSrc === "past") && generatedQaPairs ? (
          <div style={{ marginTop: 20 }}>
            {!reviewExpanded ? (
              <div
                onClick={() => { setReviewExpanded(true); setIsApproved(false); }}
                style={{
                  padding: "12px 16px",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  background: "var(--bg-card-hover)",
                  cursor: "pointer",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center"
                }}
              >
                <div>
                  <span style={{ marginRight: 8 }}>✅</span>
                  <strong>Approved Dataset {questionSrc === "auto" ? "- New" : `- Reused (${allPastJobs.find(j => j.id === pastJobId)?.created_at ? new Date(allPastJobs.find(j => j.id === pastJobId).created_at).toLocaleString() : ''})`}</strong> — Document: {docs.find(d => d.id === docId)?.name || docId}
                </div>
                <span style={{ color: "var(--text-dim)", fontSize: 13 }}>{generatedQaPairs.length} questions (Click to edit)</span>
              </div>
            ) : (
              <>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <h4 style={{ margin: 0 }}>Review Questions ({generatedQaPairs.length})</h4>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button className="btn btn-secondary" onClick={() => setGeneratedQaPairs([...generatedQaPairs, { question: "", ground_truth_answer: "" }])} style={{ padding: "4px 8px", fontSize: 12, height: "auto" }}>➕ Add Question</button>
                    {questionSrc === "auto" && <button className="btn btn-secondary" onClick={() => { setGeneratedQaPairs(null); setIsApproved(false); }} style={{ padding: "4px 8px", fontSize: 12, height: "auto" }}>Regenerate</button>}
                  </div>
                </div>

                <div style={{ maxHeight: 400, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 6, padding: 12, marginBottom: 12 }}>
                  {generatedQaPairs.map((pair, idx) => (
                    <div key={idx} style={{ marginBottom: 16, paddingBottom: 16, borderBottom: "1px solid var(--border)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <strong style={{ fontSize: 13, color: "var(--text-dim)" }}>Q{idx + 1}</strong>
                        <button style={{ background: "transparent", border: "none", color: "var(--text-dim)", cursor: "pointer" }} onClick={() => setGeneratedQaPairs(prev => prev.filter((_, i) => i !== idx))}>🗑️</button>
                      </div>
                      <input
                        type="text"
                        value={pair.question}
                        placeholder="Question"
                        onChange={e => setGeneratedQaPairs(prev => { const n = [...prev]; n[idx].question = e.target.value; return n; })}
                        style={{ width: "100%", marginBottom: 8, padding: 8, border: "1px solid var(--border)", borderRadius: 4, background: "var(--bg-card)", color: "var(--text)" }}
                      />
                      <textarea
                        value={pair.ground_truth_answer}
                        placeholder="Ground truth answer"
                        onChange={e => setGeneratedQaPairs(prev => { const n = [...prev]; n[idx].ground_truth_answer = e.target.value; return n; })}
                        style={{ width: "100%", height: 60, resize: "vertical", fontFamily: "inherit", padding: 8, border: "1px solid var(--border)", borderRadius: 4, background: "var(--bg-card)", color: "var(--text)" }}
                      />
                    </div>
                  ))}
                  {generatedQaPairs.length === 0 && <p style={{ color: "var(--text-dim)", textAlign: "center", margin: 0 }}>No questions left.</p>}
                </div>

                <button
                  className="btn btn-primary"
                  onClick={() => { setIsApproved(true); setReviewExpanded(false); }}
                  disabled={generatedQaPairs.length === 0}
                  style={{ width: "100%", justifyContent: "center" }}
                >
                  ✅ Approve Dataset
                </button>
              </>
            )}
          </div>
        ) : null}

        <div className="flex gap-8 items-center" style={{ marginTop: 20 }}>
          {questionSrc === "auto" && !generatedQaPairs ? (
            <button
              className="btn btn-primary"
              disabled={generating || !selected.length || !docId}
              onClick={handleGenerateQuestions}
            >
              {generating ? (
                <>
                  <span style={{ display: "inline-block", animation: "spin 1s linear infinite", marginRight: 8 }}>⚙</span>
                  Generating Questions…
                </>
              ) : (
                "⚙️ Auto-Generate Questions"
              )}
            </button>
          ) : (
            <button
              className="btn btn-primary"
              disabled={loading || !selected.length || ((questionSrc === "auto" || questionSrc === "past") && !isApproved)}
              onClick={start}
            >
              {loading ? (
                <>
                  <span style={{ display: "inline-block", animation: "spin 1s linear infinite", marginRight: 8 }}>⚙</span>
                  {status ? `${status}…` : "Starting…"}
                </>
              ) : (
                "▶ Start Evaluation"
              )}
            </button>
          )}
        </div>
      </div>

      {/* Results */}
      {results && (
        <div className="card">
          <div className="card-header">
            <div className="card-icon">📈</div>
            <div style={{ flex: 1 }}>
              <h3>Results</h3>
              <div className="card-subtitle">Comparing {results.length} search strategies</div>
            </div>
            <button
              className="btn btn-secondary"
              onClick={() => exportEvalResults(results)}
              style={{ fontSize: 13, height: 32 }}
            >
              ⬇️ Export CSV
            </button>
          </div>

          <div className="chart-grid">
            {[
              { key: "answer_accuracy", title: "Answer Accuracy (0–10)" },
              { key: "context_relevance", title: "Context Relevance (0–10)" },
              { key: "time_per_request", title: "Avg Time per Request (s)" },
              { key: "token_cost", title: "Total Token Cost" },
            ].map(({ key, title }) => (
              <div className="chart-card" key={key}>
                <Bar
                  data={makeChart(title, labels, results.map((r) => r[key]))}
                  options={chartOpts(title)}
                />
              </div>
            ))}
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
              evaluator — job {jobId?.slice(0, 8)}
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
            {loading && (
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
