import { useState, useEffect } from "react";
import { getPipelineRuns, getEvaluationJobs, getEvaluationJobDetails, getDocuments } from "../api";
import { exportEvalResults, exportEvalDetails } from "../utils/exportCsv";
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
    title: { display: true, text: title, font: { size: 12, family: "'Inter', sans-serif", weight: "600" }, color: "#64748b", padding: { bottom: 12 } },
    tooltip: { backgroundColor: "rgba(15,17,32,0.95)", borderColor: "rgba(139,92,246,0.3)", borderWidth: 1, titleColor: "#e2e8f0", bodyColor: "#94a3b8", padding: 10 },
  },
  scales: {
    y: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b", font: { size: 11 } } },
    x: { grid: { display: false }, ticks: { color: "#94a3b8", font: { size: 11 } } },
  },
});

function Badge({ status }) {
  const s = (status || "").toLowerCase();
  const cls =
    s === "completed" ? "badge-completed" :
      s === "failed" ? "badge-failed" :
        s === "running" ? "badge-running" :
          s === "created" ? "badge-created" :
            "badge-pending";
  return <span className={`badge ${cls}`}>{status || "unknown"}</span>;
}

function RelTime({ iso }) {
  if (!iso) return <span style={{ color: "var(--text-dim)" }}>—</span>;
  const d = new Date(iso);
  const diff = Math.round((Date.now() - d) / 1000);
  const label =
    diff < 60 ? `${diff}s ago` :
      diff < 3600 ? `${Math.round(diff / 60)}m ago` :
        diff < 86400 ? `${Math.round(diff / 3600)}h ago` :
          d.toLocaleDateString();
  return <span title={d.toLocaleString()} style={{ color: "var(--text-muted)", fontSize: 12 }}>{label}</span>;
}

const GLOBAL_RETRIEVER_TYPES = new Set(["global", "ms-graphrag-global", "ppr"]);

export default function History() {
  const [runs, setRuns] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [tab, setTab] = useState("pipeline");
  const [selectedRun, setSelectedRun] = useState(null);
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobDetails, setJobDetails] = useState(null);
  const [docs, setDocs] = useState([]);

  async function handleJobClick(job) {
    setSelectedJob(job);
    setJobDetails(null);
    const details = await getEvaluationJobDetails(job.id);
    setJobDetails(details);
  }

  useEffect(() => {
    getPipelineRuns().then(setRuns);
    getEvaluationJobs().then(setJobs);
    getDocuments().then(setDocs);
  }, []);

  return (
    <>
      <div className="page-header">
        <h2>History</h2>
        <p>Browse past pipeline runs and evaluation jobs.</p>
      </div>

      <div className="tabs">
        <button
          className={`tab-btn ${tab === "pipeline" ? "active" : ""}`}
          onClick={() => setTab("pipeline")}
        >
          ⚡ Pipeline Runs
          {runs.length > 0 && (
            <span style={{
              marginLeft: 6, background: "rgba(255,255,255,0.15)",
              borderRadius: 10, padding: "1px 6px", fontSize: 10, fontWeight: 700
            }}>{runs.length}</span>
          )}
        </button>
        <button
          className={`tab-btn ${tab === "evaluation" ? "active" : ""}`}
          onClick={() => setTab("evaluation")}
        >
          📊 Evaluation Jobs
          {jobs.length > 0 && (
            <span style={{
              marginLeft: 6, background: "rgba(255,255,255,0.15)",
              borderRadius: 10, padding: "1px 6px", fontSize: 10, fontWeight: 700
            }}>{jobs.length}</span>
          )}
        </button>
      </div>

      {tab === "pipeline" && (
        <div className="card">
          <div className="card-header">
            <div className="card-icon">⚡</div>
            <div>
              <h3>Pipeline Runs</h3>
              <div className="card-subtitle">{runs.length} total run{runs.length !== 1 ? "s" : ""}</div>
            </div>
          </div>

          {runs.length === 0 ? (
            <div className="empty">
              <span className="empty-icon">⚡</span>
              No pipeline runs yet. Go to Pipeline to get started.
            </div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Document</th>
                    <th>Status</th>
                    <th>Started</th>
                    <th>Finished</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.id} className="clickable-row" onClick={() => setSelectedRun(r)}>
                      <td style={{ fontWeight: 500 }}>
                        📄 {r.document_name || (
                          <span className="td-mono">{r.document_id?.slice(0, 8)}…</span>
                        )}
                      </td>
                      <td><Badge status={r.status} /></td>
                      <td><RelTime iso={r.started_at || r.created_at} /></td>
                      <td><RelTime iso={r.finished_at || r.completed_at} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {tab === "evaluation" && (
        <div className="card">
          <div className="card-header">
            <div className="card-icon">📊</div>
            <div>
              <h3>Evaluation Jobs</h3>
              <div className="card-subtitle">{jobs.length} total job{jobs.length !== 1 ? "s" : ""}</div>
            </div>
          </div>

          {jobs.length === 0 ? (
            <div className="empty">
              <span className="empty-icon">📊</span>
              No evaluation jobs yet. Go to Evaluation to run one.
            </div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Job ID</th>
                    <th>Document</th>
                    <th>Search Types</th>
                    <th>LLM</th>
                    <th>Status</th>
                    <th>Created At</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j) => (
                    <tr key={j.id} className="clickable-row" onClick={() => handleJobClick(j)}>
                      <td className="td-mono">{j.id.slice(0, 8)}…</td>
                      <td style={{ fontWeight: 500 }}>
                        📄 {docs.find(d => d.id === j.document_id)?.name || (j.document_id ? j.document_id.slice(0, 8) + '…' : 'Unknown')}
                      </td>
                      <td>
                        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                          {(Array.isArray(j.search_types)
                            ? j.search_types
                            : (j.search_types || "").split(",")
                          ).map((t) => {
                            const isGlobal = GLOBAL_RETRIEVER_TYPES.has(t.trim());
                            return (
                              <span key={t} className="badge badge-created" style={{ fontSize: 10, borderColor: isGlobal ? "var(--cyan)" : "var(--accent)", color: isGlobal ? "var(--cyan)" : "var(--accent)" }}>
                                {isGlobal ? "🌍" : "📍"} {t.trim()}
                              </span>
                            );
                          })}
                        </div>
                      </td>
                      <td>
                        {j.provider && j.model ? (
                          <span style={{ fontSize: 12 }}>{j.provider} / {j.model}</span>
                        ) : (
                          <span style={{ color: "var(--text-muted)", fontSize: 12 }}>—</span>
                        )}
                      </td>
                      <td><Badge status={j.status} /></td>
                      <td>{new Date(j.created_at).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {selectedRun && (
        <div className="modal-overlay" onClick={() => setSelectedRun(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedRun(null)}>×</button>
            
            <div style={{ marginBottom: 24 }}>
              <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: 8 }}>Pipeline Details</h2>
              <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
                📄 {selectedRun.document_name || selectedRun.document_id}
              </p>
            </div>

            <div className="stat-row">
              <div className="stat-pill">Status <span><Badge status={selectedRun.status} /></span></div>
              <div className="stat-pill">ID <span>{selectedRun.id.slice(0,8)}…</span></div>
              {selectedRun.config?.llm && (
                <div className="stat-pill">LLM <span>{selectedRun.config.llm.provider} / {selectedRun.config.llm.model}</span></div>
              )}
            </div>

            {selectedRun.error && (
              <div className="error-msg">⚠️ {selectedRun.error}</div>
            )}

            {selectedRun.neo4j_stats && (
              <div className="modal-section">
                <h4>Graph Statistics</h4>
                <div className="modal-data-grid">
                  <div className="data-card">
                    <div className="data-card-label">Entities</div>
                    <div className="data-card-val">{selectedRun.neo4j_stats.entities || 0}</div>
                  </div>
                  <div className="data-card">
                    <div className="data-card-label">Relationships</div>
                    <div className="data-card-val">{selectedRun.neo4j_stats.relationships || 0}</div>
                  </div>
                  <div className="data-card">
                    <div className="data-card-label">Communities</div>
                    <div className="data-card-val">{selectedRun.neo4j_stats.communities || 0}</div>
                  </div>
                  <div className="data-card">
                    <div className="data-card-label">Chunks</div>
                    <div className="data-card-val">{selectedRun.neo4j_stats.chunks || 0}</div>
                  </div>
                </div>
              </div>
            )}

            {selectedRun.llm_usage && (
              <div className="modal-section">
                <h4>LLM Usage</h4>
                <div className="modal-data-grid">
                  <div className="data-card">
                    <div className="data-card-label">Total Tokens</div>
                    <div className="data-card-val">{selectedRun.llm_usage.total_tokens || 0}</div>
                  </div>
                  <div className="data-card">
                    <div className="data-card-label">Prompt Tokens</div>
                    <div className="data-card-val">{selectedRun.llm_usage.prompt_tokens || 0}</div>
                  </div>
                  <div className="data-card">
                    <div className="data-card-label">Completion Tokens</div>
                    <div className="data-card-val">{selectedRun.llm_usage.completion_tokens || 0}</div>
                  </div>
                  <div className="data-card">
                    <div className="data-card-label">Total Requests</div>
                    <div className="data-card-val">{selectedRun.llm_usage.total_requests || 0}</div>
                  </div>
                </div>
              </div>
            )}

            {selectedRun.step_times && Object.keys(selectedRun.step_times).length > 0 && (
              <div className="modal-section">
                <h4>Phase Durations</h4>
                <div className="table-wrap">
                  <table>
                    <tbody>
                      {Object.entries(selectedRun.step_times).map(([step, time]) => (
                        <tr key={step}>
                          <td style={{ color: "var(--text-muted)" }}>{step}</td>
                          <td className="td-mono" style={{ textAlign: "right" }}>{(time || 0).toFixed(2)} s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            
          </div>
        </div>
      )}

      {selectedJob && (
        <div className="modal-overlay" onClick={() => setSelectedJob(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedJob(null)}>×</button>

            <div style={{ marginBottom: 24 }}>
              <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: 8 }}>Evaluation Details</h2>
              <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
                {selectedJob.question_source === "auto" ? "🤖 Auto-generated Questions" :
                 selectedJob.question_source === "auto_modified" ? "🤖 Auto-generated (Modified)" :
                 selectedJob.question_source === "manual" ? "✏️ Manual Questions" :
                 "📂 Uploaded CSV"}
              </p>
            </div>

            <div className="stat-row">
              <div className="stat-pill">Status <span><Badge status={selectedJob.status} /></span></div>
              <div className="stat-pill">ID <span>{selectedJob.id.slice(0,8)}…</span></div>
              <div className="stat-pill">Created <span>{new Date(selectedJob.created_at).toLocaleString()}</span></div>
              {selectedJob.provider && selectedJob.model && (
                <div className="stat-pill">LLM <span>{selectedJob.provider} / {selectedJob.model}</span></div>
              )}
            </div>

            {selectedJob.error && (
              <div className="error-msg">⚠️ {selectedJob.error}</div>
            )}

            {!jobDetails ? (
              <div className="empty" style={{ padding: 40 }}>
                <span style={{ display: "inline-block", animation: "spin 1s linear infinite" }}>⚙</span>
                <div style={{ marginTop: 8 }}>Loading full job details...</div>
              </div>
            ) : (
              <>
                {jobDetails.results && jobDetails.results.length > 0 && (() => {
                  const globalResults = jobDetails.results.filter(r => GLOBAL_RETRIEVER_TYPES.has(r.search_type));
                  const localResults = jobDetails.results.filter(r => !GLOBAL_RETRIEVER_TYPES.has(r.search_type));
                  const hasTyped = jobDetails.qa_pairs?.some(q => q.question.startsWith("[Global]") || q.question.startsWith("[Local]"));

                  const ResultGroup = ({ label, icon, color, results, qaPairs }) => {
                    const labels = results.map(r => r.search_type);
                    const groupQaPairs = hasTyped ? qaPairs : jobDetails.qa_pairs;
                    return (
                      <div className="modal-section">
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                          <h4 style={{ margin: 0, color }}>
                            {icon} {label}
                            <span style={{ fontSize: 11, fontWeight: 400, color: "var(--text-dim)", marginLeft: 8 }}>
                              {results.length} strategy{results.length !== 1 ? "s" : ""}
                              {hasTyped && groupQaPairs?.length ? ` · ${groupQaPairs.length} questions` : ""}
                            </span>
                          </h4>
                          <button
                            className="btn btn-secondary"
                            onClick={() => exportEvalResults(results)}
                            style={{ fontSize: 12, height: 28, padding: "0 12px" }}
                          >
                            ⬇️ CSV
                          </button>
                        </div>

                        <div className="table-wrap" style={{ marginBottom: 20 }}>
                          <table>
                            <thead>
                              <tr>
                                <th>Strategy</th>
                                <th>Q Type</th>
                                <th>Accuracy</th>
                                <th>Context Rel.</th>
                                <th>Time/Req</th>
                                <th>Tokens</th>
                              </tr>
                            </thead>
                            <tbody>
                              {results.map(r => (
                                <tr key={r.id || r.search_type}>
                                  <td style={{ fontWeight: 600, color }}>{r.search_type}</td>
                                  <td style={{ fontSize: 11 }}>
                                    {hasTyped
                                      ? (GLOBAL_RETRIEVER_TYPES.has(r.search_type) ? "🌍 global" : "📍 local")
                                      : "—"}
                                  </td>
                                  <td>{(r.answer_accuracy || 0).toFixed(2)}</td>
                                  <td>{(r.context_relevance || 0).toFixed(2)}</td>
                                  <td>{(r.time_per_request || 0).toFixed(2)}s</td>
                                  <td className="td-mono">{r.token_cost}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
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
                                data={makeChart(title, labels, results.map(r => r[key]))}
                                options={chartOpts(title)}
                              />
                            </div>
                          ))}
                        </div>

                        {groupQaPairs && groupQaPairs.length > 0 && (
                          <div style={{ marginTop: 24 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                              <h5 style={{ margin: 0, fontSize: 13, color: "var(--text-muted)" }}>
                                Questions Used ({groupQaPairs.length})
                              </h5>
                              <button
                                className="btn btn-secondary"
                                onClick={() => exportEvalDetails(groupQaPairs)}
                                style={{ fontSize: 11, height: 26, padding: "0 10px" }}
                              >
                                ⬇️ Export Details
                              </button>
                            </div>
                            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                              {groupQaPairs.map((qa, index) => (
                                <details key={qa.id} style={{ border: "1px solid var(--border)", borderRadius: 6, overflow: "hidden" }}>
                                  <summary style={{ padding: "10px 14px", background: "var(--bg-card)", cursor: "pointer", fontSize: 13, fontWeight: 500, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                    <span>{index + 1}. {qa.question}</span>
                                    {qa.qa_evaluations && (
                                      <span style={{ fontSize: 11, color: "var(--text-dim)", marginLeft: 8, flexShrink: 0 }}>
                                        {qa.qa_evaluations.filter(e => results.some(r => r.search_type === e.search_type)).length} evals
                                      </span>
                                    )}
                                  </summary>
                                  <div style={{ padding: "12px 14px", background: "var(--bg-card-hover)" }}>
                                    <div style={{ fontSize: 12, color: "var(--green)", marginBottom: 10 }}>
                                      <strong>Ground Truth:</strong> {qa.ground_truth_answer}
                                    </div>
                                    {qa.qa_evaluations && qa.qa_evaluations
                                      .filter(e => results.some(r => r.search_type === e.search_type))
                                      .map(e => (
                                        <div key={e.id} style={{ padding: 10, border: "1px solid var(--border)", borderRadius: 6, background: "var(--bg-card)", marginBottom: 8 }}>
                                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                                            <span style={{ color, fontWeight: 600, fontSize: 12, textTransform: "uppercase" }}>{e.search_type}</span>
                                            <span className="td-mono" style={{ fontSize: 11 }}>
                                              <span style={{ color: (e.answer_correctness_score ?? 0) >= 7 ? "#10b981" : (e.answer_correctness_score ?? 0) >= 4 ? "#f59e0b" : "#ef4444" }}>
                                                A: {e.answer_correctness_score ?? "N/A"}
                                              </span>
                                              {" · "}
                                              <span style={{ color: (e.context_relevance_score ?? 0) >= 7 ? "#10b981" : (e.context_relevance_score ?? 0) >= 4 ? "#f59e0b" : "#ef4444" }}>
                                                C: {e.context_relevance_score ?? "N/A"}
                                              </span>
                                              {" · "}{(e.latency_ms / 1000).toFixed(2)}s
                                            </span>
                                          </div>
                                          {e.answer_correctness_reason && (
                                            <div style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.5, marginBottom: 4 }}>
                                              <strong>Accuracy:</strong> {e.answer_correctness_reason}
                                            </div>
                                          )}
                                          {e.context_relevance_reason && (
                                            <div style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.5, marginBottom: 4 }}>
                                              <strong>Context:</strong> {e.context_relevance_reason}
                                            </div>
                                          )}
                                          {e.rag_reasoning && (
                                            <details style={{ fontSize: 11, marginTop: 4 }}>
                                              <summary style={{ cursor: "pointer", color: "var(--text-muted)" }}>💭 Answer Reasoning</summary>
                                              <div style={{ marginTop: 4, padding: 6, background: "var(--bg-card)", borderRadius: 4, color: "var(--text-dim)", lineHeight: 1.5, whiteSpace: "pre-wrap", fontStyle: "italic" }}>
                                                {e.rag_reasoning}
                                              </div>
                                            </details>
                                          )}
                                          {e.rag_answer && (
                                            <details style={{ fontSize: 11, marginTop: 4 }}>
                                              <summary style={{ cursor: "pointer", color: "var(--text-muted)" }}>RAG Answer</summary>
                                              <div style={{ marginTop: 4, padding: 6, background: "var(--bg-card)", borderRadius: 4, color: "var(--text-dim)", lineHeight: 1.5, whiteSpace: "pre-wrap" }}>
                                                {e.rag_answer}
                                              </div>
                                            </details>
                                          )}
                                          {e.retrieved_contexts && e.retrieved_contexts.length > 0 && (
                                            <details style={{ fontSize: 11, marginTop: 4 }}>
                                              <summary style={{ cursor: "pointer", color: "var(--cyan)" }}>🔍 Retrieved Knowledge</summary>
                                              <div className="insight-container" style={{ marginTop: 8 }}>
                                                {e.retrieved_contexts.map((ctx, idx) => {
                                                  const isGlobalSt = GLOBAL_RETRIEVER_TYPES.has(e.search_type);
                                                  const isSource = typeof ctx === 'string' && ctx.toLowerCase().includes('source text passages:');
                                                  const isRel = typeof ctx === 'string' && ctx.toLowerCase().includes('relationships:') && !isGlobalSt;
                                                  const isEntity = typeof ctx === 'string' && ctx.toLowerCase().includes('seed entities:');
                                                  let label2 = isEntity ? 'Seed Entities' : isRel ? 'Relationships' : isSource ? 'Source Text Chunks' : 'Context Part';
                                                  let icon2 = isEntity ? '👤' : isRel ? '🔗' : isSource ? '📄' : '💡';
                                                  if (isGlobalSt || (!isEntity && !isRel && !isSource)) { label2 = `Community Report #${idx + 1}`; icon2 = '🏘️'; }
                                                  const items = typeof ctx === 'string'
                                                    ? ctx.split(isSource ? /(?=--- Chunk)/g : /\n/g).map(s => s.trim()).filter(Boolean)
                                                    : [ctx];
                                                  const displayItems = items.filter(it => !it.toLowerCase().startsWith('seed entities:') && !it.toLowerCase().startsWith('relationships:') && !it.toLowerCase().startsWith('source text passages:'));
                                                  return (
                                                    <details key={idx} className="insight-group-details" style={{ marginBottom: 10, border: "1px solid rgba(255,255,255,0.05)", borderRadius: 6, overflow: "hidden" }}>
                                                      <summary style={{ padding: "8px 12px", background: "rgba(255,255,255,0.02)", cursor: "pointer", fontSize: 11, fontWeight: 600, color: "var(--text-muted)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                                        <span>{icon2} {label2}</span>
                                                        <span style={{ fontSize: 9, opacity: 0.6 }}>{displayItems.length} items</span>
                                                      </summary>
                                                      <div className="insight-group" style={{ padding: 10, background: "rgba(0,0,0,0.15)" }}>
                                                        {displayItems.map((item, itemIdx) => {
                                                          const parts = typeof item === 'string' && item.includes('|') ? item.split('|').map(p => p.trim()) : null;
                                                          return (
                                                            <div key={itemIdx} className="insight-node" style={{ marginBottom: 6 }}>
                                                              {parts ? (
                                                                <div className="graph-parts-viz">{parts.map((p, i) => <span key={i} className="graph-part">{p}</span>)}</div>
                                                              ) : (
                                                                <div style={{ fontStyle: typeof item === 'string' && item.startsWith('---') ? 'normal' : 'italic', whiteSpace: 'pre-wrap' }}>{item}</div>
                                                              )}
                                                            </div>
                                                          );
                                                        })}
                                                      </div>
                                                    </details>
                                                  );
                                                })}
                                              </div>
                                            </details>
                                          )}
                                        </div>
                                      ))}
                                  </div>
                                </details>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  };

                  if (!hasTyped) {
                    return (
                      <ResultGroup
                        label="Aggregated Results"
                        icon="📊"
                        color="var(--accent)"
                        results={jobDetails.results}
                        qaPairs={jobDetails.qa_pairs}
                      />
                    );
                  }

                  return (
                    <>
                      {globalResults.length > 0 && (
                        <ResultGroup
                          label="Global Search Strategies"
                          icon="🌍"
                          color="var(--cyan)"
                          results={globalResults}
                          qaPairs={jobDetails.qa_pairs?.filter(q => q.question.startsWith("[Global]"))}
                        />
                      )}
                      {localResults.length > 0 && (
                        <ResultGroup
                          label="Local & Other Search Strategies"
                          icon="📍"
                          color="var(--accent)"
                          results={localResults}
                          qaPairs={jobDetails.qa_pairs?.filter(q => q.question.startsWith("[Local]"))}
                        />
                      )}
                    </>
                  );
                })()}
              </>
            )}

          </div>
        </div>
      )}
    </>
  );
}
