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
                          ).map((t) => (
                            <span key={t} className="badge badge-created" style={{ fontSize: 10 }}>
                              {t.trim()}
                            </span>
                          ))}
                        </div>
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
                {selectedJob.question_source === "auto" ? "🤖 Auto-generated Questions" : "📂 Uploaded CSV"}
              </p>
            </div>

            <div className="stat-row">
              <div className="stat-pill">Status <span><Badge status={selectedJob.status} /></span></div>
              <div className="stat-pill">ID <span>{selectedJob.id.slice(0,8)}…</span></div>
              <div className="stat-pill">Created <span>{new Date(selectedJob.created_at).toLocaleString()}</span></div>
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
                {jobDetails.results && jobDetails.results.length > 0 && (
                  <div className="modal-section">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                      <h4 style={{ margin: 0 }}>Aggregated Results</h4>
                      <button 
                        className="btn btn-secondary" 
                        onClick={() => exportEvalResults(jobDetails.results)}
                        style={{ fontSize: 12, height: 28, padding: "0 12px" }}
                      >
                        ⬇️ CSV
                      </button>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Strategy</th>
                            <th>Accuracy</th>
                            <th>Context Rel.</th>
                            <th>Time/Req</th>
                            <th>Tokens</th>
                          </tr>
                        </thead>
                        <tbody>
                          {jobDetails.results.map(r => (
                            <tr key={r.id}>
                              <td style={{ fontWeight: 600, color: "var(--accent)" }}>{r.search_type}</td>
                              <td>{(r.answer_accuracy || 0).toFixed(2)}</td>
                              <td>{(r.context_relevance || 0).toFixed(2)}</td>
                              <td>{(r.time_per_request || 0).toFixed(2)}s</td>
                              <td className="td-mono">{r.token_cost}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    
                    <div className="chart-grid" style={{ marginTop: 24 }}>
                      {[
                        { key: "answer_accuracy", title: "Answer Accuracy (0–10)" },
                        { key: "context_relevance", title: "Context Relevance (0–10)" },
                        { key: "time_per_request", title: "Avg Time per Request (s)" },
                        { key: "token_cost", title: "Total Token Cost" },
                      ].map(({ key, title }) => (
                        <div className="chart-card" key={key}>
                          <Bar
                            data={makeChart(title, jobDetails.results.map((r) => r.search_type), jobDetails.results.map((r) => r[key]))}
                            options={chartOpts(title)}
                          />
                        </div>
                      ))}
                    </div>

                  </div>
                )}

                {jobDetails.qa_pairs && jobDetails.qa_pairs.length > 0 && (
                  <div className="modal-section" style={{ marginTop: 32 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                      <h4 style={{ margin: 0 }}>Questions Evaluated ({jobDetails.qa_pairs.length})</h4>
                      <button 
                        className="btn btn-secondary" 
                        onClick={() => exportEvalDetails(jobDetails.qa_pairs)}
                        style={{ fontSize: 12, height: 28, padding: "0 12px" }}
                      >
                        ⬇️ Export Details CSV
                      </button>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                      {jobDetails.qa_pairs.map((qa, index) => (
                        <div key={qa.id} className="card" style={{ padding: 16, marginBottom: 0 }}>
                          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
                            {index + 1}. {qa.question}
                          </div>
                          <div style={{ fontSize: 13, color: "var(--green)", marginBottom: 12 }}>
                            <span style={{ fontWeight: 600 }}>Ground Truth:</span> {qa.ground_truth_answer}
                          </div>

                          {qa.qa_evaluations && qa.qa_evaluations.length > 0 && (
                            <div className="table-wrap" style={{ marginTop: 12 }}>
                              <table style={{ fontSize: 12 }}>
                                <thead>
                                  <tr>
                                    <th>Strategy</th>
                                    <th>RAG Answer</th>
                                    <th>Score</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {qa.qa_evaluations.map(e => (
                                    <tr key={e.id}>
                                      <td style={{ color: "var(--accent)", whiteSpace: "nowrap" }}>{e.search_type}</td>
                                      <td>
                                        <div style={{ maxHeight: 60, overflowY: "auto", paddingRight: 4, whiteSpace: "pre-wrap" }}>
                                          {e.rag_answer}
                                        </div>
                                      </td>
                                      <td className="td-mono" style={{ whiteSpace: "nowrap" }}>
                                        A: {e.answer_correctness_score}<br/>
                                        C: {e.context_relevance_score}<br/>
                                        {(e.latency_ms / 1000).toFixed(2)}s
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

          </div>
        </div>
      )}
    </>
  );
}
