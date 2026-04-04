import { useState, useEffect } from "react";
import { getPipelineRuns, getEvaluationJobs } from "../api";

function Badge({ status }) {
  const cls =
    status === "completed"
      ? "badge-completed"
      : status === "failed"
      ? "badge-failed"
      : status === "running"
      ? "badge-running"
      : "badge-pending";
  return <span className={`badge ${cls}`}>{status}</span>;
}

export default function History() {
  const [runs, setRuns] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [tab, setTab] = useState("pipeline");

  useEffect(() => {
    getPipelineRuns().then(setRuns);
    getEvaluationJobs().then(setJobs);
  }, []);

  return (
    <>
      <h2>History</h2>

      <div className="gap-8 mb-8">
        <button
          className={`btn ${tab === "pipeline" ? "btn-primary" : "btn-outline"}`}
          onClick={() => setTab("pipeline")}
        >
          Pipeline Runs
        </button>
        <button
          className={`btn ${tab === "evaluation" ? "btn-primary" : "btn-outline"}`}
          onClick={() => setTab("evaluation")}
        >
          Evaluation Jobs
        </button>
      </div>

      {tab === "pipeline" && (
        <div className="card table-wrap">
          {runs.length === 0 ? (
            <div className="empty">No pipeline runs yet.</div>
          ) : (
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
                  <tr key={r.id}>
                    <td>{r.documents?.filename || r.document_id}</td>
                    <td><Badge status={r.status} /></td>
                    <td>{new Date(r.started_at).toLocaleString()}</td>
                    <td>
                      {r.finished_at
                        ? new Date(r.finished_at).toLocaleString()
                        : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {tab === "evaluation" && (
        <div className="card table-wrap">
          {jobs.length === 0 ? (
            <div className="empty">No evaluation jobs yet.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Search Types</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id}>
                    <td style={{ fontFamily: "monospace", fontSize: 11 }}>
                      {j.id.slice(0, 8)}...
                    </td>
                    <td>{j.search_types}</td>
                    <td><Badge status={j.status} /></td>
                    <td>{new Date(j.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </>
  );
}
