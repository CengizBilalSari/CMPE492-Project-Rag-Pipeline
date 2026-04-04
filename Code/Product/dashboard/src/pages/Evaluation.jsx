import { useState, useRef, useEffect } from "react";
import { startEvaluation, getEvalStatus, getEvalResults } from "../api";
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

const CHART_COLORS = [
  "#6D8196",
  "#4A4A4A",
  "#27ae60",
  "#c0392b",
  "#2980b9",
  "#8e44ad",
];

function makeChart(title, labels, values) {
  return {
    labels,
    datasets: [
      {
        label: title,
        data: values,
        backgroundColor: CHART_COLORS.slice(0, labels.length),
        borderRadius: 4,
      },
    ],
  };
}

const chartOpts = (title) => ({
  responsive: true,
  plugins: {
    legend: { display: false },
    title: { display: true, text: title, font: { size: 13 } },
  },
  scales: { y: { beginAtZero: true } },
});

export default function Evaluation() {
  const [selected, setSelected] = useState(["global", "local"]);
  const [questionSrc, setQuestionSrc] = useState("auto");
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const pollRef = useRef(null);
  const fileRef = useRef();

  function toggle(t) {
    setSelected((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );
  }

  async function start() {
    if (!selected.length) return;
    setError("");
    setResults(null);
    setLoading(true);
    try {
      const res = await startEvaluation(selected, questionSrc, file);
      setJobId(res.job_id);
      setStatus("pending");
      startPolling(res.job_id);
    } catch (e) {
      setError(e.message);
      setLoading(false);
    }
  }

  function startPolling(id) {
    pollRef.current = setInterval(async () => {
      try {
        const s = await getEvalStatus(id);
        setStatus(s.status);
        if (s.status === "completed" || s.status === "failed") {
          clearInterval(pollRef.current);
          setLoading(false);
          if (s.status === "completed") {
            const r = await getEvalResults(id);
            setResults(r);
          }
          if (s.status === "failed") {
            setError(s.error || "Evaluation failed.");
          }
        }
      } catch {
        clearInterval(pollRef.current);
        setLoading(false);
      }
    }, 3000);
  }

  useEffect(() => {
    return () => clearInterval(pollRef.current);
  }, []);

  const labels = results ? results.map((r) => r.search_type) : [];

  return (
    <>
      <h2>Evaluation</h2>

      {error && <div className="error-msg">{error}</div>}

      <div className="card">
        <h3>Search Types</h3>
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
      </div>

      <div className="card">
        <h3>Question Source</h3>
        <div className="form-row">
          <div className="form-group">
            <label>Source</label>
            <select
              value={questionSrc}
              onChange={(e) => setQuestionSrc(e.target.value)}
            >
              <option value="auto">Auto-generate from documents</option>
              <option value="csv">Upload CSV</option>
            </select>
          </div>
        </div>

        {questionSrc === "csv" && (
          <div className="form-group">
            <label>CSV File (question, expected_answer columns)</label>
            <div
              className="file-drop"
              onClick={() => fileRef.current.click()}
            >
              {file ? file.name : "Click to select CSV"}
              <input
                ref={fileRef}
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files[0])}
              />
            </div>
          </div>
        )}

        <button
          className="btn btn-primary mt-16"
          disabled={loading || !selected.length}
          onClick={start}
        >
          {loading ? `Running (${status || "..."})` : "Start Evaluation"}
        </button>
      </div>

      {results && (
        <div className="card">
          <h3>Results</h3>
          <div className="chart-grid">
            <div className="chart-card">
              <Bar
                data={makeChart(
                  "Answer Accuracy",
                  labels,
                  results.map((r) => r.answer_accuracy)
                )}
                options={chartOpts("Answer Accuracy (0-10)")}
              />
            </div>
            <div className="chart-card">
              <Bar
                data={makeChart(
                  "Context Relevance",
                  labels,
                  results.map((r) => r.context_relevance)
                )}
                options={chartOpts("Context Relevance (0-10)")}
              />
            </div>
            <div className="chart-card">
              <Bar
                data={makeChart(
                  "Time per Request",
                  labels,
                  results.map((r) => r.time_per_request)
                )}
                options={chartOpts("Avg Time per Request (s)")}
              />
            </div>
            <div className="chart-card">
              <Bar
                data={makeChart(
                  "Token Cost",
                  labels,
                  results.map((r) => r.token_cost)
                )}
                options={chartOpts("Total Token Cost")}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
