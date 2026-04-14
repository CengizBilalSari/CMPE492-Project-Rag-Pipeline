/**
 * Export data as a downloadable CSV file.
 * @param {string} filename - Name of the downloaded file (e.g., "results.csv")
 * @param {string[]} headers - Column headers
 * @param {any[][]} rows - Array of row arrays
 */
export function downloadCsv(filename, headers, rows) {
  const escape = (val) => {
    const str = String(val ?? "").replace(/"/g, '""');
    return str.includes(",") || str.includes('"') || str.includes("\n")
      ? `"${str}"`
      : str;
  };

  const csv = [
    headers.map(escape).join(","),
    ...rows.map((row) => row.map(escape).join(",")),
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Export aggregated evaluation results as CSV.
 */
export function exportEvalResults(results) {
  downloadCsv(
    `evaluation_results_${new Date().toISOString().slice(0, 10)}.csv`,
    ["search_type", "answer_accuracy", "context_relevance", "time_per_request_s", "token_cost"],
    results.map((r) => [
      r.search_type,
      (r.answer_accuracy || 0).toFixed(2),
      (r.context_relevance || 0).toFixed(2),
      (r.time_per_request || 0).toFixed(2),
      r.token_cost || 0,
    ])
  );
}

/**
 * Export detailed per-question evaluation data as CSV.
 */
export function exportEvalDetails(qaPairs) {
  const headers = [
    "question", "ground_truth_answer", "search_type",
    "rag_answer", "answer_correctness", "context_relevance",
    "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens",
  ];

  const rows = [];
  for (const qa of qaPairs) {
    if (!qa.qa_evaluations) continue;
    for (const ev of qa.qa_evaluations) {
      rows.push([
        qa.question,
        qa.ground_truth_answer,
        ev.search_type,
        ev.rag_answer,
        ev.answer_correctness_score,
        ev.context_relevance_score,
        ev.latency_ms,
        ev.prompt_tokens,
        ev.completion_tokens,
        ev.total_tokens,
      ]);
    }
  }

  downloadCsv(
    `evaluation_details_${new Date().toISOString().slice(0, 10)}.csv`,
    headers,
    rows
  );
}
