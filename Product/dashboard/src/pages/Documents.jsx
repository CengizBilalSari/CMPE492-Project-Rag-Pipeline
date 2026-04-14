import { useState, useEffect, useRef } from "react";
import { uploadDocument, getDocuments } from "../api";

export default function Documents() {
  const [docs, setDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef();

  useEffect(() => {
    getDocuments().then(setDocs);
  }, []);

  async function handleUpload(file) {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      await uploadDocument(file);
      const updated = await getDocuments();
      setDocs(updated);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleUpload(file);
  }

  function fmt(bytes) {
    if (!bytes) return "—";
    return bytes < 1024 * 1024
      ? `${(bytes / 1024).toFixed(1)} KB`
      : `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }

  function fmtType(ct) {
    if (!ct) return "—";
    if (ct.includes("pdf")) return "PDF";
    if (ct.includes("text")) return "TXT";
    return ct.split("/")[1]?.toUpperCase() || ct;
  }

  return (
    <>
      <div className="page-header">
        <h2>Documents</h2>
        <p>Upload and manage your source documents for the GraphRAG pipeline.</p>
      </div>

      {error && (
        <div className="error-msg">
          ⚠️ {error}
        </div>
      )}

      <div
        className={`file-drop${uploading ? " uploading" : ""}${dragOver ? " drag-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current.click()}
        style={{ marginBottom: 20, cursor: uploading ? "not-allowed" : "pointer" }}
      >
        <span className="file-drop-icon">
          {uploading ? "⏳" : dragOver ? "📂" : "📁"}
        </span>
        <div className="file-drop-text">
          {uploading
            ? "Uploading document…"
            : dragOver
            ? "Drop to upload"
            : "Drop a file here, or click to browse"}
        </div>
        <div className="file-drop-sub">Supports PDF and TXT files</div>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt"
          onChange={(e) => handleUpload(e.target.files[0])}
          disabled={uploading}
        />
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-icon">📋</div>
          <div>
            <h3>Uploaded Documents</h3>
            <div className="card-subtitle">{docs.length} document{docs.length !== 1 ? "s" : ""} available</div>
          </div>
        </div>

        {docs.length === 0 ? (
          <div className="empty">
            <span className="empty-icon">🗂️</span>
            No documents yet. Upload a file above to get started.
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Type</th>
                  <th>Uploaded</th>
                  <th>ID</th>
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.id}>
                    <td style={{ fontWeight: 500 }}>
                      {d.content_type?.includes("pdf") ? "📄 " : "📝 "}
                      {d.name}
                    </td>
                    <td>
                      <span className="badge badge-created">{fmtType(d.content_type)}</span>
                    </td>
                    <td style={{ color: "var(--text-muted)", fontSize: 12 }}>
                      {new Date(d.created_at).toLocaleString()}
                    </td>
                    <td className="td-mono">{d.id?.slice(0, 8)}…</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
