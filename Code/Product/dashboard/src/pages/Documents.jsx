import { useState, useEffect, useRef } from "react";
import { uploadDocument, getDocuments } from "../api";

export default function Documents() {
  const [docs, setDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
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
    const file = e.dataTransfer.files[0];
    handleUpload(file);
  }

  return (
    <>
      <h2>Documents</h2>

      {error && <div className="error-msg">{error}</div>}

      <div
        className="file-drop card"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => fileRef.current.click()}
      >
        {uploading ? "Uploading..." : "Drop a PDF or TXT file here, or click to browse"}
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt"
          onChange={(e) => handleUpload(e.target.files[0])}
        />
      </div>

      {docs.length === 0 ? (
        <div className="empty">No documents uploaded yet.</div>
      ) : (
        <div className="card table-wrap">
          <table>
            <thead>
              <tr>
                <th>Filename</th>
                <th>Type</th>
                <th>Uploaded</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.id}>
                  <td>{d.filename}</td>
                  <td>{d.content_type}</td>
                  <td>{new Date(d.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
