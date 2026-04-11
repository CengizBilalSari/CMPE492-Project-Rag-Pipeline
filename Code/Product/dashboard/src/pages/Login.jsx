import { useState } from "react";
import { login } from "../api";

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    const trimmed = username.trim();
    if (!trimmed) return;
    setLoading(true);
    setError("");
    try {
      await login(trimmed);
      onLogin?.();
    } catch (err) {
      setError(err.message || "Login failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo">
          <h1>GraphRAG</h1>
          <p>Enter a username to start (or continue) a chat workspace.</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. alice"
              autoFocus
              disabled={loading}
              maxLength={100}
            />
          </div>

          {error && <div className="error-msg">⚠️ {error}</div>}

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading || !username.trim()}
            style={{ width: "100%" }}
          >
            {loading ? "Connecting…" : "Continue"}
          </button>
        </form>

        <div className="login-hint">
          Existing username → same chat_id and graph.
          <br />
          New username → fresh workspace.
        </div>
      </div>
    </div>
  );
}
