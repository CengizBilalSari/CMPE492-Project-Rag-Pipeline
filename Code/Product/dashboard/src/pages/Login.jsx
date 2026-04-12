import { useState, useEffect } from "react";
import { listChats, createChat, selectChat } from "../api";

export default function Login({ onLogin }) {
  const [chats, setChats] = useState([]);
  const [newChatName, setNewChatName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    listChats()
      .then(setChats)
      .catch((err) => setError(err.message || "Failed to load chats."))
      .finally(() => setLoading(false));
  }, []);

  function handleSelectChat(chat) {
    selectChat(chat);
    onLogin?.();
  }

  async function handleCreateChat(e) {
    e.preventDefault();
    const name = newChatName.trim();
    if (!name) return;
    setLoading(true);
    setError("");
    try {
      await createChat(name);
      onLogin?.();
    } catch (err) {
      setError(err.message || "Failed to create chat base.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card" style={{ maxWidth: 480 }}>
        <div className="login-logo">
          <h1>GraphRAG</h1>
          <p>
            Welcome! Pick a chat base or create a new one to get started.
          </p>
        </div>

        {/* Existing chats list */}
        {chats.length > 0 && (
          <div style={{ marginBottom: 20 }}>
            <p style={{ marginBottom: 8, opacity: 0.7, fontSize: 13 }}>
              AVAILABLE CHAT BASES
            </p>
            <div className="chat-list">
              {chats.map((chat) => (
                <button
                  key={chat.chat_id}
                  className="chat-list-item"
                  onClick={() => handleSelectChat(chat)}
                >
                  <span className="chat-list-icon">💬</span>
                  <span className="chat-list-name">{chat.name}</span>
                  <span className="chat-list-arrow">→</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {chats.length === 0 && !loading && (
          <p style={{ opacity: 0.6, fontSize: 14, marginBottom: 16 }}>
            There are no chat bases yet. Create your first one below.
          </p>
        )}

        {/* Create new chat form */}
        <form onSubmit={handleCreateChat}>
          <div className="form-group">
            <label>New Chat Base Name</label>
            <input
              type="text"
              value={newChatName}
              onChange={(e) => setNewChatName(e.target.value)}
              placeholder="e.g. Research Project A"
              disabled={loading}
              maxLength={200}
            />
          </div>

          {error && <div className="error-msg" style={{ marginBottom: 16 }}>⚠️ {error}</div>}

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading || !newChatName.trim()}
            style={{ width: "100%" }}
          >
            {loading ? "Working…" : "+ Create New Chat Base"}
          </button>
        </form>
      </div>
    </div>
  );
}
