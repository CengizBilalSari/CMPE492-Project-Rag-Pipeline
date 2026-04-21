import { useState, useEffect } from "react";
import { listChats, createChat, selectChat, getPipelineConfig } from "../api";

export default function Login({ onLogin }) {
  const [chats, setChats] = useState([]);
  const [selectedChatId, setSelectedChatId] = useState("");
  const [newChatName, setNewChatName] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("all-MiniLM-L6-v2");
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState([
    { name: "all-MiniLM-L6-v2", dimensions: 384, description: "Fast, lightweight, classic default" },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    
    // Fetch both chats and config in parallel
    Promise.all([
      listChats().catch(err => {
        setError(err.message || "Failed to load chats.");
        return [];
      }),
      getPipelineConfig().catch(() => ({}))
    ]).then(([chatsData, configData]) => {
      setChats(chatsData);
      if (chatsData.length > 0) {
        setSelectedChatId(chatsData[0].chat_id);
      }
      if (configData.embedding_models) {
        setAvailableEmbeddingModels(configData.embedding_models);
      }
      setLoading(false);
    });
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
      await createChat(name, embeddingModel);
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
          <div style={{ marginBottom: 32 }}>
            <div className="form-group">
              <label>Available Chat Bases</label>
              <div className="flex gap-8">
                <select 
                  value={selectedChatId} 
                  onChange={(e) => setSelectedChatId(e.target.value)}
                  style={{ flex: 1 }}
                >
                  {chats.map((chat) => (
                    <option key={chat.chat_id} value={chat.chat_id}>
                      {chat.name}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="btn btn-outline"
                  disabled={!selectedChatId}
                  onClick={() => {
                    const c = chats.find(x => x.chat_id === selectedChatId);
                    if (c) handleSelectChat(c);
                  }}
                >
                  Open →
                </button>
              </div>
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

          <div className="form-group" style={{ marginTop: 16, marginBottom: 24 }}>
            <label>Embedding Model</label>
            <select 
              value={embeddingModel} 
              onChange={(e) => setEmbeddingModel(e.target.value)}
              disabled={loading}
            >
              {availableEmbeddingModels.map((em) => (
                <option key={em.name} value={em.name}>
                  {em.name} ({em.dimensions}d — {em.description})
                </option>
              ))}
            </select>
            <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 8, lineHeight: 1.4 }}>
              <strong>Important:</strong> The embedding model cannot be changed once the chat base is created. All documents in this workspace will use this vector space.
            </div>
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
