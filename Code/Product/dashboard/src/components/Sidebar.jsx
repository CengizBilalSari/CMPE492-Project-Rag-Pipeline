import { NavLink } from "react-router-dom";
import { getChatName } from "../api";

const NAV = [
  { to: "/", label: "Documents", icon: "📄", end: true },
  { to: "/pipeline", label: "Graph Generation", icon: "⚡" },
  { to: "/evaluation", label: "Evaluation", icon: "📊" },
  { to: "/history", label: "History", icon: "🕑" },
];

export default function Sidebar({ onSwitchChat }) {
  const chatName = getChatName() || "";

  return (
    <nav className="sidebar">
      <div className="sidebar-logo">
        <h1>GraphRAG</h1>
        <p>RAG Pipeline Dashboard</p>
      </div>

      <div className="sidebar-nav">
        {NAV.map(({ to, label, icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => isActive ? "active" : ""}
          >
            <span className="nav-icon">{icon}</span>
            {label}
          </NavLink>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="user-id">
          {chatName ? (
            <strong style={{ display: "block", fontSize: 13, marginBottom: 4 }}>
              💬 {chatName}
            </strong>
          ) : (
            <strong style={{ display: "block", fontSize: 13, marginBottom: 4 }}>
              No chat selected
            </strong>
          )}
        </div>
        <button
          className="btn btn-outline"
          onClick={onSwitchChat}
          style={{ marginTop: 8, width: "100%" }}
        >
          Switch Chat
        </button>
      </div>
    </nav>
  );
}
