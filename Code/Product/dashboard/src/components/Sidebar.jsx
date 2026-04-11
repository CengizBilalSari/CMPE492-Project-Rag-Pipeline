import { NavLink } from "react-router-dom";
import { getChatId, getUsername } from "../api";

const NAV = [
  { to: "/",           label: "Documents",  icon: "📄", end: true },
  { to: "/pipeline",   label: "Pipeline",   icon: "⚡" },
  { to: "/evaluation", label: "Evaluation", icon: "📊" },
  { to: "/history",    label: "History",    icon: "🕑" },
];

export default function Sidebar({ onLogout }) {
  const username = getUsername() || "—";
  const chatId = getChatId() || "";
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
          <strong>{username}</strong>
          {chatId ? `${chatId.slice(0, 8)}…${chatId.slice(-4)}` : ""}
        </div>
        <button className="btn btn-outline" onClick={onLogout} style={{ marginTop: 8, width: "100%" }}>
          Log out
        </button>
      </div>
    </nav>
  );
}
