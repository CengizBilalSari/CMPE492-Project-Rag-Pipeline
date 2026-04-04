import { NavLink } from "react-router-dom";
import { getUserId } from "../api";

export default function Sidebar() {
  return (
    <nav className="sidebar">
      <h1>GraphRAG</h1>
      <NavLink to="/" end>Documents</NavLink>
      <NavLink to="/pipeline">Pipeline</NavLink>
      <NavLink to="/evaluation">Evaluation</NavLink>
      <NavLink to="/history">History</NavLink>
      <div className="user-id">uid: {getUserId()}</div>
    </nav>
  );
}
