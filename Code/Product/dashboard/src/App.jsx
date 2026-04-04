import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Documents from "./pages/Documents";
import Pipeline from "./pages/Pipeline";
import Evaluation from "./pages/Evaluation";
import History from "./pages/History";

export default function App() {
  return (
    <div className="layout">
      <Sidebar />
      <main className="main">
        <Routes>
          <Route path="/" element={<Documents />} />
          <Route path="/pipeline" element={<Pipeline />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}
