import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Documents from "./pages/Documents";
import Pipeline from "./pages/Pipeline";
import Evaluation from "./pages/Evaluation";
import History from "./pages/History";
import Login from "./pages/Login";
import { getChatId, clearChat } from "./api";

export default function App() {
  const [chatId, setChatId] = useState(getChatId());

  // Neither step done → show login (chat base selection)
  if (!chatId) {
    return (
      <Login
        onLogin={() => {
          setChatId(getChatId());
        }}
      />
    );
  }

  /** Go back to chat-selection. */
  function handleSwitchChat() {
    clearChat();
    setChatId(null);
  }

  return (
    <div className="layout">
      <Sidebar onSwitchChat={handleSwitchChat} />
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
