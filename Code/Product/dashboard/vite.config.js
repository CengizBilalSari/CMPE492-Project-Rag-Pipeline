import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
      "/eval": { target: "http://localhost:8001", rewrite: (p) => p.replace(/^\/eval/, "/api") },
    },
  },
});
