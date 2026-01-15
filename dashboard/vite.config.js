import path from "path";
import { fileURLToPath } from "url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, "");
  const apiHttpTarget = env.VITE_API_HTTP_TARGET || "http://127.0.0.1:8010";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        // Browser opens ws(s)://<dev-host>/ws/generate → forwarded to FastAPI (avoids wrong port / localhost IPv6).
        "/ws": {
          target: apiHttpTarget,
          changeOrigin: true,
          ws: true,
          secure: false,
        },
      },
    },
  };
});
