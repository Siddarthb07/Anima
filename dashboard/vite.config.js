import path from "path";
import { fileURLToPath } from "url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, "");
  const apiHttpTarget = env.VITE_API_HTTP_TARGET || "http://127.0.0.1:8010";

  const proxyCommon = {
    target: apiHttpTarget,
    changeOrigin: true,
    secure: false,
  };

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/ws": { ...proxyCommon, ws: true },
        "/models": proxyCommon,
        "/health": proxyCommon,
        "/generate": proxyCommon,
        "/encode": proxyCommon,
      },
    },
  };
});
