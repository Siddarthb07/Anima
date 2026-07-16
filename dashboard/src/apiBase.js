/** REST base URL — empty = same origin (Vite proxy, Docker Space, nginx). */
export function apiBase() {
  if (import.meta.env.DEV) return "";
  if (import.meta.env.VITE_SAME_ORIGIN === "1" || import.meta.env.VITE_SAME_ORIGIN === "true") {
    return "";
  }
  const raw = import.meta.env.VITE_API_HTTP_TARGET || "http://127.0.0.1:8010";
  if (!String(raw).trim()) return "";
  return raw.replace(/\/$/, "");
}
