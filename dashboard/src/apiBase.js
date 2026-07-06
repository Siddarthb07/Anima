/** REST base URL — empty in Vite dev (proxied); explicit in Docker/production builds. */
export function apiBase() {
  if (import.meta.env.DEV) return "";
  const raw = import.meta.env.VITE_API_HTTP_TARGET || "http://127.0.0.1:8010";
  return raw.replace(/\/$/, "");
}
