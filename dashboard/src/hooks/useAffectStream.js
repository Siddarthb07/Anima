import { useCallback, useRef, useState } from "react";

const CONNECT_MS = 25000;

/** Resolve human /health URL for error hints. */
function healthHint(wsBase) {
  if (import.meta.env.DEV) {
    const forceDirect =
      import.meta.env.VITE_WS_DIRECT === "1" || import.meta.env.VITE_WS_DIRECT === "true";
    if (!forceDirect) {
      const t = (import.meta.env.VITE_API_HTTP_TARGET || "http://127.0.0.1:8010").replace(/\/$/, "");
      return `${t}/health`;
    }
  }
  const wsEnv = typeof import.meta.env.VITE_WS_BASE === "string" ? import.meta.env.VITE_WS_BASE.trim() : "";
  if (wsEnv) {
    try {
      const u = new URL(wsEnv.replace(/^ws/, "http"));
      return `${u.protocol}//${u.host}/health`;
    } catch {
      /* fall through */
    }
  }
  if (import.meta.env.DEV) {
    const t = (import.meta.env.VITE_API_HTTP_TARGET || "http://127.0.0.1:8010").replace(/\/$/, "");
    return `${t}/health`;
  }
  try {
    const u = new URL(wsBase.replace(/^ws/, "http"));
    return `${u.protocol}//${u.host}/health`;
  } catch {
    return "http://127.0.0.1:8010/health";
  }
}

/**
 * WebSocket protocol:
 * - Messages with kind "token" carry { readout } (matches FastAPI StreamTokenMessage)
 * - Final message kind "done" carries suppression_events + summary
 * - kind "error" carries server-side failure after the socket is open
 */
export function useAffectStream(apiBase) {
  const [tokens, setTokens] = useState([]);
  const [suppressionEvents, setSuppressionEvents] = useState([]);
  const [summary, setSummary] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState(null);
  const wsRef = useRef(null);
  const connectTimerRef = useRef(null);
  const receivedTokenRef = useRef(false);

  const stop = useCallback(() => {
    if (connectTimerRef.current != null) {
      clearTimeout(connectTimerRef.current);
      connectTimerRef.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
    setStreaming(false);
    setStatusMessage(null);
  }, []);

  const start = useCallback(
    (model, prompt, maxNewTokens = 200, detectSuppression = true, guardMode = "observe", interventionMode = "none") => {
      stop();
      receivedTokenRef.current = false;
      setError(null);
      setTokens([]);
      setSuppressionEvents([]);
      setSummary(null);
      setStreaming(true);

      const url = `${apiBase}/ws/generate`;
      const hint = healthHint(apiBase);
      setStatusMessage("Connecting to API…");

      let ws;
      try {
        ws = new WebSocket(url);
      } catch (e) {
        setError(`Invalid WebSocket URL: ${url} (${String(e)})`);
        setStreaming(false);
        setStatusMessage(null);
        return;
      }

      wsRef.current = ws;

      connectTimerRef.current = window.setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          ws.close();
          setError(`Timed out connecting to ${url}. Is the API running? Try ${hint} in the browser.`);
          setStreaming(false);
          setStatusMessage(null);
        }
        connectTimerRef.current = null;
      }, CONNECT_MS);

      ws.onopen = () => {
        if (connectTimerRef.current != null) {
          clearTimeout(connectTimerRef.current);
          connectTimerRef.current = null;
        }
        setStatusMessage("Running model (first tokens may take a while if weights are loading)…");
        ws.send(
          JSON.stringify({
            model,
            prompt,
            max_new_tokens: maxNewTokens,
            detect_suppression: detectSuppression,
            guard_mode: guardMode,
            intervention_mode: interventionMode,
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.kind === "token" && msg.readout) {
            receivedTokenRef.current = true;
            setTokens((prev) => [...prev, msg.readout]);
            setStatusMessage(null);
          } else if (msg.kind === "done") {
            setSuppressionEvents(msg.suppression_events || []);
            setSummary(msg.summary || null);
            setStatusMessage(null);
          } else if (msg.kind === "error") {
            setError(msg.message || "Server error");
            setStreaming(false);
            setStatusMessage(null);
          }
        } catch (e) {
          setError(String(e));
        }
      };

      ws.onerror = () => {
        const devProxy =
          import.meta.env.DEV &&
          !(import.meta.env.VITE_WS_DIRECT === "1" || import.meta.env.VITE_WS_DIRECT === "true");
        const proxyTip = devProxy
          ? " Ensure uvicorn matches VITE_API_HTTP_TARGET in dashboard/.env (then restart npm run dev)."
          : "";
        setError(
          `WebSocket error while connecting to ${url}. Open ${hint} — API must be running.${proxyTip} See dashboard/.env.example.`
        );
        setStreaming(false);
        setStatusMessage(null);
      };

      ws.onclose = (ev) => {
        if (connectTimerRef.current != null) {
          clearTimeout(connectTimerRef.current);
          connectTimerRef.current = null;
        }
        wsRef.current = null;
        setStreaming(false);
        setStatusMessage(null);
        if (!ev.wasClean && ev.code !== 1000 && !receivedTokenRef.current) {
          setError(
            (prev) =>
              prev ||
              `Connection closed (${ev.code}${ev.reason ? `: ${ev.reason}` : ""}). Check API logs (model load RAM, wrong model id, etc.). Try ${hint}.`
          );
        }
      };
    },
    [apiBase, stop]
  );

  const loadFromGenerateResponse = useCallback((data) => {
    stop();
    setError(null);
    setTokens(data.tokens || []);
    setSuppressionEvents(data.suppression_events || []);
    setSummary(data.summary || null);
    setStatusMessage(null);
  }, [stop]);

  return {
    tokens,
    suppressionEvents,
    summary,
    streaming,
    error,
    statusMessage,
    start,
    stop,
    loadFromGenerateResponse,
  };
}
