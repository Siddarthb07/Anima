import { useCallback, useMemo, useState } from "react";
import { useAffectStream } from "./hooks/useAffectStream.js";
import { CircumplexPlot } from "./components/CircumplexPlot.jsx";
import { UncertaintyBar } from "./components/UncertaintyBar.jsx";
import { TokenStream } from "./components/TokenStream.jsx";
import { SuppressionAlert } from "./components/SuppressionAlert.jsx";
import { BrainAlignmentPanel } from "./components/BrainAlignmentPanel.jsx";
import { GeneratedOutput } from "./components/GeneratedOutput.jsx";
import { ModelCard } from "./components/ModelCard.jsx";
import { useRestGenerate } from "./hooks/useRestGenerate.js";

function defaultModelId() {
  const v = import.meta.env.VITE_DEFAULT_MODEL;
  return typeof v === "string" && v.trim() ? v.trim() : "hf-internal-testing/tiny-random-gpt2";
}

const DEFAULT_API_HTTP = "http://127.0.0.1:8010";

function httpTargetToWsBase(http) {
  const t = http.replace(/\/$/, "");
  if (!t) return "";
  return t.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
}

function wsBaseUrl() {
  // Dev: same-origin ws(s)://<vite-host> → Vite proxies /ws to VITE_API_HTTP_TARGET (vite.config.js).
  // Ignores VITE_WS_BASE unless VITE_WS_DIRECT=1 (avoids accidental direct sockets to the API port).
  if (import.meta.env.DEV) {
    const forceDirect =
      import.meta.env.VITE_WS_DIRECT === "1" || import.meta.env.VITE_WS_DIRECT === "true";
    if (!forceDirect) {
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      return `${proto}://${window.location.host}`;
    }
  }

  const raw = typeof import.meta.env.VITE_WS_BASE === "string" ? import.meta.env.VITE_WS_BASE.trim() : "";
  if (raw) return raw.replace(/\/$/, "");

  const apiHttp =
    typeof import.meta.env.VITE_API_HTTP_TARGET === "string"
      ? import.meta.env.VITE_API_HTTP_TARGET.trim()
      : "";
  const fromHttp = httpTargetToWsBase(apiHttp);
  if (fromHttp) return fromHttp;

  let host = window.location.hostname;
  if (host === "localhost") host = "127.0.0.1";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${host}:8010`;
}

export default function App() {
  const wsUrl = useMemo(() => wsBaseUrl(), []);
  const {
    tokens,
    suppressionEvents,
    summary,
    streaming,
    error,
    statusMessage,
    start,
    stop,
    loadFromGenerateResponse,
  } = useAffectStream(wsUrl);

  const [model, setModel] = useState(defaultModelId);
  const [prompt, setPrompt] = useState("Say hello in a few tokens.");
  const [maxTok, setMaxTok] = useState(24);
  const [detectSuppression, setDetectSuppression] = useState(true);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const { generate: restGenerate, loading: restLoading } = useRestGenerate();

  const selected = selectedIdx != null ? tokens[selectedIdx] : tokens[tokens.length - 1];

  const run = useCallback(() => {
    setSelectedIdx(null);
    start(model, prompt, maxTok, detectSuppression);
  }, [model, prompt, maxTok, detectSuppression, start]);

  const runRest = useCallback(async () => {
    setSelectedIdx(null);
    const data = await restGenerate(model, prompt, maxTok, detectSuppression);
    if (data?.tokens) loadFromGenerateResponse(data);
  }, [model, prompt, maxTok, detectSuppression, restGenerate, loadFromGenerateResponse]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-ink-950 via-ink-900 to-slate-950">
      <header className="border-b border-white/10 bg-ink-900/80 backdrop-blur px-6 py-4">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">anima</h1>
            <p className="text-xs text-slate-400">
              Dimensional readout dashboard - not claims that the model &quot;feels&quot; anything.
            </p>
          </div>
          <div className="font-mono text-[11px] text-slate-500">WS {wsUrl}</div>
        </div>
      </header>

      {import.meta.env.DEV ? (
        <div className="border-b border-white/10 bg-slate-900/90 px-6 py-2 text-center text-[11px] text-slate-400">
          {import.meta.env.VITE_WS_DIRECT === "1" || import.meta.env.VITE_WS_DIRECT === "true" ? (
            <>
              Dev mode (direct WebSocket): connecting to{" "}
              <span className="font-mono text-slate-300">{wsUrl}</span> — API must listen on that host/port (no Vite
              proxy).
            </>
          ) : (
            <>
              Dev mode: WebSocket is{" "}
              <span className="font-mono text-slate-300">
                {typeof window !== "undefined" ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}` : ""}
                /ws/...
              </span>{" "}
              (proxied to{" "}
              <span className="font-mono text-slate-300">
                {(import.meta.env.VITE_API_HTTP_TARGET || DEFAULT_API_HTTP).replace(/\/$/, "")}
              </span>
              ). Start uvicorn there, then restart <span className="font-mono">npm run dev</span> after changing{" "}
              <span className="font-mono">.env</span>. To bypass the proxy, set{" "}
              <span className="font-mono">VITE_WS_DIRECT=1</span> and <span className="font-mono">VITE_WS_BASE</span>.
            </>
          )}
        </div>
      ) : null}

      <main className="mx-auto max-w-7xl space-y-6 px-6 py-8">
        <section className="rounded-xl border border-white/10 bg-ink-900/60 p-5 shadow-xl">
          <div className="grid gap-4 md:grid-cols-[1fr_2fr_auto] md:items-end">
            <label className="block text-sm">
              <span className="mb-1 block text-slate-400">HuggingFace model id</span>
              <input
                data-testid="model-input"
                className="w-full rounded-lg border border-white/10 bg-ink-950 px-3 py-2 font-mono text-sm outline-none ring-accent-cyan focus:ring-1"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="hf-internal-testing/tiny-random-gpt2"
              />
              <p className="mt-1 text-[11px] leading-snug text-slate-500">
                Default is the tiny HF regression weights (low RAM, reliable loads). Switch to{" "}
                <span className="font-mono text-slate-400">distilgpt2</span> for coherent English if your machine can
                hold the weights.
              </p>
            </label>
            <label className="block text-sm md:col-span-1">
              <span className="mb-1 block text-slate-400">Prompt</span>
              <textarea
                data-testid="prompt-input"
                rows={2}
                className="w-full resize-y rounded-lg border border-white/10 bg-ink-950 px-3 py-2 text-sm outline-none ring-accent-cyan focus:ring-1"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </label>
            <div className="flex flex-wrap gap-2">
              <label className="text-sm">
                <span className="mb-1 block text-slate-400">Max new tokens</span>
                <input
                  data-testid="max-tokens"
                  type="number"
                  min={1}
                  max={512}
                  className="w-24 rounded-lg border border-white/10 bg-ink-950 px-2 py-2 font-mono text-sm"
                  value={maxTok}
                  onChange={(e) => setMaxTok(Number(e.target.value))}
                />
              </label>
              <button
                data-testid="stream-btn"
                type="button"
                disabled={streaming}
                onClick={run}
                className="mt-6 rounded-lg bg-accent-cyan px-4 py-2 text-sm font-semibold text-ink-950 hover:bg-cyan-300 disabled:opacity-40"
              >
                {streaming ? "Streaming…" : "Stream readout"}
              </button>
              <button
                data-testid="stop-btn"
                type="button"
                onClick={stop}
                className="mt-6 rounded-lg border border-white/20 px-4 py-2 text-sm text-slate-300 hover:bg-white/5"
              >
                Stop
              </button>
              <button
                type="button"
                disabled={streaming || restLoading}
                onClick={runRest}
                className="mt-6 rounded-lg border border-accent-cyan/40 px-4 py-2 text-sm text-accent-cyan hover:bg-accent-cyan/10 disabled:opacity-40"
              >
                REST batch
              </button>
            </div>
          </div>
          <label className="mt-2 flex items-center gap-2 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={detectSuppression}
              onChange={(e) => setDetectSuppression(e.target.checked)}
            />
            Detect layer disagreement (suppression)
          </label>
          {error && <p className="mt-3 text-sm text-accent-rose">{error}</p>}
          {!error && statusMessage && (
            <p className="mt-3 text-sm text-slate-400">{statusMessage}</p>
          )}
        </section>

        <div className="grid gap-6 lg:grid-cols-2">
          <CircumplexPlot tokens={tokens} highlightIndex={selectedIdx ?? tokens.length - 1} />
          <div className="space-y-4">
            <UncertaintyBar readout={selected} />
            <BrainAlignmentPanel readout={selected} summary={summary} />
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-4">
            <GeneratedOutput tokens={tokens} />
            <TokenStream
              tokens={tokens}
              suppressionEvents={suppressionEvents}
              selectedIndex={selectedIdx}
              onSelectToken={setSelectedIdx}
            />
          </div>
          <div className="space-y-4">
            <ModelCard summary={summary} />
            <SuppressionAlert events={suppressionEvents} summary={summary} />
          </div>
        </div>
      </main>
    </div>
  );
}
