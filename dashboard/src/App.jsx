import { useCallback, useMemo, useState } from "react";
import { useAffectStream } from "./hooks/useAffectStream.js";
import { CircumplexPlot } from "./components/CircumplexPlot.jsx";
import { UncertaintyBar } from "./components/UncertaintyBar.jsx";
import { TokenStream } from "./components/TokenStream.jsx";
import { BrainAlignmentPanel } from "./components/BrainAlignmentPanel.jsx";
import { GeneratedOutput } from "./components/GeneratedOutput.jsx";
import { ModelCard } from "./components/ModelCard.jsx";
import { ModelSelector } from "./components/ModelSelector.jsx";
import { StabilityPanel } from "./components/StabilityPanel.jsx";
import { GlossaryPanel } from "./components/GlossaryPanel.jsx";
import { TribeSurrogatePanel } from "./components/TribeSurrogatePanel.jsx";
import { LayerDisagreementPanel } from "./components/LayerDisagreementPanel.jsx";
import { useRestGenerate } from "./hooks/useRestGenerate.js";

function defaultModelId() {
  const v = import.meta.env.VITE_DEFAULT_MODEL;
  return typeof v === "string" && v.trim() ? v.trim() : "Qwen/Qwen2.5-0.5B-Instruct";
}

const DEFAULT_API_HTTP = "http://127.0.0.1:8010";

function httpTargetToWsBase(http) {
  const t = http.replace(/\/$/, "");
  if (!t) return "";
  return t.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
}

function wsBaseUrl() {
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
  const [guardMode, setGuardMode] = useState("gate");
  const [interventionMode, setInterventionMode] = useState("none");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const { generate: restGenerate, loading: restLoading } = useRestGenerate();

  const selected = selectedIdx != null ? tokens[selectedIdx] : tokens[tokens.length - 1];

  const run = useCallback(() => {
    setSelectedIdx(null);
    start(model, prompt, maxTok, detectSuppression, guardMode, interventionMode);
  }, [model, prompt, maxTok, detectSuppression, guardMode, interventionMode, start]);

  const runRest = useCallback(async () => {
    setSelectedIdx(null);
    const data = await restGenerate(model, prompt, maxTok, detectSuppression, guardMode, interventionMode);
    if (data?.tokens) loadFromGenerateResponse(data);
  }, [model, prompt, maxTok, detectSuppression, guardMode, interventionMode, restGenerate, loadFromGenerateResponse]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-ink-950 via-ink-900 to-slate-950 text-[15px]">
      <header className="border-b border-white/10 bg-ink-900/80 px-6 py-4 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">anima</h1>
            <p className="text-sm text-slate-400">
              Dimensional readout dashboard — not claims that the model &quot;feels&quot; anything.
            </p>
          </div>
          <div className="font-mono text-xs text-slate-500">WS {wsUrl}</div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-6 py-8">
        <section className="rounded-xl border border-white/10 bg-ink-900/60 p-5 shadow-xl">
          <div className="grid gap-4 md:grid-cols-[1fr_2fr_auto] md:items-end">
            <ModelSelector value={model} onChange={setModel} />
            <label className="block text-sm">
              <span className="mb-1 block text-slate-400">Prompt</span>
              <textarea
                data-testid="prompt-input"
                rows={2}
                className="w-full resize-y rounded-lg border border-white/10 bg-ink-950 px-3 py-2.5 text-base outline-none ring-accent-cyan focus:ring-1"
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
                  className="w-24 rounded-lg border border-white/10 bg-ink-950 px-2 py-2.5 font-mono text-base"
                  value={maxTok}
                  onChange={(e) => setMaxTok(Number(e.target.value))}
                />
              </label>
              <button
                data-testid="stream-btn"
                type="button"
                disabled={streaming}
                onClick={run}
                className="mt-6 rounded-lg bg-accent-cyan px-4 py-2.5 text-sm font-semibold text-ink-950 hover:bg-cyan-300 disabled:opacity-40"
              >
                {streaming ? "Streaming…" : "Stream readout"}
              </button>
              <button
                data-testid="stop-btn"
                type="button"
                onClick={stop}
                className="mt-6 rounded-lg border border-white/20 px-4 py-2.5 text-sm text-slate-300 hover:bg-white/5"
              >
                Stop
              </button>
              <button
                type="button"
                disabled={streaming || restLoading}
                onClick={runRest}
                className="mt-6 rounded-lg border border-accent-cyan/40 px-4 py-2.5 text-sm text-accent-cyan hover:bg-accent-cyan/10 disabled:opacity-40"
              >
                REST batch
              </button>
            </div>
          </div>

          <button
            type="button"
            className="mt-3 text-xs text-accent-cyan hover:underline"
            onClick={() => setShowAdvanced((v) => !v)}
          >
            {showAdvanced ? "Hide" : "Show"} guard & layer options
          </button>
          {showAdvanced && (
            <div className="mt-2 flex flex-wrap gap-4 rounded-lg border border-white/5 bg-ink-950/50 p-3 text-sm text-slate-400">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={detectSuppression}
                  onChange={(e) => setDetectSuppression(e.target.checked)}
                />
                Detect layer disagreement (suppression)
              </label>
              <label className="flex items-center gap-2">
                Guard mode
                <select
                  className="rounded border border-white/10 bg-ink-950 px-2 py-1 font-mono text-slate-300"
                  value={guardMode}
                  onChange={(e) => setGuardMode(e.target.value)}
                >
                  <option value="observe">observe</option>
                  <option value="gate">gate</option>
                </select>
              </label>
              <label className="flex items-center gap-2">
                Intervention
                <select
                  className="rounded border border-white/10 bg-ink-950 px-2 py-1 font-mono text-slate-300"
                  value={interventionMode}
                  onChange={(e) => setInterventionMode(e.target.value)}
                >
                  <option value="none">none</option>
                  <option value="dampen">dampen</option>
                </select>
              </label>
            </div>
          )}

          {error && <p className="mt-3 text-sm text-accent-rose">{error}</p>}
          {!error && statusMessage && <p className="mt-3 text-sm text-slate-400">{statusMessage}</p>}
        </section>

        <div className="grid gap-6 lg:grid-cols-2">
          <CircumplexPlot tokens={tokens} highlightIndex={selectedIdx ?? tokens.length - 1} />
          <div className="space-y-4">
            <UncertaintyBar readout={selected} />
            <BrainAlignmentPanel readout={selected} />
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            <GeneratedOutput tokens={tokens} />
            <TokenStream
              tokens={tokens}
              suppressionEvents={suppressionEvents}
              selectedIndex={selectedIdx}
              onSelectToken={setSelectedIdx}
            />
          </div>
          <div className="space-y-4">
            <GlossaryPanel />
            <StabilityPanel summary={summary} />
            <LayerDisagreementPanel events={suppressionEvents} summary={summary} />
            <TribeSurrogatePanel readout={selected} />
            <ModelCard summary={summary} />
          </div>
        </div>
      </main>
    </div>
  );
}
