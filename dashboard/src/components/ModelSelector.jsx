import { useEffect, useMemo, useState } from "react";
import { apiBase } from "../apiBase.js";

/** CPU-tier models shown when /models is unreachable (matches docker-up + train_text_zoo). */
const FALLBACK_CPU_MODELS = [
  "Qwen/Qwen2.5-0.5B-Instruct",
  "distilgpt2",
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  "hf-internal-testing/tiny-random-gpt2",
];

function hasTrainedProbe(m) {
  return (m.zoo_checkpoints || []).length > 0;
}

function optionLabel(m) {
  const id = typeof m === "string" ? m : m.model_id;
  if (typeof m === "string") return m;
  const parts = [id];
  if (m.probe_origin && m.probe_origin !== "random") {
    parts.push(m.probe_origin);
  } else if (!hasTrainedProbe(m)) {
    parts.push("random probe");
  }
  const r = m.text_val_pearson_valence ?? m.brain_val_r_valence ?? m.val_pearson_valence;
  if (r != null && !Number.isNaN(Number(r))) {
    parts.push(`r_v=${Number(r).toFixed(2)}`);
  }
  return parts.join(" · ");
}

export function ModelSelector({ value, onChange }) {
  const [models, setModels] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`${apiBase()}/models`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => {
        setModels(d.models || []);
        setError(null);
      })
      .catch((e) => {
        setError(String(e));
        setModels(FALLBACK_CPU_MODELS.map((id) => ({ model_id: id, probe_origin: "unknown" })));
      })
      .finally(() => setLoading(false));
  }, []);

  const { trained, untrained } = useMemo(() => {
    const t = [];
    const u = [];
    for (const m of models) {
      if (hasTrainedProbe(m)) t.push(m);
      else u.push(m);
    }
    t.sort((a, b) => a.model_id.localeCompare(b.model_id));
    u.sort((a, b) => a.model_id.localeCompare(b.model_id));
    return { trained: t, untrained: u };
  }, [models]);

  const showGroups = trained.length > 0 && untrained.length > 0;

  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-slate-300">Model</span>
      <select
        data-testid="model-select"
        className="model-select w-full rounded-lg border border-white/15 bg-ink-950 px-3 py-3 font-sans text-lg font-medium leading-snug text-slate-100 outline-none ring-accent-cyan focus:border-accent-cyan/40 focus:ring-2"
        value={value}
        disabled={loading && models.length === 0}
        onChange={(e) => onChange(e.target.value)}
      >
        {loading && models.length === 0 ? (
          <option value={value}>Loading models…</option>
        ) : null}
        {showGroups ? (
          <>
            <optgroup label="Trained probes">
              {trained.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {optionLabel(m)}
                </option>
              ))}
            </optgroup>
            <optgroup label="Random probe — run anima train-text">
              {untrained.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {optionLabel(m)}
                </option>
              ))}
            </optgroup>
          </>
        ) : (
          models.map((m) => (
            <option key={m.model_id} value={m.model_id}>
              {optionLabel(m)}
            </option>
          ))
        )}
      </select>
      {error && (
        <p className="mt-1 text-xs text-amber-400/90">
          API /models unreachable — showing CPU fallback list. Start API on port 8010.
        </p>
      )}
    </label>
  );
}
