import { useEffect, useState } from "react";

const apiHttp =
  (import.meta.env.VITE_API_HTTP_TARGET || "http://127.0.0.1:8010").replace(/\/$/, "");

export function ModelCard({ summary }) {
  const [models, setModels] = useState([]);

  useEffect(() => {
    fetch(`${apiHttp}/models`)
      .then((r) => r.json())
      .then((d) => setModels(d.models || []))
      .catch(() => setModels([]));
  }, []);

  return (
    <section className="rounded-xl border border-white/10 bg-ink-900/60 p-4 text-xs text-slate-400">
      <h3 className="mb-2 font-semibold text-slate-200">Model card</h3>
      {summary?.probe_origin && (
        <p>
          Loaded probe: <span className="font-mono text-slate-300">{summary.probe_origin}</span>
          {summary.guard_abstain_count != null && (
            <> · guard abstains: {summary.guard_abstain_count}</>
          )}
        </p>
      )}
      {(summary?.trust_badge || summary?.trust_integrity) && (
        <p className="mt-1">
          Trust badge:{" "}
          <span
            className={`font-mono ${
              String(summary.trust_badge || "").toLowerCase() === "locked" ||
              !summary.trust_integrity?.trust_ui_claim_allowed
                ? "text-rose-300"
                : "text-emerald-300"
            }`}
          >
            {summary.trust_badge ||
              (summary.trust_integrity?.trust_ui_claim_allowed ? "unlocked" : "locked")}
          </span>
          {summary.trust_integrity?.operating_point && (
            <> · OP <span className="font-mono text-slate-300">{summary.trust_integrity.operating_point}</span></>
          )}
        </p>
      )}
      <ul className="mt-2 max-h-32 overflow-y-auto font-mono text-[10px]">
        {models.map((m) => (
          <li key={m.model_id} className="border-b border-white/5 py-1">
            {m.model_id}
            {m.zoo_checkpoints?.length ? ` [${m.zoo_checkpoints.join(", ")}]` : " [random probe]"}
            {m.probe_origin ? ` · ${m.probe_origin}` : ""}
            {m.brain_data_tier && m.brain_data_tier !== "none" ? ` · brain:${m.brain_data_tier}` : ""}
          </li>
        ))}
      </ul>
    </section>
  );
}
