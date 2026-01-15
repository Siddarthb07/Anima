const MESSAGES = {
  valence_suppression:
    "Internal geometry was more negative earlier in depth than near the output — population-level analogy only.",
  uncertainty_overclaim:
    "Late-depth readout suggests more confidence than early-depth geometry — heuristic mismatch, not moral judgment.",
};

export function SuppressionAlert({ events, summary }) {
  return (
    <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Layer disagreement</h3>
      {summary && (
        <p className="mb-3 font-mono text-[11px] text-slate-400">
          Events: {summary.suppression_event_count ?? 0} · High-U tokens:{" "}
          {summary.high_uncertainty_token_count ?? 0}
        </p>
      )}
      <div className="max-h-96 space-y-2 overflow-y-auto">
        {!events?.length && <p className="text-sm text-slate-500">No events in this run.</p>}
        {events?.map((ev, i) => (
          <div
            key={`${ev.token_index}-${i}`}
            className={`rounded-lg border px-3 py-2 text-xs ${
              ev.severity === "HIGH"
                ? "border-rose-500/40 bg-rose-950/30"
                : "border-amber-500/30 bg-amber-950/20"
            }`}
          >
            <div className="font-mono text-accent-cyan">
              #{ev.token_index}{" "}
              <span className="text-slate-300">{ev.suppression_type}</span> · {ev.severity}
            </div>
            <div className="mt-1 text-slate-400">{MESSAGES[ev.suppression_type] || ev.suppression_type}</div>
            <div className="mt-1 font-mono text-[10px] text-slate-500">
              Δvalence {ev.valence_shift} · Δuncertainty {ev.uncertainty_shift}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
