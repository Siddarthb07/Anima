import { AnalysisCaption } from "./AnalysisCaption.jsx";

const MESSAGES = {
  valence_suppression: "Early layers looked more negative than the final layer on this token.",
  uncertainty_overclaim: "Final layer looked more confident than earlier layers suggested.",
};

export function LayerDisagreementPanel({ events, summary }) {
  const count = summary?.suppression_event_count ?? events?.length ?? 0;
  const recent = (events || []).slice(-3).reverse();

  return (
    <div className="rounded-xl border border-amber-500/20 bg-ink-900/40 p-4">
      <h3 className="text-sm font-semibold text-amber-100">Layer disagreement</h3>
      <AnalysisCaption>
        When probed layers disagree on valence or confidence — a heuristic flag, not “the model is lying.”
      </AnalysisCaption>
      <p className="mt-2 font-mono text-[11px] text-slate-400">
        {count} event{count === 1 ? "" : "s"}
        {summary?.high_uncertainty_token_count != null
          ? ` · ${summary.high_uncertainty_token_count} high-uncertainty tokens`
          : ""}
      </p>
      <div className="mt-2 space-y-2">
        {!recent.length && <p className="text-xs text-slate-500">No disagreements this run.</p>}
        {recent.map((ev, i) => (
          <div
            key={`${ev.token_index}-${i}`}
            className="rounded border border-white/5 bg-ink-950/60 px-2 py-1.5 text-[11px] text-slate-400"
          >
            <span className="font-mono text-accent-cyan">#{ev.token_index}</span>{" "}
            {MESSAGES[ev.suppression_type] || ev.suppression_type}
          </div>
        ))}
      </div>
    </div>
  );
}
