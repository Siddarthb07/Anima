import { AnalysisCaption } from "./AnalysisCaption.jsx";

export function StabilityPanel({ summary }) {
  if (!summary || summary.stability_score == null) {
    return (
      <section className="rounded-xl border border-white/10 bg-ink-900/60 p-4 text-xs text-slate-500">
        <h3 className="font-semibold text-slate-300">Readout stability</h3>
        <AnalysisCaption>How much valence swings token-to-token — low score means choppy readouts.</AnalysisCaption>
        <p className="mt-2">Run a stream first.</p>
      </section>
    );
  }

  const score = Number(summary.stability_score);
  const pct = Math.round(score * 100);
  const unstable = summary.unstable_token_count ?? 0;
  const gated = summary.guard_gated_token_count ?? 0;

  return (
    <section className="rounded-xl border border-white/10 bg-ink-900/60 p-4 text-xs text-slate-400">
      <h3 className="font-semibold text-slate-200">Readout stability</h3>
      <AnalysisCaption>How much valence swings token-to-token — low score means choppy readouts.</AnalysisCaption>      <div className="mb-2 h-2 overflow-hidden rounded-full bg-ink-950">
        <div
          className="h-full rounded-full bg-accent-cyan transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p>
        Score <span className="font-mono text-slate-200">{score.toFixed(3)}</span>
        {summary.max_valence_swing != null && (
          <> · max swing <span className="font-mono text-slate-300">{summary.max_valence_swing}</span></>
        )}
      </p>
      <p className="mt-1">
        Unstable windows: <span className="text-slate-200">{unstable}</span>
        {gated > 0 && (
          <> · gated tokens: <span className="text-accent-rose">{gated}</span></>
        )}
      </p>
      {summary.guard_mode && (
        <p className="mt-1 font-mono text-[10px] text-slate-500">
          guard={summary.guard_mode}
          {summary.intervention_mode ? ` · intervention=${summary.intervention_mode}` : ""}
        </p>
      )}
    </section>
  );
}
