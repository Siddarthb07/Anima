function TierBadge({ tier }) {
  const colors = {
    HIGH: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
    MEDIUM: "bg-amber-500/20 text-amber-200 border-amber-500/40",
    LOW: "bg-rose-500/20 text-rose-200 border-rose-500/40",
  };
  const cls = colors[tier] || colors.MEDIUM;
  return (
    <span className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${cls}`}>
      Confidence tier: {tier}
    </span>
  );
}

export function UncertaintyBar({ readout }) {
  if (!readout) {
    return (
      <div className="rounded-xl border border-white/10 bg-ink-900/40 p-5">
        <h3 className="text-base font-semibold text-slate-200">Uncertainty decomposition</h3>
        <p className="mt-2 text-sm text-slate-500">Select a token or stream output first.</p>
      </div>
    );
  }

  const u = readout.uncertainty_signals || {};
  const e = u.entropy ?? 0;
  const g = u.logit_gap ?? 0;
  const a = u.attn_entropy ?? 0;
  const fused = u.fused ?? readout.affect?.uncertainty ?? 0;
  const sum = Math.max(e + g + a, 1e-6);

  return (
    <div className="rounded-xl border border-white/10 bg-ink-900/40 p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-200">Uncertainty decomposition</h3>
        <TierBadge tier={readout.confidence_tier} />
      </div>

      {readout.guard && (
        <div className="mb-4 rounded border border-white/10 bg-ink-950/80 p-3 text-xs text-slate-400">
          <span className="font-semibold text-slate-300">Readout guard</span>
          {readout.guard.abstain_recommended && (
            <span className="ml-2 text-accent-rose">abstain recommended</span>
          )}
          {readout.guard.reasons?.length > 0 && (
            <p className="mt-1 font-mono text-[11px]">{readout.guard.reasons.join(" · ")}</p>
          )}
        </div>
      )}

      <div className="mb-2 flex h-5 overflow-hidden rounded-full bg-ink-950">
        <div className="bg-violet-500" style={{ width: `${(e / sum) * 100}%` }} title="entropy" />
        <div className="bg-sky-500" style={{ width: `${(g / sum) * 100}%` }} title="logit gap" />
        <div className="bg-teal-500" style={{ width: `${(a / sum) * 100}%` }} title="attention entropy" />
      </div>
      <div className="mb-4 flex justify-between font-mono text-xs text-slate-500">
        <span>entropy {e.toFixed(3)}</span>
        <span>logit-gap-unc {g.toFixed(3)}</span>
        <span>attn {a.toFixed(3)}</span>
      </div>

      <div>
        <div className="mb-1.5 flex justify-between text-sm text-slate-400">
          <span>Fused uncertainty</span>
          <span className="font-mono">{fused.toFixed(4)}</span>
        </div>
        <div className="h-2.5 rounded-full bg-ink-950">
          <div
            className="h-2.5 rounded-full bg-gradient-to-r from-accent-cyan to-accent-rose transition-all"
            style={{ width: `${Math.min(100, fused * 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
