export function TokenStream({ tokens, suppressionEvents, selectedIndex, onSelectToken }) {
  const suppressionIdx = new Set((suppressionEvents || []).map((e) => e.token_index));

  return (
    <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4" data-testid="token-stream-panel">
      <h3 className="mb-3 text-sm font-semibold text-slate-200">Token stream</h3>
      <div className="max-h-96 overflow-y-auto rounded-lg border border-white/5 bg-ink-950/50 p-3 font-mono text-sm leading-relaxed">
        {!tokens.length && <span className="text-slate-500">—</span>}
        {tokens.map((t, i) => {
          const v = t.affect?.valence ?? 0;
          const ar = t.affect?.arousal ?? 0.5;
          const hu = t.flags?.high_uncertainty;
          const sup = suppressionIdx.has(i);
          const heat = Math.max(0, Math.min(1, (v + 1) / 2));
          const bg = `rgba(${Math.round(255 * (1 - heat))}, ${Math.round(120 + 80 * heat)}, ${Math.round(
            160 + 40 * (1 - heat)
          )}, ${0.15 + 0.35 * ar})`;
          const active = selectedIndex === i;
          return (
            <button
              type="button"
              key={`${i}-${t.token_id}`}
              onClick={() => onSelectToken(i)}
              className={`mr-1 inline-block rounded px-0.5 transition ring-offset-2 ring-offset-ink-950 ${
                active ? "ring-2 ring-accent-cyan" : "hover:ring-1 hover:ring-white/30"
              }`}
              style={{ backgroundColor: bg }}
              title={`v=${v?.toFixed(3)} a=${ar?.toFixed(3)}`}
            >
              {hu && <span className="mr-0.5 text-amber-400">⚠</span>}
              {sup && <span className="mr-0.5 text-rose-400">●</span>}
              <span className="text-slate-100">{t.token_text}</span>
            </button>
          );
        })}
      </div>
      <p className="mt-2 text-[11px] text-slate-500">
        ⚠ high-uncertainty flag · ● layer-disagreement (suppression list). Click a token to inspect signals.
      </p>
    </div>
  );
}
