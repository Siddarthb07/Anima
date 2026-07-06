import { AnalysisCaption } from "./AnalysisCaption.jsx";

function RoiBar({ label, value }) {
  const pct = Math.max(0, Math.min(100, ((Number(value) + 1) / 2) * 100));
  return (
    <div className="mb-2">
      <div className="mb-0.5 flex justify-between font-mono text-[10px] text-slate-500">
        <span>{label}</span>
        <span>{Number(value).toFixed(3)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-ink-950">
        <div className="h-full rounded-full bg-violet-400/70 transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function TribeSurrogatePanel({ readout }) {
  const tribe = readout?.tribe_v2;
  if (!tribe) {
    return (
      <div className="rounded-xl border border-violet-500/20 bg-ink-900/40 p-4">
        <h3 className="text-sm font-semibold text-violet-200">fMRI surrogate (TRIBE sketch)</h3>
        <AnalysisCaption>
          Optional brain-region sketch from fixed weights — analogy only, not a real scan.
        </AnalysisCaption>
        <p className="mt-3 text-xs text-slate-500">Stream tokens to see ROI bars.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-violet-500/25 bg-violet-950/20 p-4">
      <h3 className="text-sm font-semibold text-violet-200">fMRI surrogate (TRIBE sketch)</h3>
      <AnalysisCaption>
        Rough “which brain area might light up” sketch from hidden states — not literal voxel data.
      </AnalysisCaption>
      <div className="mt-3 space-y-1">
        {Object.entries(tribe.roi_scores || {}).map(([roi, v]) => (
          <RoiBar key={roi} label={roi} value={v} />
        ))}
      </div>
      <p className="mt-2 font-mono text-[10px] text-slate-500">
        sketch valence/arousal: v {tribe.derived_va?.valence ?? "—"} · a {tribe.derived_va?.arousal ?? "—"}
      </p>
    </div>
  );
}
