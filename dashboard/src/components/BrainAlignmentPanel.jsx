import { useState } from "react";

function RoiBar({ label, value }) {
  const pct = Math.max(0, Math.min(100, ((Number(value) + 1) / 2) * 100));
  return (
    <div className="mb-2">
      <div className="mb-0.5 flex justify-between font-mono text-[10px] text-slate-500">
        <span>{label}</span>
        <span>{Number(value).toFixed(3)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-ink-950">
        <div className="h-full rounded-full bg-accent-cyan/80 transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function BrainAlignmentPanel({ readout, summary }) {
  const [showNote, setShowNote] = useState(false);
  const [showTribeNote, setShowTribeNote] = useState(false);
  const tribe = readout?.tribe_v2;

  return (
    <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
      <h3 className="text-sm font-semibold text-slate-200">Brain alignment context</h3>
      <p className="mt-2 text-xs leading-relaxed text-slate-400">
        Probes map hidden states to valence / arousal / uncertainty readouts. Population psychology analogies are
        explicitly labeled - they are not claims that the language model has subjective experience.
      </p>

      {readout && (
        <div className="mt-4 space-y-2 rounded-lg border border-white/5 bg-ink-950/60 p-3 text-sm">
          <div>
            <span className="text-slate-500">Region tag · </span>
            <span className="font-medium text-slate-200">{readout.region}</span>
          </div>
          <div className="text-xs leading-relaxed text-slate-300">{readout.region_analog}</div>
          <button
            type="button"
            className="text-[11px] text-accent-cyan hover:underline"
            onMouseEnter={() => setShowNote(true)}
            onMouseLeave={() => setShowNote(false)}
            onFocus={() => setShowNote(true)}
            onBlur={() => setShowNote(false)}
          >
            Hover for brain_alignment_note
          </button>
          {showNote && (
            <p className="rounded border border-white/10 bg-ink-900 p-2 text-[11px] text-slate-400">
              {readout.brain_alignment_note}
            </p>
          )}
        </div>
      )}

      {tribe && (
        <div className="mt-4 rounded-lg border border-accent-cyan/20 bg-ink-950/50 p-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-accent-cyan">TRIBEv2 surrogate (ROI axes)</h4>
          <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
            Fixed projections from the same probed-layer states as the linear probe. Inspect ROI ladders alongside probe
            readouts - not literal TRIBE voxel predictions.
          </p>
          <div className="mt-3 space-y-1">
            {Object.entries(tribe.roi_scores || {}).map(([roi, v]) => (
              <RoiBar key={roi} label={roi} value={v} />
            ))}
          </div>
          <div className="mt-3 font-mono text-[11px] text-slate-400">
            derived VA (from ROI sketch): v {tribe.derived_va?.valence ?? "-"} · a {tribe.derived_va?.arousal ?? "-"}
          </div>
          <button
            type="button"
            className="mt-2 text-[11px] text-accent-cyan hover:underline"
            onMouseEnter={() => setShowTribeNote(true)}
            onMouseLeave={() => setShowTribeNote(false)}
            onFocus={() => setShowTribeNote(true)}
            onBlur={() => setShowTribeNote(false)}
          >
            methodology_note
          </button>
          {showTribeNote && (
            <p className="mt-2 rounded border border-white/10 bg-ink-900 p-2 text-[11px] text-slate-400">
              {tribe.methodology_note}
            </p>
          )}
        </div>
      )}

      {summary && (
        <div className="mt-4 font-mono text-[11px] text-slate-500">
          <div>mean valence {summary.mean_valence}</div>
          <div>mean arousal {summary.mean_arousal}</div>
          <div>mean uncertainty {summary.mean_uncertainty}</div>
          <div>dominant region {summary.dominant_region ?? "-"}</div>
          {summary.tribe_v2_mean_derived_va && (
            <div className="mt-2 border-t border-white/5 pt-2 text-slate-400">
              TRIBEv2 mean derived VA: v {summary.tribe_v2_mean_derived_va.valence} · a{" "}
              {summary.tribe_v2_mean_derived_va.arousal}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
