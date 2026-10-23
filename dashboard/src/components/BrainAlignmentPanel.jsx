import { AnalysisCaption } from "./AnalysisCaption.jsx";

export function BrainAlignmentPanel({ readout }) {
  if (!readout) {
    return (
      <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
        <h3 className="text-sm font-semibold text-slate-200">Psychology analogy</h3>
        <AnalysisCaption>
          Population-level region tag mapped from valence/arousal — not subjective experience or voxel fMRI.
        </AnalysisCaption>
        <p className="mt-3 text-xs text-slate-500">Select a token to see its analogy tag.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
      <h3 className="text-sm font-semibold text-slate-200">Psychology analogy</h3>
      <AnalysisCaption>
        Population-level region tag mapped from valence/arousal — not subjective experience. Real
        Narratives depth: stimulus encoding can correlate with BOLD; PCA→VA holdout does not equal
        brain emotion.
      </AnalysisCaption>
      <div className="mt-3 rounded-lg border border-white/5 bg-ink-950/60 p-3">
        <div className="text-xs text-slate-500">Region tag</div>
        <div className="text-sm font-medium text-slate-100">{readout.region}</div>
        <p className="mt-2 text-[11px] leading-relaxed text-slate-400">{readout.region_analog}</p>
      </div>
      <p className="mt-2 text-[10px] text-slate-500">
        TRIBEv2 panel = linear sketch of the same activations, not voxel fMRI decoding.
      </p>
    </div>
  );
}
