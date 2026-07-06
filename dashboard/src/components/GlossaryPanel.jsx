import { AnalysisCaption } from "./AnalysisCaption.jsx";

const TERMS = [
  {
    term: "Valence",
    def: "How positive (+) or negative (−) the readout is — like pleasant vs unpleasant, not moral judgment.",
  },
  {
    term: "Arousal",
    def: "How activated or intense the readout is (0 = calm, 1 = keyed-up).",
  },
  {
    term: "Uncertainty",
    def: "How unsure the model's next-token geometry looks — high U often means hedging or conflict.",
  },
];

export function GlossaryPanel() {
  return (
    <div className="rounded-lg border border-white/10 bg-ink-950/50 p-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Readout axes</h3>
      <AnalysisCaption>Plain-language meanings — these are probe projections, not feelings.</AnalysisCaption>
      <dl className="mt-2 space-y-2.5">
        {TERMS.map(({ term, def }) => (
          <div key={term}>
            <dt className="text-sm font-medium text-accent-cyan">{term}</dt>
            <dd className="text-xs leading-relaxed text-slate-400">{def}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
