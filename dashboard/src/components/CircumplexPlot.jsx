import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

function QuadrantLabels() {
  return (
    <div className="pointer-events-none absolute inset-0 text-[10px] font-medium uppercase tracking-wide text-slate-500">
      <span className="absolute left-4 top-8">Low arousal</span>
      <span className="absolute right-4 top-8">Excited</span>
      <span className="absolute bottom-10 left-4">Sad / flat</span>
      <span className="absolute bottom-10 right-4">Stressed</span>
    </div>
  );
}

export function CircumplexPlot({ tokens, highlightIndex }) {
  const data = tokens.map((t, i) => ({
    valence: t.affect.valence,
    arousal: t.affect.arousal,
    idx: i,
    label: (t.token_text || "").slice(0, 48),
    uncertainty: t.affect.uncertainty,
    highlight: i === highlightIndex,
  }));

  if (!data.length) {
    return (
      <div className="relative flex h-80 items-center justify-center rounded-xl border border-white/10 bg-ink-900/40">
        <QuadrantLabels />
        <p className="text-sm text-slate-500">Run a stream to plot valence × arousal.</p>
      </div>
    );
  }

  return (
    <div className="relative rounded-xl border border-white/10 bg-ink-900/40 p-3 shadow-inner">
      <QuadrantLabels />
      <h3 className="relative z-10 mb-2 text-sm font-semibold text-slate-200">Circumplex (readout)</h3>
      <div className="relative z-10 h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
            <XAxis
              type="number"
              dataKey="valence"
              domain={[-1, 1]}
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              label={{ value: "Valence (− / +)", fill: "#64748b", fontSize: 11, position: "bottom" }}
            />
            <YAxis
              type="number"
              dataKey="arousal"
              domain={[0, 1]}
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              label={{
                value: "Arousal",
                fill: "#64748b",
                fontSize: 11,
                angle: -90,
                position: "insideLeft",
              }}
            />
            <ZAxis range={[40, 120]} />
            <ReferenceLine x={0} stroke="#475569" strokeDasharray="4 4" />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload;
                return (
                  <div className="rounded-lg border border-white/10 bg-ink-950/95 px-3 py-2 text-xs shadow-xl">
                    <div className="font-mono text-accent-cyan">#{p.idx}</div>
                    <div className="mt-1 max-w-xs whitespace-pre-wrap text-slate-200">{p.label}</div>
                    <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-[11px] text-slate-400">
                      <span>v {p.valence?.toFixed(3)}</span>
                      <span>a {p.arousal?.toFixed(3)}</span>
                      <span>u {p.uncertainty?.toFixed(3)}</span>
                    </div>
                  </div>
                );
              }}
            />
            <Scatter
              name="tokens"
              data={data.filter((d) => !d.highlight)}
              fill="#64748b"
              fillOpacity={0.35}
            />
            <Scatter name="current" data={data.filter((d) => d.highlight)} fill="#22d3ee" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
