import { useMemo } from "react";

/** Concatenated decode of streamed **generated** tokens (subword splits preserved). */
export function GeneratedOutput({ tokens }) {
  const text = useMemo(() => (tokens || []).map((t) => t.token_text ?? "").join(""), [tokens]);

  return (
    <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Model output (decoded stream)</h3>
      <p className="mb-2 text-[11px] text-slate-500">
        Tokens shown below are concatenated here as plain text for readability (tokenizer spacing may differ from a
        chat UI).
      </p>
      <div className="min-h-[3rem] whitespace-pre-wrap rounded-lg border border-white/5 bg-ink-950/70 p-3 font-mono text-sm leading-relaxed text-slate-100">
        {text.length ? text : <span className="text-slate-600">...</span>}
      </div>
    </div>
  );
}
