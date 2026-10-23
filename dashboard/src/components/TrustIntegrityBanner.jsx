import { useEffect, useState } from "react";
import { apiBase } from "../apiBase.js";

const SLUG_BY_MODEL = {
  "Qwen/Qwen2.5-0.5B-Instruct": "qwen_qwen2_5-0_5b-instruct",
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama_tinyllama-1_1b-chat-v1_0",
};

/**
 * Sealed Trust Integrity Gate surface.
 * Never paints “trusted” from fixture AUROC alone — uses /trust-badge or run summary.
 */
export function TrustIntegrityBanner({ modelId, summary }) {
  const [remote, setRemote] = useState(null);

  useEffect(() => {
    if (!modelId) return;
    const base = apiBase();
    const slug = SLUG_BY_MODEL[modelId];
    const url = slug ? `${base}/trust-badge?slug=${encodeURIComponent(slug)}` : `${base}/trust-badge`;
    let cancelled = false;
    fetch(url)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!cancelled) setRemote(d);
      })
      .catch(() => {
        if (!cancelled) setRemote(null);
      });
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  const fromSummary = summary?.trust_integrity || null;
  const models = remote?.models || {};
  const slug =
    SLUG_BY_MODEL[modelId] ||
    Object.keys(models).find((k) => modelId?.toLowerCase().includes("qwen") && k.includes("qwen")) ||
    Object.keys(models).find(
      (k) => modelId?.toLowerCase().includes("tinyllama") && k.includes("tinyllama")
    );

  const entry = slug && models[slug] ? models[slug] : remote && !remote.models ? remote : null;

  const st = fromSummary
    ? {
        badge: summary.trust_badge || (fromSummary.trust_ui_claim_allowed ? "UNLOCKED" : "LOCKED"),
        claim: !!fromSummary.trust_ui_claim_allowed,
        op: fromSummary.operating_point,
        message: fromSummary.ui_message || fromSummary.message,
        fails: fromSummary.failed_gates || [],
        baseline: undefined,
        adapted: undefined,
      }
    : entry
      ? {
          badge: entry.badge,
          claim: !!entry.trust_ui_claim_allowed,
          op: entry.operating_point,
          message: entry.honesty?.message,
          fails: entry.failed_gates || [],
          baseline: entry.baseline_claim_allowed,
          adapted: entry.adapted_claim_allowed,
        }
      : null;

  if (!st) {
    return (
      <section
        data-testid="trust-integrity-banner"
        className="rounded-xl border border-amber-500/30 bg-amber-950/30 px-4 py-3 text-xs text-amber-100/90"
      >
        <p className="font-semibold text-amber-200">Trust Integrity</p>
        <p className="mt-1 text-amber-100/70">
          No sealed receipt for this model yet. Fixture AUROC alone never unlocks a green badge. See{" "}
          <code className="font-mono">docs/BENCHMARKS.md</code> and{" "}
          <code className="font-mono">GET /trust-badge</code>.
        </p>
      </section>
    );
  }

  const locked = !st.claim || String(st.badge).toUpperCase() === "LOCKED";
  const border = locked ? "border-rose-500/40 bg-rose-950/40" : "border-emerald-500/40 bg-emerald-950/30";
  const title = locked ? "Trust badge LOCKED" : "Trust badge UNLOCKED (adapted OP)";

  return (
    <section
      data-testid="trust-integrity-banner"
      className={`rounded-xl border px-4 py-3 text-xs ${border}`}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className={`text-sm font-bold tracking-wide ${locked ? "text-rose-300" : "text-emerald-300"}`}>
          {title}
        </p>
        {st.op && <span className="font-mono text-[10px] text-slate-400">{st.op}</span>}
      </div>
      <p className="mt-1 text-slate-300">
        {locked
          ? "Live Protocol L / polarity / transfer-gap gates not met for a production-style green shield. This UI must not imply the model is “trusted.”"
          : "Sealed adapted operating point allows trust-UI language. Baseline Protocol L collapse remains the scientific lead finding."}
      </p>
      {(st.baseline === false || st.adapted != null) && (
        <p className="mt-1 font-mono text-[10px] text-slate-500">
          baseline_claim={String(st.baseline)} · adapted_claim={String(st.adapted ?? st.claim)}
        </p>
      )}
      {st.message && <p className="mt-1 text-slate-400">{st.message}</p>}
      {locked && st.fails?.length > 0 && (
        <p className="mt-2 max-h-16 overflow-y-auto font-mono text-[10px] text-rose-200/80">
          {st.fails.slice(0, 6).join(" · ")}
          {st.fails.length > 6 ? "…" : ""}
        </p>
      )}
    </section>
  );
}
