"""
Trust Badge Console — operator-facing integrity view (CLI + HTML + API helper).

Shows claim_allowed, transfer gaps, misleading-green rates, polarity, GreenShieldBench.
Honesty: badge is LOCKED when TIG fails; never paints green from fixture AUROC alone.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPORTS = Path(__file__).resolve().parent / "reports"

LOCKED_MODELS = [
    ("qwen_qwen2_5-0_5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("tinyllama_tinyllama-1_1b-chat-v1_0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
]


def load_console_state(slug: str, reports: Path = REPORTS) -> dict[str, Any]:
    tig_base = _json(reports / f"trust_integrity_{slug}.json")
    tig_ad = _json(reports / f"trust_integrity_adapted_{slug}.json")
    gsb_base = _json(reports / f"greenshield_bench_{slug}.json")
    gsb_ad = _json(reports / f"greenshield_bench_adapted_{slug}.json")
    pol = _json(reports / f"polarity_diagnostic_{slug}.json")
    lp = _json(reports / f"protocol_l_prime_{slug}.json")
    live = _json(reports / f"live_guard_{slug}.json")

    # Operator badge follows adapted L′ when present; baseline collapse stays visible.
    tig = tig_ad or tig_base
    gsb = gsb_ad or gsb_base
    claim = bool((tig or {}).get("trust_ui_claim_allowed", False))
    badge = "UNLOCKED" if claim else "LOCKED"
    color = "#2e7d32" if claim else "#b71c1c"
    op = (tig or {}).get("operating_point") or "protocol_L_baseline"

    suites_ui = {}
    for suite in ("halueval", "truthfulqa"):
        live_b = ((live or {}).get("benchmarks") or {}).get(suite) or {}
        lp_s = ((lp or {}).get("suites") or {}).get(suite) or {}
        gsb_s = ((gsb or {}).get("suites") or {}).get(suite) or {}
        pol_s = ((pol or {}).get("suites") or {}).get(suite) or {}
        mgr = (lp_s.get("misleading_green") or gsb_s.get("misleading_green") or {})
        suites_ui[suite] = {
            "baseline_live_auroc": live_b.get("auroc_composite"),
            "l_prime_loqo_auroc": lp_s.get("auroc_loqo") or gsb_s.get("protocol_l_auroc"),
            "fixture_auroc": gsb_s.get("protocol_f_auroc"),
            "transfer_gap_same_scorer": lp_s.get("transfer_gap_same_scorer")
            or gsb_s.get("transfer_gap"),
            "misleading_green_rate": mgr.get("misleading_green_rate"),
            "polarity_verdict_baseline": pol_s.get("verdict"),
            "polarity_verdict_l_prime": lp_s.get("polarity_verdict"),
            "never_abstains_baseline": live_b.get("never_abstains"),
        }

    return {
        "schema": "anima.trust_badge_console.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_slug": slug,
        "badge": badge,
        "badge_color": color,
        "operating_point": op,
        "trust_ui_claim_allowed": claim,
        "baseline_claim_allowed": bool((tig_base or {}).get("trust_ui_claim_allowed", False)),
        "adapted_claim_allowed": bool((tig_ad or {}).get("trust_ui_claim_allowed", False)),
        "failed_gates": (tig or {}).get("failed_gates") or [],
        "seal_sha256": (tig or {}).get("seal_sha256"),
        "greenshield_passed": (gsb or {}).get("passed"),
        "greenshield_failed_checks": (gsb or {}).get("failed_checks") or [],
        "suites": suites_ui,
        "honesty": {
            "fixture_auroc_cannot_unlock": True,
            "not_user_study": True,
            "not_hallucination_sota": True,
            "baseline_collapse_preserved": True,
            "message": (
                "Badge follows Protocol L′ TIG when present. "
                "Baseline Protocol L remains locked on these models (collapse finding). "
                "Fixture AUROC alone never unlocks."
            ),
        },
    }


def _json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_html(states: list[dict[str, Any]]) -> str:
    cards = []
    for s in states:
        rows = "".join(
            f"<tr><td>{suite}</td>"
            f"<td>{v.get('baseline_live_auroc')}</td>"
            f"<td>{v.get('l_prime_loqo_auroc')}</td>"
            f"<td>{v.get('transfer_gap_same_scorer')}</td>"
            f"<td>{v.get('misleading_green_rate')}</td>"
            f"<td>{v.get('polarity_verdict_l_prime') or v.get('polarity_verdict_baseline')}</td></tr>"
            for suite, v in (s.get("suites") or {}).items()
        )
        fails = ", ".join(s.get("failed_gates") or []) or "(none)"
        cards.append(
            f"""
<section class="card" style="border-left:6px solid {s['badge_color']}">
  <h2>{s['model_slug']}</h2>
  <p class="badge" style="color:{s['badge_color']}">{s['badge']}</p>
  <p>op=<code>{s.get('operating_point')}</code>
     · adapted claim=<code>{s.get('adapted_claim_allowed')}</code>
     · baseline claim=<code>{s.get('baseline_claim_allowed')}</code>
     · GSB=<code>{s.get('greenshield_passed')}</code></p>
  <p class="fails">Failed gates: {fails}</p>
  <table>
    <thead><tr><th>Suite</th><th>Baseline L</th><th>L′ LOQO</th><th>Δ F′−L′</th><th>MGR</th><th>Polarity</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p class="note">{s['honesty']['message']}</p>
</section>
"""
        )
    body = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Anima Trust Badge Console</title>
<style>
  body {{ font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background:#0f1218; color:#e8eaed; margin:0; padding:2rem; }}
  h1 {{ font-weight:600; letter-spacing:.02em; }}
  .sub {{ color:#9aa0a6; max-width:52rem; }}
  .card {{ background:#1a1f2a; margin:1.5rem 0; padding:1.25rem 1.5rem; border-radius:4px; }}
  .badge {{ font-size:1.75rem; font-weight:700; margin:.25rem 0 1rem; }}
  table {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:.9rem; }}
  th, td {{ text-align:left; padding:.4rem .5rem; border-bottom:1px solid #2a3140; }}
  .note, .fails {{ color:#bdc1c6; font-size:.85rem; }}
  code {{ background:#0f1218; padding:.1rem .35rem; border-radius:3px; }}
</style>
</head>
<body>
  <h1>Anima Trust Badge Console</h1>
  <p class="sub">Operator view of Trust Integrity Gate + GreenShieldBench.
  Fixture AUROC alone cannot unlock a green badge. Not a user study; not hallucination SOTA.</p>
  {body}
  <p class="sub">Generated {datetime.now(timezone.utc).isoformat()}</p>
</body>
</html>
"""


def write_console(reports: Path = REPORTS, slugs: Optional[list[str]] = None) -> dict[str, Any]:
    slugs = slugs or [s for s, _ in LOCKED_MODELS]
    states = [load_console_state(s, reports) for s in slugs]
    reports.mkdir(parents=True, exist_ok=True)
    rollup = {
        "schema": "anima.trust_badge_console_rollup.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {s["model_slug"]: s for s in states},
        "any_unlocked": any(s["trust_ui_claim_allowed"] for s in states),
    }
    json_path = reports / "trust_badge_console.json"
    html_path = reports / "trust_badge_console.html"
    json_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    html_path.write_text(render_html(states), encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path), "rollup": rollup}


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Trust Badge Console")
    p.add_argument("--all-locked", action="store_true")
    p.add_argument("--slug", default="")
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--print", dest="do_print", action="store_true")
    args = p.parse_args(argv)
    if args.all_locked or not args.slug:
        slugs = [s for s, _ in LOCKED_MODELS]
    else:
        slugs = [args.slug]
    out = write_console(args.reports_dir, slugs)
    print(f"Wrote {out['json']}")
    print(f"Wrote {out['html']}")
    if args.do_print:
        for slug, st in out["rollup"]["models"].items():
            print(
                f"  [{slug}] badge={st['badge']} claim={st['trust_ui_claim_allowed']} "
                f"gsb={st.get('greenshield_passed')} fails={len(st.get('failed_gates') or [])}"
            )


if __name__ == "__main__":
    main()
