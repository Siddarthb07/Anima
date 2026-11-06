#!/usr/bin/env python3
"""τ-path occupancy certificate (novel dual-protocol shipping artefact).

For A(τ)=1[S≥τ], the KS occupancy theorem gives, for EVERY τ:
  gap(τ) := |P_L(A=1) − P_F(A=1)| ≤ KS(F_F, F_L).

Prior work (and our own transfer-gap g) reports a single operating point.
This module ships the full path:
  • gap(τ) and fixture/live abstain rates on a τ-grid
  • tightness ρ(τ) = gap(τ) / KS ∈ [0,1]  (0 if KS=0)
  • stress threshold τ* = argmax_τ ρ(τ)  — where the protocol shift bites hardest
  • shipping certificate: (KS, τ_ship, gap, ρ, τ*, ρ*)

Novelty claim (honest): selective prediction usually fixes one τ; conformal
calibrates τ for coverage. Packaging the *path* + tightness map as a dual-
protocol *shipping certificate* for abstain chrome is the contribution —
not a new inequality beyond KS.

Also: multi-guard KS fingerprint = vector of KS distances across guards;
protocol-shift signature when fixture vs live CDFs diverge differently per channel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from benchmarks.occupancy_ks_bound import (
    abstain_rate,
    evaluate_cell,
    fixture_quantile_tau,
    ks_distance,
    synthesize_demo_cells,
)

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER_ART = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact\prereg")
PAPER_FIG = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\figures")


@dataclass
class TauPathPoint:
    tau: float
    fixture_abstain: float
    live_abstain: float
    gap: float
    tightness: float


@dataclass
class TauPathCertificate:
    name: str
    channel: str
    ks: float
    n_fixture: int
    n_live: int
    tau_ship: float
    gap_ship: float
    tightness_ship: float
    tau_star: float
    tightness_star: float
    bound_holds_everywhere: bool
    path: list[TauPathPoint]
    notes: str = ""


def tau_grid_from_scores(
    fixture: Sequence[float],
    live: Sequence[float],
    *,
    n_grid: int = 64,
) -> np.ndarray:
    a = np.asarray(fixture, dtype=float)
    b = np.asarray(live, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    lo = float(np.min(np.concatenate([a, b])))
    hi = float(np.max(np.concatenate([a, b])))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([0.5], dtype=float)
    # Interior grid plus unique score atoms
    grid = np.linspace(lo, hi, num=n_grid)
    atoms = np.unique(np.concatenate([a, b]))
    if len(atoms) > 200:
        atoms = atoms[:: max(1, len(atoms) // 200)]
    return np.unique(np.concatenate([grid, atoms]))


def build_tau_path(
    name: str,
    fixture_scores: Sequence[float],
    live_scores: Sequence[float],
    *,
    tau_ship: Optional[float] = None,
    q_ship: float = 0.95,
    channel: str = "",
    n_grid: int = 64,
    tol: float = 1e-9,
) -> TauPathCertificate:
    fs = np.asarray(fixture_scores, dtype=float)
    ls = np.asarray(live_scores, dtype=float)
    fs = fs[np.isfinite(fs)]
    ls = ls[np.isfinite(ls)]
    ks = ks_distance(fs, ls)
    if tau_ship is None:
        tau_ship = fixture_quantile_tau(fs, q=q_ship)
    taus = tau_grid_from_scores(fs, ls, n_grid=n_grid)
    path: list[TauPathPoint] = []
    holds = True
    for t in taus:
        fr = abstain_rate(fs, float(t))
        lr = abstain_rate(ls, float(t))
        gap = abs(lr - fr)
        if np.isfinite(ks) and gap > ks + tol:
            holds = False
        tight = float(gap / ks) if (np.isfinite(ks) and ks > 0) else 0.0
        path.append(
            TauPathPoint(
                tau=float(t),
                fixture_abstain=fr,
                live_abstain=lr,
                gap=gap,
                tightness=tight,
            )
        )
    # Stress threshold: max tightness; tie-break toward shipped τ neighborhood
    best = max(path, key=lambda p: (p.tightness, -abs(p.tau - float(tau_ship))))
    fr_s = abstain_rate(fs, float(tau_ship))
    lr_s = abstain_rate(ls, float(tau_ship))
    gap_s = abs(lr_s - fr_s)
    tight_s = float(gap_s / ks) if (np.isfinite(ks) and ks > 0) else 0.0
    return TauPathCertificate(
        name=name,
        channel=channel,
        ks=float(ks),
        n_fixture=int(len(fs)),
        n_live=int(len(ls)),
        tau_ship=float(tau_ship),
        gap_ship=gap_s,
        tightness_ship=tight_s,
        tau_star=best.tau,
        tightness_star=best.tightness,
        bound_holds_everywhere=holds,
        path=path,
    )


def multi_guard_ks_fingerprint(
    fixture_by_guard: dict[str, Sequence[float]],
    live_by_guard: dict[str, Sequence[float]],
) -> dict[str, Any]:
    """KS vector across guards — protocol-shift fingerprint."""
    keys = sorted(set(fixture_by_guard) & set(live_by_guard))
    vec = {k: ks_distance(fixture_by_guard[k], live_by_guard[k]) for k in keys}
    finite_keys = [k for k in keys if np.isfinite(vec[k])]
    arr = np.array([vec[k] for k in finite_keys], dtype=float)
    return {
        "guards": keys,
        "ks_by_guard": vec,
        "ks_l2": float(np.linalg.norm(arr)) if len(arr) else float("nan"),
        "ks_max": float(np.max(arr)) if len(arr) else float("nan"),
        "dominant_guard": finite_keys[int(np.argmax(arr))] if len(arr) else None,
    }


def _cert_to_dict(c: TauPathCertificate) -> dict[str, Any]:
    d = asdict(c)
    # Cap path length in JSON for readability (full path still used for plots)
    if len(d["path"]) > 128:
        step = max(1, len(d["path"]) // 128)
        d["path"] = d["path"][::step]
        d["path_downsampled"] = True
    return d


def plot_tau_path(cert: TauPathCertificate, out_png: Path, out_pdf: Optional[Path] = None) -> None:
    import matplotlib.pyplot as plt

    taus = [p.tau for p in cert.path]
    gaps = [p.gap for p in cert.path]
    tights = [p.tightness for p in cert.path]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    ax[0].plot(taus, gaps, color="#1a1a1a", lw=1.6, label=r"gap$(\tau)$")
    ax[0].axhline(cert.ks, color="#c0392b", ls="--", lw=1.2, label=f"KS={cert.ks:.3f}")
    ax[0].axvline(cert.tau_ship, color="#2980b9", ls=":", lw=1.2, label=rf"$\tau_{{\mathrm{{ship}}}}$")
    ax[0].axvline(cert.tau_star, color="#27ae60", ls=":", lw=1.2, label=rf"$\tau^\star$")
    ax[0].set_xlabel(r"threshold $\tau$")
    ax[0].set_ylabel("occupancy gap")
    ax[0].set_title("τ-path occupancy (bound)")
    ax[0].legend(fontsize=7, loc="best")
    ax[0].text(
        0.02,
        0.98,
        "exploratory" if "proxy" in cert.name else "certificate",
        transform=ax[0].transAxes,
        va="top",
        fontsize=7,
        color="#666",
    )
    ax[1].plot(taus, tights, color="#1a1a1a", lw=1.6)
    ax[1].axvline(cert.tau_ship, color="#2980b9", ls=":", lw=1.2)
    ax[1].axvline(cert.tau_star, color="#27ae60", ls=":", lw=1.2)
    ax[1].set_xlabel(r"threshold $\tau$")
    ax[1].set_ylabel(r"tightness $\rho(\tau)=\mathrm{gap}/\mathrm{KS}$")
    ax[1].set_title("protocol stress map")
    ax[1].set_ylim(-0.05, 1.05)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def demo_certificate(seed: int = 0) -> TauPathCertificate:
    rng = np.random.default_rng(seed)
    fix = rng.normal(0.7, 0.08, size=400)
    live = rng.normal(0.3, 0.08, size=400)
    tau = float(np.quantile(fix, 0.95))
    return build_tau_path("synth_shifted_idle", fix, live, tau_ship=tau, channel="demo")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", type=Path, default=REPORTS / "occupancy_tau_path_certificate.json")
    p.add_argument("--fig-dir", type=Path, default=PAPER_FIG)
    p.add_argument("--also-paper", action="store_true")
    args = p.parse_args(argv)

    certs: list[TauPathCertificate] = []
    if args.demo:
        certs.append(demo_certificate())

    # Proxy-fixture cells from live reports (same honesty as occupancy_ks_bound)
    live_candidates = [
        ("qwen05_halu_composite", REPORTS / "halueval_live_qwen_qwen2_5-0_5b-instruct.json", "composite"),
        ("qwen05_halu_entropy", REPORTS / "halueval_live_qwen_qwen2_5-0_5b-instruct.json", "entropy"),
        ("qwen05_halu_invgap", REPORTS / "halueval_live_qwen_qwen2_5-0_5b-instruct.json", "inv_logit_gap"),
        (
            "qwen05_lenbal_composite",
            REPORTS / "halueval_live_qwen_qwen2_5-0_5b-instruct_lenbal_n300.json",
            "composite",
        ),
    ]
    from benchmarks.occupancy_ks_bound import _scores_from_rows

    for name, path, channel in live_candidates:
        if not path.exists():
            continue
        live = json.loads(path.read_text(encoding="utf-8"))
        rows = live["rows"] if isinstance(live, dict) else live
        ls = _scores_from_rows(rows, channel)
        arr = np.asarray(ls, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 20:
            continue
        proxy = 0.5 * arr + 0.45
        tau = fixture_quantile_tau(proxy, q=0.95)
        cert = build_tau_path(
            f"{name}_proxyfix",
            proxy,
            arr,
            tau_ship=tau,
            channel=channel,
        )
        cert.notes = "fixture=affine_proxy; replace with true Protocol-F encode"
        certs.append(cert)

    main_certs = [c for c in certs if "proxy" not in c.name]
    proxy_certs = [c for c in certs if "proxy" in c.name]
    payload = {
        "schema": "anima.occupancy_tau_path.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "novelty": (
            "τ-path occupancy certificate: gap(τ)≤KS for all τ; "
            "tightness ρ=gap/KS; stress τ*=argmax ρ; shipping tuple "
            "(KS, τ_ship, gap, ρ, τ*, ρ*)"
        ),
        "honesty": {
            "main_text_excludes_affine_proxy_fixtures": True,
            "proxy_cells_quarantined": len(proxy_certs),
        },
        "n_certificates": len(main_certs),
        "frac_bound_holds_everywhere": float(
            np.mean([c.bound_holds_everywhere for c in main_certs])
        )
        if main_certs
        else 0.0,
        "certificates": [_cert_to_dict(c) for c in main_certs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out} n_main={len(main_certs)} n_proxy_excluded={len(proxy_certs)}")
    certs = main_certs + proxy_certs  # keep proxies available for quarantine block

    # Paper figure: synthetic stress cell ONLY (no affine proxy fixtures).
    fig_certs = [c for c in certs if "proxy" not in c.name]
    if not fig_certs and certs:
        fig_certs = [demo_certificate()]
    if fig_certs:
        primary = next((c for c in fig_certs if "shifted" in c.name), fig_certs[0])
        png = args.fig_dir / "fig_ks_occupancy_gap.png"
        pdf = args.fig_dir / "fig_ks_occupancy_gap.pdf"
        plot_tau_path(primary, png, pdf)
        print(f"fig {png} (no_proxy primary={primary.name})")
    # Quarantine proxy cells in a separate exploratory receipt
    proxy_only = [c for c in certs if "proxy" in c.name]
    if proxy_only and args.also_paper:
        PAPER_ART.mkdir(parents=True, exist_ok=True)
        expl = {
            "schema": "anima.occupancy_tau_path.proxy_exploratory.v1",
            "warning": "AFFINE PROXY FIXTURES — not confirmatory; not for main-text figures",
            "shipping_claim": False,
            "n": len(proxy_only),
            "certificates": [_cert_to_dict(c) for c in proxy_only],
        }
        expl_path = PAPER_ART / "occupancy_tau_path_proxy_EXPLORATORY_ONLY.json"
        expl_path.write_text(json.dumps(expl, indent=2), encoding="utf-8")
        print(f"quarantined proxies → {expl_path}")

    if args.also_paper:
        PAPER_ART.mkdir(parents=True, exist_ok=True)
        dest = PAPER_ART / args.out.name
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"also {dest}")


if __name__ == "__main__":
    main()
