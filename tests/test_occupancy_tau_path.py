"""Unit tests for τ-path occupancy certificate."""

from __future__ import annotations

import numpy as np

from benchmarks.occupancy_tau_path import (
    build_tau_path,
    demo_certificate,
    multi_guard_ks_fingerprint,
)


def test_bound_holds_everywhere_matched():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 300)
    cert = build_tau_path("m", a, a.copy(), tau_ship=0.7)
    assert cert.bound_holds_everywhere
    assert cert.ks < 0.05
    assert cert.tightness_star < 0.05


def test_shifted_idle_has_high_ks_and_tight_star():
    cert = demo_certificate(seed=1)
    assert cert.bound_holds_everywhere
    assert cert.ks > 0.5
    assert cert.gap_ship <= cert.ks + 1e-9
    # Novelty: shipped p95 τ can be LOOSE; stress τ* is where ρ peaks
    assert cert.tightness_star >= 0.8
    assert abs(cert.tau_star - cert.tau_ship) > 0.05


def test_fingerprint_dominant():
    rng = np.random.default_rng(2)
    fix = {"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)}
    live = {"a": rng.normal(0, 1, 100), "b": rng.normal(3, 1, 100)}
    fp = multi_guard_ks_fingerprint(fix, live)
    assert fp["dominant_guard"] == "b"
    assert fp["ks_by_guard"]["b"] > fp["ks_by_guard"]["a"]
