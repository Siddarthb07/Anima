"""Unit tests for KS occupancy-gap theorem."""

from __future__ import annotations

import numpy as np

from benchmarks.occupancy_ks_bound import (
    abstain_rate,
    evaluate_cell,
    fixture_quantile_tau,
    ks_distance,
    synthesize_demo_cells,
    transfer_gap_g,
)


def test_ks_identical_zero():
    x = np.linspace(0, 1, 50)
    assert ks_distance(x, x) == 0.0


def test_ks_shifted_positive():
    a = np.zeros(100)
    b = np.ones(100)
    assert ks_distance(a, b) == 1.0


def test_bound_holds_matched():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 300)
    cell = evaluate_cell("m", a, a.copy(), tau=0.7)
    assert cell.occupancy_gap <= cell.ks + 1e-9
    assert cell.bound_holds


def test_bound_holds_shifted_idle():
    rng = np.random.default_rng(1)
    fix = rng.normal(0.75, 0.05, 500)
    live = rng.normal(0.2, 0.05, 500)
    tau = fixture_quantile_tau(fix, 0.95)
    cell = evaluate_cell("idle", fix, live, tau=tau)
    assert cell.live_abstain_rate == 0.0
    assert cell.fixture_abstain_rate > 0.0
    assert cell.bound_holds
    assert cell.ks + 1e-9 >= cell.occupancy_gap
    # Idle reject corollary: g < 1
    assert transfer_gap_g(live, tau) < 1.0


def test_empirical_quantile_abstain_near_one_minus_q():
    rng = np.random.default_rng(2)
    s = rng.uniform(0, 1, 2000)
    q = 0.95
    tau = fixture_quantile_tau(s, q)
    rate = abstain_rate(s, tau)
    assert abs(rate - (1 - q)) < 0.02


def test_demo_cells_all_hold():
    cells = synthesize_demo_cells(0)
    assert all(c.bound_holds for c in cells)
