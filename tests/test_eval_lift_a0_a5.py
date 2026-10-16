"""IEEE-grade regression tests for A0–A5 eval lift."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci
from scripts.polarity_diagnostic import diagnose_rows
from scripts.protocol_r_retune import evaluate_suite, question_holdout_split

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "benchmarks" / "reports"
QWEN_SLUG = "qwen_qwen2_5-0_5b-instruct"
HALU = REPORTS / f"halueval_live_{QWEN_SLUG}.json"
TQA = REPORTS / f"truthfulqa_live_{QWEN_SLUG}.json"


def test_auroc_polarity_identity():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    a = auroc_from_scores(scores, labels)
    b = auroc_from_scores(scores, [1 - y for y in labels])
    assert abs(a + b - 1.0) < 1e-9


def test_bootstrap_ci_shape():
    scores = [0.1, 0.2, 0.3, 0.8, 0.85, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    ci = bootstrap_auroc_ci(scores, labels, B=200, seed=42)
    assert ci["n"] == 6
    assert ci["B"] == 200
    assert ci["seed"] == 42
    assert 0.0 <= ci["lo"] <= ci["point"] <= ci["hi"] <= 1.0


@pytest.mark.skipif(not HALU.is_file(), reason="locked HaluEval live JSON missing")
def test_polarity_halueval_anti_aligned():
    rows = json.loads(HALU.read_text(encoding="utf-8"))["rows"]
    diag = diagnose_rows(rows, B=200, seed=42)
    assert diag["auroc_original"] < 0.5
    assert diag["abs_delta_flip_vs_one_minus_original"] <= 0.01
    assert diag["verdict"] == "anti_aligned_ranking"


@pytest.mark.skipif(not HALU.is_file(), reason="locked HaluEval live JSON missing")
def test_qwen_halueval_auroc_locked():
    rows = json.loads(HALU.read_text(encoding="utf-8"))["rows"]
    scores = [float(r["composite_score"]) for r in rows]
    labels = [int(bool(r["is_hallucination"])) for r in rows]
    auroc = auroc_from_scores(scores, labels)
    assert abs(auroc - 0.2469) < 1e-3
    ci = bootstrap_auroc_ci(scores, labels, B=2000, seed=42)
    # Locked paper CI [0.139, 0.369] — allow small float tolerance
    assert abs(ci["lo"] - 0.139) < 0.02
    assert abs(ci["hi"] - 0.369) < 0.02


@pytest.mark.skipif(not TQA.is_file(), reason="locked TruthfulQA live JSON missing")
def test_qwen_truthfulqa_auroc_locked():
    rows = json.loads(TQA.read_text(encoding="utf-8"))["rows"]
    scores = [float(r["composite_score"]) for r in rows]
    labels = [int(bool(r["should_abstain"])) for r in rows]
    auroc = auroc_from_scores(scores, labels)
    assert abs(auroc - 0.4587) < 1e-3


def test_heldout_no_question_leakage():
    questions = ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"]
    tune, test = question_holdout_split(questions, seed=42, frac_tune=0.5)
    tune_q = {questions[i] for i in range(len(questions)) if tune[i]}
    test_q = {questions[i] for i in range(len(questions)) if test[i]}
    assert tune_q.isdisjoint(test_q)
    assert tune.any() and test.any()


@pytest.mark.skipif(not HALU.is_file(), reason="locked HaluEval live JSON missing")
def test_protocol_r_qwen_halueval_best_acc_chance():
    block = evaluate_suite(HALU, seed=42, held_out=True)
    assert block["protocol_r"]["best_acc"] == pytest.approx(0.5, abs=0.02)
    assert block["protocol_r"]["optimistic"] is True
    assert "protocol_r_prime" in block
    assert block["protocol_r_prime"]["held_out"] is True
    # no question leakage in split sizes
    assert block["protocol_r_prime"]["n_tune"] + block["protocol_r_prime"]["n_test"] == block["n"]
