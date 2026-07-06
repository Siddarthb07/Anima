"""Multi-judge council — scores benchmark manifests and live prompt readouts for validity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JudgeVerdict:
    judge: str
    score: float  # 0–100
    weight: float
    passed: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class CouncilReport:
    model: str
    aggregate_score: float
    passed: bool
    verdicts: list[JudgeVerdict]
    good_examples: list[dict[str, Any]] = field(default_factory=list)
    bad_examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdicts"] = [asdict(v) for v in self.verdicts]
        return d


def _entry(manifest: dict, name: str) -> dict | None:
    for e in manifest.get("entries") or []:
        if e.get("benchmark") == name:
            return e
    return None


def judge_schema_integrity(manifest: dict) -> JudgeVerdict:
    notes: list[str] = []
    score = 100.0
    if manifest.get("manifest_schema_version") != 1:
        score -= 40
        notes.append("missing or wrong manifest_schema_version")
    if not manifest.get("model"):
        score -= 30
        notes.append("missing model id")
    if not manifest.get("generated_at"):
        score -= 10
        notes.append("missing generated_at")
    entries = manifest.get("entries") or []
    if not entries:
        score -= 20
        notes.append("no benchmark entries")
    return JudgeVerdict(
        judge="schema_integrity",
        score=max(0, score),
        weight=0.15,
        passed=score >= 70,
        notes=notes,
    )


def judge_probe_signal(manifest: dict) -> JudgeVerdict:
    notes: list[str] = []
    score = 50.0
    go = _entry(manifest, "go_emotions")
    if go and go.get("status") == "ok":
        pv = float(go.get("pearson_valence") or 0)
        pa = float(go.get("pearson_arousal") or 0)
        if pv >= 0.15:
            score += 25
            notes.append(f"go_emotions valence r={pv:.3f} meets ≥0.15 gate")
        elif pv >= 0.10:
            score += 10
            notes.append(f"go_emotions valence r={pv:.3f} weak but above 0.10")
        else:
            score -= 15
            notes.append(f"go_emotions valence r={pv:.3f} below gate")
        if pa >= 0.10:
            score += 10
        if go.get("probe_origin") == "text_emotion":
            score += 5
    elif go and go.get("status") == "skipped":
        notes.append(f"go_emotions skipped: {go.get('reason', 'no text checkpoint')}")
        score -= 5
    narr = _entry(manifest, "narratives_holdout")
    if narr and narr.get("status") == "ok":
        rv = float(narr.get("val_r_valence") or 0)
        if rv >= 0.15:
            score += 20
            notes.append(f"brain holdout valence r={rv:.3f} strong")
        elif rv >= 0:
            score += 5
            notes.append(f"brain holdout valence r={rv:.3f} weak-positive")
        else:
            score -= 15
            notes.append(f"brain holdout valence r={rv:.3f} negative — synthetic brain tier limit")
    smoke = _entry(manifest, "smoke_extract")
    if smoke and smoke.get("status") == "ok":
        score += 10
        notes.append(f"smoke_extract ok ({smoke.get('n_tokens')} tokens)")
    return JudgeVerdict(
        judge="probe_signal",
        score=max(0, min(100, score)),
        weight=0.35,
        passed=score >= 55,
        notes=notes,
    )


def judge_honesty_flags(manifest: dict) -> JudgeVerdict:
    """Penalise overclaim patterns (synthetic guard AUROC 1.0, tiny fixture n)."""
    notes: list[str] = []
    score = 100.0
    for name in ("halueval", "truthfulqa_guard"):
        g = _entry(manifest, name)
        if not g or g.get("status") != "ok":
            continue
        n = int(g.get("n_samples") or 0)
        auroc = float(g.get("auroc_composite") or 0)
        acc = float(g.get("abstain_accuracy") or 0)
        if auroc >= 0.99 and acc >= 0.99:
            score -= 15
            notes.append(f"{name}: perfect AUROC/accuracy — label as fixture-policy smoke only")
        if n < 50:
            score -= 10
            notes.append(f"{name}: n={n} < 50 — expand fixtures before claiming generalisation")
    return JudgeVerdict(
        judge="honesty_flags",
        score=max(0, score),
        weight=0.20,
        passed=score >= 60,
        notes=notes,
    )


def judge_prompt_separation(examples: list[dict]) -> JudgeVerdict:
    """Score positive vs negative prompt mean-valence separation from live runs."""
    notes: list[str] = []
    score = 40.0
    by_id = {e["id"]: e for e in examples if e.get("id")}
    pos = by_id.get("positive", {})
    neg = by_id.get("negative", {})
    pv = pos.get("mean_valence")
    nv = neg.get("mean_valence")
    if pv is not None and nv is not None:
        gap = float(pv) - float(nv)
        if gap >= 0.25:
            score += 40
            notes.append(f"valence gap positive−negative = {gap:.3f} (strong separation)")
        elif gap >= 0.10:
            score += 20
            notes.append(f"valence gap {gap:.3f} (moderate)")
        else:
            score -= 10
            notes.append(f"valence gap {gap:.3f} (weak — thesis risk)")
        if float(pv) >= 0.2:
            score += 10
            notes.append(f"positive mean valence {pv:.3f} meets +0.2 gate")
        else:
            notes.append(f"positive mean valence {pv:.3f} below +0.2 gate")
        if float(nv) < -0.1:
            score += 10
            notes.append(f"negative mean valence {nv:.3f} below -0.1")
        else:
            notes.append(f"negative mean valence {nv:.3f} — weak negative separation")
    else:
        notes.append("missing positive/negative prompt runs")
        score -= 20
    return JudgeVerdict(
        judge="prompt_separation",
        score=max(0, min(100, score)),
        weight=0.30,
        passed=score >= 50,
        notes=notes,
    )


def classify_examples(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    good: list[dict] = []
    bad: list[dict] = []
    by_id = {e["id"]: e for e in examples if e.get("id")}
    pos = by_id.get("positive")
    neg = by_id.get("negative")
    if pos and float(pos.get("mean_valence", 0)) >= 0.15:
        good.append({**pos, "verdict": "expected positive valence"})
    elif pos:
        bad.append({**pos, "verdict": "positive prompt did not reach +0.15 mean valence"})
    if neg and float(neg.get("mean_valence", 0)) < -0.1:
        good.append({**neg, "verdict": "expected negative valence"})
    elif neg:
        bad.append({**neg, "verdict": "negative prompt failed to separate (valence not < -0.1)"})
    hedge = by_id.get("hedge_volatile")
    if hedge and float(hedge.get("stability_score", 1)) < 0.35:
        good.append({**hedge, "verdict": "hedge prompt flagged unstable — guard/stability working"})
    elif hedge:
        bad.append({**hedge, "verdict": "hedge prompt did not trigger low stability"})
    return good, bad


def score_manifest(
    manifest: dict,
    examples: list[dict] | None = None,
) -> CouncilReport:
    examples = examples or []
    verdicts = [
        judge_schema_integrity(manifest),
        judge_probe_signal(manifest),
        judge_honesty_flags(manifest),
        judge_prompt_separation(examples),
    ]
    total_w = sum(v.weight for v in verdicts)
    aggregate = sum(v.score * v.weight for v in verdicts) / total_w if total_w else 0
    good, bad = classify_examples(examples)
    return CouncilReport(
        model=manifest.get("model", "unknown"),
        aggregate_score=round(aggregate, 1),
        passed=aggregate >= 60 and all(v.passed for v in verdicts[:3]),
        verdicts=verdicts,
        good_examples=good,
        bad_examples=bad,
    )


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_council_json(report: CouncilReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
