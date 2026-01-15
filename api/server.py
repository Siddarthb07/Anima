from pathlib import Path
from typing import Any, Dict

import asyncio
import torch
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from alignment.tribe_encoder import TRIBEv2Encoder, TRIBEv2_SURROGATE_NOTE, tribe_seed
from api.schemas import (
    AffectReadout,
    GenerateRequest,
    GenerateResponse,
    StreamDoneMessage,
    StreamErrorMessage,
    StreamTokenMessage,
    SuppressionEvent,
    TRIBEv2TokenSurrogate,
)
from core.extractor import ActivationExtractor
from core.suppression import detect_suppression
from probes.linear_probe import AffectProbe

app = FastAPI(title="anima", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


_registry: dict[str, tuple[ActivationExtractor, AffectProbe]] = {}
_tribe_registry: dict[tuple[str, int], TRIBEv2Encoder] = {}


def _probe_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace("-", "_")


def _zoo_path(slug: str) -> Path:
    return Path(__file__).resolve().parent.parent / "probes" / "zoo" / f"{slug}.pt"


def get_extractor_and_probe(model_name: str) -> tuple[ActivationExtractor, AffectProbe]:
    if model_name not in _registry:
        extractor = ActivationExtractor(model_name)
        n_layers = len(extractor.layer_indices)
        probe = AffectProbe(extractor.hidden_dim, n_layers)
        slug = _probe_slug(model_name)
        ckpt = _zoo_path(slug)
        if ckpt.exists():
            probe.load_state_dict(torch.load(ckpt, map_location="cpu"))
        probe.eval()
        _registry[model_name] = (extractor, probe)
    return _registry[model_name]


def get_tribe_encoder(model_name: str, hidden_dim: int) -> TRIBEv2Encoder:
    key = (model_name, int(hidden_dim))
    if key not in _tribe_registry:
        _tribe_registry[key] = TRIBEv2Encoder(int(hidden_dim), seed=tribe_seed(model_name))
    return _tribe_registry[key]


def label_region(valence: float, arousal: float, uncertainty: float) -> tuple[str, str]:
    if uncertainty > 0.75:
        return (
            "high-uncertainty",
            "No direct human analog — model-specific signal: low internal confidence",
        )
    if valence > 0.4 and arousal > 0.5:
        return (
            "high-positive-activation",
            "In human psychology, this region corresponds to: excitement, enthusiasm",
        )
    if valence > 0.4 and arousal <= 0.5:
        return (
            "low-positive-activation",
            "In human psychology, this region corresponds to: calm, contentment",
        )
    if valence <= -0.4 and arousal > 0.5:
        return (
            "high-negative-activation",
            "In human psychology, this region corresponds to: anxiety, stress",
        )
    if valence <= -0.4 and arousal <= 0.5:
        return (
            "low-negative-activation",
            "In human psychology, this region corresponds to: sadness, disengagement",
        )
    return ("neutral", "Neutral state — no strong dimensional signal")


def compute_flags(affect: dict[str, float]) -> dict[str, bool]:
    return {
        "high_uncertainty": affect["uncertainty"] > 0.75,
        "negative_valence": affect["valence"] < -0.4,
        "high_arousal": affect["arousal"] > 0.7,
        "likely_hedging": affect["uncertainty"] > 0.6 and affect["valence"] < 0,
    }


def tier(uncertainty: float) -> str:
    if uncertainty < 0.4:
        return "HIGH"
    if uncertainty < 0.7:
        return "MEDIUM"
    return "LOW"


def _readout_from_raw_one(result: dict, probe: AffectProbe, tribe: TRIBEv2Encoder) -> AffectReadout:
    affect = probe.predict(result["activations"])
    region, analog = label_region(**affect)
    roi_scores = tribe.encode_layer_activations(result["activations"])
    derived_va = tribe.derived_va_from_rois(roi_scores)
    tribe_block = TRIBEv2TokenSurrogate(
        roi_scores=roi_scores,
        derived_va=derived_va,
        methodology_note=TRIBEv2_SURROGATE_NOTE,
    )
    return AffectReadout(
        token_id=int(result["token_id"]),
        token_text=str(result["token_text"]),
        affect=affect,
        region=region,
        region_analog=analog,
        flags=compute_flags(affect),
        confidence_tier=tier(affect["uncertainty"]),
        uncertainty_signals={k: float(v) for k, v in result["uncertainty_signals"].items()},
        brain_alignment_note=(
            "Coordinates derive from a probe trained with temporal splits on neural encoding targets; "
            "region labels are population-level analogs, not subjective experience claims."
        ),
        tribe_v2=tribe_block,
    )


def _readouts_from_raw(raw_results: list, probe: AffectProbe, tribe: TRIBEv2Encoder) -> list[AffectReadout]:
    return [_readout_from_raw_one(r, probe, tribe) for r in raw_results]


def _summary(readouts: list[AffectReadout], suppression_events: list[SuppressionEvent]) -> Dict[str, Any]:
    if not readouts:
        return {
            "mean_valence": 0.0,
            "mean_arousal": 0.0,
            "mean_uncertainty": 0.0,
            "suppression_event_count": len(suppression_events),
            "high_uncertainty_token_count": 0,
            "dominant_region": "n/a",
        }
    regions = [r.region for r in readouts]
    dominant = max(set(regions), key=lambda x: sum(1 for r in readouts if r.region == x))
    n = len(readouts)
    out: Dict[str, Any] = {
        "mean_valence": round(sum(r.affect["valence"] for r in readouts) / n, 4),
        "mean_arousal": round(sum(r.affect["arousal"] for r in readouts) / n, 4),
        "mean_uncertainty": round(sum(r.affect["uncertainty"] for r in readouts) / n, 4),
        "suppression_event_count": len(suppression_events),
        "high_uncertainty_token_count": sum(1 for r in readouts if r.flags["high_uncertainty"]),
        "dominant_region": dominant,
    }
    rois = list(readouts[0].tribe_v2.roi_scores.keys())
    out["tribe_v2_mean_rois"] = {
        roi: round(sum(r.tribe_v2.roi_scores[roi] for r in readouts) / n, 4) for roi in rois
    }
    mv = sum(r.tribe_v2.derived_va["valence"] for r in readouts) / n
    ma = sum(r.tribe_v2.derived_va["arousal"] for r in readouts) / n
    out["tribe_v2_mean_derived_va"] = {"valence": round(mv, 4), "arousal": round(ma, 4)}
    return out


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    extractor, probe = get_extractor_and_probe(req.model)
    tribe = get_tribe_encoder(req.model, extractor.hidden_dim)
    raw_results = extractor.extract(req.prompt, req.max_new_tokens)

    readouts = _readouts_from_raw(raw_results, probe, tribe)

    suppression_events: list[SuppressionEvent] = []
    if req.detect_suppression:
        events = detect_suppression(raw_results, probe, extractor.early_layer, extractor.late_layer)
        suppression_events = [SuppressionEvent(**e) for e in events]

    summary = _summary(readouts, suppression_events)

    return GenerateResponse(
        model=req.model,
        prompt=req.prompt,
        tokens=readouts,
        suppression_events=suppression_events,
        summary=summary,
    )


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        model_name = data["model"]
        prompt = data["prompt"]
        max_new = int(data.get("max_new_tokens", 200))
        detect_suppression_flag = bool(data.get("detect_suppression", True))

        extractor, probe = get_extractor_and_probe(model_name)
        tribe = get_tribe_encoder(model_name, extractor.hidden_dim)
        raw_results: list = []
        readouts: list[AffectReadout] = []
        extract_iter_fn = getattr(extractor, "extract_iter", None)
        if callable(extract_iter_fn):
            for result in extract_iter_fn(prompt, max_new):
                raw_results.append(result)
                ro = _readout_from_raw_one(result, probe, tribe)
                readouts.append(ro)
                await websocket.send_text(StreamTokenMessage(readout=ro).model_dump_json())
                await asyncio.sleep(0)
        else:
            raw_results = extractor.extract(prompt, max_new)
            readouts = _readouts_from_raw(raw_results, probe, tribe)
            for r in readouts:
                await websocket.send_text(StreamTokenMessage(readout=r).model_dump_json())

        suppression_events: list[SuppressionEvent] = []
        if detect_suppression_flag:
            events = detect_suppression(raw_results, probe, extractor.early_layer, extractor.late_layer)
            suppression_events = [SuppressionEvent(**e) for e in events]

        summary = _summary(readouts, suppression_events)
        done = StreamDoneMessage(suppression_events=suppression_events, summary=summary)
        await websocket.send_text(done.model_dump_json())
    except Exception as exc:
        payload = StreamErrorMessage(message=f"{type(exc).__name__}: {exc}")
        try:
            await websocket.send_text(payload.model_dump_json())
        finally:
            await websocket.close(code=1011)
        return
    await websocket.close()
