from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import asyncio
import os
import torch
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from alignment.tribe_encoder import TRIBEv2Encoder, TRIBEv2_SURROGATE_NOTE, tribe_seed
from alignment.tribe_runtime import get_tribe_mode, predict_text_roi_summary
from api.schemas import (
    AffectReadout,
    EncodeRequest,
    EncodeResponse,
    GenerateRequest,
    GenerateResponse,
    GuardInfo,
    ModelInfo,
    ModelsResponse,
    StreamDoneMessage,
    StreamErrorMessage,
    StreamTokenMessage,
    SuppressionEvent,
    TRIBEv2TokenSurrogate,
)
from core.extractor import ActivationExtractor
from core.guard import evaluate_guard, region_under_guard
from core.layer_config import LAYER_CONFIG
from core.regions import compute_flags, confidence_tier_from_fused
from core.suppression import detect_suppression
from probes.linear_probe import AffectProbe
from probes.zoo_io import calib_path, load_meta, load_probe_into, meta_path, probe_slug, tribe_weights_path

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for ex, _ in list(_registry.values()):
        try:
            ex.cleanup()
        except Exception:
            pass
    _registry.clear()
    _tribe_registry.clear()
    _probe_meta_cache.clear()
    _calib_cache.clear()


app = FastAPI(title="anima", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ANIMA_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


_registry: dict[str, tuple[ActivationExtractor, AffectProbe]] = {}
_tribe_registry: dict[tuple[str, int], TRIBEv2Encoder] = {}
_probe_meta_cache: dict[str, dict[str, Any]] = {}
_calib_cache: dict[str, Optional[Any]] = {}


def _load_calibrator(slug: str, suffix: str = "") -> Optional[Any]:
    key = f"{slug}{suffix}"
    if key in _calib_cache:
        return _calib_cache[key]
    from probes.calibration import PlattScaler

    for suf in ("_narratives_pca", "_text", ""):
        p = calib_path(slug, suf)
        if p.exists():
            cal = PlattScaler()
            cal.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            cal.eval()
            _calib_cache[key] = cal
            return cal
    _calib_cache[key] = None
    return None


def get_extractor_and_probe(model_name: str) -> tuple[ActivationExtractor, AffectProbe, dict[str, Any]]:
    if model_name not in _registry:
        if model_name not in LAYER_CONFIG:
            raise HTTPException(status_code=400, detail=f"unknown_model:{model_name}")
        extractor = ActivationExtractor(model_name)
        n_layers = len(extractor.layer_indices)
        probe = AffectProbe(extractor.hidden_dim, n_layers)
        meta = load_probe_into(probe, model_name)
        _probe_meta_cache[model_name] = meta
        _registry[model_name] = (extractor, probe)
    ex, pr = _registry[model_name]
    return ex, pr, _probe_meta_cache.get(model_name, {})


def get_tribe_encoder(model_name: str, hidden_dim: int) -> TRIBEv2Encoder:
    key = (model_name, int(hidden_dim))
    if key not in _tribe_registry:
        slug = probe_slug(model_name)
        wpath = tribe_weights_path(slug)
        enc = TRIBEv2Encoder(
            int(hidden_dim),
            seed=tribe_seed(model_name),
            weights_path=str(wpath) if wpath.exists() else None,
        )
        _tribe_registry[key] = enc
    return _tribe_registry[key]


def _tribe_block(
    result: dict,
    tribe: TRIBEv2Encoder,
    model_name: str,
    prompt: str,
) -> TRIBEv2TokenSurrogate:
    mode = get_tribe_mode()
    roi_scores = tribe.encode_layer_activations(result["activations"])
    note = TRIBEv2_SURROGATE_NOTE
    if tribe.mode == "surrogate_trained":
        note = "Trained surrogate TRIBEv2 ROI projections from LM hidden states (not voxel TRIBE)."
    if mode in ("runtime", "blend"):
        rt = predict_text_roi_summary(prompt, cache_key=probe_slug(model_name))
        if rt:
            if mode == "blend":
                roi_scores = {
                    k: round(0.5 * roi_scores[k] + 0.5 * rt.get(k, roi_scores[k]), 4)
                    for k in roi_scores
                }
            else:
                roi_scores = {k: round(float(rt.get(k, roi_scores[k])), 4) for k in roi_scores}
            note = "TRIBEv2 runtime ROI summary (optional) blended with surrogate when ANIMA_TRIBE_MODE=blend."
    derived_va = tribe.derived_va_from_rois(roi_scores)
    return TRIBEv2TokenSurrogate(
        roi_scores=roi_scores,
        derived_va=derived_va,
        methodology_note=note,
    )


def _readout_from_raw_one(
    result: dict,
    probe: AffectProbe,
    tribe: TRIBEv2Encoder,
    *,
    model_name: str,
    prompt: str,
    probe_meta: dict[str, Any],
) -> AffectReadout:
    affect = probe.predict(result["activations"])
    slug = probe_slug(model_name)
    cal = _load_calibrator(slug)
    unc = {k: float(v) for k, v in result["uncertainty_signals"].items()}
    guard = evaluate_guard(
        affect=affect,
        uncertainty_signals=unc,
        token_text=str(result.get("token_text", "")),
        calibrator=cal,
    )
    region, analog = region_under_guard(
        affect["valence"],
        affect["arousal"],
        affect["uncertainty"],
        guard,
    )
    tribe_block = _tribe_block(result, tribe, model_name, prompt)
    tier = confidence_tier_from_fused(unc.get("fused", 0.5))
    return AffectReadout(
        token_id=int(result["token_id"]),
        token_text=str(result["token_text"]),
        affect=affect,
        region=region,
        region_analog=analog,
        flags=compute_flags(affect),
        confidence_tier=tier,
        uncertainty_signals=unc,
        brain_alignment_note=(
            "Coordinates derive from a probe trained with temporal splits on neural encoding targets; "
            "region labels are population-level analogs, not subjective experience claims."
        ),
        tribe_v2=tribe_block,
        guard=GuardInfo(**guard.to_dict()),
    )


def _readouts_from_raw(
    raw_results: list,
    probe: AffectProbe,
    tribe: TRIBEv2Encoder,
    *,
    model_name: str,
    prompt: str,
    probe_meta: dict[str, Any],
) -> list[AffectReadout]:
    return [
        _readout_from_raw_one(
            r, probe, tribe, model_name=model_name, prompt=prompt, probe_meta=probe_meta
        )
        for r in raw_results
    ]


def _summary(
    readouts: list[AffectReadout],
    suppression_events: list[SuppressionEvent],
    probe_meta: dict[str, Any],
) -> Dict[str, Any]:
    if not readouts:
        return {
            "mean_valence": 0.0,
            "mean_arousal": 0.0,
            "mean_uncertainty": 0.0,
            "suppression_event_count": len(suppression_events),
            "high_uncertainty_token_count": 0,
            "dominant_region": "n/a",
            "probe_origin": probe_meta.get("probe_origin", "random"),
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
        "probe_origin": probe_meta.get("probe_origin", "random"),
        "probe_checkpoint": probe_meta.get("checkpoint"),
        "guard_abstain_count": sum(1 for r in readouts if r.guard.abstain_recommended),
    }
    rois = list(readouts[0].tribe_v2.roi_scores.keys())
    out["tribe_v2_mean_rois"] = {
        roi: round(sum(r.tribe_v2.roi_scores[roi] for r in readouts) / n, 4) for roi in rois
    }
    mv = sum(r.tribe_v2.derived_va["valence"] for r in readouts) / n
    ma = sum(r.tribe_v2.derived_va["arousal"] for r in readouts) / n
    out["tribe_v2_mean_derived_va"] = {"valence": round(mv, 4), "arousal": round(ma, 4)}
    return out


@app.get("/models", response_model=ModelsResponse)
def list_models():
    from probes.zoo_io import checkpoint_path

    models = []
    for mid in sorted(LAYER_CONFIG.keys()):
        slug = probe_slug(mid)
        zoo = []
        origin = "random"
        for suf in ("_narratives_pca", "_text", ""):
            if checkpoint_path(slug, suf).exists():
                zoo.append(suf or "default")
                meta = load_meta(slug, suf)
                origin = str(meta.get("probe_origin", "zoo"))
        if not zoo and meta_path(slug).exists():
            origin = str(load_meta(slug).get("probe_origin", "random"))
        models.append(
            ModelInfo(
                model_id=mid,
                hidden_dim=LAYER_CONFIG[mid]["hidden_dim"],
                layers=len(LAYER_CONFIG[mid]["layers"]),
                has_sae=bool(LAYER_CONFIG[mid].get("has_sae")),
                zoo_checkpoints=zoo,
                probe_origin=origin,
            )
        )
    return ModelsResponse(models=models)


@app.post("/encode", response_model=EncodeResponse)
async def encode_stimulus(req: EncodeRequest):
    extractor, probe, meta = get_extractor_and_probe(req.model)
    tribe = get_tribe_encoder(req.model, extractor.hidden_dim)
    raw = extractor.encode_sequence(req.text, max_length=req.max_length)
    readouts = _readouts_from_raw(
        raw, probe, tribe, model_name=req.model, prompt=req.text, probe_meta=meta
    )
    return EncodeResponse(model=req.model, text=req.text, tokens=readouts, summary=_summary(readouts, [], meta))


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    extractor, probe, meta = get_extractor_and_probe(req.model)
    tribe = get_tribe_encoder(req.model, extractor.hidden_dim)
    raw_results = extractor.extract(req.prompt, req.max_new_tokens)
    readouts = _readouts_from_raw(
        raw_results, probe, tribe, model_name=req.model, prompt=req.prompt, probe_meta=meta
    )
    suppression_events: list[SuppressionEvent] = []
    if req.detect_suppression:
        events = detect_suppression(raw_results, probe, extractor.early_layer, extractor.late_layer)
        suppression_events = [SuppressionEvent(**e) for e in events]
    summary = _summary(readouts, suppression_events, meta)
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

        extractor, probe, meta = get_extractor_and_probe(model_name)
        tribe = get_tribe_encoder(model_name, extractor.hidden_dim)
        raw_results: list = []
        readouts: list[AffectReadout] = []
        for result in extractor.extract_iter(prompt, max_new):
            raw_results.append(result)
            ro = _readout_from_raw_one(
                result,
                probe,
                tribe,
                model_name=model_name,
                prompt=prompt,
                probe_meta=meta,
            )
            readouts.append(ro)
            await websocket.send_text(StreamTokenMessage(readout=ro).model_dump_json())
            await asyncio.sleep(0)

        suppression_events: list[SuppressionEvent] = []
        if detect_suppression_flag:
            events = detect_suppression(raw_results, probe, extractor.early_layer, extractor.late_layer)
            suppression_events = [SuppressionEvent(**e) for e in events]

        summary = _summary(readouts, suppression_events, meta)
        done = StreamDoneMessage(suppression_events=suppression_events, summary=summary)
        await websocket.send_text(done.model_dump_json())
    except HTTPException as exc:
        payload = StreamErrorMessage(message=f"http_error:{exc.detail}")
        await websocket.send_text(payload.model_dump_json())
        await websocket.close(code=1008)
        return
    except Exception as exc:
        payload = StreamErrorMessage(message=f"{type(exc).__name__}: {exc}")
        try:
            await websocket.send_text(payload.model_dump_json())
        finally:
            await websocket.close(code=1011)
        return
    await websocket.close()
