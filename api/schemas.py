from typing import Any, Literal

from pydantic import BaseModel, Field

from core.defaults import DEFAULT_CAUSAL_LM


class SuppressionEvent(BaseModel):
    token_index: int
    token_text: str
    suppression_type: str
    early_affect: dict[str, float]
    late_affect: dict[str, float]
    valence_shift: float
    uncertainty_shift: float
    severity: str


class TRIBEv2TokenSurrogate(BaseModel):
    """Per-token surrogate ROI ladder computed alongside probe readouts."""

    roi_scores: dict[str, float]
    derived_va: dict[str, float] = Field(
        ...,
        description="Valence/arousal sketch inferred from surrogate ROI scalars (not duplicate of probe heads).",
    )
    methodology_note: str


class AffectReadout(BaseModel):
    token_id: int
    token_text: str
    affect: dict[str, float]
    region: str
    region_analog: str
    flags: dict[str, bool]
    confidence_tier: str
    uncertainty_signals: dict[str, float]
    brain_alignment_note: str
    tribe_v2: TRIBEv2TokenSurrogate


class GenerateRequest(BaseModel):
    model: str = DEFAULT_CAUSAL_LM
    prompt: str
    max_new_tokens: int = 200
    detect_suppression: bool = True


class GenerateResponse(BaseModel):
    model: str
    prompt: str
    tokens: list[AffectReadout]
    suppression_events: list[SuppressionEvent]
    summary: dict[str, Any]


class StreamTokenMessage(BaseModel):
    kind: Literal["token"] = "token"
    readout: AffectReadout


class StreamDoneMessage(BaseModel):
    kind: Literal["done"] = "done"
    suppression_events: list[SuppressionEvent]
    summary: dict[str, Any]


class StreamErrorMessage(BaseModel):
    kind: Literal["error"] = "error"
    message: str
