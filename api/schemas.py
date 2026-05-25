from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.defaults import DEFAULT_CAUSAL_LM


class GuardInfo(BaseModel):
    tier: str
    abstain_recommended: bool
    composite_score: float
    reasons: list[str] = Field(default_factory=list)


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
    roi_scores: dict[str, float]
    derived_va: dict[str, float] = Field(
        ...,
        description="Valence/arousal sketch inferred from surrogate ROI scalars.",
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
    guard: GuardInfo = Field(default_factory=lambda: GuardInfo(tier="MEDIUM", abstain_recommended=False, composite_score=0.5))


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


class EncodeRequest(BaseModel):
    model: str = DEFAULT_CAUSAL_LM
    text: str
    max_length: Optional[int] = None


class EncodeResponse(BaseModel):
    model: str
    text: str
    tokens: list[AffectReadout]
    summary: dict[str, Any]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    hidden_dim: int
    layers: int
    has_sae: bool
    zoo_checkpoints: list[str] = Field(default_factory=list)
    probe_origin: str = "random"
    brain_data_tier: str = Field(
        default="none",
        description="none | synthetic_minimal | real_fMRI — from brain probe meta when present.",
    )
    narratives_root: Optional[str] = None
    train_stories: list[str] = Field(default_factory=list)
    holdout_stories: list[str] = Field(default_factory=list)
    brain_val_r_valence: Optional[float] = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


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
