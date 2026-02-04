"""Region labeling from dimensional readouts."""


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


def confidence_tier_from_fused(fused: float) -> str:
    if fused < 0.4:
        return "HIGH"
    if fused < 0.7:
        return "MEDIUM"
    return "LOW"
