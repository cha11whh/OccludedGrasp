"""Late fusion for grasp-candidate quality and motion-feasibility signals.

This module combines component scores only when a fitted fusion model is
explicitly supplied. It does not invent missing components or train/calibrate a
model. A fused output is called an operation-success probability only when the
model artifact includes held-out calibration metadata and a defined outcome.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

COMPONENT_KEYS = (
    "grasp_quality_score",
    "motion_feasibility_score",
)
MODEL_TYPE = "logistic_two_signal_fusion_v1"
UNCALIBRATED_OUTPUT = "uncalibrated_fusion_utility"
CALIBRATED_OUTPUT = "calibrated_operation_success_probability"
_EPSILON = 1e-6


def validate_fusion_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the versioned logistic two-signal fusion model metadata."""
    if model.get("model_type") != MODEL_TYPE:
        raise ValueError(f"score fusion model_type must be {MODEL_TYPE!r}")
    coefficients = model.get("coefficients")
    if not isinstance(coefficients, Mapping):
        raise ValueError("score fusion model must include a coefficient mapping")
    for key in COMPONENT_KEYS:
        try:
            value = float(coefficients[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"score fusion model requires coefficient {key!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"score fusion coefficient {key!r} must be finite")
    try:
        intercept = float(model["intercept"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("score fusion model requires a finite intercept") from exc
    if not math.isfinite(intercept):
        raise ValueError("score fusion model requires a finite intercept")

    calibration = model.get("calibration", {})
    if not isinstance(calibration, Mapping):
        raise ValueError("score fusion calibration metadata must be a mapping")
    output_kind = str(calibration.get("output_kind", UNCALIBRATED_OUTPUT))
    if output_kind not in {UNCALIBRATED_OUTPUT, CALIBRATED_OUTPUT}:
        raise ValueError(f"unsupported score fusion output_kind: {output_kind}")
    if output_kind == CALIBRATED_OUTPUT:
        required = (
            "method", "validation_sample_count", "validation_positive_count",
            "validation_negative_count", "scene_disjoint", "brier_score",
            "platt_slope", "platt_intercept", "success_definition",
        )
        missing = [key for key in required if key not in calibration]
        if missing:
            raise ValueError(f"calibrated model is missing calibration metadata: {missing}")
        if not calibration["scene_disjoint"]:
            raise ValueError("calibrated probability requires a scene-disjoint validation split")
        sample_count = int(calibration["validation_sample_count"])
        positive_count = int(calibration["validation_positive_count"])
        negative_count = int(calibration["validation_negative_count"])
        brier_score = float(calibration["brier_score"])
        platt_slope = float(calibration["platt_slope"])
        platt_intercept = float(calibration["platt_intercept"])
        if sample_count <= 0 or positive_count <= 0 or negative_count <= 0:
            raise ValueError("calibrated probability metadata requires positive and negative validation cases")
        if positive_count + negative_count != sample_count:
            raise ValueError("calibration validation counts must sum to validation_sample_count")
        if not math.isfinite(brier_score) or not 0.0 <= brier_score <= 1.0:
            raise ValueError("calibration brier_score must be finite and in [0, 1]")
        if not math.isfinite(platt_slope) or not math.isfinite(platt_intercept):
            raise ValueError("Platt calibration parameters must be finite")
    return {
        "model_type": MODEL_TYPE,
        "model_id": str(model.get("model_id", "unspecified")),
        "intercept": intercept,
        "coefficients": {key: float(coefficients[key]) for key in COMPONENT_KEYS},
        "calibration": dict(calibration),
        "output_kind": output_kind,
    }


def fuse_score_components(
    components: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse grasp-candidate quality and motion feasibility with a fitted logistic stacker.

    The inputs are model outputs normalized to [0, 1] using the preprocessing
    recorded when the fusion model was fitted. Values are logit-transformed,
    clipped only for numerical stability, and passed through the model's
    fitted coefficients. Missing component values are errors, never silently
    replaced by zero or the legacy geometric proxy.
    """
    validated = validate_fusion_model(model)
    values: dict[str, float] = {}
    for key in COMPONENT_KEYS:
        if key not in components:
            raise ValueError(f"late fusion requires score component {key!r}")
        try:
            value = float(components[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score component {key!r} must be numeric") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"score component {key!r} must be finite and in [0, 1]")
        values[key] = value

    linear_score = validated["intercept"]
    for key in COMPONENT_KEYS:
        clipped = min(1.0 - _EPSILON, max(_EPSILON, values[key]))
        logit = math.log(clipped / (1.0 - clipped))
        linear_score += validated["coefficients"][key] * logit
    calibration = validated["calibration"]
    if validated["output_kind"] == CALIBRATED_OUTPUT:
        calibrated_logit = float(calibration["platt_slope"]) * linear_score + float(calibration["platt_intercept"])
    else:
        calibrated_logit = linear_score
    if calibrated_logit >= 0.0:
        fused_score = 1.0 / (1.0 + math.exp(-calibrated_logit))
    else:
        exp_score = math.exp(calibrated_logit)
        fused_score = exp_score / (1.0 + exp_score)

    return {
        "fused_score": fused_score,
        "score_kind": validated["output_kind"],
        "model_id": validated["model_id"],
        "component_scores": values,
        "linear_score": linear_score,
        "calibrated_logit": calibrated_logit,
    }
