import pytest

from affordance_score_fusion import (
    CALIBRATED_OUTPUT,
    COMPONENT_KEYS,
    UNCALIBRATED_OUTPUT,
    fuse_score_components,
    validate_fusion_model,
)


def _model(output_kind=UNCALIBRATED_OUTPUT):
    calibration = {"output_kind": output_kind}
    if output_kind == CALIBRATED_OUTPUT:
        calibration.update({
            "method": "platt",
            "validation_sample_count": 240,
            "validation_positive_count": 71,
            "validation_negative_count": 169,
            "scene_disjoint": True,
            "brier_score": 0.12,
            "platt_slope": 0.8,
            "platt_intercept": -0.15,
            "success_definition": "test outcome predicate",
        })
    return {
        "model_type": "logistic_two_signal_fusion_v1",
        "model_id": "test-model",
        "intercept": 0.0,
        "coefficients": {
            "grasp_quality_score": 0.2,
            "motion_feasibility_score": 0.4,
        },
        "calibration": calibration,
    }


def _components(**overrides):
    values = {
        "grasp_quality_score": 0.8,
        "motion_feasibility_score": 0.9,
    }
    values.update(overrides)
    return values


def test_fusion_requires_both_named_components():
    for key in COMPONENT_KEYS:
        components = _components()
        del components[key]
        with pytest.raises(ValueError, match=key):
            fuse_score_components(components, _model())


def test_fusion_rejects_nonfinite_or_out_of_range_component():
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        fuse_score_components(_components(motion_feasibility_score=1.1), _model())
    with pytest.raises(ValueError, match="finite"):
        fuse_score_components(_components(grasp_quality_score=float("nan")), _model())


def test_logistic_two_signal_fusion_is_monotonic_for_positive_component_coefficients():
    model = _model()
    low = fuse_score_components(_components(), model)["fused_score"]
    high = fuse_score_components(_components(motion_feasibility_score=0.99), model)["fused_score"]
    assert 0.0 < low < high < 1.0


def test_uncalibrated_model_never_claims_probability():
    result = fuse_score_components(_components(), _model())
    assert result["score_kind"] == UNCALIBRATED_OUTPUT
    assert result["model_id"] == "test-model"


def test_calibrated_operation_probability_requires_disjoint_calibration_metadata():
    model = _model(CALIBRATED_OUTPUT)
    assert validate_fusion_model(model)["output_kind"] == CALIBRATED_OUTPUT
    model["calibration"]["scene_disjoint"] = False
    with pytest.raises(ValueError, match="scene-disjoint"):
        validate_fusion_model(model)


def test_calibrated_output_applies_platt_slope_and_intercept():
    model = _model(CALIBRATED_OUTPUT)
    components = _components()
    raw = fuse_score_components(components, _model())
    calibrated = fuse_score_components(components, model)
    assert calibrated["fused_score"] != raw["fused_score"]
    assert calibrated["calibrated_logit"] == pytest.approx(
        0.8 * calibrated["linear_score"] - 0.15,
    )


def test_calibration_validation_counts_must_be_consistent():
    model = _model(CALIBRATED_OUTPUT)
    model["calibration"]["validation_negative_count"] = 100
    with pytest.raises(ValueError, match="sum to validation_sample_count"):
        validate_fusion_model(model)


def test_calibrated_model_requires_platt_parameters():
    model = _model(CALIBRATED_OUTPUT)
    del model["calibration"]["platt_intercept"]
    with pytest.raises(ValueError, match="platt_intercept"):
        validate_fusion_model(model)
