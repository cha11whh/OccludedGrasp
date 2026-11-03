import numpy as np
import pytest

from obstruction_train.frozen_grasp_backend import predict_candidates


def test_scene_prediction_is_bound_to_observation_and_model_without_claiming_feasibility():
    calls = []

    def predictor(points, object_ids):
        calls.append((points.shape, object_ids))
        return ({2: np.eye(4)[None]}, {2: np.array([0.8])},
                {2: np.array([[0.0, 0.0, 1.0]])}, {2: np.array([0.04])})

    result = predict_candidates(
        np.array([[0.0, 0.0, 1.0]]), [2],
        observation_id="obs-1", distance_unit="m", coordinate_frame="camera",
        model_version="frozen-checkpoint-sha", predictor=predictor,
    )

    assert calls == [((1, 3), (2,))]
    assert result["status"] == "candidates_generated"
    assert result["observation_id"] == "obs-1"
    assert result["model_version"] == "frozen-checkpoint-sha"
    assert result["candidates"][0]["object_id"] == 2
    assert result["candidates"][0]["score"] == 0.8
    assert result["candidates"][0]["width"] == 0.04
    assert "feasible" not in result["candidates"][0]


def _request(predictor, **changes):
    kwargs = dict(observation_id="obs-1", distance_unit="m",
                  coordinate_frame="camera", model_version="frozen-sha", predictor=predictor)
    kwargs.update(changes)
    return predict_candidates(np.array([[0.0, 0.0, 1.0]]), [2], **kwargs)


def _output():
    return ({2: np.eye(4)[None]}, {2: np.array([0.8])},
            {2: np.array([[0.0, 0.0, 1.0]])}, {2: np.array([0.04])})


@pytest.mark.parametrize("changes", [
    {"distance_unit": "relative"}, {"coordinate_frame": "base"},
    {"observation_id": ""}, {"model_version": ""},
])
def test_invalid_request_never_invokes_backend(changes):
    calls = []
    result = _request(lambda *args: calls.append(args), **changes)
    assert result["reason"] == "invalid_request"
    assert calls == []


@pytest.mark.parametrize("failure", ["scene_key", "nan", "score_range", "width", "pose", "alignment"])
def test_invalid_backend_output_never_authorizes_a_candidate(failure):
    output = _output()
    if failure == "scene_key":
        output = tuple({-1: item[2]} for item in output)
    elif failure == "nan":
        output[1][2][0] = np.nan
    elif failure == "score_range":
        output[1][2][0] = 1.1
    elif failure == "width":
        output[3][2][0] = -0.01
    elif failure == "pose":
        output[0][2][0, 0, 0] = 2.0
    else:
        output[3][2] = np.array([])
    result = _request(lambda *args: output)
    assert result["reason"] == "invalid_backend_output"
    assert result["candidates"] == []


def test_no_proposals_is_unknown_not_infeasible():
    result = _request(lambda *args: ({}, {}, {}, {}))
    assert result["status"] == "unknown"
    assert result["reason"] == "no_candidates_found"


@pytest.mark.parametrize("dtype", [bool, str, complex])
@pytest.mark.parametrize("field", [0, 1, 2, 3])
def test_nonreal_numeric_backend_arrays_are_rejected_without_coercion(dtype, field):
    output = _output()
    output[field][2] = output[field][2].astype(dtype)
    result = _request(lambda *args: output)
    assert result["reason"] == "invalid_backend_output"
    assert result["candidates"] == []


@pytest.mark.parametrize("error,reason", [(TimeoutError, "backend_timeout"), (RuntimeError, "backend_error")])
def test_backend_failure_does_not_fall_back_to_synthetic_scores(error, reason):
    def fail(*args):
        raise error("backend unavailable")
    result = _request(fail)
    assert result["status"] == "unknown"
    assert result["reason"] == reason
    assert result["candidates"] == []