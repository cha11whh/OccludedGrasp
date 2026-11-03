# S02 CPU contract slice, not actual model deployment

Public seam: `frozen_grasp_backend.predict_candidates`. One injected predictor
call accepts metric camera points and requested object IDs and returns four
object-indexed dictionaries: poses, quality scores, contacts and jaw openings.
It does not load a model, provide segmentation, prove feasibility, or enforce
timeouts on an arbitrary in-process callback. Model-owning production workers
must use the S00 bounded process runner. A reported TimeoutError is propagated
as unknown, not evidence that this contract can interrupt a hanging predictor.

The request explicitly supplies observation ID and frozen-model version; the
result carries them as caller-owned provenance, not cryptographic proof that a
stub loaded weights. A production worker must bind its segment masks, model
identity and outputs to that request. No stub result is real model evidence.

Checks include metric/camera conventions, real finite point coordinates,
requested positive IDs, object-assigned output keys (reject whole-scene -1),
aligned arrays, real finite scores in [0,1], positive widths and rigid transforms.
Scores are model confidence, not calibrated grasp success probability. Empty
outputs, backend errors and malformed outputs return unknown with no authorized
candidate; proposals do not establish IK, collision or path feasibility.

Actual upstream scene API integration is still pending: per-object segmentation
must be supplied and the estimator's single-candidate squeeze shapes normalized
explicitly. NumPy IDs also require explicit canonicalization in that adapter;
this boundary deliberately takes ordinary integer IDs, rejecting booleans.

## Observed evidence

Coordinator test commands from the isolated task-worktree root used
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=obstruction-train
/home/media/zy/pi_work/.venv_perception/bin/python -m pytest -q ...`.

- Initial public-contract test: collection failed because the new backend module
  did not yet exist (ModuleNotFoundError). After implementation and supplementary
  negative tests, full CPU suite: **88 passed in 2.30 seconds**. Supplementary
  negative tests initially passed; no assertion RED is claimed for those.
- Independent review rejected coercing string/bool/complex output arrays to
  float. Added public-seam parameterized tests actually failed before repair;
  after validating the original array dtype, full CPU suite:
  **100 passed in 2.31 seconds**. `git diff --check` passed.
- Independent final review is required before commit. No real model prediction,
  calibrated simulation input, feasibility backend, latency measurement or
  training result is claimed. New-stage GPU occupation remains zero.