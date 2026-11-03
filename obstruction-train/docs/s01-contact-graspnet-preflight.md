# S01 Contact-GraspNet preflight — official artifact download blocked

Date: 2026-11-03. Static inspection only; no model, Python import, compilation,
weight download, dependency installation, training or GPU inference was run.
This document is not a deployment-readiness or latency result.

## Source and protection

Existing external checkout: `/home/media/zy/third_party/contact_graspnet`.
Observed Git HEAD: `721706067cc7f7889c32afe7b4f7955eb90bd866`.
Existing PointNet2 compiled-artifact modifications were not changed or copied.
The actual implementation is `contact_graspnet/contact_grasp_estimator.py`;
the CLI is `contact_graspnet/inference.py`. The inspected checkpoint directory
contains only `.gitkeep`. This limited search does not prove no weights exist
elsewhere on the machine.

The official README points to the model folder:
https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl?usp=sharing

The link, complete model/config contents and checksums remain unverified. No
weight acquisition or license acceptance was performed.

## License gate

The coordinator read the actual local PDF using the existing system tool:
`pdftotext /home/media/zy/third_party/contact_graspnet/License.pdf -`.

The title is NVIDIA Source Code License for Contact-GraspNet. Section 3.3
restricts use of the work and derivative works to non-commercial research or
evaluation. Redistribution and attribution conditions also apply. This is not
an MIT/Apache license. Weight-specific terms must also be checked when acquiring
weights; source permission must not be silently assumed to cover all artifacts.

The user explicitly confirmed non-commercial research/evaluation on 2026-11-03.
That source use-scope gate is resolved. Weight-specific terms remain to be
checked; do not silently switch models or treat source permission as a universal
permission for every model artifact.

## Continuation after use-scope confirmation

- Isolated clean source clone created at
  `/home/media/zy/pi_work/.scratch/frozen-grasp-scoring/runtime/contact_graspnet`,
  HEAD `721706067cc7f7889c32afe7b4f7955eb90bd866`; original dirty binaries were
  not copied or modified.
- Official public subfolder was read over verified HTTPS:
  `https://drive.google.com/drive/folders/1vcHNeKgMpHpeWKggkRdWmMr4-ymEPQx8`.
  It lists `checkpoint`, `config.yaml`, and `model.ckpt-144144` components.
- Actual config download using `curl --cacert /etc/ssl/certs/ca-certificates.crt`
  failed with curl error 60 (unable to get local issuer certificate). The direct
  `drive.usercontent.google.com` endpoint also failed TLS verification; its
  response showed a Fortinet proxy connection. This is not a request for user
  credentials. Certificate checks were not disabled; no untrusted mirror used.
- No official model config/weights were obtained or hashed. No inference
  environment was installed; no TensorFlow/custom-op or checkpoint restore test
  was performed. Existing environment metadata contains multiple TensorFlow and
  NumPy dist-info versions and is not assumed to be a clean runtime.
- A separate CPU contract slice can proceed with injected test predictors, but
  is not evidence that Contact-GraspNet actually ran. Download/ABI/model gates
  remain open. No GPU occupation, training or simulation occurred.

## Deployment facts and pending verification

- Official README describes metric-depth/K/optional-segmentation or metric point
  cloud input, object-wise crop/filter options, and grasp candidates with scores.
  These scores are not calibrated object-level grasp success probabilities.
- README lists an older TensorFlow/CUDA stack; custom PointNet2 native operators
  require isolated compatibility/build validation. Existing binaries are not
  evidence of compatible ABI or successful inference.
- GPU query observed NVIDIA A100-PCIE-40GB. No inference speed or memory claim
  follows from hardware availability.
- A separate environment/source-build location is approved in principle, not
  modification of the existing contact_grasp_env or original dirty checkout.
- Robot-frame transforms, frontal approach, jaw geometry, IK, full-arm collision
  and carrying-path checks remain required downstream. Model confidence does
  not establish any of them.

## Status

S00 CPU budget infrastructure was reviewed and committed as `503a4fd`. S01 is
**partial/blocked**, not complete. S02 CPU contract work is separate from actual
backend deployment. Actual model execution and S03—S06 benchmarks have not
started. New-stage GPU
occupation remains **0 seconds of 7200 seconds**; no production budget run was
performed. Unit-test temporary ledgers are not the production budget ledger.