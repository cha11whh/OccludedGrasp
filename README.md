# Occlusion Grasping Workspace Backup

This repository is a source-code backup of the local workspace at:

`/home/ubuntu22-zy/occlusion`

It keeps the code and configuration needed for the RGB/depth/mask obstruction pipeline, relation training, amodal segmentation integration, Depth Anything V2 integration, UnoGrasp reference code, and Isaac Lab tabletop grasp simulation wrapper.

Large or reproducible artifacts are intentionally excluded from Git:

- model checkpoints and downloaded weights
- full datasets and training outputs
- generated masks, depth arrays, images, videos, and USD captures
- Python environments, package caches, and Isaac Lab installation files

Important local paths from the original workspace:

- `amodal/`
- `depth-anything-v2/`
- `obstruction-preprocess/`
- `obstruction-train/`
- `UnoGrasp/`

## Included UnoBench smoke-test subset

`datasets/UnoBench_small/` contains the official minimal UnoBench Challenge
sample: 8 RGB images, 8 Set-of-Mark images, matching instance annotations,
sample predictions, ground truth, and local evaluation scripts. It is intended
only for inference, visualization, and simulation integration smoke tests; the
full UnoBench dataset remains external.

Source: https://github.com/tev-fbk/unobenchchallenge

UnoBench is released for academic, non-commercial use under CC BY-NC 4.0. See
the included dataset README and the official dataset page for attribution and
license details: https://huggingface.co/datasets/FBK-TeV/UnoBench

To refresh this backup from the source workspace, rerun the `rsync` command used by Codex in the parent workspace, preserving the same exclusion rules.
