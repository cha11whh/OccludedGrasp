# Occlusion Grasping Workspace Backup

This repository is a source-code backup of the local workspace at:

`/home/ubuntu22-zy/occlusion`

It keeps the code and configuration needed for the RGB/depth/mask obstruction pipeline, relation training, amodal segmentation integration, Depth Anything V2 integration, UnoGrasp reference code, and Isaac Lab tabletop grasp simulation wrapper.

Large or reproducible artifacts are intentionally excluded from Git:

- model checkpoints and downloaded weights
- datasets and training outputs
- generated masks, depth arrays, images, videos, and USD captures
- Python environments, package caches, and Isaac Lab installation files

Important local paths from the original workspace:

- `amodal/`
- `depth-anything-v2/`
- `obstruction-preprocess/`
- `obstruction-train/`
- `UnoGrasp/`

To refresh this backup from the source workspace, rerun the `rsync` command used by Codex in the parent workspace, preserving the same exclusion rules.
