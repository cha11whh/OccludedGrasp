# Dual-Panda IsaacLab readiness worker

The bounded preflight launches `scripts/isaac_dual_panda_readiness.py` with the
IsaacLab Python runtime, headless rendering, and camera support. The worker
places two fixed-base `FRANKA_PANDA_CFG` assets at $x=-1.2$ m and $x=1.2$ m,
adds a ground plane and a known $0.15$ m rigid cube, and runs 480 physics steps
at $1/60$ s. Each Panda receives a measured 7-joint arm target and a
close/reopen target on its two finger joints. The v2 report records the
camera's ROS optical pose (quaternion order $wxyz$, optical forward $+Z$),
intrinsics, center sample pixel, and the cube's measured world pose and
dimensions. Both worker and validator intersect that sample ray with the
oriented cube and compare the resulting axial depth to the rendered
`distance_to_image_plane` sample; Euclidean camera-to-face-center distance is
not used.

Loaded fixed-base state comes from IsaacLab's articulation `is_fixed_base`
property, not the asset configuration. Initial/final root poses are reported;
translation and quaternion-derived orientation drift are checked against those
poses.

The report is atomically replaced on startup and at completion. A failed
startup or simulation leaves a fresh `status: "error"` report with the same
`run_id`, startup timestamp, completion timestamp, and error text; a complete
report contains joint, pose, camera, depth, and loaded-articulation evidence
read from the live simulation. The CPU test for atomic serialization does not launch IsaacLab or
claim GPU evidence.

This is readiness evidence only: it does not validate collision-free paths,
grasp success, or physical hardware safety. The coordinator owns the bounded
GPU run; do not invoke the worker from ordinary CPU test runs.

## Recovery verification — 2026-11-09

The coordinator verified the actual recovered files and ran the full CPU
suite: **153 passed in 2.54s**, including 13 readiness tests. An independent
spec review first rejected the depth and fixed-base evidence; after the v2
changes, re-review passed. The repair agent returned no execution report, so
RED-before-GREEN provenance for these repairs is unavailable; only the
coordinator's observed passing regression is claimed.

No Isaac experiment was launched in this recovery. The A100 had 37,728 MiB
allocated to an unrelated worker and only 2,528 MiB free. That process was not
stopped. Actual asset loading, joint/finger tracking and RGB-D readiness remain
unverified; CPU acceptance of constructed reports is not physical evidence.