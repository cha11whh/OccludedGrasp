#!/usr/bin/env python3
"""Capture RGB-D and instance masks from a proxy cabinet scene in Isaac Lab.

This is a sensor/data-contract smoke, not a JAKA robot simulation. The scene
uses simple labeled cuboids as clutter proxies so perception wiring can be
validated before the official K1 model and cabinet assets are available.
"""

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--warmup-steps", type=int, default=12)
parser.add_argument("--with-openarm", action="store_true", help="Optionally add the Isaac Lab OpenArm bimanual surrogate.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.width <= 0 or args.height <= 0 or args.warmup_steps < 0:
    parser.error("image dimensions must be positive and warmup steps non-negative")

app = AppLauncher(args).app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_assets.robots.openarm import OPENARM_BI_CFG


def spawn_box(path: str, size: tuple[float, float, float], position: tuple[float, float, float],
              color: tuple[float, float, float], label: str) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        semantic_tags=[("class", label)],
    )
    cfg.func(path, cfg, translation=position)


def main() -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    # Open-front cabinet proxy, dimensions in meters. Side walls, back panel,
    # and shelves leave the front visible to the eye-level camera.
    spawn_box("/World/Cabinet/Back", (0.96, 0.06, 1.45), (0.0, 0.35, 1.10), (0.48, 0.34, 0.22), "cabinet")
    spawn_box("/World/Cabinet/LeftWall", (0.06, 0.66, 1.50), (-0.51, 0.0, 1.10), (0.55, 0.40, 0.25), "cabinet")
    spawn_box("/World/Cabinet/RightWall", (0.06, 0.66, 1.50), (0.51, 0.0, 1.10), (0.55, 0.40, 0.25), "cabinet")
    spawn_box("/World/Cabinet/Bottom", (1.08, 0.70, 0.08), (0.0, 0.0, 0.36), (0.55, 0.40, 0.25), "cabinet")
    spawn_box("/World/Cabinet/Top", (1.08, 0.70, 0.08), (0.0, 0.0, 1.84), (0.55, 0.40, 0.25), "cabinet")
    spawn_box("/World/Cabinet/Shelf0", (0.96, 0.62, 0.045), (0.0, 0.02, 0.93), (0.64, 0.48, 0.30), "shelf")
    spawn_box("/World/Cabinet/Shelf1", (0.96, 0.62, 0.045), (0.0, 0.02, 1.40), (0.64, 0.48, 0.30), "shelf")

    # Deliberately overlapping eye-level silhouettes; labeled box proxies,
    # not realistic mug/bottle geometry or a trained-task benchmark.
    clutter = [
        ("GreenMugProxy", (0.17, 0.16, 0.26), (-0.10, -0.15, 1.08), (0.10, 0.65, 0.25)),
        ("BlueBoxProxy", (0.25, 0.18, 0.22), (0.06, 0.04, 1.05), (0.20, 0.35, 0.85)),
        ("RedBoxProxy", (0.19, 0.18, 0.28), (0.25, 0.10, 1.08), (0.85, 0.18, 0.12)),
        ("YellowBoxProxy", (0.24, 0.18, 0.23), (-0.20, 0.08, 1.52), (0.85, 0.70, 0.12)),
    ]
    for name, size, position, color in clutter:
        spawn_box(f"/World/Clutter/{name}", size, position, color, name)

    robot = None
    if args.with_openarm:
        robot_cfg = OPENARM_BI_CFG.replace(prim_path="/World/OpenArm")
        robot_cfg.init_state.pos = (0.0, -0.95, 0.0)
        robot = Articulation(cfg=robot_cfg)

    sim_utils.create_prim("/World/CameraRoot", "Xform")
    camera = Camera(
        CameraCfg(
            prim_path="/World/CameraRoot/Camera",
            update_period=0.0,
            width=args.width,
            height=args.height,
            data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
            colorize_instance_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
        )
    )

    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[0.0, -1.8, 1.25]], device=sim.device),
        torch.tensor([[0.0, 0.10, 1.22]], device=sim.device),
    )
    for _ in range(args.warmup_steps):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    sim.step()
    camera.update(dt=sim.get_physics_dt(), force_recompute=True)
    output = camera.data.output
    required = ("rgb", "distance_to_image_plane", "instance_segmentation_fast")
    missing = [key for key in required if key not in output]
    if missing:
        raise RuntimeError(f"camera did not produce required modalities: {missing}; got {list(output)}")

    arrays = {}
    for key in required:
        value = output[key][0].detach().cpu().numpy()
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value[..., 0]
        np.save(args.out_dir / f"{key}.npy", value)
        arrays[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    info = camera.data.info[0].get("instance_segmentation_fast")
    metadata = {
        "schema": "occluded_grasp_isaac_proxy_capture.v1",
        "simulator": "Isaac Sim 5.1 / Isaac Lab 2.3.2",
        "robot": "OpenArm bimanual surrogate" if robot is not None else None,
        "scene": "procedural cabinet and labeled cuboid clutter proxies",
        "camera_pose": {"eye": [0.0, -1.8, 1.25], "target": [0.0, 0.10, 1.22]},
        "modalities": arrays,
        "segmentation_info": info,
        "warning": "Not JAKA K1 geometry; clutter proxies are not realistic objects or calibrated robot inputs.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


try:
    main()
except BaseException:
    app.close()
    raise
else:
    # Isaac Sim Kit may retain extension worker threads after app.close() in
    # headless inference jobs. These scripts are isolated subprocesses, so an
    # immediate successful exit avoids hanging the staged pipeline.
    os._exit(0)
