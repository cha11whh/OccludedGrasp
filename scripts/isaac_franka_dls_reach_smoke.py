#!/usr/bin/env python3
"""Small Isaac Lab Franka DLS reach regression for the grasp feedback controller."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--base-height", type=float, default=0.0)
parser.add_argument("--yaw-deg", type=float, default=0.0)
parser.add_argument("--delta-x", type=float, default=0.0)
parser.add_argument("--delta-y", type=float, default=0.10)
parser.add_argument("--delta-z", type=float, default=0.0)
parser.add_argument("--with-rigid-object", action="store_true")
parser.add_argument("--with-camera", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_from_angle_axis, quat_inv, subtract_frame_transforms
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground=AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    robot=FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0/120.0, device=args.device))
    scene=InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    rigid_object = None
    if args.with_rigid_object:
        rigid_object = RigidObject(RigidObjectCfg(
            prim_path="/World/TestObject",
            spawn=sim_utils.CuboidCfg(
                size=(0.25, 0.18, 0.22),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.35, 0.85)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.06, 0.04, 1.05)),
        ))
    camera = None
    if args.with_camera:
        sim_utils.create_prim("/World/CameraRoot", "Xform")
        camera = Camera(CameraCfg(
            prim_path="/World/CameraRoot/Camera", update_period=0.0, width=320, height=240,
            data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
            colorize_instance_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                            horizontal_aperture=20.955, clipping_range=(0.05, 10.0)),
        ))
    sim.reset()
    robot=scene["robot"]
    robot_cfg=scene.cfg.robot
    # Put the articulation at requested yaw/height through its root state and reset all buffers.
    root=robot.data.root_state_w.clone()
    angle=torch.tensor([args.yaw_deg * torch.pi / 180.0], device=robot.device)
    z_axis=torch.tensor([[0.0,0.0,1.0]], device=robot.device)
    root[:,0:3]=torch.tensor([[0.0,-0.72,args.base_height]],device=robot.device)
    root[:,3:7]=quat_from_angle_axis(angle,z_axis)
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())
    ee_id=robot.find_bodies("panda_hand")[0][0]
    arm_ids=[robot.joint_names.index(f"panda_joint{i}") for i in range(1,8)]
    jac_id=ee_id-1
    ctrl=DifferentialIKController(DifferentialIKControllerCfg(command_type="position",use_relative_mode=False,ik_method="dls"),num_envs=1,device=sim.device)
    hand_w=robot.data.body_pose_w[:,ee_id]
    base=root[:,:7]
    hand_b,_=subtract_frame_transforms(base[:,:3],base[:,3:7],hand_w[:,:3],hand_w[:,3:7])
    initial_hand_b = hand_b.clone()
    target_b = hand_b + torch.tensor([[args.delta_x, args.delta_y, args.delta_z]], device=sim.device)
    target_b_reference = target_b.clone()
    ctrl.set_command(target_b.clone(), ee_quat=robot.data.body_pose_w[:, ee_id, 3:7].clone())
    q_start=robot.data.joint_pos[:,arm_ids].clone()
    err=float("inf")
    step_durations=[]
    for _ in range(args.steps):
        jac=robot.root_physx_view.get_jacobians()[:,jac_id,:,arm_ids]
        base_rot_inv=matrix_from_quat(quat_inv(robot.data.root_pose_w[:,3:7]))
        jac[:,:3,:]=torch.bmm(base_rot_inv,jac[:,:3,:])
        jac[:,3:,:]=torch.bmm(base_rot_inv,jac[:,3:,:])
        hand_w=robot.data.body_pose_w[:,ee_id]
        root_w=robot.data.root_pose_w
        hand_b,_=subtract_frame_transforms(root_w[:,:3],root_w[:,3:7],hand_w[:,:3],hand_w[:,3:7])
        q=robot.data.joint_pos[:,arm_ids]
        q_des=ctrl.compute(hand_b,robot.data.body_pose_w[:,ee_id,3:7],jac[:,:3,:],q)
        q_des=q+torch.clamp(q_des-q,-0.04,0.04)
        limits=robot.data.soft_joint_pos_limits[:,arm_ids,:]
        q_des=torch.maximum(torch.minimum(q_des,limits[:,:,1]),limits[:,:,0])
        robot.set_joint_position_target(q_des,joint_ids=arm_ids)
        robot.write_data_to_sim()
        step_started=time.perf_counter()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
        if rigid_object is not None:
            rigid_object.update(sim.get_physics_dt())
        step_durations.append(time.perf_counter()-step_started)
        hand_w=robot.data.body_pose_w[:,ee_id]
        root_w=robot.data.root_pose_w
        hand_b,_=subtract_frame_transforms(root_w[:,:3],root_w[:,3:7],hand_w[:,:3],hand_w[:,3:7])
        err=float(torch.linalg.vector_norm(target_b_reference-hand_b).item())
    q_end=robot.data.joint_pos[:,arm_ids].clone()
    result={"steps":args.steps,"base_height":args.base_height,"yaw_deg":args.yaw_deg,"target_delta_base_m":[args.delta_x,args.delta_y,args.delta_z],
            "initial_ee_base_m":initial_hand_b[0].detach().cpu().tolist(),"target_ee_base_m":target_b_reference[0].detach().cpu().tolist(),"final_ee_base_m":hand_b[0].detach().cpu().tolist(),
            "position_error_m":err,"q_start":q_start[0].detach().cpu().tolist(),"q_end":q_end[0].detach().cpu().tolist(),
            "with_rigid_object":args.with_rigid_object,"with_camera":args.with_camera,
            "max_step_seconds":max(step_durations),
            "mean_step_seconds":sum(step_durations)/len(step_durations)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)

try:
    main()
except BaseException:
    app.close()
    raise
else:
    os._exit(0)
