#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def q(path):
    return shlex.quote(str(path))


def safe_stem(value):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(value).strip()).strip("_") or "target"


def run_command(name, cmd, dry_run=False):
    start = time.perf_counter()
    print(f"\n[{name}] start")
    print(cmd)
    if dry_run:
        return name, 0.0

    proc = subprocess.run(cmd, shell=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
    print(f"[{name}] done in {elapsed:.2f}s")
    return name, elapsed


def docker_bash(image, repo_root, script, gpus=True, ipc_host=True, docker_user=None, cache_root=None):
    parts = ["docker", "run", "--rm"]
    if gpus:
        parts.extend(["--gpus", "all"])
    if ipc_host:
        parts.append("--ipc=host")
    if docker_user:
        parts.extend(["--user", docker_user])
    if cache_root:
        parts.extend([
            "-e",
            f"HOME={cache_root / 'home'}",
            "-e",
            f"HF_HOME={cache_root / 'huggingface'}",
            "-e",
            f"TRANSFORMERS_CACHE={cache_root / 'huggingface' / 'transformers'}",
            "-e",
            f"MPLCONFIGDIR={cache_root / 'matplotlib'}",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "TRANSFORMERS_OFFLINE=1",
        ])
    parts.extend([
        "-v",
        f"{repo_root}:{repo_root}",
        image,
        "bash",
        "-lc",
        script,
    ])
    return " ".join(shlex.quote(part) for part in parts)


def write_list(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip() + "\n")


def prepare_inputs(args):
    repo_root = args.repo_root.resolve()
    amodal_root = args.amodal_root.resolve()
    image_path = args.rgb.resolve()

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if args.input_dir:
        input_dir = args.input_dir.resolve()
    else:
        input_dir = image_path.parent

    try:
        image_filename = str(image_path.relative_to(input_dir))
    except ValueError as exc:
        raise ValueError(f"--rgb must be inside --input-dir. rgb={image_path}, input_dir={input_dir}") from exc

    stem = image_path.stem
    run_dir = args.work_dir.resolve() / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = repo_root / ".cache" / "docker_user"
    for subdir in ["home", "huggingface/transformers", "matplotlib"]:
        (cache_root / subdir).mkdir(parents=True, exist_ok=True)

    img_list = args.img_filenames_txt.resolve() if args.img_filenames_txt else run_dir / "img_filenames.txt"
    if not args.img_filenames_txt:
        write_list(img_list, [image_filename])

    if not args.target_prompts_txt:
        raise ValueError("--target-prompts-txt is required for GroundingDINO/SAM modal mask generation")

    if args.amodal_classes_txt:
        amodal_classes_txt = args.amodal_classes_txt.resolve()
    else:
        amodal_classes_txt = None

    return {
        "repo_root": repo_root,
        "amodal_root": amodal_root,
        "depth_root": args.depth_root.resolve(),
        "obstruction_root": args.obstruction_root.resolve(),
        "image_path": image_path,
        "image_filename": image_filename,
        "stem": stem,
        "input_dir": input_dir,
        "run_dir": run_dir,
        "cache_root": cache_root,
        "img_list": img_list,
        "target_prompts_txt": args.target_prompts_txt.resolve(),
        "amodal_classes_txt": amodal_classes_txt,
    }


def build_commands(args, ctx):
    stem = ctx["stem"]
    repo_root = ctx["repo_root"]
    image_path = ctx["image_path"]
    input_dir = ctx["input_dir"]
    img_list = ctx["img_list"]
    target_prompts_txt = ctx["target_prompts_txt"]
    amodal_classes_txt = ctx["amodal_classes_txt"]

    depth_out = args.depth_out.resolve() if args.depth_out else ctx["depth_root"] / "outputs" / f"{args.depth_encoder}_parallel" / stem
    modal_out = args.modal_out.resolve() if args.modal_out else ctx["amodal_root"] / f"targeted_modal_masks_parallel_{stem}"
    amodal_out = args.amodal_out.resolve() if args.amodal_out else ctx["amodal_root"] / f"experiment_output_parallel_{stem}"
    graph_out = args.graph_out.resolve() if args.graph_out else ctx["obstruction_root"] / "outputs" / f"{stem}_parallel"

    depth_raw = depth_out / f"{stem}_raw.npy"
    modal_metadata = modal_out / stem / "metadata.json"
    amodal_metadata = amodal_out / stem / "mask_metadata.json"
    amodal_masks_dir = amodal_out / stem

    depth_script = (
        f"cd {q(ctx['depth_root'])} && "
        f"python infer_single.py "
        f"--img-path {q(image_path)} "
        f"--outdir {q(depth_out)} "
        f"--encoder {shlex.quote(args.depth_encoder)} "
        f"--input-size {int(args.depth_input_size)}"
    )
    if args.skip_existing and depth_raw.exists():
        depth_script = f"echo 'depth exists, skip: {q(depth_raw)}'"

    amodal_env = (
        f"export TORCH_HOME={q(ctx['amodal_root'] / 'lama')} && "
        f"export PYTHONPATH={q(ctx['amodal_root'] / 'lama')}:{q(ctx['amodal_root'] / 'diffusers' / 'src')}:$PYTHONPATH && "
    )

    modal_script = (
        amodal_env
        + f"cd {q(ctx['amodal_root'])} && "
        f"python generate_modal_masks.py "
        f"--input_dir {q(input_dir)} "
        f"--img_filenames_txt {q(img_list)} "
        f"--target_prompts_txt {q(target_prompts_txt)} "
        f"--output_dir {q(modal_out)} "
        f"--box_thresh {float(args.box_thresh)} "
        f"--text_thresh {float(args.text_thresh)} "
        f"--device {shlex.quote(args.device)} "
        f"--prompt_mode {shlex.quote(args.prompt_mode)}"
    )

    classes_arg = target_prompts_txt
    amodal_script = (
        f"cd {q(ctx['amodal_root'])} && "
        f"python fast_modal_first.py "
        f"--input_dir {q(input_dir)} "
        f"--img_filenames_txt {q(img_list)} "
        f"--classes_txt {q(classes_arg)} "
        f"--modal_masks_dir {q(modal_out)} "
        f"--output_dir {q(amodal_out)} "
        f"--bbox_overlap_thresh {float(args.bbox_overlap_thresh)} "
        f"--run_amodal {str(args.run_amodal).lower()} "
        f"--amodal_completion_mode {shlex.quote(args.amodal_completion_mode)} "
        f"--geometric_radius {int(args.geometric_radius)} "
        f"--geometric_max_added_fraction {float(args.geometric_max_added_fraction)} "
        f"--sd_num_inference_steps {int(args.sd_num_inference_steps)} "
        f"--max_iter_id {int(args.max_iter_id)} "
        f"--complete_boundary_objects {str(args.complete_boundary_objects).lower()}"
    )
    if amodal_classes_txt:
        amodal_script += f" --amodal_classes_txt {q(amodal_classes_txt)}"

    mask_script = modal_script + " && " + amodal_script
    if args.skip_existing and modal_metadata.exists() and amodal_metadata.exists():
        mask_script = f"echo 'masks exist, skip: {q(amodal_metadata)}'"

    graph_script = (
        f"cd {q(ctx['obstruction_root'])} && "
        f"python -m obstruction_preprocess.estimate_obstruction "
        f"--rgb {q(image_path)} "
        f"--depth {q(depth_raw)} "
        f"--modal-dir {q(modal_out / stem)} "
        f"--modal-metadata {q(modal_metadata)} "
        f"--amodal-dir {q(amodal_masks_dir)} "
        f"--amodal-metadata {q(amodal_metadata)} "
        f"--scene-id {shlex.quote(args.scene_id or stem)} "
        f"--view-id {shlex.quote(args.view_id)} "
        f"--dedupe-iou {float(args.dedupe_iou)} "
        f"--min-contact-fraction {float(args.min_contact_fraction)} "
        f"--min-relation-confidence {float(args.min_relation_confidence)} "
        f"--save-visualizations {str(args.save_visualizations).lower()} "
        f"--process-max-side {int(args.process_max_side)} "
        f"--depth-transform {shlex.quote(args.depth_transform)} "
        f"--relation-mode {shlex.quote(args.relation_mode)} "
        f"--relation-model-checkpoint {q(args.relation_model_checkpoint)} "
        f"--relation-model-root {q(args.relation_model_root)} "
        f"--relation-model-min-confidence {float(args.relation_model_min_confidence)} "
        f"--relation-model-device {shlex.quote(args.relation_model_device)} "
        f"--out-dir {q(graph_out)}"
    )

    rl_out = graph_out / (f"rl_affordance_{safe_stem(args.rl_target)}" if args.rl_target else "rl_affordance")
    rl_script = (
        f"cd {q(ctx['obstruction_root'])} && "
        f"python -m obstruction_preprocess.rl_affordance_preprocess "
        f"--objects {q(graph_out / 'objects.json')} "
        f"--relations {q(graph_out / 'occ_detail.json')} "
        f"--depth {q(depth_raw)} "
        f"--out-dir {q(rl_out)} "
        f"--min-confidence {float(args.rl_min_confidence)}"
    )
    if args.rl_target:
        rl_script += f" --target {shlex.quote(args.rl_target)}"

    return {
        "depth": docker_bash(args.docker_image, repo_root, depth_script, gpus=True, docker_user=args.docker_user, cache_root=ctx["cache_root"]),
        "mask": docker_bash(args.docker_image, repo_root, mask_script, gpus=True, docker_user=args.docker_user, cache_root=ctx["cache_root"]),
        "graph": docker_bash(args.docker_image, repo_root, graph_script, gpus=args.graph_uses_gpu or args.relation_mode == "model", docker_user=args.docker_user, cache_root=ctx["cache_root"]),
        "rl": docker_bash(args.docker_image, repo_root, rl_script, gpus=False, docker_user=args.docker_user, cache_root=ctx["cache_root"]),
        "paths": {
            "depth_raw": depth_raw,
            "modal_metadata": modal_metadata,
            "amodal_metadata": amodal_metadata,
            "graph_out": graph_out,
            "rl_out": rl_out,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run depth and modal/amodal segmentation in parallel, then build obstruction graph.")
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--img-filenames-txt", type=Path, default=None)
    parser.add_argument("--target-prompts-txt", type=Path, required=True)
    parser.add_argument("--amodal-classes-txt", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("/home/ubuntu22-zy/occlusion"))
    parser.add_argument("--amodal-root", type=Path, default=Path("/home/ubuntu22-zy/occlusion/amodal"))
    parser.add_argument("--depth-root", type=Path, default=Path("/home/ubuntu22-zy/occlusion/depth-anything-v2"))
    parser.add_argument("--obstruction-root", type=Path, default=Path("/home/ubuntu22-zy/occlusion/obstruction-preprocess"))
    parser.add_argument("--work-dir", type=Path, default=Path("/home/ubuntu22-zy/occlusion/obstruction-preprocess/runs"))
    parser.add_argument("--depth-out", type=Path, default=None)
    parser.add_argument("--modal-out", type=Path, default=None)
    parser.add_argument("--amodal-out", type=Path, default=None)
    parser.add_argument("--graph-out", type=Path, default=None)
    parser.add_argument("--docker-image", default="gsa:amodal")
    parser.add_argument("--docker-user", default=f"{os.getuid()}:{os.getgid()}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--box-thresh", type=float, default=0.25)
    parser.add_argument("--text-thresh", type=float, default=0.25)
    parser.add_argument("--prompt-mode", choices=["combined", "per_prompt"], default="combined")
    parser.add_argument("--bbox-overlap-thresh", type=float, default=0.02)
    parser.add_argument("--run-amodal", type=str2bool, default=True)
    parser.add_argument("--amodal-completion-mode", choices=["geometric", "iterative"], default="geometric")
    parser.add_argument("--geometric-radius", type=int, default=36)
    parser.add_argument("--geometric-max-added-fraction", type=float, default=0.65)
    parser.add_argument("--sd-num-inference-steps", type=int, default=20)
    parser.add_argument("--max-iter-id", type=int, default=1)
    parser.add_argument("--complete-boundary-objects", type=str2bool, default=False)
    parser.add_argument("--dedupe-iou", type=float, default=0.98)
    parser.add_argument("--min-contact-fraction", type=float, default=0.001)
    parser.add_argument("--min-relation-confidence", type=float, default=0.01)
    parser.add_argument("--save-visualizations", type=str2bool, default=False)
    parser.add_argument("--process-max-side", type=int, default=1600)
    parser.add_argument("--depth-transform", choices=["identity", "inverse"], default="inverse")
    parser.add_argument("--relation-mode", choices=["heuristic", "model"], default="model")
    parser.add_argument("--relation-model-checkpoint", type=Path, default=Path("/home/ubuntu22-zy/occlusion/obstruction-train/outputs/pair_transformer_v1_base_da_vitl_ft_e1/best.pt"))
    parser.add_argument("--relation-model-root", type=Path, default=Path("/home/ubuntu22-zy/occlusion/obstruction-train"))
    parser.add_argument("--relation-model-min-confidence", type=float, default=0.50)
    parser.add_argument("--relation-model-device", default="cuda")
    parser.add_argument("--run-rl-affordance", type=str2bool, default=False)
    parser.add_argument("--rl-target", default=None)
    parser.add_argument("--rl-min-confidence", type=float, default=0.50)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--view-id", default="0")
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--graph-uses-gpu", type=str2bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ctx = prepare_inputs(args)
    commands = build_commands(args, ctx)

    start = time.perf_counter()
    timings = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(run_command, "depth", commands["depth"], args.dry_run),
                pool.submit(run_command, "mask", commands["mask"], args.dry_run),
            ]
            for future in as_completed(futures):
                name, elapsed = future.result()
                timings[name] = elapsed

        name, elapsed = run_command("graph", commands["graph"], args.dry_run)
        timings[name] = elapsed

        if args.run_rl_affordance or args.rl_target:
            name, elapsed = run_command("rl", commands["rl"], args.dry_run)
            timings[name] = elapsed
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    total = time.perf_counter() - start
    paths = commands["paths"]
    print("\nPipeline complete")
    print(f"total_time={total:.2f}s")
    for name in ["depth", "mask", "graph", "rl"]:
        if name in timings:
            print(f"{name}_time={timings[name]:.2f}s")
    print(f"depth_raw={paths['depth_raw']}")
    print(f"modal_metadata={paths['modal_metadata']}")
    print(f"amodal_metadata={paths['amodal_metadata']}")
    print(f"graph_out={paths['graph_out']}")
    if args.run_rl_affordance or args.rl_target:
        print(f"rl_out={paths['rl_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
