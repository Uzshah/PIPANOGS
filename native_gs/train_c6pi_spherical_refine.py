#!/usr/bin/env python3
"""Short direct-ERP refinement of a fixed-count C6 Gaussian scene.

The panorama-centric C6 initialization and topology remain fixed.  Only the
chart-invariant Gaussian attributes are calibrated through the credited SPaGS
spherical CUDA projection, eliminating train/inference projection mismatch.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import torch
from fused_ssim import fused_ssim

from native_gs.spags_backend import SPaGSRasterizer
from native_gs.benchmark_c6pi_spherical_cuda import (
    load_graphdeco_ply,
    panorama_camera,
)


def write_graphdeco_ply(
    source: Path,
    target: Path,
    *,
    positions: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity_logits: torch.Tensor,
    sh_0: torch.Tensor,
    sh_rest: torch.Tensor,
) -> None:
    vertex = PlyData.read(str(source))["vertex"].data.copy()
    cpu = lambda value: value.detach().float().cpu().numpy()
    xyz = cpu(positions)
    for index, axis in enumerate(("x", "y", "z")):
        vertex[axis] = xyz[:, index]
    vertex["opacity"] = cpu(opacity_logits).reshape(-1)
    scales = cpu(log_scales)
    for index in range(3):
        vertex[f"scale_{index}"] = scales[:, index]
    quaternions = cpu(torch.nn.functional.normalize(rotations, dim=-1))
    for index in range(4):
        vertex[f"rot_{index}"] = quaternions[:, index]
    dc = cpu(sh_0[:, 0])
    for index in range(3):
        vertex[f"f_dc_{index}"] = dc[:, index]
    rest = cpu(sh_rest.transpose(1, 2).reshape(sh_rest.shape[0], -1))
    rest_names = sorted(
        (name for name in vertex.dtype.names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    for index, name in enumerate(rest_names):
        vertex[name] = rest[:, index]
    target.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(target))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--baseline-iteration", type=int, default=7500)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--erp-width", type=int, default=1024)
    parser.add_argument("--mode", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--position-lr", type=float, default=0.0)
    parser.add_argument("--scale-lr", type=float, default=5e-4)
    parser.add_argument("--rotation-lr", type=float, default=1e-4)
    parser.add_argument("--opacity-lr", type=float, default=5e-3)
    parser.add_argument("--sh0-lr", type=float, default=2.5e-3)
    parser.add_argument("--sh-rest-lr", type=float, default=1.25e-4)
    args = parser.parse_args()
    if args.iterations < 1 or args.erp_width < 2 or args.erp_width % 2:
        parser.error("iterations must be positive and ERP width must be positive/even")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    baseline = args.baseline_model.resolve()
    source = args.source_path.resolve()
    model = args.model_path.resolve()
    input_ply = baseline / "point_cloud" / f"iteration_{args.baseline_iteration}" / "point_cloud.ply"
    output_ply = model / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    summary_path = model / "spherical_refine_training_summary.json"
    if output_ply.is_file() and summary_path.is_file():
        print(f"[skip] spherical refinement exists: {output_ply}")
        return 0

    fixed = load_graphdeco_ply(input_ply, torch.device("cuda"))
    positions = torch.nn.Parameter(fixed["positions"], requires_grad=args.position_lr > 0)
    log_scales = torch.nn.Parameter(torch.log(fixed["scales"].clamp_min(1e-12)))
    rotations = torch.nn.Parameter(fixed["rotations"])
    opacity_logits = torch.nn.Parameter(torch.logit(fixed["opacities"].clamp(1e-6, 1 - 1e-6)))
    sh_0 = torch.nn.Parameter(fixed["sh_0"])
    sh_rest = torch.nn.Parameter(fixed["sh_rest"])
    groups = [
        {"params": [log_scales], "lr": args.scale_lr, "name": "scales"},
        {"params": [rotations], "lr": args.rotation_lr, "name": "rotations"},
        {"params": [opacity_logits], "lr": args.opacity_lr, "name": "opacities"},
        {"params": [sh_0], "lr": args.sh0_lr, "name": "sh0"},
        {"params": [sh_rest], "lr": args.sh_rest_lr, "name": "sh_rest"},
    ]
    if args.position_lr > 0:
        groups.insert(0, {"params": [positions], "lr": args.position_lr, "name": "positions"})
    optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    metadata = json.loads((source / "metadata/cameras.json").read_text())
    faces = json.loads((source / "metadata/faces.json").read_text())
    prep = json.loads((source / "prep_summary.json").read_text())
    raw_scene = Path(prep["source_scene"])
    face_zero = np.asarray(faces[0]["tile_rotation"], dtype=np.float32)
    train_items = []
    for pano in prep["split_panoramas"]["train"]:
        entries = sorted(
            (item for item in metadata if item["split"] == "train" and item["pano_id"] == pano),
            key=lambda item: int(item["face_index"]),
        )
        camera = panorama_camera(entries[0], face_zero, args.erp_width)
        target_np = np.asarray(
            Image.open(raw_scene / f"{pano}_rgb_pano.jpg").convert("RGB").resize(
                (args.erp_width, args.erp_width // 2), Image.Resampling.LANCZOS
            ), dtype=np.float32,
        ) / 255.0
        target = torch.from_numpy(target_np).permute(2, 0, 1).cuda()
        train_items.append((pano, camera, target))
    if not train_items:
        raise RuntimeError("No training panoramas")

    rasterizer = SPaGSRasterizer().cuda()
    history = []
    optimizer_steps = 0
    started = time.perf_counter()
    for step in range(1, args.iterations + 1):
        pano, camera, target = train_items[(step - 1) % len(train_items)]
        optimizer.zero_grad(set_to_none=True)
        rendered = rasterizer(
            positions=positions,
            scales=torch.exp(log_scales),
            rotations=torch.nn.functional.normalize(rotations, dim=-1),
            opacities=torch.sigmoid(opacity_logits),
            sh_0=sh_0,
            sh_rest=sh_rest,
            densification_info=torch.empty(0, device="cuda"),
            camera=camera,
            mode=args.mode,
            K=args.k,
            active_sh_bases=16,
            scale_modifier=1.0,
            use_distance_scaling=False,
        )
        l1 = torch.nn.functional.l1_loss(rendered, target)
        dssim = 1.0 - fused_ssim(rendered[None], target[None])
        loss = 0.8 * l1 + 0.2 * dssim
        loss.backward()
        optimizer.step()
        optimizer_steps += 1
        with torch.no_grad():
            opacity_logits.clamp_(-12.0, 12.0)
            log_scales.clamp_(-12.0, 3.0)
            rotations.copy_(torch.nn.functional.normalize(rotations, dim=-1))
        if step == 1 or step % 50 == 0 or step == args.iterations:
            record = {
                "step": step,
                "panorama": pano,
                "loss": float(loss.detach()),
                "l1": float(l1.detach()),
                "dssim": float(dssim.detach()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    if optimizer_steps != args.iterations:
        raise RuntimeError("Spherical refinement optimizer-update budget mismatch")
    wall_seconds = time.perf_counter() - started
    write_graphdeco_ply(
        input_ply, output_ply,
        positions=positions,
        log_scales=log_scales,
        rotations=rotations,
        opacity_logits=opacity_logits,
        sh_0=sh_0,
        sh_rest=sh_rest,
    )
    summary = {
        "format": "c6pi_direct_spherical_refine_v1",
        "baseline_model": str(baseline),
        "source_ply": str(input_ply),
        "output_ply": str(output_ply),
        "source_path": str(source),
        "iterations": args.iterations,
        "optimizer_steps": optimizer_steps,
        "optimizer_steps_verified": True,
        "erp_resolution": [args.erp_width, args.erp_width // 2],
        "seed": args.seed,
        "gaussians": int(positions.shape[0]),
        "topology_changed": False,
        "heldout_data_used": False,
        "rasterizer": "credited SPaGS differentiable spherical CUDA backend",
        "mode": args.mode,
        "k": args.k,
        "wall_seconds": wall_seconds,
        "learning_rates": {group["name"]: group["lr"] for group in groups},
        "history": history,
    }
    model.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] {output_ply} ({wall_seconds:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
