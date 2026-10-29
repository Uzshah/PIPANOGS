#!/usr/bin/env python3
"""C6-PI: fused panorama training with projection-invariant reliability."""

from __future__ import annotations

import json
import random
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from native_gs.batched_cuda_renderer import (
    packed_max_radii, prepare_batched_invariants, render_batched_cuda,
)
from native_gs.projection_invariant_gaussian_model import ProjectionInvariantGaussianModel
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def load_groups(cameras, source: Path, map_root: Path):
    metadata = json.loads((source / "metadata/cameras.json").read_text())
    info = {Path(item["img_name"]).stem: item for item in metadata}
    face_weights, groups = {}, defaultdict(list)
    valid_depth = []
    for camera in cameras:
        stem = Path(camera.image_name).stem
        entry = info[stem]
        face = int(entry["face_index"])
        if face not in face_weights:
            face_weights[face] = torch.from_numpy(np.load(
                map_root / "spherical_weights" / f"face_{face:03d}.npy"
            ).astype(np.float32))[None].cuda()
        depth = np.load(map_root / "depth_confidence_hyss" / f"{stem}.npz")
        camera.pano_ray_weight = face_weights[face]
        camera.pano_invdepth = torch.from_numpy(depth["invdepth"].astype(np.float32))[None]
        camera.pano_depth_confidence = torch.from_numpy(depth["confidence"].astype(np.float32))[None]
        groups[str(entry["pano_id"])].append(camera)
        valid_depth.append(float((depth["confidence"] > 0).mean()))
    for group in groups.values():
        center = group[0].camera_center
        if any(not torch.allclose(center, camera.camera_center, atol=1e-5, rtol=0) for camera in group[1:]):
            raise RuntimeError("A panorama group contains different camera centers")
    return dict(groups), {"panoramas": len(groups), "cameras": len(cameras),
                          "mean_depth_valid": float(np.mean(valid_depth))}


def enforce_budget(gaussians: GaussianModel, maximum: int) -> int:
    count = int(gaussians.get_xyz.shape[0])
    if maximum <= 0 or count <= maximum:
        return 0
    opacity = gaussians.get_opacity.detach().squeeze()
    scale = gaussians.get_scaling.detach().amax(dim=1)
    score = opacity * scale
    keep = torch.topk(score, maximum, sorted=False).indices
    keep_mask = torch.zeros(count, dtype=torch.bool, device="cuda")
    keep_mask[keep] = True
    gaussians.prune_points(~keep_mask)
    return count - maximum


def train(dataset, opt, pipe, args):
    output, source = Path(dataset.model_path).resolve(), Path(dataset.source_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    attribute_path = source / "uinit_attributes.npz"
    if not attribute_path.is_file():
        raise FileNotFoundError(f"C6-PI requires aligned reliability attributes: {attribute_path}")
    gaussians = ProjectionInvariantGaussianModel(dataset.sh_degree, attribute_path, args.support_cap)
    gaussians.densification_gradient_mode = getattr(args, "densification_gradient", "screen")
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    groups, group_summary = load_groups(scene.getTrainCameras(), source, Path(args.map_root).resolve())
    pano_ids = sorted(groups)
    if not pano_ids or any(len(group) < args.chart_batch for group in groups.values()):
        raise ValueError("Every training panorama must supply the configured chart batch")
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    ema = {"loss": 0.0, "photo": 0.0, "depth": 0.0}
    rendered_charts, budget_pruned, observation_updates = 0, 0, 0
    densification_history = []
    optimizer_steps = 0
    started = time.time()
    progress = tqdm(range(1, opt.iterations + 1), desc="C6-PI fused panorama")
    for iteration in progress:
        pano = random.choice(pano_ids)
        group = groups[pano]
        batch_size = min(args.chart_batch, len(group))
        # Match Graphdeco schedules by supervised chart renders rather than
        # shared-origin optimizer updates.
        schedule_iteration = rendered_charts + batch_size
        gaussians.update_learning_rate(schedule_iteration)
        if schedule_iteration // 1000 > rendered_charts // 1000:
            gaussians.oneupSHdegree()
        cameras = random.sample(group, batch_size)
        cache = prepare_batched_invariants(gaussians, cameras[0].camera_center)
        bg = torch.rand(3, device="cuda") if opt.random_background else background
        package = render_batched_cuda(
            cameras, gaussians, cache, bg, render_inverse_depth=True, packed=True)
        totals = {"loss": 0.0, "photo": 0.0, "depth": 0.0}
        losses, confidence_maps = [], []
        for index, camera in enumerate(cameras):
            image, target = package["render"][index], camera.original_image.cuda()
            if camera.alpha_mask is not None:
                image = image * camera.alpha_mask.cuda()
            residual = (image - target).abs().mean(dim=0, keepdim=True)
            balanced = (1.0 - args.ray_balance) * l1_loss(image, target)
            balanced += args.ray_balance * weighted_mean(residual, camera.pano_ray_weight)
            photo = ((1.0 - opt.lambda_dssim) * balanced
                     + opt.lambda_dssim * (1.0 - ssim(image, target)))
            confidence = camera.pano_depth_confidence.cuda(non_blocking=True)
            confidence_maps.append(confidence)
            invdepth = camera.pano_invdepth.cuda(non_blocking=True)
            valid = (confidence > 0) & torch.isfinite(invdepth) & (invdepth > 0)
            depth_residual = F.huber_loss(package["depth"][index], invdepth, reduction="none",
                                          delta=args.depth_huber_delta)
            depth_raw = weighted_mean(
                depth_residual, torch.where(valid, confidence, torch.zeros_like(confidence)))
            warmup = min(schedule_iteration / max(args.depth_warmup, 1), 1.0)
            depth = args.depth_weight * warmup * depth_raw
            loss = photo + depth
            losses.append(loss)
            totals["loss"] += float(loss.detach()) / batch_size
            totals["photo"] += float(photo.detach()) / batch_size
            totals["depth"] += float(depth.detach()) / batch_size
        torch.stack(losses).mean().backward()
        with torch.no_grad():
            info = package["info"]
            batch_radii = packed_max_radii(info, gaussians.u_count)
            visible_any = batch_radii > 0
            observation_updates += gaussians.observe_panorama_packed(
                cameras, confidence_maps, info, args.reliability_ema_beta)
            if schedule_iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visible_any] = torch.maximum(
                    gaussians.max_radii2D[visible_any], batch_radii[visible_any])
                gaussians.add_packed_panorama_densification_stats(cameras, info)
        rendered_charts += batch_size
        with torch.no_grad():
            densify_crossed = (
                schedule_iteration // opt.densification_interval
                > (schedule_iteration - batch_size) // opt.densification_interval
            )
            if (schedule_iteration < opt.densify_until_iter
                    and schedule_iteration > opt.densify_from_iter and densify_crossed):
                threshold = 20 if schedule_iteration > opt.opacity_reset_interval else None
                event = gaussians.densify_and_prune_uopt(
                    args.reliability_score_threshold, 0.005, scene.cameras_extent,
                    threshold, batch_radii, args.reliability_confidence_gamma,
                    args.reliability_support_floor, args.reliability_prune_confidence,
                    args.reliability_prune_support,
                    args.reliability_score_mode,
                    args.reliability_modulation_floor,
                    args.reliability_modulation_ceiling,
                )
                event["rendered_charts"] = rendered_charts
                densification_history.append(event)
                budget_pruned += enforce_budget(gaussians, args.max_gaussians)
            opacity_crossed = (
                schedule_iteration // opt.opacity_reset_interval
                > (schedule_iteration - batch_size) // opt.opacity_reset_interval
            )
            if schedule_iteration < opt.densify_until_iter and opacity_crossed:
                gaussians.reset_opacity()
            gaussians.optimizer.step()
            optimizer_steps += 1
            gaussians.optimizer.zero_grad(set_to_none=True)
            for key in ema:
                ema[key] = 0.4 * totals[key] + 0.6 * ema[key]
            if iteration % 10 == 0:
                progress.set_postfix(loss=f"{ema['loss']:.5f}", n=f"{len(gaussians.get_xyz):,}")
    if optimizer_steps != opt.iterations:
        raise RuntimeError(f"Optimizer update mismatch: {optimizer_steps} != {opt.iterations}")
    scene.save(opt.iterations)
    summary = {
        "format": "c6pi_camera_batched_projection_invariant_training_v1", "source_path": str(source),
        "cumulative_parent": "C6",
        "output_model": str(output), "graphdeco_pretrained_model_used": False,
        "initialization": "HYSS FRED predicted depth", "optimizer_steps": optimizer_steps,
        "optimizer_steps_verified": True, "requested_optimizer_steps": opt.iterations,
        "seed": args.seed,
        "chart_batch": args.chart_batch, "rendered_charts": rendered_charts,
        "schedule_unit": "rendered_chart",
        "shared_per_batch": ["SH color", "Gaussian parameters", "one camera-batched CUDA rasterization"],
        "gaussians": int(gaussians.get_xyz.shape[0]), "max_gaussians": args.max_gaussians,
        "budget_pruned": budget_pruned, "ray_balance": args.ray_balance,
        "depth_weight": args.depth_weight, "training_seconds": time.time() - started,
        "reliability_state": "one overlap-normalized confidence/support EMA update per panorama",
        "reliability_projection_invariant": True,
        "densification_projection_invariant": False,
        "densification_gradient_convention": gaussians.densification_gradient_mode,
        "densification_gradient_units": "historical NDC-style norm",
        "cuda_rasterizer_calls_per_batch": 1,
        "reliability_guided_densification": True,
        "reliability_score_mode": args.reliability_score_mode,
        "reliability_score_threshold": args.reliability_score_threshold,
        "reliability_confidence_gamma": args.reliability_confidence_gamma,
        "reliability_support_floor": args.reliability_support_floor,
        "reliability_modulation_floor": args.reliability_modulation_floor,
        "reliability_modulation_ceiling": args.reliability_modulation_ceiling,
        "reliability_prune_confidence": args.reliability_prune_confidence,
        "reliability_prune_support": args.reliability_prune_support,
        "observation_updates": observation_updates,
        "final_confidence_mean": float(gaussians.u_confidence.mean().item()),
        "final_support_mean": float(gaussians.u_support.mean().item()),
        "densification_history": densification_history,
        "ema": ema, "groups": group_summary,
        "rasterizer": "gsplat 1.5 camera-batched CUDA 3DGS (Graphdeco parameterization)",
    }
    (output / "panorama_centric_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] C6-PI fused panorama: {summary['gaussians']:,} Gaussians")


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    model, optimization, pipeline = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument("--map-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--densification-gradient", choices=("screen",), default="screen")
    parser.add_argument("--chart-batch", type=int, default=4)
    parser.add_argument("--ray-balance", type=float, default=0.5)
    parser.add_argument("--depth-weight", type=float, default=0.01)
    parser.add_argument("--depth-huber-delta", type=float, default=0.05)
    parser.add_argument("--depth-warmup", type=int, default=200)
    parser.add_argument("--max-gaussians", type=int, default=2_000_000)
    parser.add_argument("--support-cap", type=float, default=4.0)
    parser.add_argument("--reliability-ema-beta", type=float, default=0.95)
    parser.add_argument("--reliability-score-threshold", type=float, default=2e-4)
    parser.add_argument("--reliability-confidence-gamma", type=float, default=0.5)
    parser.add_argument("--reliability-support-floor", type=float, default=0.25)
    parser.add_argument("--reliability-prune-confidence", type=float, default=0.15)
    parser.add_argument("--reliability-prune-support", type=float, default=0.35)
    parser.add_argument("--reliability-score-mode", choices=("suppress", "mean_preserving"),
                        default="mean_preserving")
    parser.add_argument("--reliability-modulation-floor", type=float, default=0.95)
    parser.add_argument("--reliability-modulation-ceiling", type=float, default=1.05)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.chart_batch < 1 or not 0 <= args.ray_balance <= 1:
        parser.error("chart-batch must be positive and ray-balance must be in [0,1]")
    if not 0 < args.reliability_modulation_floor <= args.reliability_modulation_ceiling:
        parser.error("reliability modulation bounds must satisfy 0 < floor <= ceiling")
    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(model.extract(args), optimization.extract(args), pipeline.extract(args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
