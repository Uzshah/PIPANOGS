#!/usr/bin/env python3
"""Evaluate C0 at held-out panorama centers with the original 3DGS rasterizer.

The script renders the explicit val/test camera lists written by
``prepare_c0_hyss_gnomonic.py``.  It reports perspective-view metrics and
reprojects all 80 overlapping gnomonic views into an ERP image for panorama
metrics and an overlap-disagreement (seam) diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torchvision

from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix
from utils.loss_utils import ssim


def camera_from_metadata(entry: dict, device: torch.device) -> MiniCam:
    width, height = int(entry["width"]), int(entry["height"])
    fov_x = 2.0 * math.atan(width / (2.0 * float(entry["fx"])))
    fov_y = 2.0 * math.atan(height / (2.0 * float(entry["fy"])))
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(entry["rotation"], dtype=np.float32)
    c2w[:3, 3] = np.asarray(entry["position"], dtype=np.float32)
    world_view = torch.from_numpy(np.linalg.inv(c2w)).to(device).transpose(0, 1)
    projection = getProjectionMatrix(0.01, 100.0, fov_x, fov_y).to(device).transpose(0, 1)
    return MiniCam(
        width,
        height,
        fov_y,
        fov_x,
        0.01,
        100.0,
        world_view,
        world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
    )


def tile_map(rotation: np.ndarray, fov: float, patch_size: int, width: int, height: int):
    half = math.tan(0.5 * fov)
    axis = np.linspace(-half, half, patch_size, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(axis, axis, indexing="ij")
    local = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    rays = np.einsum("...j,ij->...i", local, rotation.astype(np.float32))
    longitude = np.arctan2(rays[..., 2], rays[..., 0])
    latitude = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
    source_x = (longitude + math.pi) / (2.0 * math.pi) * (width - 1)
    source_y = (latitude + math.pi / 2.0) / math.pi * (height - 1)
    return source_x, source_y


def tukey_window(size: int, rho: float, device: torch.device) -> torch.Tensor:
    coordinate = torch.linspace(-1.0, 1.0, size, device=device).abs()
    start = 1.0 - rho
    one_d = torch.ones_like(coordinate)
    edge = coordinate > start
    one_d[edge] = 0.5 * (
        1.0 + torch.cos(math.pi * (coordinate[edge] - start) / max(rho, 1e-6))
    )
    return one_d[:, None] * one_d[None, :]


class ERPFuser:
    def __init__(self, width: int, height: int, patch_size: int, device: torch.device):
        self.width = width
        self.height = height
        self.device = device
        self.rgb = torch.zeros((height * width, 3), device=device)
        self.rgb2 = torch.zeros_like(self.rgb)
        self.weight = torch.zeros(height * width, device=device)
        self.contributors = torch.zeros(height * width, device=device)
        self.window = tukey_window(patch_size, 0.35, device)

    def add(self, tile: torch.Tensor, map_x_np: np.ndarray, map_y_np: np.ndarray) -> None:
        map_x = torch.from_numpy(map_x_np).to(self.device)
        map_y = torch.from_numpy(map_y_np).to(self.device)
        x0 = torch.floor(map_x).long()
        y0 = torch.floor(map_y).long().clamp(0, self.height - 1)
        dx = map_x - x0.float()
        dy = map_y - y0.float()
        values = tile.permute(1, 2, 0).reshape(-1, 3)
        base = self.window
        for xx, yy, bilinear in (
            (x0, y0, (1.0 - dx) * (1.0 - dy)),
            (x0 + 1, y0, dx * (1.0 - dy)),
            (x0, y0 + 1, (1.0 - dx) * dy),
            (x0 + 1, y0 + 1, dx * dy),
        ):
            xx = torch.remainder(xx, self.width)
            yy = yy.clamp(0, self.height - 1)
            weight = (base * bilinear).reshape(-1)
            index = (yy * self.width + xx).reshape(-1)
            weighted = values * weight[:, None]
            self.rgb.index_add_(0, index, weighted)
            self.rgb2.index_add_(0, index, values.square() * weight[:, None])
            self.weight.index_add_(0, index, weight)
            self.contributors.index_add_(0, index, (weight > 1e-6).float())

    def finish(self) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        valid = self.weight > 1e-8
        mean = self.rgb / self.weight.clamp_min(1e-8)[:, None]
        variance = (self.rgb2 / self.weight.clamp_min(1e-8)[:, None] - mean.square()).clamp_min(0)
        overlap = valid & (self.contributors > 1.0)
        if overlap.any():
            seam_rmse = torch.sqrt(variance[overlap].mean()).item()
        else:
            seam_rmse = float("nan")
        image = mean.reshape(self.height, self.width, 3).permute(2, 0, 1).clamp(0, 1)
        mask = valid.reshape(self.height, self.width)
        coverage = valid.float().mean().item()
        return image, mask, coverage, seam_rmse


def metrics(rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    if mask is None:
        squared = (rendered - target).square()
        absolute = (rendered - target).abs()
    else:
        expanded = mask[None].expand_as(rendered)
        squared = (rendered - target).square()[expanded]
        absolute = (rendered - target).abs()[expanded]
    mse = squared.mean().clamp_min(1e-12)
    result = {
        "psnr": float((-10.0 * torch.log10(mse)).item()),
        "l1": float(absolute.mean().item()),
    }
    # SSIM is defined over a dense image; ERP coverage is reported separately.
    result["ssim"] = float(ssim(rendered[None], target[None]).item())
    return result


def mean_metrics(items: list[dict]) -> dict:
    return {key: float(np.mean([item[key] for item in items])) for key in items[0]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--erp-width", type=int, default=1024)
    parser.add_argument("--white-background", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.resolve()
    source_path = args.source_path.resolve()
    if args.erp_width < 2 or args.erp_width % 2:
        raise ValueError("--erp-width must be an even integer >= 2")
    metadata = json.loads((source_path / "metadata" / "cameras.json").read_text())
    faces = json.loads((source_path / "metadata" / "faces.json").read_text())
    prep = json.loads((source_path / "prep_summary.json").read_text())
    raw_scene = Path(prep["source_scene"])

    device = torch.device("cuda")
    gaussians = GaussianModel(3)
    ply_path = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    gaussians.load_ply(str(ply_path))
    background = torch.ones(3, device=device) if args.white_background else torch.zeros(3, device=device)
    pipe = SimpleNamespace(debug=False, antialiasing=False, compute_cov3D_python=False, convert_SHs_python=False)
    maps = [
        tile_map(
            np.asarray(face["tile_rotation"]),
            float(face["fov_radians"]),
            int(metadata[0]["width"]),
            args.erp_width,
            args.erp_width // 2,
        )
        for face in faces
    ]

    summary = {
        "format": "heldout_graphdeco_evaluation_v1",
        "rasterizer": "graphdeco.diff-gaussian-rasterization",
        "model": str(model_path),
        "source_ply": str(ply_path),
        "iteration": args.iteration,
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "split_panoramas": prep["split_panoramas"],
        "erp_resolution": [args.erp_width, args.erp_width // 2],
        "splits": {},
    }
    for split in args.splits:
        entries = [entry for entry in metadata if entry["split"] == split]
        output_dir = model_path / split / f"ours_{args.iteration}"
        render_dir, gt_dir = output_dir / "renders", output_dir / "gt"
        erp_dir = output_dir / "erp"
        render_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        erp_dir.mkdir(parents=True, exist_ok=True)
        tile_scores: list[dict] = []
        pano_scores: list[dict] = []
        start = time.perf_counter()
        for pano_id in prep["split_panoramas"][split]:
            pano_entries = sorted(
                (entry for entry in entries if entry["pano_id"] == pano_id),
                key=lambda entry: entry["face_index"],
            )
            render_fuser = ERPFuser(args.erp_width, args.erp_width // 2, int(entries[0]["width"]), device)
            gt_fuser = ERPFuser(args.erp_width, args.erp_width // 2, int(entries[0]["width"]), device)
            for local_index, entry in enumerate(pano_entries):
                camera = camera_from_metadata(entry, device)
                with torch.no_grad():
                    rendered = render(camera, gaussians, pipe, background)["render"].clamp(0, 1)
                target_np = np.asarray(
                    Image.open(source_path / "images" / entry["img_name"]).convert("RGB"),
                    dtype=np.float32,
                ) / 255.0
                target = torch.from_numpy(target_np).permute(2, 0, 1).to(device)
                score = metrics(rendered, target)
                score.update({"image": entry["img_name"], "pano_id": pano_id, "face_index": entry["face_index"]})
                tile_scores.append(score)
                torchvision.utils.save_image(rendered, render_dir / entry["img_name"])
                torchvision.utils.save_image(target, gt_dir / entry["img_name"])
                map_x, map_y = maps[entry["face_index"]]
                render_fuser.add(rendered, map_x, map_y)
                gt_fuser.add(target, map_x, map_y)
                if (local_index + 1) % 20 == 0:
                    print(f"[{split}/{pano_id}] rendered {local_index + 1}/{len(pano_entries)}")

            erp_render, coverage_mask, coverage, seam_rmse = render_fuser.finish()
            erp_tile_gt, gt_coverage_mask, gt_coverage, gt_seam_rmse = gt_fuser.finish()
            resampling = getattr(Image, "Resampling", Image)
            erp_gt_np = np.asarray(
                Image.open(raw_scene / f"{pano_id}_rgb_pano.jpg")
                .convert("RGB")
                .resize((args.erp_width, args.erp_width // 2), resampling.LANCZOS),
                dtype=np.float32,
            ) / 255.0
            erp_gt = torch.from_numpy(erp_gt_np).permute(2, 0, 1).to(device)
            pano_score = metrics(erp_render, erp_gt, coverage_mask)
            ceiling_score = metrics(erp_tile_gt, erp_gt, gt_coverage_mask)
            pano_score.update({
                "pano_id": pano_id,
                "coverage": coverage,
                "prediction_overlap_seam_rmse": seam_rmse,
                "gt_tile_fusion_coverage": gt_coverage,
                "gt_tile_overlap_seam_rmse": gt_seam_rmse,
                "gt_tile_fusion_ceiling": ceiling_score,
            })
            pano_scores.append(pano_score)
            torchvision.utils.save_image(erp_render, erp_dir / f"{pano_id}_render.png")
            torchvision.utils.save_image(erp_gt, erp_dir / f"{pano_id}_gt.png")
            torchvision.utils.save_image(erp_tile_gt, erp_dir / f"{pano_id}_tile_gt_fused.png")
            torchvision.utils.save_image(coverage_mask.float()[None], erp_dir / f"{pano_id}_coverage.png")
            print(
                f"[{split}/{pano_id}] ERP PSNR={pano_score['psnr']:.3f} "
                f"SSIM={pano_score['ssim']:.4f} coverage={coverage:.5f}"
            )

        summary["splits"][split] = {
            "tiles": len(entries),
            "panoramas": len(pano_scores),
            "perspective_mean": mean_metrics([
                {key: score[key] for key in ("psnr", "ssim", "l1")} for score in tile_scores
            ]),
            "erp_mean": mean_metrics([
                {key: score[key] for key in ("psnr", "ssim", "l1", "coverage", "prediction_overlap_seam_rmse")}
                for score in pano_scores
            ]),
            "per_panorama": pano_scores,
            "per_view": tile_scores,
            "render_and_fusion_seconds": time.perf_counter() - start,
        }
        print(
            f"[{split}] perspective PSNR={summary['splits'][split]['perspective_mean']['psnr']:.3f}, "
            f"ERP PSNR={summary['splits'][split]['erp_mean']['psnr']:.3f}"
        )

    output_json = model_path / "heldout_metrics.json"
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] metrics -> {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
