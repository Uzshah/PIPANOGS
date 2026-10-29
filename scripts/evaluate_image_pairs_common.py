#!/usr/bin/env python3
"""Evaluate paired render/GT directories at one explicit common resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchmetrics.functional.image import (
    learned_perceptual_image_patch_similarity,
    structural_similarity_index_measure,
)


def image_files(path: Path, pattern: str) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg"}
    return sorted(item for item in path.glob(pattern) if item.suffix.lower() in extensions)


def load(path: Path, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--render-glob", default="*")
    parser.add_argument("--gt-glob", default="*")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.width < 2 or args.width % 2:
        parser.error("--width must be a positive even integer")
    renders = image_files(args.render_dir.resolve(), args.render_glob)
    targets = image_files(args.gt_dir.resolve(), args.gt_glob)
    if len(renders) != len(targets) or not renders:
        raise ValueError(f"Pair count mismatch: renders={len(renders)}, targets={len(targets)}")
    size = (args.width, args.width // 2)
    device = torch.device("cuda")
    items = []
    with torch.inference_mode():
        for rendered_path, target_path in zip(renders, targets):
            rendered, target = load(rendered_path, size, device), load(target_path, size, device)
            mse = (rendered - target).square().mean().clamp_min(1e-12)
            items.append({
                "render": rendered_path.name,
                "target": target_path.name,
                "psnr": float((-10.0 * torch.log10(mse)).item()),
                "ssim": float(structural_similarity_index_measure(
                    rendered[None], target[None], data_range=1.0
                ).item()),
                "lpips": float(learned_perceptual_image_patch_similarity(
                    rendered[None], target[None], net_type="vgg", normalize=True
                ).item()),
            })
    metrics = {key: float(np.mean([item[key] for item in items]))
               for key in ("psnr", "ssim", "lpips")}
    result = {
        "format": "common_resolution_image_metrics_v1",
        "resolution": list(size),
        "render_dir": str(args.render_dir.resolve()),
        "gt_dir": str(args.gt_dir.resolve()),
        "render_glob": args.render_glob,
        "gt_glob": args.gt_glob,
        "pairs": len(items),
        "metrics": metrics,
        "per_image": items,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
