#!/usr/bin/env python3
"""Evaluate a C6 Graphdeco PLY with the pinned SPaGS spherical CUDA kernel.

This is an inference/rasterizer control, not a reimplementation of SPaGS.  It
keeps the learned C6 Gaussian scene fixed and changes only the chart-dependent
projection, sorting, and blending path from gnomonic charts to direct ERP.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
from torchmetrics.functional.image import (
    learned_perceptual_image_patch_similarity,
    structural_similarity_index_measure,
)
import torchvision

from Cameras.Equirectangular import EquirectangularCamera
from Cameras.utils import CameraProperties
from native_gs.spags_backend import SPaGSRasterizer


# SPaGS local ERP rays are (sin(lon) cos(lat), sin(lat),
# -cos(lon) cos(lat)); C6's ERP rays are (cos(lat) cos(lon), sin(lat),
# cos(lat) sin(lon)).  This proper rotation maps the former to the latter.
SPAGS_TO_C6_ERP = np.asarray([
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=np.float32)


def load_graphdeco_ply(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    vertex = PlyData.read(str(path))["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "opacity"}
    required |= {f"f_dc_{index}" for index in range(3)}
    required |= {f"scale_{index}" for index in range(3)}
    required |= {f"rot_{index}" for index in range(4)}
    missing = required - names
    if missing:
        raise ValueError(f"Missing Graphdeco PLY fields: {sorted(missing)}")

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)

    positions = tensor(np.column_stack([vertex[axis] for axis in ("x", "y", "z")]))
    sh_0 = tensor(np.column_stack([vertex[f"f_dc_{index}"] for index in range(3)]))[:, None]
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(rest_names) != 45:
        raise ValueError(f"Expected degree-3 SH (45 rest fields), found {len(rest_names)}")
    # Graphdeco stores channel-major [3, 15]; SPaGS consumes [15, 3].
    sh_rest = tensor(np.column_stack([vertex[name] for name in rest_names]))
    sh_rest = sh_rest.reshape(-1, 3, 15).transpose(1, 2).contiguous()
    scales = torch.exp(tensor(np.column_stack([vertex[f"scale_{i}"] for i in range(3)])))
    rotations = torch.nn.functional.normalize(
        tensor(np.column_stack([vertex[f"rot_{i}"] for i in range(4)])), dim=-1
    )
    opacities = torch.sigmoid(tensor(np.asarray(vertex["opacity"]))[:, None])
    return {
        "positions": positions.contiguous(),
        "scales": scales.contiguous(),
        "rotations": rotations.contiguous(),
        "opacities": opacities.contiguous(),
        "sh_0": sh_0.contiguous(),
        "sh_rest": sh_rest,
    }


def panorama_camera(entry: dict, face_zero: np.ndarray, width: int) -> EquirectangularCamera:
    pano_c2w = np.asarray(entry["rotation"], dtype=np.float32) @ face_zero.T
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = pano_c2w @ SPAGS_TO_C6_ERP
    c2w[:3, 3] = np.asarray(entry["position"], dtype=np.float32)
    properties = CameraProperties(
        width=width,
        height=width // 2,
        c2w=torch.from_numpy(c2w).cuda(),
        _is_device_copy=True,
    )
    return EquirectangularCamera(near_plane=0.01, far_plane=100.0).setProperties(properties)


def quality(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = (rendered - target).square().mean().clamp_min(1e-12)
    return {
        "psnr": float((-10.0 * torch.log10(mse)).item()),
        "ssim": float(structural_similarity_index_measure(
            rendered[None], target[None], data_range=1.0
        ).item()),
        "lpips": float(learned_perceptual_image_patch_similarity(
            rendered[None], target[None], net_type="vgg", normalize=True
        ).item()),
    }


def mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
    keys = ("psnr", "ssim", "lpips")
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=7500)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--erp-width", type=int, default=1024)
    parser.add_argument("--mode", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--speed-repeats", type=int, default=100)
    parser.add_argument("--parameter-refined", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.erp_width < 2 or args.erp_width % 2:
        parser.error("--erp-width must be a positive even integer")

    model = args.model_path.resolve()
    source = args.source_path.resolve()
    ply = model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    parameters = load_graphdeco_ply(ply, torch.device("cuda"))
    metadata = json.loads((source / "metadata/cameras.json").read_text())
    faces = json.loads((source / "metadata/faces.json").read_text())
    prep = json.loads((source / "prep_summary.json").read_text())
    raw_scene = Path(prep["source_scene"])
    face_zero = np.asarray(faces[0]["tile_rotation"], dtype=np.float32)
    by_pano: dict[str, list[dict]] = {}
    for entry in metadata:
        if entry["split"] == args.split:
            by_pano.setdefault(str(entry["pano_id"]), []).append(entry)
    cameras = []
    for pano in prep["split_panoramas"][args.split]:
        entries = sorted(by_pano[pano], key=lambda item: int(item["face_index"]))
        if not entries:
            raise KeyError(f"Missing camera metadata for panorama {pano}")
        cameras.append((pano, panorama_camera(entries[0], face_zero, args.erp_width)))

    rasterizer = SPaGSRasterizer().cuda()
    common = dict(
        **parameters,
        mode=args.mode,
        K=args.k,
        active_sh_bases=16,
        scale_modifier=1.0,
    )
    output_dir = model / args.split / f"spherical_cuda_{args.iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    with torch.inference_mode():
        for pano, camera in cameras:
            rendered, _ = rasterizer.render(
                **common, camera=camera, to_chw=True, use_median_depth=False
            )
            rendered = rendered.clamp(0.0, 1.0)
            target_np = np.asarray(
                Image.open(raw_scene / f"{pano}_rgb_pano.jpg").convert("RGB").resize(
                    (args.erp_width, args.erp_width // 2), Image.Resampling.LANCZOS
                ), dtype=np.float32,
            ) / 255.0
            target = torch.from_numpy(target_np).permute(2, 0, 1).cuda()
            item = quality(rendered, target)
            item["panorama"] = pano
            scores.append(item)
            torchvision.utils.save_image(rendered, output_dir / f"{pano}_render.png")
            torchvision.utils.save_image(target, output_dir / f"{pano}_gt.png")

        # Benchmark the no-allocation inference kernel, matching SPaGS's own
        # published timing path. Warm-up is excluded and CUDA is synchronized.
        for _, camera in cameras:
            rasterizer.benchmark(**common, camera=camera, to_chw=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.speed_repeats):
            for _, camera in cameras:
                rasterizer.benchmark(**common, camera=camera, to_chw=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    frames = args.speed_repeats * len(cameras)
    result = {
        "format": "c6pi_spags_spherical_cuda_control_v1",
        "method": "C6-PI fixed Gaussians + credited SPaGS spherical CUDA rasterizer",
        "model": str(model),
        "source_ply": str(ply),
        "split": args.split,
        "iteration": args.iteration,
        "gaussians": int(parameters["positions"].shape[0]),
        "erp_resolution": [args.erp_width, args.erp_width // 2],
        "mode": args.mode,
        "k": args.k,
        "panoramas": len(cameras),
        "quality": mean_dict(scores),
        "per_panorama": scores,
        "render_ms": 1000.0 * elapsed / frames,
        "fps": frames / elapsed,
        "chart_invariant": ["means", "scales", "rotations", "opacities", "SH coefficients"],
        "chart_dependent": ["direct ERP projection", "tile assignment", "sorting", "blending"],
        "training_changed": args.parameter_refined,
        "rasterizer_credit": "SPaGS official CUDA backend (pinned third_party/SPaGS)",
    }
    output = args.output.resolve() if args.output else model / "c6pi_spherical_cuda.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
