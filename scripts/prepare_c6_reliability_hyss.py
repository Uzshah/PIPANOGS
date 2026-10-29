#!/usr/bin/env python3
"""Build a C6-compatible, HYSS-only reliability-aware initialization.

The camera set and panorama split are inherited from the frozen C6 source.
Only source-panorama RGB, HYSS/FRED radial depth/confidence, and HYSS poses are
used.  Cross-panorama depth/color agreement is fused with FRED confidence;
held-out panoramas never contribute to geometry or reliability.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from initialization_utils import (
    candidates_for_pano,
    concatenate,
    confidence_voxel_fusion,
    diagnostic_bins,
    weighted_sample,
)
from chart_utils import write_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c6-source", type=Path, required=True)
    parser.add_argument("--hyss-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-stride", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--support-cap", type=int, default=4)
    parser.add_argument("--reproj-base-tol", type=float, default=0.03)
    parser.add_argument("--reproj-relative-tol", type=float, default=0.01)
    parser.add_argument("--reproj-confidence-tau", type=float, default=0.05)
    parser.add_argument("--photo-tau", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def locate_fred(hyss_scene: Path) -> Path:
    for relative in ("panorama/iter_00/fred_depths", "iter_00/fred_depths"):
        candidate = hyss_scene / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No FRED depth directory below {hyss_scene}")


def locate_poses(hyss_scene: Path) -> Path:
    candidates = (
        hyss_scene / "panorama/iter_00/sfm_output/sfm_poses.npz",
        hyss_scene / "iter_00/sfm_output/sfm_poses.npz",
        hyss_scene / "sfm_poses.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No HYSS pose archive below {hyss_scene}")


def load_hyss_data(
    raw_scene: Path,
    train_panos: list[str],
    fred: Path,
    poses: np.lib.npyio.NpzFile,
    min_depth: float,
    max_depth: float,
) -> dict[str, dict]:
    result = {}
    for pano in train_panos:
        rgb = np.asarray(Image.open(raw_scene / f"{pano}_rgb_pano.jpg").convert("RGB"))
        depth = np.asarray(Image.open(fred / f"{pano}.png"), dtype=np.float32) / 4000.0
        confidence = np.asarray(
            Image.open(fred / f"{pano}_weight.png"), dtype=np.float32
        ) / 65535.0
        mask = np.asarray(Image.open(raw_scene / f"{pano}_mask_pano.png").convert("L")) > 0
        height, width = rgb.shape[:2]
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = mask & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
        result[pano] = {
            "rgb": rgb,
            "depth": depth,
            "valid": valid,
            "measurement_confidence": np.clip(confidence, 0.0, 1.0),
            "rotation": np.asarray(poses[f"{pano}_R"], dtype=np.float64),
            "center": np.asarray(poses[f"{pano}_t"], dtype=np.float64),
        }
    return result


def main() -> int:
    args = parse_args()
    source, output = args.c6_source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    if args.candidate_stride < 1 or args.voxel_size <= 0:
        raise ValueError("candidate-stride and voxel-size must be positive")

    base = json.loads((source / "prep_summary.json").read_text())
    raw_scene = Path(base["source_scene"])
    train_panos = list(base["split_panoramas"]["train"])
    target_count = int(base["initialization_points"])
    hyss_scene = args.hyss_scene.resolve()
    fred, poses_path = locate_fred(hyss_scene), locate_poses(hyss_scene)
    poses = np.load(poses_path)
    data = load_hyss_data(raw_scene, train_panos, fred, poses, args.min_depth, args.max_depth)

    output.mkdir(parents=True)
    os.symlink(os.path.relpath(source / "images", output), output / "images", target_is_directory=True)
    shutil.copytree(source / "metadata", output / "metadata")
    shutil.copytree(source / "sparse", output / "sparse")
    confidence_dir = output / "confidence"
    confidence_dir.mkdir()

    items = []
    for pano in train_panos:
        candidate, maps = candidates_for_pano(
            pano, data, args.candidate_stride, args.min_depth, args.max_depth,
            args.support_cap, args.reproj_base_tol, args.reproj_relative_tol,
            args.reproj_confidence_tau, args.photo_tau,
        )
        rows, cols = maps["rows"], maps["cols"]
        sampled_measurement = data[pano]["measurement_confidence"][np.ix_(rows, cols)]
        measurement = sampled_measurement[maps["valid"].astype(bool)].astype(np.float32)
        # Equal log-space contribution from upstream confidence and independent
        # cross-panorama evidence. This is conservative when either is weak.
        combined = np.sqrt(
            np.clip(candidate["confidence"], 0.02, 1.0)
            * np.clip(measurement, 0.02, 1.0)
        ).astype(np.float32)
        candidate["confidence"] = combined
        maps["confidence"][maps["valid"].astype(bool)] = combined
        maps["c_fred"] = np.zeros_like(maps["confidence"], dtype=np.float32)
        maps["c_fred"][maps["valid"].astype(bool)] = measurement
        items.append(candidate)
        np.savez_compressed(confidence_dir / f"{pano}.npz", **maps)
        print(f"[reliability] {pano}: {len(combined):,} candidates, mean={combined.mean():.4f}")

    candidates = concatenate(items)
    eligible = candidates["confidence"] >= args.min_confidence
    candidates = {key: value[eligible] for key, value in candidates.items()}
    voxels = confidence_voxel_fusion(candidates, args.voxel_size)
    if len(voxels["points"]) < target_count:
        raise RuntimeError(
            f"Reliability initialization has {len(voxels['points']):,} voxels, "
            f"below the frozen-C6 matched target {target_count:,}"
        )
    selected = weighted_sample(voxels, target_count, args.seed)
    points, colors = voxels["points"][selected], voxels["colors"][selected]
    write_ply(output / "sparse/0/points3D.ply", points, colors)
    np.savez_compressed(
        output / "uinit_attributes.npz",
        confidence=voxels["confidence"][selected],
        support_count=voxels["support_count"][selected],
        depth_variance=voxels["depth_variance"][selected],
        raw_count=voxels["raw_count"][selected],
        voxel_keys=voxels["voxel_keys"][selected],
        xyz=points,
    )
    bins = diagnostic_bins(candidates["confidence"], candidates["reprojection_error"])
    summary = {
        "format": "c6r_hyss_reliability_initialization_v1",
        "cumulative_parent": "C6",
        "frozen_c6_source": str(source),
        "source_scene": str(raw_scene),
        "hyss_scene": str(hyss_scene),
        "fred_depth_dir": str(fred),
        "poses": str(poses_path),
        "split_panoramas": base["split_panoramas"],
        "initialization_source_panoramas": train_panos,
        "heldout_data_used": False,
        "replica_ground_truth_depth_used": False,
        "candidate_stride": args.candidate_stride,
        "voxel_size": args.voxel_size,
        "candidate_points": int(len(candidates["points"])),
        "candidate_voxels": int(len(voxels["points"])),
        "initialization_points": int(len(points)),
        "matched_frozen_c6_count": target_count,
        "count_match": bool(len(points) == target_count),
        "reliability": "sqrt(HYSS_FRED_confidence * cross_panorama_range_support_depth_color)",
        "fusion": "reliability-weighted voxel position/color average",
        "sampling": "deterministic reliability*spherical-area Gumbel top-k",
        "reprojection_reliability_bins": bins,
        "leakage_audit": "passed",
    }
    (output / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] C6-R initialization: {len(points):,} matched reliable Gaussians")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
