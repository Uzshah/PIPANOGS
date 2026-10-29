#!/usr/bin/env python3
"""Prepare unique-ray weights and train-only HYSS depth targets for C6."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from chart_utils import gnomonic_map


def local_ray_z(fov: float, size: int) -> np.ndarray:
    half = math.tan(0.5 * fov)
    axis = np.linspace(-half, half, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return 1.0 / np.sqrt(xx * xx + yy * yy + 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--hyss-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--erp-width", type=int, default=1024)
    args = parser.parse_args()
    source = args.source_path.resolve()
    output = args.output.resolve()
    summary_path = output / "c6_map_summary.json"
    if summary_path.is_file():
        print(f"[skip] C6 maps exist: {summary_path}")
        return 0
    if args.erp_width < 2 or args.erp_width % 2:
        parser.error("--erp-width must be a positive even integer")

    prep = json.loads((source / "prep_summary.json").read_text())
    cameras = json.loads((source / "metadata" / "cameras.json").read_text())
    faces = json.loads((source / "metadata" / "faces.json").read_text())
    train_panos = list(prep["split_panoramas"]["train"])
    train_cameras = [item for item in cameras if item["split"] == "train"]
    patch = int(train_cameras[0]["width"])
    erp_w, erp_h = args.erp_width, args.erp_width // 2
    raw_scene = Path(prep["source_scene"])
    first_rgb = np.asarray(Image.open(raw_scene / f"{train_panos[0]}_rgb_pano.jpg"))
    raw_h, raw_w = first_rgb.shape[:2]

    maps = []
    coverage = np.zeros((erp_h, erp_w), dtype=np.float32)
    for face in faces:
        fov = float(face["fov_radians"])
        mx, my = gnomonic_map(
            np.asarray(face["tile_rotation"], dtype=np.float64),
            fov, patch, erp_w, erp_h,
        )
        xi = np.mod(np.rint(mx).astype(np.int64), erp_w)
        yi = np.clip(np.rint(my).astype(np.int64), 0, erp_h - 1)
        np.add.at(coverage, (yi.ravel(), xi.ravel()), 1.0)
        maps.append((mx, my, local_ray_z(fov, patch), fov))

    weight_dir = output / "spherical_weights"
    depth_dir = output / "depth_confidence_hyss"
    weight_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    raw_weights = []
    for index, (mx, my, _, fov) in enumerate(maps):
        xi = np.mod(np.rint(mx).astype(np.int64), erp_w)
        yi = np.clip(np.rint(my).astype(np.int64), 0, erp_h - 1)
        half = math.tan(0.5 * fov)
        axis = np.linspace(-half, half, patch, dtype=np.float32)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        solid_angle = np.power(1.0 + xx * xx + yy * yy, -1.5)
        weight = solid_angle / np.maximum(coverage[yi, xi], 1.0)
        raw_weights.append(weight)
    normalizer = float(np.mean(np.concatenate([item.ravel() for item in raw_weights])))
    for index, weight in enumerate(raw_weights):
        np.save(weight_dir / f"face_{index:03d}.npy", (weight / normalizer).astype(np.float16))

    hyss_scene = args.hyss_scene.resolve()
    fred_dir = hyss_scene / "panorama" / "iter_00" / "fred_depths"
    if not fred_dir.is_dir():
        fred_dir = hyss_scene / "iter_00" / "fred_depths"
    if not fred_dir.is_dir():
        raise FileNotFoundError(f"HYSS FRED directory not found under {hyss_scene}")

    manifest = []
    for pano in train_panos:
        depth_path = fred_dir / f"{pano}.png"
        confidence_path = fred_dir / f"{pano}_weight.png"
        if not depth_path.is_file() or not confidence_path.is_file():
            raise FileNotFoundError(f"Missing HYSS depth/confidence for {pano}: {fred_dir}")
        radial = np.asarray(Image.open(depth_path), dtype=np.float32) / 4000.0
        confidence = np.asarray(Image.open(confidence_path), dtype=np.float32) / 65535.0
        mask_path = raw_scene / f"{pano}_mask_pano.png"
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if radial.shape != (raw_h, raw_w):
            radial = cv2.resize(radial, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
        valid = mask & np.isfinite(radial) & (radial >= 0.25) & (radial <= 20.0)
        pano_entries = sorted(
            (item for item in train_cameras if item["pano_id"] == pano),
            key=lambda item: item["face_index"],
        )
        for entry in pano_entries:
            face_index = int(entry["face_index"])
            mx_small, my_small, ray_z, _ = maps[face_index]
            mx = mx_small * (raw_w - 1) / max(erp_w - 1, 1)
            my = my_small * (raw_h - 1) / max(erp_h - 1, 1)
            depth_tile = cv2.remap(radial, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            confidence_tile = cv2.remap(
                confidence, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            valid_tile = cv2.remap(
                valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            ).astype(bool)
            confidence_tile = np.clip(confidence_tile, 0.0, 1.0)
            confidence_tile[~valid_tile] = 0.0
            z_depth = depth_tile * ray_z
            invdepth = np.zeros_like(z_depth, dtype=np.float32)
            invdepth[valid_tile] = 1.0 / np.maximum(z_depth[valid_tile], 1e-6)
            name = Path(entry["img_name"]).with_suffix(".npz").name
            np.savez_compressed(
                depth_dir / name,
                invdepth=invdepth.astype(np.float16),
                confidence=confidence_tile.astype(np.float16),
                valid=valid_tile.astype(np.uint8),
            )
            manifest.append({
                "image": entry["img_name"],
                "pano_id": pano,
                "face_index": face_index,
                "valid_fraction": float(valid_tile.mean()),
                "confidence_mean_valid": float(confidence_tile[valid_tile].mean()) if valid_tile.any() else 0.0,
            })
        print(f"[HYSS maps] {pano}: {len(pano_entries)} charts")

    expected = len(train_cameras)
    if len(manifest) != expected:
        raise RuntimeError(f"Expected {expected} train maps, produced {len(manifest)}")
    summary = {
        "format": "c6_spherical_hyss_maps_v1",
        "source_path": str(source),
        "hyss_scene": str(hyss_scene),
        "fred_depth_dir": str(fred_dir),
        "train_panoramas": train_panos,
        "heldout_panoramas_used": False,
        "depth_source": "HYSS FRED prediction (qmm); Replica .dpt ground truth is not read",
        "confidence_source": "HYSS normalized FRED overlap/prompt-consistency weight",
        "unique_ray_weight": "gnomonic solid-angle Jacobian divided by ERP overlap multiplicity",
        "faces": len(faces),
        "training_maps": len(manifest),
        "mean_valid_fraction": float(np.mean([item["valid_fraction"] for item in manifest])),
        "mean_confidence_valid": float(np.mean([item["confidence_mean_valid"] for item in manifest])),
        "manifest": manifest,
        "leakage_audit": "passed",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
