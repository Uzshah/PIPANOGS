#!/usr/bin/env python3
"""Prepare fixed chart cameras and source/evaluation splits without reading depth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from chart_utils import (
    build_faces,
    gnomonic_map,
    rotmat2qvec,
    write_ply,
)


def discover_panoramas(scene_dir):
    records = []
    for rgb in sorted(scene_dir.glob("*_rgb_pano.jpg")):
        stem = rgb.name.split("_rgb_pano.")[0]
        records.append((stem, rgb, None, scene_dir / f"{stem}_mask_pano.png"))
    if not records:
        raise FileNotFoundError(f"No panorama RGB images in {scene_dir}")
    return records


DEFAULT_TRAIN = ["0000", "0001", "0002", "0005", "0006", "0008", "0009"]
DEFAULT_VAL = ["0004"]
DEFAULT_TEST = ["0003", "0007"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--poses", type=Path, help="Default: <scene-dir>/sfm_poses.npz")
    parser.add_argument(
        "--splits-json", type=Path,
        help="Optional JSON object with train/val/test panorama lists; overrides list arguments",
    )
    parser.add_argument("--train-panos", nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--val-panos", nargs="+", default=DEFAULT_VAL)
    parser.add_argument("--test-panos", nargs="+", default=DEFAULT_TEST)
    parser.add_argument("--subdivision", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--margin-deg", type=float, default=8.0)
    parser.add_argument("--point-stride", type=int, default=16)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=20.0)
    return parser.parse_args()


def validate_splits(available: set[str], split_panos: dict[str, list[str]]) -> None:
    assigned: dict[str, str] = {}
    for split, panos in split_panos.items():
        if len(panos) != len(set(panos)):
            raise ValueError(f"Duplicate panorama within {split}: {panos}")
        for pano in panos:
            if pano not in available:
                raise ValueError(f"Unknown {split} panorama {pano}; available={sorted(available)}")
            if pano in assigned:
                raise ValueError(f"Panorama {pano} occurs in both {assigned[pano]} and {split}")
            assigned[pano] = split
    if set(assigned) != available:
        raise ValueError(f"Every panorama must have a split; unassigned={sorted(available-set(assigned))}")


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output = args.output.resolve()
    poses_path = (args.poses or scene_dir / "sfm_poses.npz").resolve()
    if args.subdivision < 0 or args.patch_size < 2 or args.point_stride < 1:
        raise ValueError("subdivision >= 0, patch-size >= 2, and point-stride >= 1 are required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    records = discover_panoramas(scene_dir)
    records_by_stem = {record[0]: record for record in records}
    split_panos = (
        json.loads(args.splits_json.resolve().read_text())
        if args.splits_json is not None
        else {
            "train": list(args.train_panos),
            "val": list(args.val_panos),
            "test": list(args.test_panos),
        }
    )
    if set(split_panos) != {"train", "val", "test"}:
        raise ValueError("--splits-json must contain exactly train, val, and test")
    validate_splits(set(records_by_stem), split_panos)
    split_by_pano = {pano: split for split, panos in split_panos.items() for pano in panos}

    poses = np.load(poses_path)
    for stem in records_by_stem:
        for suffix in ("R", "t"):
            if f"{stem}_{suffix}" not in poses:
                raise KeyError(f"Missing {stem}_{suffix} in {poses_path}")

    faces = build_faces(args.subdivision, args.margin_deg)
    image_dir = output / "images"
    sparse_dir = output / "sparse" / "0"
    metadata_dir = output / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    first_rgb = np.asarray(Image.open(records[0][1]).convert("RGB"))
    maps = [
        gnomonic_map(rotation, fov, args.patch_size, first_rgb.shape[1], first_rgb.shape[0])
        for rotation, fov in faces
    ]

    cameras_lines = ["# HYSS icosahedral gnomonic PINHOLE cameras"]
    face_metadata = []
    for face_index, (tile_rotation, fov) in enumerate(faces):
        focal = (args.patch_size - 1) / (2.0 * math.tan(0.5 * fov))
        principal = 0.5 * (args.patch_size - 1)
        cameras_lines.append(
            f"{face_index + 1} PINHOLE {args.patch_size} {args.patch_size} "
            f"{focal:.12g} {focal:.12g} {principal:.12g} {principal:.12g}"
        )
        face_metadata.append({
            "face_index": face_index,
            "camera_id": face_index + 1,
            "tile_rotation": tile_rotation.tolist(),
            "fov_radians": float(fov),
            "fov_degrees": math.degrees(fov),
            "fx": focal,
            "fy": focal,
            "cx": principal,
            "cy": principal,
        })
    camera_text = "\n".join(cameras_lines) + "\n"
    (sparse_dir / "cameras.txt").write_text(camera_text)
    (metadata_dir / "cameras.txt").write_text(camera_text)

    headers = ["# Known HYSS poses composed with gnomonic tile rotations"]
    colmap_lines = {split: list(headers) for split in split_panos}
    all_cameras: list[dict] = []
    image_id = 1
    for stem, rgb_path, _, _ in records:
        split = split_by_pano[stem]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        if rgb.shape != first_rgb.shape:
            raise ValueError(f"All panoramas must have one shape: {rgb_path} has {rgb.shape}")
        pano_rotation = np.asarray(poses[f"{stem}_R"], dtype=np.float64)
        center = np.asarray(poses[f"{stem}_t"], dtype=np.float64)
        for face_index, ((tile_rotation, fov), (map_x, map_y)) in enumerate(zip(faces, maps)):
            name = f"{stem}_g{face_index:03d}.png"
            tile = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            Image.fromarray(tile).save(image_dir / name)
            camera_to_world = pano_rotation @ tile_rotation
            world_to_camera = camera_to_world.T
            translation = -world_to_camera @ center
            quaternion = rotmat2qvec(world_to_camera)
            values = [*quaternion.tolist(), *translation.tolist()]
            colmap_lines[split].append(
                f"{image_id} " + " ".join(f"{value:.17g}" for value in values)
                + f" {face_index + 1} {name}\n"
            )
            focal = face_metadata[face_index]["fx"]
            all_cameras.append({
                "id": image_id - 1,
                "colmap_image_id": image_id,
                "img_name": name,
                "pano_id": stem,
                "face_index": face_index,
                "split": split,
                "width": args.patch_size,
                "height": args.patch_size,
                "fx": focal,
                "fy": focal,
                "cx": face_metadata[face_index]["cx"],
                "cy": face_metadata[face_index]["cy"],
                "fov_radians": float(fov),
                "position": center.tolist(),
                "rotation": camera_to_world.tolist(),
            })
            image_id += 1
        print(f"[images] {stem} ({split}): {len(faces)} gnomonic views")

    for split, lines in colmap_lines.items():
        (metadata_dir / f"{split}_images.txt").write_text("\n".join(lines) + "\n")
    # Graphdeco trains only cameras named here.  No --eval split is used.
    (sparse_dir / "images.txt").write_text("\n".join(colmap_lines["train"]) + "\n")
    (sparse_dir / "points3D.txt").write_text("# Point initialization is stored in points3D.ply\n")
    (metadata_dir / "cameras.json").write_text(json.dumps(all_cameras, indent=2) + "\n")
    (metadata_dir / "faces.json").write_text(json.dumps(face_metadata, indent=2) + "\n")
    (metadata_dir / "splits.json").write_text(json.dumps(split_panos, indent=2) + "\n")

    # Geometry is created in the subsequent train-only HYSS reliability stage.
    # No Replica ground-truth depth is discovered or read here.
    if args.target_count < 1:
        raise ValueError("target-count must be positive")

    split_counts = {
        split: sum(camera["split"] == split for camera in all_cameras) for split in split_panos
    }
    train_names = {camera["img_name"] for camera in all_cameras if camera["split"] == "train"}
    colmap_train_names = {
        line.split()[-1] for line in colmap_lines["train"] if line and not line.startswith("#")
    }
    if train_names != colmap_train_names:
        raise RuntimeError("Leakage audit failed: sparse/0/images.txt does not equal train tiles")

    fovs = np.degrees([fov for _, fov in faces])
    summary = {
        "format": "pipanogs_chart_cover_v1",
        "source_scene": str(scene_dir),
        "poses": str(poses_path),
        "split_panoramas": split_panos,
        "split_images": split_counts,
        "training_images_in_sparse_model": len(train_names),
        "initialization_source_panoramas": split_panos["train"],
        "heldout_depth_used_for_initialization": False,
        "subdivision": args.subdivision,
        "views_per_panorama": len(faces),
        "images_total": len(all_cameras),
        "patch_size": args.patch_size,
        "margin_deg": args.margin_deg,
        "fov_deg_min": float(fovs.min()),
        "fov_deg_max": float(fovs.max()),
        "depth_scale_applied": False,
        "point_stride": args.point_stride,
        "voxel_size": args.voxel_size,
        "raw_initialization_points": 0,
        "initialization_points": args.target_count,
        "leakage_audit": "passed",
    }
    (output / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[done] train/val/test={split_counts['train']}/{split_counts['val']}/{split_counts['test']} "
        f"tiles; geometry deferred to train-only HYSS initialization -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
