"""Cross-panorama reliability, weighted fusion, and deterministic sampling.

Input depth is supplied by the caller; this module never loads ground-truth depth.
"""
from __future__ import annotations
import math
import numpy as np

def candidates_for_pano(
    pano: str,
    data: dict[str, dict],
    stride: int,
    min_depth: float,
    max_depth: float,
    support_cap: int,
    base_tol: float,
    relative_tol: float,
    confidence_tau: float,
    photo_tau: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    source = data[pano]
    height, width = source["depth"].shape
    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    depth = source["depth"][vv, uu]
    valid_grid = source["valid"][vv, uu] & (depth >= min_depth) & (depth <= max_depth)
    longitude = uu / (width - 1) * (2.0 * math.pi) - math.pi
    latitude = vv / (height - 1) * math.pi - math.pi / 2.0
    cos_latitude = np.cos(latitude)
    rays = np.stack(
        [cos_latitude * np.cos(longitude), np.sin(latitude), cos_latitude * np.sin(longitude)],
        axis=-1,
    )
    camera_points = rays[valid_grid] * depth[valid_grid, None]
    world = camera_points @ source["rotation"].T + source["center"]
    colors = source["rgb"][vv, uu][valid_grid].astype(np.float32) / 255.0
    source_area = np.clip(cos_latitude[valid_grid], 0.02, 1.0).astype(np.float32)
    source_depth = depth[valid_grid].astype(np.float32)

    n = len(world)
    support = np.ones(n, dtype=np.int16)
    comparable_count = np.zeros(n, dtype=np.int16)
    residual_sum = np.zeros(n, dtype=np.float64)
    residual_abs_sum = np.zeros(n, dtype=np.float64)
    residual_square_sum = np.zeros(n, dtype=np.float64)
    photo_sum = np.zeros(n, dtype=np.float64)

    for target_pano, target in data.items():
        if target_pano == pano:
            continue
        camera = (world - target["center"]) @ target["rotation"]
        radius = np.linalg.norm(camera, axis=1)
        valid_radius = radius > 1e-6
        target_longitude = np.arctan2(camera[:, 2], camera[:, 0])
        target_latitude = np.arcsin(np.clip(camera[:, 1] / np.maximum(radius, 1e-6), -1.0, 1.0))
        target_u = np.rint((target_longitude + math.pi) / (2.0 * math.pi) * (width - 1)).astype(np.int64)
        target_v = np.rint((target_latitude + math.pi / 2.0) / math.pi * (height - 1)).astype(np.int64)
        target_u = np.remainder(target_u, width)
        target_v = np.clip(target_v, 0, height - 1)
        observed_depth = target["depth"][target_v, target_u]
        observed_valid = target["valid"][target_v, target_u] & valid_radius
        tolerance = base_tol + relative_tol * np.minimum(radius, observed_depth)
        # If the candidate is behind a closer target surface, it is occluded and
        # supplies no evidence. In-front conflicts remain comparable evidence.
        comparable = observed_valid & (radius <= observed_depth + tolerance)
        residual = observed_depth - radius
        absolute = np.abs(residual)
        consistent = comparable & (absolute <= tolerance)
        support += consistent.astype(np.int16)
        comparable_count += comparable.astype(np.int16)
        residual_sum[comparable] += residual[comparable]
        residual_abs_sum[comparable] += absolute[comparable]
        residual_square_sum[comparable] += residual[comparable] ** 2
        sampled_color = target["rgb"][target_v, target_u].astype(np.float32) / 255.0
        photo_error = np.sqrt(np.mean((sampled_color - colors) ** 2, axis=1))
        photo_sum[comparable] += photo_error[comparable]

    has_comparison = comparable_count > 0
    normalizer = np.maximum(comparable_count, 1)
    mean_residual = residual_sum / normalizer
    mean_absolute = residual_abs_sum / normalizer
    depth_variance = np.maximum(residual_square_sum / normalizer - mean_residual**2, 0.0)
    tolerance = base_tol + relative_tol * source_depth
    c_reproj = np.exp(-mean_absolute / confidence_tau)
    c_reproj[~has_comparison] = 0.02
    mean_absolute[~has_comparison] = np.nan
    mean_photo = photo_sum / normalizer
    c_photo = np.exp(-mean_photo / photo_tau)
    c_photo[~has_comparison] = 0.40
    c_fusion = np.minimum(1.0, support.astype(np.float32) / float(support_cap))
    sigma = 0.002 + 0.008 * source_depth
    sigma_at_min = 0.002 + 0.008 * min_depth
    c_range = np.clip((sigma_at_min / sigma) ** 2, 0.02, 1.0)
    # Prompt prediction residuals and registration residuals are not distributed
    # with room_0_rand; neutral channels are saved with explicit availability.
    c_prompt = np.ones(n, dtype=np.float32)
    c_pose = np.ones(n, dtype=np.float32)
    components = np.stack([c_range, c_fusion, c_reproj, c_photo], axis=1)
    weights = np.asarray([0.10, 0.15, 0.65, 0.10], dtype=np.float32)
    confidence = np.exp(np.sum(np.log(np.clip(components, 0.02, 1.0)) * weights, axis=1))
    confidence = np.clip(confidence, 0.02, 1.0).astype(np.float32)

    candidate = {
        "points": world.astype(np.float32),
        "colors": colors.astype(np.float32),
        "confidence": confidence,
        "support_count": support,
        "depth_variance": depth_variance.astype(np.float32),
        "spherical_area": source_area,
        "reprojection_error": mean_absolute.astype(np.float32),
        "c_range": c_range.astype(np.float32),
        "c_fusion": c_fusion.astype(np.float32),
        "c_reproj": c_reproj.astype(np.float32),
        "c_photo": c_photo.astype(np.float32),
        "c_prompt": c_prompt,
        "c_pose": c_pose,
    }
    maps = {
        key: np.zeros(valid_grid.shape, dtype=value.dtype)
        for key, value in candidate.items()
        if key not in {"points", "colors"}
    }
    maps["valid"] = valid_grid.astype(np.uint8)
    for key, output in maps.items():
        if key != "valid":
            output[valid_grid] = candidate[key]
    maps["rows"] = rows.astype(np.int32)
    maps["cols"] = cols.astype(np.int32)
    return candidate, maps


def concatenate(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in items[0]}


def confidence_voxel_fusion(
    candidate: dict[str, np.ndarray], voxel_size: float
) -> dict[str, np.ndarray]:
    keys = np.floor(candidate["points"] / voxel_size).astype(np.int64)
    voxel_keys, inverse, raw_count = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    count = len(voxel_keys)
    confidence = candidate["confidence"].astype(np.float64)
    sum_weight = np.bincount(inverse, weights=confidence, minlength=count)
    points = np.column_stack([
        np.bincount(inverse, weights=confidence * candidate["points"][:, axis], minlength=count)
        for axis in range(3)
    ]) / sum_weight[:, None]
    colors = np.column_stack([
        np.bincount(inverse, weights=confidence * candidate["colors"][:, axis], minlength=count)
        for axis in range(3)
    ]) / sum_weight[:, None]
    fused_confidence = np.bincount(
        inverse, weights=confidence**2, minlength=count
    ) / sum_weight
    area_mass = np.bincount(
        inverse,
        weights=confidence * candidate["spherical_area"].astype(np.float64),
        minlength=count,
    )
    support_count = np.zeros(count, dtype=np.int16)
    np.maximum.at(support_count, inverse, candidate["support_count"])
    residual_variance = np.bincount(
        inverse, weights=confidence * candidate["depth_variance"], minlength=count
    ) / sum_weight
    point_norm2 = np.sum(candidate["points"].astype(np.float64) ** 2, axis=1)
    spatial_variance = np.maximum(
        np.bincount(inverse, weights=confidence * point_norm2, minlength=count) / sum_weight
        - np.sum(points**2, axis=1),
        0.0,
    )
    return {
        "points": points.astype(np.float32),
        "colors": np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8),
        "confidence": fused_confidence.astype(np.float32),
        "support_count": support_count,
        "depth_variance": (residual_variance + spatial_variance).astype(np.float32),
        "sampling_mass": area_mass.astype(np.float64),
        "raw_count": raw_count.astype(np.int32),
        "voxel_keys": voxel_keys,
    }


def weighted_sample(voxels: dict[str, np.ndarray], target_count: int, seed: int) -> np.ndarray:
    if len(voxels["points"]) < target_count:
        raise ValueError(f"Only {len(voxels['points'])} eligible voxels for target {target_count}")
    rng = np.random.default_rng(seed)
    # Gumbel top-k implements deterministic weighted sampling without replacement.
    mass = np.maximum(voxels["sampling_mass"], 1e-12)
    score = np.log(mass) - np.log(-np.log(rng.uniform(1e-12, 1.0 - 1e-12, len(mass))))
    selected = np.argpartition(score, -target_count)[-target_count:]
    return np.sort(selected)


def diagnostic_bins(confidence: np.ndarray, error: np.ndarray, bins: int = 10) -> list[dict]:
    observed = np.isfinite(error)
    confidence = confidence[observed]
    error = error[observed]
    if confidence.size == 0:
        return []
    edges = np.quantile(confidence, np.linspace(0.0, 1.0, bins + 1))
    result = []
    for index in range(bins):
        upper_closed = index == bins - 1
        chosen = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if upper_closed else confidence < edges[index + 1]
        )
        result.append({
            "bin": index,
            "confidence_low": float(edges[index]),
            "confidence_high": float(edges[index + 1]),
            "count": int(chosen.sum()),
            "mean_confidence": float(confidence[chosen].mean()),
            "mean_reprojection_error_m": float(error[chosen].mean()),
        })
    return result

