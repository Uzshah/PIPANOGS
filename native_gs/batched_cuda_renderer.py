"""Camera-batched CUDA 3DGS rendering with panorama-shared invariants.

This production path keeps one Gaussian scene and passes batched view matrices
directly to gsplat CUDA. SH evaluation is chart invariant for co-located
panorama charts and is performed once per panorama batch.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import torch
from gsplat import rasterization
from utils.sh_utils import eval_sh

@dataclass(frozen=True)
class BatchedInvariantCache:
    camera_center: torch.Tensor
    colors: torch.Tensor
    scale_modifier: float
    selected: torch.Tensor | None = None

def prepare_batched_invariants(pc, camera_center: torch.Tensor,
                               scale_modifier: float = 1.0,
                               selected: torch.Tensor | None = None):
    center = camera_center.reshape(1, 3)
    xyz = pc.get_xyz if selected is None else pc.get_xyz[selected]
    features = pc.get_features if selected is None else pc.get_features[selected]
    directions = xyz - center
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sh = features.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree + 1) ** 2)
    colors = torch.clamp_min(eval_sh(pc.active_sh_degree, sh, directions) + 0.5, 0.0)
    return BatchedInvariantCache(center.reshape(3).clone(), colors, scale_modifier, selected)

def _camera_tensors(cameras: Sequence, dtype: torch.dtype, device: torch.device):
    views, intrinsics = [], []
    first = cameras[0]
    width, height = int(first.image_width), int(first.image_height)
    for camera in cameras:
        if (int(camera.image_width), int(camera.image_height)) != (width, height):
            raise ValueError("all batched charts must have the same resolution")
        if not hasattr(camera, "_c6pi_viewmatrix"):
            camera._c6pi_viewmatrix = camera.world_view_transform.transpose(0, 1).contiguous()
            fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
            fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
            camera._c6pi_intrinsics = torch.tensor(
                ((fx, 0.0, width * 0.5), (0.0, fy, height * 0.5), (0.0, 0.0, 1.0)),
                dtype=dtype, device=device)
        views.append(camera._c6pi_viewmatrix)
        intrinsics.append(camera._c6pi_intrinsics)
    return torch.stack(views).to(dtype=dtype, device=device), torch.stack(intrinsics)

def render_batched_cuda(cameras: Sequence, pc, cache: BatchedInvariantCache,
                        background: torch.Tensor, *, render_inverse_depth: bool = True,
                        packed: bool = True, radius_clip: float = 0.0) -> dict:
    """Render co-located charts in one camera-batched CUDA rasterizer call."""
    cameras = list(cameras)
    if not cameras:
        raise ValueError("at least one camera is required")
    for camera in cameras:
        if not torch.allclose(camera.camera_center, cache.camera_center, atol=1e-5, rtol=0.0):
            raise ValueError("all cameras must share the cached panorama center")
    selected = cache.selected
    xyz = pc.get_xyz if selected is None else pc.get_xyz[selected]
    scales = pc.get_scaling if selected is None else pc.get_scaling[selected]
    rotations = pc.get_rotation if selected is None else pc.get_rotation[selected]
    opacities = pc.get_opacity if selected is None else pc.get_opacity[selected]
    scales = scales * cache.scale_modifier
    views, intrinsics = _camera_tensors(cameras, xyz.dtype, xyz.device)
    colors = cache.colors
    render_mode = "RGB+ED" if render_inverse_depth else "RGB"
    rendered, alpha, info = rasterization(
        means=xyz, quats=rotations, scales=scales,
        opacities=opacities.squeeze(-1), colors=colors,
        viewmats=views, Ks=intrinsics,
        width=int(cameras[0].image_width), height=int(cameras[0].image_height),
        near_plane=0.2, far_plane=float(cameras[0].zfar), radius_clip=radius_clip,
        eps2d=0.3, packed=packed, backgrounds=None, render_mode=render_mode,
        rasterize_mode="classic")
    if torch.is_grad_enabled() and info["means2d"].requires_grad:
        info["means2d"].retain_grad()
    rgb = rendered[..., :3] + (1.0 - alpha) * background.reshape(1, 1, 1, 3)
    result = {"render": rgb.permute(0, 3, 1, 2).clamp(0, 1),
              "alpha": alpha.permute(0, 3, 1, 2), "info": info,
              "packed": packed, "rasterizer_calls": 1}
    if render_inverse_depth:
        expected_depth = rendered[..., 3:4]
        inverse_depth = torch.where(alpha > 1e-8, expected_depth.clamp_min(0.2).reciprocal(),
                                    torch.zeros_like(expected_depth))
        result["depth"] = inverse_depth.permute(0, 3, 1, 2)
    return result

def packed_max_radii(info: dict, gaussian_count: int,
                     camera_count: int | None = None) -> torch.Tensor:
    """Scatter packed elliptical radii to one panorama maximum per Gaussian."""
    keep = (torch.ones_like(info["camera_ids"], dtype=torch.bool)
            if camera_count is None else info["camera_ids"] < camera_count)
    radii = info["radii"][keep].amax(dim=-1).float()
    maximum = radii.new_zeros(gaussian_count)
    maximum.scatter_reduce_(0, info["gaussian_ids"][keep].long(), radii,
                            reduce="amax", include_self=True)
    return maximum
