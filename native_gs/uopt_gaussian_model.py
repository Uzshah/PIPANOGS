"""Graphdeco GaussianModel with non-parameter U-Opt reliability state."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import torch

from scene.gaussian_model import GaussianModel
from utils.system_utils import mkdir_p


class UOptGaussianModel(GaussianModel):
    """Preserve confidence/support/variance across Graphdeco topology changes."""

    def __init__(self, sh_degree: int, attribute_path: Path, support_cap: float = 4.0):
        super().__init__(sh_degree, "default")
        self.attribute_path = Path(attribute_path)
        self.support_cap = support_cap
        self.u_confidence = torch.empty((0, 1), device="cuda")
        self.u_support = torch.empty((0, 1), device="cuda")
        self.u_depth_variance = torch.empty((0, 1), device="cuda")
        self.u_observations = torch.empty((0, 1), device="cuda")
        self._pending_u_attributes = None

    def create_from_pcd(self, pcd, cam_infos, spatial_lr_scale):
        super().create_from_pcd(pcd, cam_infos, spatial_lr_scale)
        attributes = np.load(self.attribute_path)
        xyz = np.asarray(attributes["xyz"], dtype=np.float32)
        pcd_xyz = np.asarray(pcd.points, dtype=np.float32)
        if xyz.shape != pcd_xyz.shape or not np.allclose(xyz, pcd_xyz, atol=1e-6):
            raise ValueError("uinit_attributes.npz is not aligned with C1 points3D.ply")
        confidence = np.asarray(attributes["confidence"], dtype=np.float32)
        support = np.asarray(attributes["support_count"], dtype=np.float32) / self.support_cap
        variance = np.asarray(attributes["depth_variance"], dtype=np.float32)
        self.u_confidence = torch.from_numpy(confidence[:, None]).cuda().clamp(0.02, 1.0)
        self.u_support = torch.from_numpy(support[:, None]).cuda().clamp(0.0, 1.0)
        self.u_depth_variance = torch.from_numpy(variance[:, None]).cuda().clamp_min(0.0)
        self.u_observations = torch.zeros_like(self.u_confidence)

    @property
    def u_count(self) -> int:
        return int(self.u_confidence.shape[0])

    def observe(self, camera, confidence_map: torch.Tensor, radii: torch.Tensor, beta: float) -> int:
        visible = torch.nonzero(radii > 0, as_tuple=False).squeeze(1)
        if visible.numel() == 0:
            return 0
        xyz = self.get_xyz[visible]
        homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)
        clip = homogeneous @ camera.full_proj_transform
        ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-8)
        px = torch.round((ndc[:, 0] + 1.0) * 0.5 * (camera.image_width - 1)).long()
        py = torch.round((ndc[:, 1] + 1.0) * 0.5 * (camera.image_height - 1)).long()
        inside = (
            (px >= 0) & (px < camera.image_width) &
            (py >= 0) & (py < camera.image_height)
        )
        visible = visible[inside]
        if visible.numel() == 0:
            return 0
        observed = confidence_map[0, py[inside], px[inside]].float().clamp(0.0, 1.0)
        old_confidence = self.u_confidence[visible, 0]
        old_support = self.u_support[visible, 0]
        self.u_confidence[visible, 0] = beta * old_confidence + (1.0 - beta) * observed
        supported = (observed >= 0.15).float()
        self.u_support[visible, 0] = beta * old_support + (1.0 - beta) * supported
        self.u_observations[visible, 0] += 1.0
        return int(visible.numel())

    def _set_pending(self, mask: torch.Tensor, repeats: int = 1) -> None:
        self._pending_u_attributes = tuple(
            value[mask].repeat(repeats, 1)
            for value in (
                self.u_confidence,
                self.u_support,
                self.u_depth_variance,
                self.u_observations,
            )
        )

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_tmp_radii,
    ):
        pending = self._pending_u_attributes
        if pending is None or pending[0].shape[0] != new_xyz.shape[0]:
            raise RuntimeError("Missing or misaligned U-Opt child attributes")
        super().densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )
        self.u_confidence = torch.cat((self.u_confidence, pending[0]), dim=0)
        self.u_support = torch.cat((self.u_support, pending[1]), dim=0)
        self.u_depth_variance = torch.cat((self.u_depth_variance, pending[2]), dim=0)
        self.u_observations = torch.cat((self.u_observations, pending[3]), dim=0)
        self._pending_u_attributes = None

    def prune_points(self, mask):
        valid = ~mask
        # Graphdeco's prune_points unconditionally slices tmp_radii. That
        # buffer is only populated inside a densification transaction, while
        # our explicit Gaussian-budget pruning also runs between transactions.
        # Supply a shape-compatible transient buffer for the parent operation
        # and restore the inactive state afterwards.
        restore_empty_tmp_radii = self.tmp_radii is None
        if restore_empty_tmp_radii:
            self.tmp_radii = torch.zeros(
                self.get_xyz.shape[0],
                dtype=self.get_xyz.dtype,
                device=self.get_xyz.device,
            )
        super().prune_points(mask)
        if restore_empty_tmp_radii:
            self.tmp_radii = None
        self.u_confidence = self.u_confidence[valid]
        self.u_support = self.u_support[valid]
        self.u_depth_variance = self.u_depth_variance[valid]
        self.u_observations = self.u_observations[valid]
        if self.u_count != self.get_xyz.shape[0]:
            raise RuntimeError("U-Opt state diverged after pruning")

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected &= torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        self._set_pending(selected)
        super().densify_and_clone(grads, grad_threshold, scene_extent)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init = self.get_xyz.shape[0]
        padded = torch.zeros(n_init, device="cuda")
        padded[:grads.shape[0]] = grads.squeeze()
        selected = padded >= grad_threshold
        selected &= torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        self._set_pending(selected, N)
        super().densify_and_split(grads, grad_threshold, scene_extent, N)

    def densify_and_prune_uopt(
        self,
        score_threshold: float,
        min_opacity: float,
        extent: float,
        max_screen_size,
        radii: torch.Tensor,
        confidence_gamma: float,
        support_floor: float,
        prune_confidence: float,
        prune_support: float,
        score_mode: str = "suppress",
        modulation_floor: float = 0.5,
        modulation_ceiling: float = 1.5,
    ) -> dict:
        gradients = self.xyz_gradient_accum / self.denom
        gradients[gradients.isnan()] = 0.0
        quality = (self.u_confidence + 1e-6).pow(confidence_gamma)
        visibility_support = support_floor + (1.0 - support_floor) * self.u_support
        raw_modulation = quality * visibility_support
        if score_mode == "suppress":
            modulation = raw_modulation
        elif score_mode == "mean_preserving":
            # Redistribute densification toward reliable geometry without
            # globally suppressing texture/detail growth.
            normalizer = raw_modulation.detach().mean().clamp_min(1e-6)
            modulation = (raw_modulation / normalizer).clamp(
                min=modulation_floor, max=modulation_ceiling
            )
        else:
            raise ValueError(f"Unknown reliability score mode: {score_mode}")
        score = gradients * modulation
        before = self.get_xyz.shape[0]
        eligible = int((score.squeeze() >= score_threshold).sum().item())

        self.tmp_radii = radii
        self.densify_and_clone(score, score_threshold, extent)
        self.densify_and_split(score, score_threshold, extent)

        low_quality = (
            (self.u_confidence.squeeze() < prune_confidence)
            & (self.u_support.squeeze() < prune_support)
        )
        low_opacity = (self.get_opacity < min_opacity).squeeze()
        prune_mask = low_opacity & low_quality
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask |= big_points_vs | big_points_ws
        pruned = int(prune_mask.sum().item())
        self.prune_points(prune_mask)
        after = self.get_xyz.shape[0]
        self.tmp_radii = None
        torch.cuda.empty_cache()
        return {
            "before": int(before),
            "eligible_score": eligible,
            "pruned": pruned,
            "after": int(after),
            "confidence_mean": float(self.u_confidence.mean().item()),
            "support_mean": float(self.u_support.mean().item()),
            "observations_mean": float(self.u_observations.mean().item()),
            "score_mode": score_mode,
            "raw_modulation_mean": float(raw_modulation.mean().item()),
            "modulation_mean": float(modulation.mean().item()),
            "modulation_min": float(modulation.min().item()),
            "modulation_max": float(modulation.max().item()),
        }

    def construct_list_of_attributes(self):
        return super().construct_list_of_attributes() + [
            "u_confidence",
            "u_support",
            "u_depth_variance",
            "u_observations",
        ]

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        arrays = (
            xyz,
            normals,
            f_dc,
            f_rest,
            self._opacity.detach().cpu().numpy(),
            self._scaling.detach().cpu().numpy(),
            self._rotation.detach().cpu().numpy(),
            self.u_confidence.detach().cpu().numpy(),
            self.u_support.detach().cpu().numpy(),
            self.u_depth_variance.detach().cpu().numpy(),
            self.u_observations.detach().cpu().numpy(),
        )
        dtype = [(name, "f4") for name in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, np.concatenate(arrays, axis=1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def capture(self):
        return (
            "uopt_v1",
            super().capture(),
            self.u_confidence,
            self.u_support,
            self.u_depth_variance,
            self.u_observations,
        )

    def restore(self, model_args, training_args):
        tag, base, confidence, support, variance, observations = model_args
        if tag != "uopt_v1":
            raise ValueError(f"Unsupported U-Opt checkpoint tag: {tag}")
        super().restore(base, training_args)
        self.u_confidence = confidence
        self.u_support = support
        self.u_depth_variance = variance
        self.u_observations = observations
