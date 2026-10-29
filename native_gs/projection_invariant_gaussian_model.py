"""Projection-invariant reliability and densification for panorama-centric 3DGS.

Every update represents one physical panorama observation. Contributions from
overlapping gnomonic charts are normalized before updating the EMA or the
Graphdeco densification accumulator.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from native_gs.uopt_gaussian_model import UOptGaussianModel

from native_gs.aggregation import PanoramaAggregate, normalized_scatter_mean

class ProjectionInvariantGaussianModel(UOptGaussianModel):
    """U-Opt model with one overlap-normalized update per panorama center."""

    @torch.no_grad()
    def _sample_visible(self, camera, value_map: torch.Tensor,
                        radii: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visible = torch.nonzero(radii > 0, as_tuple=False).squeeze(1)
        if visible.numel() == 0:
            return visible, value_map.new_empty(0, dtype=torch.float32)
        xyz = self.get_xyz[visible]
        homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)
        clip = homogeneous @ camera.full_proj_transform
        ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-8)
        px = torch.round((ndc[:, 0] + 1.0) * 0.5 * (camera.image_width - 1)).long()
        py = torch.round((ndc[:, 1] + 1.0) * 0.5 * (camera.image_height - 1)).long()
        inside = ((px >= 0) & (px < camera.image_width) &
                  (py >= 0) & (py < camera.image_height))
        visible = visible[inside]
        if visible.numel() == 0:
            return visible, value_map.new_empty(0, dtype=torch.float32)
        return visible, value_map[0, py[inside], px[inside]].float()

    @torch.no_grad()
    def _chart_samples(self, cameras: Sequence, maps: Sequence[torch.Tensor],
                       radii: torch.Tensor):
        all_indices, all_values, all_weights = [], [], []
        for chart, (camera, value_map) in enumerate(zip(cameras, maps)):
            indices, values = self._sample_visible(camera, value_map, radii[chart])
            if indices.numel() == 0:
                continue
            weight_indices, weights = self._sample_visible(
                camera, camera.pano_ray_weight, radii[chart])
            if not torch.equal(indices, weight_indices):
                raise RuntimeError("value and spherical-weight projections diverged")
            finite = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
            all_indices.append(indices[finite])
            all_values.append(values[finite])
            all_weights.append(weights[finite])
        if not all_indices:
            empty_i = torch.empty(0, dtype=torch.long, device=self.get_xyz.device)
            empty_v = torch.empty(0, dtype=torch.float32, device=self.get_xyz.device)
            return empty_i, empty_v, empty_v
        return torch.cat(all_indices), torch.cat(all_values), torch.cat(all_weights)

    @torch.no_grad()
    def observe_panorama(self, cameras: Sequence, confidence_maps: Sequence[torch.Tensor],
                         radii: torch.Tensor, beta: float) -> int:
        """Update reliability once after overlap-normalizing all sampled charts."""
        indices, confidence, weights = self._chart_samples(cameras, confidence_maps, radii)
        if indices.numel() == 0:
            return 0
        confidence = confidence.clamp(0.0, 1.0)
        aggregate = normalized_scatter_mean(indices, confidence, weights, self.u_count)
        support = normalized_scatter_mean(indices, (confidence >= 0.15).float(),
                                          weights, self.u_count)
        valid = aggregate.valid
        self.u_confidence[valid, 0] = beta * self.u_confidence[valid, 0] + (1.0 - beta) * aggregate.values[valid]
        self.u_support[valid, 0] = beta * self.u_support[valid, 0] + (1.0 - beta) * support.values[valid]
        self.u_observations[valid, 0] += 1.0
        return int(valid.sum().item())

    @torch.no_grad()
    def add_panorama_densification_stats(self, cameras: Sequence,
                                         viewspace_points: torch.Tensor,
                                         radii: torch.Tensor) -> int:
        """Accumulate one normalized screen-gradient observation per panorama."""
        if viewspace_points.grad is None:
            raise RuntimeError("viewspace gradients are unavailable; call after backward")
        indices_parts, gradient_parts, weight_parts = [], [], []
        for chart, camera in enumerate(cameras):
            indices, weights = self._sample_visible(camera, camera.pano_ray_weight, radii[chart])
            if indices.numel() == 0:
                continue
            gradients = torch.norm(viewspace_points.grad[chart, indices, :2], dim=-1)
            finite = torch.isfinite(gradients) & torch.isfinite(weights) & (weights > 0)
            indices_parts.append(indices[finite])
            gradient_parts.append(gradients[finite])
            weight_parts.append(weights[finite])
        if not indices_parts:
            return 0
        aggregate = normalized_scatter_mean(torch.cat(indices_parts), torch.cat(gradient_parts),
                                            torch.cat(weight_parts), self.u_count)
        valid = aggregate.valid
        self.xyz_gradient_accum[valid, 0] += aggregate.values[valid]
        self.denom[valid, 0] += 1.0
        return int(valid.sum().item())


    @staticmethod
    def _sample_packed_map(value_map: torch.Tensor, means2d: torch.Tensor):
        height, width = value_map.shape[-2:]
        px, py = means2d[:, 0].round().long(), means2d[:, 1].round().long()
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        return inside, value_map[0, py[inside], px[inside]].float()

    @torch.no_grad()
    def observe_panorama_packed(self, cameras: Sequence,
                                confidence_maps: Sequence[torch.Tensor],
                                info: dict, beta: float) -> int:
        indices_parts, confidence_parts, weight_parts = [], [], []
        for chart, (camera, confidence_map) in enumerate(zip(cameras, confidence_maps)):
            chart_mask = info["camera_ids"] == chart
            gaussian_ids = info["gaussian_ids"][chart_mask].long()
            chart_means = info["means2d"][chart_mask]
            inside, confidence = self._sample_packed_map(confidence_map, chart_means)
            weight_inside, weights = self._sample_packed_map(camera.pano_ray_weight, chart_means)
            if not torch.equal(inside, weight_inside):
                raise RuntimeError("packed confidence and ray-weight sampling diverged")
            finite = torch.isfinite(confidence) & torch.isfinite(weights) & (weights > 0)
            if finite.any():
                indices_parts.append(gaussian_ids[inside][finite])
                confidence_parts.append(confidence[finite].clamp(0.0, 1.0))
                weight_parts.append(weights[finite])
        if not indices_parts:
            return 0
        indices, confidence, weights = (torch.cat(indices_parts),
                                        torch.cat(confidence_parts), torch.cat(weight_parts))
        aggregate = normalized_scatter_mean(indices, confidence, weights, self.u_count)
        support = normalized_scatter_mean(indices, (confidence >= 0.15).float(),
                                          weights, self.u_count)
        valid = aggregate.valid
        self.u_confidence[valid, 0] = beta * self.u_confidence[valid, 0] + (1.0 - beta) * aggregate.values[valid]
        self.u_support[valid, 0] = beta * self.u_support[valid, 0] + (1.0 - beta) * support.values[valid]
        self.u_observations[valid, 0] += 1.0
        return int(valid.sum().item())

    @torch.no_grad()
    def add_packed_panorama_densification_stats(self, cameras: Sequence,
                                                info: dict) -> int:
        means2d = info["means2d"]
        if means2d.grad is None:
            raise RuntimeError("packed screen-space gradients are unavailable")
        indices_parts, gradient_parts, weight_parts = [], [], []
        for chart, camera in enumerate(cameras):
            chart_mask = info["camera_ids"] == chart
            gaussian_ids = info["gaussian_ids"][chart_mask].long()
            chart_means = means2d[chart_mask]
            # Graphdeco's placeholder gradient is NDC-scaled: W/2 times
            # gsplat's pixel-coordinate mean gradient. Preserve its threshold.
            gradient_scale = 0.5 * max(camera.image_width, camera.image_height)
            gradients = torch.norm(means2d.grad[chart_mask], dim=-1) * gradient_scale
            inside, weights = self._sample_packed_map(camera.pano_ray_weight, chart_means)
            gradients = gradients[inside]
            finite = torch.isfinite(gradients) & torch.isfinite(weights) & (weights > 0)
            if finite.any():
                indices_parts.append(gaussian_ids[inside][finite])
                gradient_parts.append(gradients[finite])
                weight_parts.append(weights[finite])
        if not indices_parts:
            return 0
        aggregate = normalized_scatter_mean(torch.cat(indices_parts), torch.cat(gradient_parts),
                                            torch.cat(weight_parts), self.u_count)
        valid = aggregate.valid
        self.xyz_gradient_accum[valid, 0] += aggregate.values[valid]
        self.denom[valid, 0] += 1.0
        return int(valid.sum().item())
