"""Partition-normalized per-Gaussian aggregation (CPU or CUDA)."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class PanoramaAggregate:
    values: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor

def normalized_scatter_mean(indices: torch.Tensor, values: torch.Tensor,
                            weights: torch.Tensor, size: int) -> PanoramaAggregate:
    """Weighted mean by Gaussian id; invariant to duplicated observations."""
    numerator = values.new_zeros(size)
    denominator = weights.new_zeros(size)
    numerator.scatter_add_(0, indices, values * weights)
    denominator.scatter_add_(0, indices, weights)
    valid = denominator > 1e-12
    result = values.new_zeros(size)
    result[valid] = numerator[valid] / denominator[valid]
    return PanoramaAggregate(result, denominator, valid)

