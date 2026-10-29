import torch

from native_gs.aggregation import normalized_scatter_mean


def test_duplicate_charts_do_not_change_reliability():
    indices = torch.tensor([0, 1, 1, 2, 2, 2])
    values = torch.tensor([0.2, 0.3, 0.7, 0.1, 0.5, 0.9])
    weights = torch.tensor([1.0, 0.4, 0.6, 0.2, 0.3, 0.5])
    reference = normalized_scatter_mean(indices, values, weights, 4)
    repeated = normalized_scatter_mean(
        indices.repeat(7), values.repeat(7), weights.repeat(7), 4
    )
    torch.testing.assert_close(reference.values, repeated.values)
    assert torch.equal(reference.valid, repeated.valid)


def test_uniform_overlap_multiplicity_cancels():
    indices = torch.tensor([0, 0, 1, 1])
    values = torch.tensor([0.25, 0.75, 0.4, 0.8])
    weights = torch.tensor([0.5, 0.5, 0.25, 0.75])
    reference = normalized_scatter_mean(indices, values, weights, 2)
    overlap_corrected = normalized_scatter_mean(
        indices.repeat_interleave(3), values.repeat_interleave(3),
        (weights / 3.0).repeat_interleave(3), 2,
    )
    torch.testing.assert_close(reference.values, overlap_corrected.values)


def test_nonuniform_chart_overlap_cancels_with_partition_weights():
    """Changing local chart density must not bias a panorama observation.

    Gaussian 0 is covered by one chart and Gaussian 1 by three charts. The
    per-chart weights form a partition of unity for each physical spherical
    sample, matching the overlap normalization produced by
    prepare_c6_spherical_maps.py.
    """
    reference = normalized_scatter_mean(
        torch.tensor([0, 1]),
        torch.tensor([0.35, 0.80]),
        torch.tensor([1.0, 1.0]),
        2,
    )
    charted = normalized_scatter_mean(
        torch.tensor([0, 1, 1, 1]),
        torch.tensor([0.35, 0.80, 0.80, 0.80]),
        torch.tensor([1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        2,
    )
    torch.testing.assert_close(reference.values, charted.values)
    torch.testing.assert_close(reference.weights, charted.weights)
