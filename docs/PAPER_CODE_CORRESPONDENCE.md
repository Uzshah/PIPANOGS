# Paper/code correspondence

The release uses the original screen-space densification formulation. The manuscript is maintained separately in the research workspace and will be revised by the authors to describe this formulation.

| Component | Implementation |
|---|---|
| Reliable initialization | `scripts/prepare_c6_reliability_hyss.py`, source FRED depth/confidence and cross-panorama evidence |
| Chart cover | `scripts/prepare_charts.py`, 80 overlapping charts per panorama |
| Solid-angle/overlap weights | `scripts/prepare_c6_spherical_maps.py` |
| Panorama-level evidence | `native_gs/aggregation.py`, weighted normalized scatter |
| Densification statistic | `add_packed_panorama_densification_stats`, screen-space gradient norm scaled by half the maximum chart dimension |
| Shared-center execution | `native_gs/batched_cuda_renderer.py`, shared SH evaluation and one packed camera batch |
| Reliability-aware topology | `native_gs/uopt_gaussian_model.py` |
| Spherical calibration | `train_c6pi_spherical_refine.py`, fixed means/count/topology |
| Direct ERP rendering | Existing credited SPaGS backend |

For a packed gsplat pixel-coordinate gradient g, the implemented scalar is

`gradient = ||g||_2 * 0.5 * max(image_width, image_height)`.

These scalar observations are weighted and normalized across sampled charts before one update of the per-Gaussian gradient accumulator and observation denominator. This normalizes redundant chart contributions; it does not convert gradients into an angular tangent-space covector. The manuscript should not claim that this code computes `J_pi^T g`, or establish full projection invariance from normalized aggregation tests alone. Discrete visibility, chart sampling, and optimization trajectories remain relevant.

The reliability confidence/support EMA explicitly writes updated values back through boolean indexing. This fixes the original research snapshot's in-place arithmetic on an indexed copy. The release preserves the screen-gradient algorithm but is not a bitwise copy of the buggy snapshot; numerical results can change.

The exact run provenance of the supplied manuscript's revised numerical table is still unavailable. Its reported values in the README are labeled as manuscript values, not recomputed release results.
