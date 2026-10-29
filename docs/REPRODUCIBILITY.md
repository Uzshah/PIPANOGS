# Reproducibility and benchmark scope

## Executable configuration

`configs/replica360.json` follows the available equal-update benchmark runner: 29,000 perspective optimizer updates, batch 4, and 1,000 spherical updates. This is 30,000 total optimizer updates and 116,000 perspective-chart exposures. The older chart-budget recipe uses 7,500 perspective updates and 1,000 calibration updates (30,000 perspective-chart exposures). Neither is asserted to be the unique recipe underlying the revised manuscript table.

All twelve scene names, physical panorama splits, seed 0, maximum Gaussian budget 2,000,000, ERP width 1024, and per-scene initialization targets are explicit in the configuration. Other objective/densification settings retain the defaults of the copied trainer and Graphdeco argument parser. Save the complete source integrity manifest together with each run to freeze those defaults. CPU tests verify that a batch of four still executes every requested optimizer update, including the final update before exporting the PLY.

For each published run, retain:

- `config.json`, `splits/`, stage records, raw-data and prediction checksums, and source SHA256 manifest;
- `outputs/environment.json`, driver/toolkit/compiler versions, GPU model, and `pip freeze`;
- perspective and spherical training summaries and final PLYs;
- per-panorama quality, common-resolution metrics, mode/K, and synchronized render timings.

Input files must be immutable within an output directory. Stage markers match command paths/configuration, not the content of externally modified assets. A partial stage does not have a completion marker. Preparation scripts refuse to overwrite non-empty output directories; use a new output root after an interrupted preparation. Model-state resume is not supported.

## Source and numerical limits

Upstream Git histories and commit IDs were absent from the clean input snapshot. `provenance/source_snapshot.json` records file-level SHA256 of copied inputs; `release_manifest.json` records the actual release files. Local third-party compatibility changes are retained, so replacing bundled source with an arbitrary current upstream checkout is not equivalent.

CUDA kernels may accumulate in nondeterministic order. Seeds alone do not imply bitwise-identical results across toolchains or GPUs. The pinned package recipe records the observed workspace versions but does not pin every transitive dependency or certify a successful fresh CUDA build.

The screen-space densification convention, fixed reliability EMA, and missing provenance for the revised numerical table are detailed in `PAPER_CODE_CORRESPONDENCE.md`. The release uses screen gradients throughout. The room-0 functional GPU run does not establish reproduction of the manuscript's twelve-scene table.

## External baselines and sequential ablation

The main release executes the proposed packed pipeline and direct ERP evaluation. Complete SPaGS training, the PFGS360 known-pose adaptation, and the historical sequential ablation are not presented as validated one-command reproductions here. They require separate upstream environments, weights, adapters, and exact run provenance.

For comparisons, use the identical 7/1/2 panorama-center split, seed set, 1024×512 ERP size, and `scripts/evaluate_image_pairs_common.py` for PSNR/SSIM/VGG LPIPS. Average test views within each scene, then average scenes with equal weight. Do not mix native-resolution metrics into the common-resolution table. Record optimizer updates and chart exposures separately. Timing must use the same GPU and retain backend mode; avoid attributing direct spherical-renderer speed to a new rasterizer contribution.

PFGS360-KP is a known-pose adaptation rather than the official pose-free input protocol. The historical PFGS provenance identifies commit `e79413f767d164c4228f7f4fc686e0a4dba66d78`; the source snapshot explicitly noted that an upstream license file was not observed and limited use pending clarification. Its source is therefore not silently relicensed into this bundle. HYSS/FRED inference, dataset redistribution, and pretrained weights remain separate dependencies.

## Conventional Graphdeco control

After preparation, `tools/run.py ... --stages graphdeco` trains the bundled conventional Graphdeco implementation from the same prepared initialization, for `base_updates + calibration_updates` optimizer updates. It uses the preserved `--exact_optimizer_updates` compatibility patch so that the final update is applied before the final PLY is saved. This is an executable control, not a recovered run of the manuscript table.

Evaluate its PLY with the bundled chart-render/fusion evaluator, then the common metric tool. Set the import paths with the runner's environment, for example from the repository root:

```bash
PYTHONPATH="$PWD:$PWD/third_party/gaussian-splatting" python native_gs/evaluate_c0_graphdeco.py \
  --model-path outputs/replica360/models/seed_0/room_0_rand/graphdeco \
  --source-path outputs/replica360/prepared/room_0_rand \
  --iteration 30000 --splits test --erp-width 1024
python scripts/evaluate_image_pairs_common.py \
  --render-dir outputs/replica360/models/seed_0/room_0_rand/graphdeco/test/ours_30000/erp \
  --gt-dir outputs/replica360/models/seed_0/room_0_rand/graphdeco/test/ours_30000/erp \
  --render-glob '*_render.png' --gt-glob '*_gt.png' --width 1024 \
  --output outputs/replica360/models/seed_0/room_0_rand/graphdeco/common_metrics.json
```

