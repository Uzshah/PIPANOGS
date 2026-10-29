# Validation

Validation was performed on 2026-10-29 in the existing Python 3.10 environment. A fresh installation and the complete twelve-scene table have not been validated.

## CPU checks

The release tests cover normalized aggregation under duplicated observations and partition weights, persistent reliability EMA updates, the actual packed screen-gradient accumulator, exact optimizer budgets including the final update, physical panorama split validation, synthetic end-to-end preparation without ground-truth depth, and complete/equal-scene result summaries.

The original release tests also passed from a standalone copy outside the research tree. Python syntax, shell syntax, source hashes, binary exclusions, and missing/added files are checked by `tools/verify_release.py`. CLI help and module imports were checked for training, calibration, ERP/perspective evaluation, and the Graphdeco control.

## Room-0 GPU check

The screen-gradient implementation completed 7,500 perspective optimizer updates with batch four and 1,000 spherical calibration updates on an RTX 4090. It produced 28.5347 dB PSNR, 0.87161 SSIM, and 0.19853 LPIPS on two 1024×512 evaluation panoramas, with 998,846 Gaussians. PLY field comparison confirmed that calibration preserved positions bitwise and kept the count unchanged. This uses the retained reliability EMA fix.

The sandbox hides GPU devices, so GPU execution required host device access. The inherited Conda compiler was too new for CUDA 12.6; explicitly selecting system GCC/G++ 11 allowed the gsplat runtime compilation to succeed. Local runtime versions included PyTorch 2.6.0+cu126 and NumPy 1.26.4.

The imported preparation records `pose_estimation_used_heldout_rgb=true`. These metrics are therefore a functional single-scene validation, not a strict source-only pose/depth reproduction of the paper. The complete calibration/evaluation artifacts remain locally under `outputs/room0_angular_comparison/screen/`, excluded from Git. The experimental angular source/report has been archived outside this release folder.
