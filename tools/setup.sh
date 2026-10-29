#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
command -v nvcc >/dev/null || { echo "CUDA toolkit/nvcc is required" >&2; exit 1; }
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./third_party/gaussian-splatting/submodules/simple-knn
python -m pip install --no-build-isolation ./third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./third_party/fused-ssim
python -m pip install --no-build-isolation ./third_party/SPaGS/SPaGSCudaBackend
python -m pip install --no-deps -e .
python tools/check_environment.py
