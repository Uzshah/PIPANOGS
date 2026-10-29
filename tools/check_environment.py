"""Check compiled dependencies and write a portable environment report."""
import importlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from run import environment, ROOT
import os
os.environ.update(environment())
for path in reversed(os.environ['PYTHONPATH'].split(os.pathsep)):
    sys.path.insert(0,path)

def main():
    report = {"python":platform.python_version(), "platform":platform.platform(), "packages":{}, "errors":[]}
    for name in ['torch','torchvision','gsplat','numpy','Pillow','torchmetrics','plyfile','opencv-python-headless']:
        try:
            report['packages'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report['errors'].append(f'Missing package: {name}')
    for name in ['torch','simple_knn._C','diff_gaussian_rasterization','fused_ssim','native_gs.spags_backend','native_gs.train_panorama_centric_invariant','native_gs.benchmark_c6pi_spherical_cuda']:
        try:
            importlib.import_module(name)
        except Exception as error:
            report['errors'].append(f'{name}: {error}')
    try:
        import torch
        report['cuda_runtime'] = torch.version.cuda
        report['cuda_available'] = torch.cuda.is_available()
        if not report['cuda_available']:
            report['errors'].append('CUDA device unavailable')
        else:
            report['gpu'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    path = ROOT/'outputs/environment.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return bool(report['errors'])

if __name__ == '__main__':
    raise SystemExit(main())
