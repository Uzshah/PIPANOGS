"""Exercise the actual C6 training loop with CPU renderer/scene substitutes."""
import ast
import json
import random
import importlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("updates", [1, 3])
def test_batch_four_performs_every_optimizer_update(tmp_path, monkeypatch, updates):
    source = tmp_path / "source"
    source.mkdir()
    (source / "uinit_attributes.npz").touch()
    output = tmp_path / "model"
    calls = []

    class Model:
        def __init__(self, *args):
            self.color = torch.nn.Parameter(torch.full((3, 2, 2), 0.5))
            self.get_xyz = torch.zeros(1, 3)
            self.u_count = 1
            self.u_confidence = torch.ones(1)
            self.u_support = torch.ones(1)
            self.optimizer = torch.optim.SGD([self.color], lr=0.1)
            original_step = self.optimizer.step

            def step():
                calls.append(1)
                original_step()

            self.optimizer.step = step

        def training_setup(self, opt):
            pass

        def update_learning_rate(self, iteration):
            pass

        def oneupSHdegree(self):
            pass

        def observe_panorama_packed(self, *args):
            return 1

    class Scene:
        cameras_extent = 1.0

        def __init__(self, dataset, model, **kwargs):
            self.model = model

        def getTrainCameras(self):
            return []

        def save(self, iteration):
            # Check final export sees the updated parameters, including N=1.
            assert self.model.color.mean().item() < 0.5

    class Progress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    tensor = torch.tensor
    torch_proxy = SimpleNamespace(**{name: getattr(torch, name) for name in dir(torch)})
    torch_proxy.tensor = lambda *args, **kwargs: tensor(*args, **{k: v for k, v in kwargs.items() if k != "device"})
    camera = SimpleNamespace(
        camera_center=torch.zeros(3), original_image=torch.zeros(3, 2, 2),
        alpha_mask=None, pano_ray_weight=torch.ones(1, 2, 2),
        pano_depth_confidence=torch.ones(1, 2, 2), pano_invdepth=torch.ones(1, 2, 2),
    )
    groups = {"0000": [camera] * 4}
    namespace = dict(
        Path=Path, json=json, random=random, time=time, torch=torch_proxy, F=F,
        ProjectionInvariantGaussianModel=Model, Scene=Scene, tqdm=Progress,
        load_groups=lambda *args: (groups, {"panoramas": 1}),
        prepare_batched_invariants=lambda *args: None,
        render_batched_cuda=lambda cameras, model, *args, **kwargs: {
            "render": model.color[None].expand(len(cameras), -1, -1, -1),
            "depth": torch.ones(len(cameras), 1, 2, 2), "info": {},
        },
        packed_max_radii=lambda *args: torch.ones(1),
        l1_loss=lambda image, target: (image - target).abs().mean(),
        weighted_mean=lambda value, weight: (value * weight).sum() / weight.sum(),
        ssim=lambda image, target: image.mean() * 0 + 1,
    )
    path = Path(__file__).parents[1] / "native_gs/train_panorama_centric_invariant.py"
    tree = ast.parse(path.read_text())
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    exec(compile(ast.Module(body=[train], type_ignores=[]), str(path), "exec"), namespace)
    dataset = SimpleNamespace(model_path=str(output), source_path=str(source), sh_degree=3, white_background=False)
    opt = SimpleNamespace(iterations=updates, random_background=False, lambda_dssim=0.2,
                          densify_until_iter=0, densification_interval=100, opacity_reset_interval=3000)
    args = SimpleNamespace(map_root=str(source), chart_batch=4, support_cap=4, ray_balance=0.5,
        depth_weight=0.01, depth_huber_delta=0.05, depth_warmup=200, reliability_ema_beta=0.95,
        seed=0, max_gaussians=2000000, reliability_score_mode="mean_preserving",
        reliability_score_threshold=0.0002, reliability_confidence_gamma=0.5,
        reliability_support_floor=0.25, reliability_modulation_floor=0.95,
        reliability_modulation_ceiling=1.05, reliability_prune_confidence=0.15,
        reliability_prune_support=0.35)
    namespace["train"](dataset, opt, None, args)
    summary = json.loads((output / "panorama_centric_training_summary.json").read_text())
    assert len(calls) == summary["optimizer_steps"] == updates
    assert summary["rendered_charts"] == 4 * updates


