"""Exercise the actual packed accumulator without CUDA-only parent imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from native_gs.aggregation import normalized_scatter_mean


def test_screen_gradient_norm_and_grouped_observation_count():
    path=Path(__file__).parents[1]/'native_gs/projection_invariant_gaussian_model.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ProjectionInvariantGaussianModel')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='add_packed_panorama_densification_stats')
    method.decorator_list=[]
    namespace={'torch':torch,'Sequence':list,'normalized_scatter_mean':normalized_scatter_mean}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    means=torch.zeros(2,2,requires_grad=True)
    means.grad=torch.tensor([[3.,4.],[3.,4.]])
    camera=SimpleNamespace(image_width=512,image_height=512,pano_ray_weight=torch.ones(1,2,2))
    model=SimpleNamespace(u_count=1,xyz_gradient_accum=torch.zeros(1,1),denom=torch.zeros(1,1))
    model._sample_packed_map=lambda values,coords:(torch.ones(len(coords),dtype=torch.bool),torch.ones(len(coords)))
    info={'means2d':means,'camera_ids':torch.tensor([0,1]),'gaussian_ids':torch.tensor([0,0])}
    namespace['add_packed_panorama_densification_stats'](model,[camera,camera],info)
    torch.testing.assert_close(model.xyz_gradient_accum,torch.tensor([[5.*256.]]))
    torch.testing.assert_close(model.denom,torch.ones(1,1))
