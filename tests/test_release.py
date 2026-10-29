import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from native_gs.aggregation import normalized_scatter_mean

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_runner', ROOT/'tools/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_budget_and_fixed_position_contract(tmp_path):
    config = runner.load_config(ROOT/'configs/replica360.json')
    assert len(config['scenes']) == 12
    assert config['densification_gradient'] == 'screen'
    assert config['base_updates'] + config['calibration_updates'] == 30000
    assert config['base_updates'] * config['chart_batch'] == 116000
    plan = runner.commands(config,'room_0_rand',0,tmp_path,tmp_path/'data',tmp_path/'hyss')
    assert plan['train'][0][plan['train'][0].index('--iterations')+1] == '29000'
    assert plan['calibrate'][0][plan['calibrate'][0].index('--position-lr')+1] == '0'
    assert plan['train'][0][plan['train'][0].index('--densification-gradient')+1] == 'screen'
    assert '--split' in plan['evaluate'][0] and 'test' in plan['evaluate'][0]


def test_invalid_splits_rejected(tmp_path):
    config = json.loads((ROOT/'configs/replica360.json').read_text())
    config['scenes']['room_0_rand']['split']['test'].append('0000')
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='overlapping'):
        runner.load_config(path)


def test_packed_reliability_updates_persistent_state():
    # Execute the actual method without importing CUDA-only parent classes.
    path=ROOT/'native_gs/projection_invariant_gaussian_model.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ProjectionInvariantGaussianModel')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='observe_panorama_packed')
    method.decorator_list=[]
    ns={'torch':torch,'Sequence':list,'normalized_scatter_mean':normalized_scatter_mean}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
    model=SimpleNamespace(u_count=2,u_confidence=torch.zeros(2,1),u_support=torch.zeros(2,1),u_observations=torch.zeros(2,1))
    def sample(value_map, means):
        return torch.ones(len(means),dtype=torch.bool),value_map.flatten()[:len(means)]
    model._sample_packed_map=sample
    camera=SimpleNamespace(pano_ray_weight=torch.ones(1,1,2))
    info={'camera_ids':torch.tensor([0,0]),'gaussian_ids':torch.tensor([0,1]),'means2d':torch.zeros(2,2)}
    assert ns['observe_panorama_packed'](model,[camera],[torch.tensor([[[0.8,0.1]]])],info,0.5)==2
    torch.testing.assert_close(model.u_confidence[:,0],torch.tensor([0.4,0.05]))
    torch.testing.assert_close(model.u_support[:,0],torch.tensor([0.5,0.0]))
    torch.testing.assert_close(model.u_observations,torch.ones(2,1))


def test_aggregation_ignores_unobserved_gaussians():
    result=normalized_scatter_mean(torch.tensor([0]),torch.tensor([0.9]),torch.tensor([0.0]),2)
    assert not result.valid.any()
    assert torch.isfinite(result.values).all()
