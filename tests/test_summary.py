import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('summary',Path(__file__).resolve().parents[1]/'tools/summarize.py')
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)


def test_summary_requires_all_scenes_and_weights_them_equally(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'scenes':{'a':{},'b':{}},'erp_width':1024}))
    for scene,psnr in [('a',20.),('b',30.)]:
        folder=tmp_path/'models/seed_0'/scene/'calibrated';folder.mkdir(parents=True)
        (folder/'c6pi_spherical_cuda.json').write_text(json.dumps({'split':'test','erp_resolution':[1024,512],'quality':{'psnr':psnr,'ssim':0.8,'lpips':0.2},'gaussians':100,'render_ms':1.}))
    result=summary.summarize(tmp_path,[0])
    assert result['complete'] and result['mean']['psnr']==25.
    with pytest.raises(ValueError,match='Missing 2'):
        summary.summarize(tmp_path,[0,1])
