"""Run all preparation stages on synthetic RGB/FRED inputs, without .dpt files."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def test_preparation_requires_no_ground_truth_depth(tmp_path):
    raw=tmp_path/'raw'; raw.mkdir()
    hyss=tmp_path/'hyss'; fred=hyss/'iter_00/fred_depths'; fred.mkdir(parents=True)
    poses={}
    for i in range(3):
        pano=f'{i:04d}'
        Image.fromarray(np.full((16,32,3),128,dtype=np.uint8)).save(raw/f'{pano}_rgb_pano.jpg')
        Image.fromarray(np.full((16,32),255,dtype=np.uint8)).save(raw/f'{pano}_mask_pano.png')
        poses[f'{pano}_R']=np.eye(3)
        poses[f'{pano}_t']=np.array([i*0.1,0.,0.])
    # Only the source panorama has predicted depth and confidence.
    Image.fromarray(np.full((16,32),8000,dtype=np.uint16)).save(fred/'0000.png')
    Image.fromarray(np.full((16,32),65535,dtype=np.uint16)).save(fred/'0000_weight.png')
    np.savez(hyss/'sfm_poses.npz',**poses)
    splits={'train':['0000'],'val':['0001'],'test':['0002']}
    split_path=tmp_path/'splits.json';split_path.write_text(json.dumps(splits))
    charts=tmp_path/'charts'; source=tmp_path/'prepared'; maps=tmp_path/'maps'
    commands=[
        ['scripts/prepare_charts.py','--scene-dir',str(raw),'--poses',str(hyss/'sfm_poses.npz'),'--output',str(charts),'--splits-json',str(split_path),'--target-count','8','--patch-size','8'],
        ['scripts/prepare_c6_reliability_hyss.py','--c6-source',str(charts),'--hyss-scene',str(hyss),'--output',str(source)],
        ['scripts/prepare_c6_spherical_maps.py','--source-path',str(source),'--hyss-scene',str(hyss),'--output',str(maps),'--erp-width','32'],
    ]
    for command in commands:
        result=subprocess.run([sys.executable,*command],cwd=ROOT,text=True,capture_output=True)
        assert result.returncode==0,result.stdout+result.stderr
    prep=json.loads((source/'prep_summary.json').read_text())
    assert prep['initialization_points']==8
    assert prep['split_panoramas']==splits
    assert not prep['replica_ground_truth_depth_used']
    attrs=np.load(source/'uinit_attributes.npz')
    assert attrs['xyz'].shape==(8,3)
    assert (source/'sparse/0/points3D.ply').is_file()
    train_images=(source/'sparse/0/images.txt').read_text()
    assert '0001_g' not in train_images and '0002_g' not in train_images
    assert len(list((maps/'depth_confidence_hyss').glob('*.npz')))==80
