#!/usr/bin/env python3
"""Portable, stage-explicit Pi-GS experiment runner. No shell or Conda assumptions."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config['base_updates'] < 1 or config['calibration_updates'] < 1 or config['chart_batch'] < 1:
        raise ValueError('Update and batch counts must be positive')
    if config['erp_width'] < 2 or config['erp_width'] % 2:
        raise ValueError('ERP width must be positive and even')
    if config.get('densification_gradient', 'screen') != 'screen':
        raise ValueError('This release uses screen densification gradients')
    for name, scene in config['scenes'].items():
        split = scene['split']
        if set(split) != {'train', 'val', 'test'} or any(not split[k] for k in split):
            raise ValueError(f'{name}: need nonempty train/val/test splits')
        ids = sum((split[k] for k in ('train', 'val', 'test')), [])
        if len(ids) != len(set(ids)):
            raise ValueError(f'{name}: overlapping or duplicate panorama IDs')
        if scene['initialization_points'] < 1:
            raise ValueError(f'{name}: invalid initialization count')
    return config


def environment():
    env = os.environ.copy()
    paths = [ROOT, ROOT/'scripts', ROOT/'third_party/gaussian-splatting', ROOT/'third_party/nerficg/src']
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths))
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    return env


def commands(config, scene, seed, work, data_root, hyss_root):
    setting = config['scenes'][scene]
    source, maps = work/'prepared'/scene, work/'maps'/scene
    charts = work/'charts'/scene
    hyss = hyss_root/scene
    pose_candidates = [hyss/'panorama/iter_00/sfm_output/sfm_poses.npz', hyss/'iter_00/sfm_output/sfm_poses.npz', hyss/'sfm_poses.npz']
    poses = next((p for p in pose_candidates if p.is_file()), pose_candidates[0])
    model = work/'models'/f'seed_{seed}'/scene
    base, final = model/'perspective', model/'calibrated'
    base_steps, refine_steps = str(config['base_updates']), str(config['calibration_updates'])
    width = str(config['erp_width'])
    split = work/'splits'/f'{scene}.json'
    return {
        'prepare': [
            ['scripts/prepare_charts.py','--scene-dir',str(data_root/scene),'--poses',str(poses),'--output',str(charts),'--splits-json',str(split),'--target-count',str(setting['initialization_points'])],
            ['scripts/prepare_c6_reliability_hyss.py','--c6-source',str(charts),'--hyss-scene',str(hyss),'--output',str(source)],
            ['scripts/prepare_c6_spherical_maps.py','--source-path',str(source),'--hyss-scene',str(hyss),'--output',str(maps),'--erp-width',width],
        ],
        'train': [['native_gs/train_panorama_centric_invariant.py','-s',str(source),'-m',str(base),'--map-root',str(maps),'--iterations',base_steps,'--chart-batch',str(config['chart_batch']),'--densification-gradient',config.get('densification_gradient','screen'),'--max-gaussians',str(config['max_gaussians']),'--seed',str(seed),'--resolution','1','--quiet']],
        'calibrate': [['native_gs/train_c6pi_spherical_refine.py','--baseline-model',str(base),'--source-path',str(source),'--baseline-iteration',base_steps,'--model-path',str(final),'--iterations',refine_steps,'--erp-width',width,'--position-lr','0','--mode','2','--seed',str(seed)]],
        'evaluate': [['native_gs/benchmark_c6pi_spherical_cuda.py','--model-path',str(final),'--source-path',str(source),'--iteration',refine_steps,'--erp-width',width,'--split','test','--mode','2','--parameter-refined']],
        'graphdeco': [['third_party/gaussian-splatting/train.py','--source_path',str(source),'--model_path',str(model/'graphdeco'),'--iterations',str(config['base_updates']+config['calibration_updates']),'--resolution','1','--exact_optimizer_updates','--seed',str(seed),'--test_iterations','-1','--save_iterations',str(config['base_updates']+config['calibration_updates']),'--disable_viewer']],
        'perspective-evaluate': [['native_gs/evaluate_c0_graphdeco.py','--model-path',str(base),'--source-path',str(source),'--iteration',base_steps,'--splits','test','--erp-width',width]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/replica360.json')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--hyss-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, default=ROOT/'outputs/replica360')
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--stages', nargs='+', choices=['prepare','train','calibrate','evaluate','perspective-evaluate','graphdeco'], default=['prepare','train','calibrate','evaluate'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    scenes = args.scenes or list(config['scenes'])
    if set(scenes) - config['scenes'].keys():
        parser.error('Unknown scene; add its split and initialization count to the config')
    seeds = args.seeds if args.seeds is not None else [config['seed']]
    if len(set(seeds)) != len(seeds):
        parser.error('Seeds must be unique')
    work = args.work_root.resolve()
    records = []
    config_hash = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if not args.dry_run:
        work.mkdir(parents=True,exist_ok=True)
        saved = work/'config.json'
        if saved.exists() and json.loads(saved.read_text()) != config:
            parser.error('Work directory has a different config; use a new --work-root')
        saved.write_text(json.dumps(config,indent=2)+'\n')
    for scene in scenes:
        if not args.dry_run:
            (work/'splits').mkdir(exist_ok=True)
            (work/'splits'/f'{scene}.json').write_text(json.dumps(config['scenes'][scene]['split'],indent=2)+'\n')
        for seed in seeds:
            plan = commands(config,scene,seed,work,args.data_root.resolve(),args.hyss_root.resolve())
            for stage in args.stages:
                run_id = 'shared' if stage == 'prepare' else f'seed{seed}'
                marker = work/'runs'/f'{scene}_{run_id}_{stage}.json'
                if marker.exists() and not args.dry_run:
                    previous = json.loads(marker.read_text())
                    if previous['config_sha256'] != config_hash or previous['commands'] != plan[stage]:
                        raise RuntimeError(f'Existing stage uses different inputs: {marker}')
                    print(f'[skip] {scene} seed {seed} {stage}')
                    continue
                started = time.monotonic()
                for index, command in enumerate(plan[stage]):
                    print('[command]', ' '.join([sys.executable,*command]),flush=True)
                    if not args.dry_run:
                        log = work/'logs'/f'{scene}_seed{seed}_{stage}_{index}.log'
                        log.parent.mkdir(parents=True,exist_ok=True)
                        with log.open('w') as stream:
                            subprocess.run([sys.executable,*command],cwd=ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT,check=True)
                if not args.dry_run:
                    marker.parent.mkdir(parents=True,exist_ok=True)
                    record = {'scene':scene,'seed':seed,'stage':stage,'config_sha256':config_hash,'commands':plan[stage],'wall_seconds':time.monotonic()-started}
                    marker.write_text(json.dumps(record,indent=2)+'\n')
                    records.append(record)
    if not args.dry_run:
        print(f'Completed {len(records)} stages; outputs: {work}')


if __name__ == '__main__':
    main()
