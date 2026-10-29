#!/usr/bin/env python3
"""Summarize per-scene ERP metrics, requiring a complete experiment by default."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def summarize(work, seeds, allow_partial=False):
    config=json.loads((work/'config.json').read_text())
    rows=[]; missing=[]
    for seed in seeds:
        for scene in config['scenes']:
            path=work/'models'/f'seed_{seed}'/scene/'calibrated/c6pi_spherical_cuda.json'
            if not path.is_file():
                missing.append({'scene':scene,'seed':seed})
                continue
            result=json.loads(path.read_text())
            if result['split']!='test' or result['erp_resolution']!=[config['erp_width'],config['erp_width']//2]:
                raise ValueError(f'Metric protocol mismatch: {path}')
            rows.append({'scene':scene,'seed':seed,**result['quality'],'gaussians':result['gaussians'],'render_ms':result['render_ms']})
    if missing and not allow_partial:
        raise ValueError(f'Missing {len(missing)} scene/seed results; use --allow-partial only for progress reports: {missing}')
    if not rows:
        raise ValueError('No results found')
    # Equal scene weight, including when a partial report has unequal seed coverage.
    keys=['psnr','ssim','lpips','gaussians','render_ms']
    scene_means={scene:{key:mean(row[key] for row in rows if row['scene']==scene) for key in keys} for scene in sorted({row['scene'] for row in rows})}
    result={'complete':not missing,'missing':missing,'rows':rows,'scene_means':scene_means,'mean':{key:mean(value[key] for value in scene_means.values()) for key in keys}}
    (work/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (work/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0])
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    result=summarize(args.work_root.resolve(),args.seeds,args.allow_partial)
    print(json.dumps({'complete':result['complete'],'missing':result['missing'],'mean':result['mean']},indent=2))

if __name__=='__main__':
    main()
