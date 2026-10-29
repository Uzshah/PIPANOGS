#!/usr/bin/env python3
"""Verify release file hashes, Python syntax, shell syntax, and binary exclusions."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'__pycache__','.pytest_cache','.git','outputs','build','dist'}
MANIFEST=ROOT/'provenance/release_manifest.json'


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and p!=MANIFEST
                  and not any(part in EXCLUDED or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts))


def main():
    paths=files()
    errors=[]
    for path in paths:
        if path.is_symlink():
            errors.append(f'Unexpected source symlink: {path.relative_to(ROOT)}')
        if path.suffix in {'.so','.o','.a','.pth','.pt','.ckpt','.zip','.npz','.npy'}:
            errors.append(f'Generated/binary artifact: {path.relative_to(ROOT)}')
        if path.stat().st_size>20*1024*1024:
            errors.append(f'Oversized source artifact: {path.relative_to(ROOT)}')
        if path.suffix=='.py':
            try:
                ast.parse(path.read_text(),filename=str(path))
            except (SyntaxError,UnicodeError) as error:
                errors.append(str(error))
        if path.suffix=='.sh':
            check=subprocess.run(['bash','-n',str(path)],text=True,capture_output=True)
            if check.returncode:
                errors.append(check.stderr)
    actual={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if '--write-manifest' in sys.argv:
        MANIFEST.write_text(json.dumps({'algorithm':'sha256','files':actual},indent=2)+'\n')
    elif not MANIFEST.exists():
        errors.append('Release manifest missing')
    else:
        expected=json.loads(MANIFEST.read_text())['files']
        for name in sorted(actual.keys()|expected.keys()):
            if actual.get(name)!=expected.get(name):
                errors.append(f'File missing, added, or modified: {name}')
    print(json.dumps({'files':len(paths),'bytes':sum(p.stat().st_size for p in paths),'errors':errors},indent=2))
    return bool(errors)


if __name__=='__main__':
    raise SystemExit(main())
