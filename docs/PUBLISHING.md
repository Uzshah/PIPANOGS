# Copying into the existing GitHub repository

The intended remote is https://github.com/Uzshah/PIPANOGS. This preparation does not push to GitHub or alter the original research tree.

Clone the existing repository into a separate destination, so its initial README commit is preserved. Copy the contents of this release folder into that clone (including `.gitignore` and `.github/`, excluding generated caches and outputs). The release README replaces the initial placeholder README.

For example, from a directory containing both this release folder and a clone named `PIPANOGS-github`:

```bash
git clone https://github.com/Uzshah/PIPANOGS.git PIPANOGS-github
rsync -av --exclude=.git --exclude=outputs --exclude=__pycache__ \
  --exclude=.pytest_cache --exclude='*.egg-info' PIPANOGS/ PIPANOGS-github/
cd PIPANOGS-github
python tools/verify_release.py
python -m pytest -q
git status --short
git add .
git diff --cached --stat
git commit -m "Release Pi-GS with screen-space densification"
git push origin HEAD
```

Before describing the repository as an exact paper reproduction, resolve the explicit paper/code differences, recover the revised result-table run configuration/checkpoints, obtain source-only preprocessing inputs, and validate a fresh installation and the full twelve-scene matrix. The original-code license is an author decision; third-party terms are already preserved. There is no reason to force-push or replace existing history.
