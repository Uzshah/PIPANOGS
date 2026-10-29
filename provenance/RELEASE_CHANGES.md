# Changes from the research snapshot

- Curated the original packed perspective → spherical calibration pipeline, retaining internal filenames for traceability.
- Preserved the original screen-space densification statistic: pixel-gradient norm times half the maximum chart dimension, followed by panorama-weighted aggregation. All configurations and CLI defaults select screen gradients.
- Fixed confidence/support EMA write-back through boolean indexing in packed and unpacked observation methods. This is a numerical behavior change from the buggy original snapshot.
- Extracted the existing normalized aggregation helper into a CPU-importable module without changing its math.
- Added depth-independent chart preparation; geometry uses source FRED predictions rather than Replica ground-truth depth. Scene splits and initialization counts are explicit.
- Split preparation helpers into `chart_utils.py` and `initialization_utils.py`; empty reprojection diagnostics return an empty list instead of crashing.
- Bundled source-only dependencies, retaining local patches, GLM inline headers, and license notices.
- Made the optional Graphdeco viewer socket lazy so CLI/help and disabled-viewer training need no socket creation.
- Added portable stage orchestration, configuration checks, logs, result summaries, setup recipes, CPU tests, CI, citation metadata, publishing instructions, and SHA256 manifests.
- Removed the exploratory angular implementation and its tooling/reports from this release at the authors' request. Its source snapshot is retained locally in `work/room0_angular_archive/source`, outside the GitHub release folder. Existing experimental checkpoints remain in the ignored `outputs/` directory.
- Excluded the frozen manuscript source from the public code folder because its angular claims are pending author revision; the original manuscript in the research tree was not changed.

`source_snapshot.json` records original input hashes before release edits and may include inputs later split, renamed, or excluded. `release_manifest.json` records the final distributable files.
