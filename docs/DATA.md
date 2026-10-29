# Data and preprocessing contract

The executable release begins with RGB panoramas, fixed camera poses, and predicted radial-depth/confidence maps. HYSS/FRED inference and pretrained weight acquisition are external prerequisites, not hidden steps in `tools/run.py`. No download URL for a prepared prediction bundle was present in the supplied source; none is invented here.

## Replica360 splits

Every scene in `configs/replica360.json` has seven training panoramas (`0000`, `0001`, `0002`, `0005`, `0006`, `0008`, `0009`), validation panorama `0004`, and test panoramas `0003`, `0007`. Splits are fixed before any pose estimation or depth refinement. The configuration also records the per-scene initialization count found in the supplied prepared-scene metadata.

The pose archive must contain `<pano>_R` (3×3 camera-to-world rotation) and `<pano>_t` (3-vector camera center) for every panorama. Translation and depth must use the same metric coordinate system. Source poses/depth/confidence must be generated using source images only. Held-out poses must be supplied through an independent evaluation protocol; having poses in one archive does not certify that its upstream estimation was leakage-free. Held-out RGB must never be registered jointly with training RGB to generate source initialization.

## Expected files

Raw scene: `<data-root>/<scene>/<pano>_rgb_pano.jpg` and `<pano>_mask_pano.png`. Mask pixels greater than zero are valid. Panorama dimensions must agree within a scene. Masks are used for source geometry and depth validity; benchmark image-quality metrics are full-image, not masked metrics.

HYSS scene: `<hyss-root>/<scene>/panorama/iter_00/fred_depths/<pano>.png` and `<pano>_weight.png`, for training panoramas only. Depth PNGs are uint16, decoded as meters = value / 4000. Confidence PNGs are uint16, decoded as confidence = value / 65535. The alternate `iter_00/fred_depths/` layout is accepted.

Pose archive: `panorama/iter_00/sfm_output/sfm_poses.npz`, `iter_00/sfm_output/sfm_poses.npz`, or `sfm_poses.npz`, below each HYSS scene. The first existing path is used.

The panorama ray convention is `(cos(latitude) cos(longitude), sin(latitude), cos(latitude) sin(longitude))`, with longitude spanning −π to π horizontally and latitude spanning −π/2 to π/2 down the image. The SPaGS adapter applies a fixed rotation between this convention and SPaGS's ERP frame.

## Preparation

1. `prepare_charts.py` builds a subdivision-1 icosahedral cover (80 charts, margin 8°, patch size 512). It exports all RGB charts and split metadata. Only training cameras appear in `sparse/0/images.txt`. It reads no depth files.
2. `prepare_c6_reliability_hyss.py` backprojects training FRED depths, combines FRED confidence with cross-panorama depth/color/support evidence, performs 2 cm confidence-weighted voxel fusion, and deterministically samples the configured target count. If too few eligible voxels exist, it fails; it does not silently change the experiment count. A different depth predictor may require a new explicitly documented count/configuration.
3. `prepare_c6_spherical_maps.py` constructs solid-angle/overlap weights and confidence-weighted inverse-z-depth targets for training charts only. Raw radial depth is converted using the local ray z component.

Prepared scenes contain relative image-directory symlinks and metadata pointing to the raw data path. Move the source data only by regenerating preparation. Do not commit these generated assets to GitHub.

A synthetic CPU integration test creates the complete prepared scene and supervision maps using three RGB panoramas and only one source prediction. No `.dpt` or held-out depth/confidence exists in that fixture.
