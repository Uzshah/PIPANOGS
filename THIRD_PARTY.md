# Third-party source and licensing

This folder contains source snapshots from the provided research workspace, with copyright notices and local compatibility patches retained. Git histories were absent, so upstream revisions are identified by file hashes rather than invented commit IDs. Only the source required for the main pipeline is bundled; no upstream trained models, outputs, or compiled extensions are included.

| Component | Use | License notice |
|---|---|---|
| Graphdeco Gaussian Splatting | Scene, cameras, Gaussian parameterization and optional perspective evaluation | `third_party/gaussian-splatting/LICENSE.md` |
| Graphdeco rasterizer | Gaussian model imports and optional perspective evaluation | `third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/LICENSE.md` |
| simple-knn | Initialization scales | `third_party/gaussian-splatting/submodules/simple-knn/LICENSE.md` |
| GLM | CUDA/C++ math headers | `third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/third_party/glm/copying.txt` |
| fused-ssim | Spherical calibration objective | `third_party/fused-ssim/LICENSE` |
| SPaGS | Existing spherical differentiable and benchmark rasterization | `third_party/SPaGS/LICENSE` |
| NeRFICG camera utilities | ERP camera configuration for SPaGS | `third_party/nerficg/LICENSE` |
| gsplat (pip dependency) | Packed camera-batched perspective rasterization | See installed package's license |

Graphdeco and simple-knn carry non-commercial research/evaluation terms; those terms are not replaced by a license for original Π-GS code. SPaGS and the retained NeRFICG camera utilities retain their MIT notices. Other nested notices remain in place.

The chart-construction helper states that its projection equations were ported from HYSS. HYSS/FRED predictions are external inputs. The authors must confirm the intended license of their original and adapted preparation code before assigning a repository-wide license. No broad MIT/Apache claim is made by this release.

PFGS360 source is not bundled: the historical provenance explicitly reports no observed upstream license file. Its known-pose adaptation requires separate upstream license/provenance review. Data and pretrained weights must be obtained under their own terms.
