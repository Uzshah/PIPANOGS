import torch

from Cameras.Equirectangular import EquirectangularCamera

from SPaGSCudaBackend import _C

def update_3d_filter(
        camera: EquirectangularCamera,
        positions: torch.Tensor,
        filter_3d: torch.Tensor,
        visibility_mask: torch.Tensor,
        distance2filter: float,
) -> None:
    return _C.update_3d_filter_cuda(
        positions,
        camera.properties.T.cuda(),
        filter_3d,
        visibility_mask,
        camera.near_plane,
        distance2filter,
    )
