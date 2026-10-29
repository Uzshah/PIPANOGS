"""Icosahedral chart geometry and PLY output.

Chart construction/projection equations originate from the HYSS research helper.
Only depth-independent functions needed by this release are retained.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

def build_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + 5.0**0.5) / 2.0
    a, b = 1.0, phi
    vertices = np.array([
        [-a, b, 0], [a, b, 0], [-a, -b, 0], [a, -b, 0],
        [0, -a, b], [0, a, b], [0, -a, -b], [0, a, -b],
        [b, 0, -a], [b, 0, a], [-b, 0, -a], [-b, 0, a],
    ], dtype=np.float64)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)
    return vertices, faces


def subdivide_geodesic(
    vertices: np.ndarray, faces: np.ndarray, level: int
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(level):
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces: list[list[int]] = []
        vertex_list = vertices.tolist()

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key not in midpoint_cache:
                value = vertices[i] + vertices[j]
                value /= np.linalg.norm(value)
                midpoint_cache[key] = len(vertex_list)
                vertex_list.append(value.tolist())
            return midpoint_cache[key]

        for i0, i1, i2 in faces:
            a = midpoint(int(i0), int(i1))
            b = midpoint(int(i1), int(i2))
            c = midpoint(int(i2), int(i0))
            new_faces.extend([[i0, a, c], [a, i1, b], [c, b, i2], [a, b, c]])
        vertices = np.asarray(vertex_list, dtype=np.float64)
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
        faces = np.asarray(new_faces, dtype=np.int32)
    return vertices, faces


def face_rotation_from_center(center: np.ndarray) -> np.ndarray:
    """HYSS local pinhole (+x,+y,+z) to its ERP ray frame."""
    z_axis = center / np.linalg.norm(center)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(z_axis, up))) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def build_faces(subdivision: int, margin_deg: float) -> list[tuple[np.ndarray, float]]:
    vertices, triangles = build_icosahedron()
    vertices, triangles = subdivide_geodesic(vertices, triangles, subdivision)
    centers = vertices[triangles].sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    result = []
    for center, indices in zip(centers, triangles):
        angular_radius = max(
            math.acos(float(np.clip(np.dot(center, vertices[index]), -1.0, 1.0)))
            for index in indices
        )
        fov = 2.0 * angular_radius + math.radians(margin_deg)
        if not 0.0 < fov < math.pi:
            raise ValueError(f"Invalid gnomonic field of view: {math.degrees(fov):.3f} deg")
        result.append((face_rotation_from_center(center), fov))
    return result


def gnomonic_map(
    rotation: np.ndarray, fov: float, patch_size: int, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    half = math.tan(0.5 * fov)
    axis = np.linspace(-half, half, patch_size, dtype=np.float64)
    y_grid, x_grid = np.meshgrid(axis, axis, indexing="ij")
    local = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    rays = np.einsum("...j,ij->...i", local, rotation)
    longitude = np.arctan2(rays[..., 2], rays[..., 0])
    latitude = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
    source_x = (longitude + math.pi) / (2.0 * math.pi) * (width - 1)
    source_y = (latitude + math.pi / 2.0) / math.pi * (height - 1)
    return source_x.astype(np.float32), source_y.astype(np.float32)


def rotmat2qvec(rotation: np.ndarray) -> np.ndarray:
    """COLMAP scalar-first quaternion, matching Graphdeco's loader."""
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.array([
        [rxx - ryy - rzz, 0, 0, 0],
        [ryx + rxy, ryy - rxx - rzz, 0, 0],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
    ]) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    quaternion = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    return -quaternion if quaternion[0] < 0 else quaternion


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"),
        ("ny", "<f4"), ("nz", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.zeros(len(points), dtype=dtype)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())

