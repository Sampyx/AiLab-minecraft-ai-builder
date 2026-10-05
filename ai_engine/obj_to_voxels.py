"""Convert a triangle mesh into integer Minecraft-style voxel coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def mesh_to_coordinates(mesh_path: str | Path, resolution: int = 32) -> np.ndarray:
    """Return occupied voxel coordinates as an ``N x 3`` integer array.

    Coordinates are normalized so the smallest occupied coordinate is zero and
    the longest mesh axis is at most ``resolution`` voxels.
    """
    if resolution < 1:
        raise ValueError("resolution must be at least 1")

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh contains no geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    if mesh.is_empty:
        raise ValueError(f"mesh contains no geometry: {mesh_path}")
    longest_extent = float(mesh.extents.max())
    if longest_extent <= 0:
        raise ValueError("mesh must have a non-zero extent")

    pitch = longest_extent / resolution
    voxel_grid = mesh.voxelized(pitch=pitch).fill()
    coordinates = np.argwhere(voxel_grid.matrix).astype(np.int32)
    if coordinates.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    return coordinates - coordinates.min(axis=0)


def write_coordinates(mesh_path: str | Path, output_path: str | Path, resolution: int = 32) -> None:
    """Convert ``mesh_path`` and write coordinates as a JSON array."""
    coordinates = mesh_to_coordinates(mesh_path, resolution)
    payload = [{"x": int(x), "y": int(y), "z": int(z)} for x, y, z in coordinates]
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="input .obj file")
    parser.add_argument("output", type=Path, help="output JSON file")
    parser.add_argument("--resolution", type=int, default=32)
    args = parser.parse_args()
    write_coordinates(args.mesh, args.output, args.resolution)


if __name__ == "__main__":
    main()