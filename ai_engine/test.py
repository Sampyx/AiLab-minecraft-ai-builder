import trimesh
import json

# Load 3D mesh
mesh = trimesh.load('sample.obj')

# Convert smooth mesh surface to a discrete voxel grid (e.g., 32x32x32)
voxel_grid = mesh.voxelized(pitch=0.05).fill()
points = voxel_grid.points.astype(int)

# Export normalized coordinates to JSON
coords = [{"x": int(pt[0]), "y": int(pt[1]), "z": int(pt[2])} for pt in points]
with open("voxel_coords.json", "w") as f:
    json.dump(coords, f)

print(f"Generated {len(coords)} voxel positions.")