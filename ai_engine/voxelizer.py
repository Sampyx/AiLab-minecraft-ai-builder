import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
import trimesh
import open_clip

# ---------------------------------------------------------
# 1. Device Configuration & OpenCLIP Initialization
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model on: {device}")

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# ---------------------------------------------------------
# 2. Block Vocabulary & Precomputed Text Embeddings
# ---------------------------------------------------------
palette = {
    "stone_bricks": "rough grey carved stone brick fortress wall",
    "oak_planks": "warm wooden timber planks",
    "cobblestone": "cracked weathered grey cobblestone",
    "mossy_cobblestone": "mossy green ancient damp stone",
    "glass": "transparent clear glass window pane",
    "bricks": "traditional red clay masonry bricks",
    "sandstone": "dry pale yellow carved desert sandstone"
}

block_names = list(palette.keys())
prompts = [palette[b] for b in block_names]
text_tokens = tokenizer(prompts).to(device)

with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

# ---------------------------------------------------------
# 3. 3D Mesh Generation / Loading & Voxelization
# ---------------------------------------------------------
def get_voxel_grid(grid_resolution=16):
    """
    Creates a sample 3D mesh (e.g., a tower) and voxelizes it.
    You can replace 'cylinder' with trimesh.load('your_mesh.obj').
    """
    mesh = trimesh.creation.cylinder(radius=1.5, height=3.0)
    # Voxelize the continuous mesh to a discrete grid
    voxelized = mesh.voxelized(pitch=mesh.extents.max() / grid_resolution).fill()
    matrix = voxelized.matrix # 3D boolean array
    return matrix

# ---------------------------------------------------------
# 4. CLIP Semantic Block Assignment (Your Novel Module)
# ---------------------------------------------------------
def generate_build_plan(voxel_matrix, structure_theme="mossy stone"):
    """
    Simulates local patch inference through CLIP to select blocks
    based on contextual similarity to the structure's theme.
    """
    # Create a synthetic texture patch representing the theme
    dummy_texture = Image.new("RGB", (64, 64), color=(110, 130, 95)) # muted moss-green/stone
    img_tensor = preprocess(dummy_texture).unsqueeze(0).to(device)
    
    with torch.no_grad():
        img_embedding = model.encode_image(img_tensor)
        img_embedding /= img_embedding.norm(dim=-1, keepdim=True)
        
        # Cosine similarity matrix multiplication
        similarities = (img_embedding @ text_embeddings.T).squeeze(0)
        top_indices = torch.topk(similarities, k=2).indices.cpu().numpy()
        
    primary_block = block_names[top_indices[0]]
    secondary_block = block_names[top_indices[1]]
    print(f"CLIP Selected Materials: Primary='{primary_block}', Secondary='{secondary_block}'")

    # Build the sequential placement list
    build_plan = []
    # Minecraft coordinate system: Y is up/down
    dx, dy, dz = voxel_matrix.shape
    for y in range(dy): # Bottom-to-top layer ordering
        for x in range(dx):
            for z in range(dz):
                if voxel_matrix[x, y, z]:
                    # Alternate secondary material for edges/accents
                    chosen = secondary_block if (x == 0 or x == dx - 1 or z == 0 or z == dz - 1) else primary_block
                    build_plan.append({
                        "x": int(x),
                        "y": int(y),
                        "z": int(z),
                        "block": chosen
                    })
                    
    return build_plan

# Run pipeline
voxel_matrix = get_voxel_grid(grid_resolution=12)
build_plan = generate_build_plan(voxel_matrix, structure_theme="ancient damp stone")

# Export to JSON
with open("build_plan.json", "w") as f:
    json.dump(build_plan, f, indent=2)

print(f"Done! {len(build_plan)} blocks planned in 'build_plan.json'.")