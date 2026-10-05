# Minecraft AI Builder

## First milestone: `.obj` to voxel coordinates

Run this locally or in Colab after installing `requirements-colab.txt`:

```bash
python -m ai_engine.obj_to_voxels model.obj voxel_coords.json --resolution 32
```

The output is a JSON list of integer `{x, y, z}` positions. The converter
combines scene geometry, fills the voxelized mesh, and normalizes its minimum
coordinate to zero.

## Colab mesh generation

The simplest initial path is Shap-E. In a Colab GPU runtime, run:

```python
!pip install git+https://github.com/openai/shap-e.git trimesh
```

Then generate and export an OBJ:

```python
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
latents = sample_latents(
    batch_size=1, model=model, diffusion=load_config('diffusion')
    .make_schedules(64), guidance_scale=15.0,
    model_kwargs={'texts': ['a small Minecraft stone house']},
    progress=True, clip_denoised=True, use_fp16=True, device=device,
)
mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
with open('house.obj', 'w') as output:
    mesh.write_obj(output)
```

The Shap-E API changes occasionally; if that cell reports an import or keyword
error, use the current export cell from the Shap-E repository. Once `house.obj`
exists, run the converter above and then use `ai_engine.block_lookup.BlockLookup`
to rank block materials from a rendered texture.

