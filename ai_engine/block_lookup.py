"""OpenCLIP-based mapping from an image or texture description to blocks."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image


BLOCK_PROMPTS = {
    "stone_bricks": "rough grey carved stone brick fortress wall",
    "oak_planks": "warm wooden oak timber planks",
    "cobblestone": "cracked weathered grey cobblestone",
    "mossy_cobblestone": "mossy green ancient damp stone",
    "glass": "transparent clear glass window pane",
    "bricks": "traditional red clay masonry bricks",
    "sandstone": "dry pale yellow carved desert sandstone",
}


class BlockLookup:
    """Cache OpenCLIP embeddings and rank Minecraft blocks for an input image."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        import open_clip

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        tokens = tokenizer(list(BLOCK_PROMPTS.values())).to(self.device)
        with torch.no_grad():
            self.text_embeddings = self.model.encode_text(tokens)
            self.text_embeddings /= self.text_embeddings.norm(dim=-1, keepdim=True)
        self.block_names = list(BLOCK_PROMPTS)

    def rank(self, image: Image.Image, limit: int | None = None) -> list[dict[str, Any]]:
        """Return blocks ordered by cosine similarity, highest first."""
        with torch.no_grad():
            image_embedding = self.model.encode_image(
                self.preprocess(image).unsqueeze(0).to(self.device)
            )
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            scores = (image_embedding @ self.text_embeddings.T).squeeze(0)

        order = torch.argsort(scores, descending=True).cpu().tolist()
        if limit is not None:
            order = order[:limit]
        return [
            {"block": self.block_names[index], "score": float(scores[index])}
            for index in order
        ]