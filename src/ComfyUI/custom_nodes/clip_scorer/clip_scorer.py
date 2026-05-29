import torch
import numpy as np
from PIL import Image

class CLIPScorer:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": False, "default": "A photo"}),
                "nonce": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)

    FUNCTION = "compute_score"
    CATEGORY = "custom/metrics"

    def __init__(self):
        import clip
        self.clip = clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def compute_score(self, image, text, nonce):
        img_tensor = image[0]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        text_input = self.clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()

        score = float(similarity)

        return {
            "ui": {"score": [score]},
            "result": (score,)
        }

        # return (float(similarity),)
        # return {"ui": {"score": float(similarity)}, "result": (float(similarity),)}

NODE_CLASS_MAPPINGS = {
    "CLIPScorer": CLIPScorer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPScorer": "CLIP Scorer"
}