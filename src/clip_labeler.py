import importlib
import torch
from PIL import Image
from typing import List, Tuple, Any, Optional, Callable


class CLIPConceptLabeler:
    """
    Optional zero-shot concept hypothesis ranker using CLIP.
    Compares candidate text descriptions against top-activating image crops.
    Falls back gracefully if CLIP is not installed, uninitialized, or weights are not loaded.
    """
    def __init__(self,
                 model_name: str = "ViT-B-32",
                 pretrained: Optional[str] = None,
                 device: str = "cpu"):
        self.device = device
        self.available: bool = False
        self.model: Optional[Any] = None
        self.preprocess: Optional[Callable[[Image.Image], torch.Tensor]] = None
        self.tokenizer: Optional[Any] = None

        # Only attempt remote weights loading if pretrained identifier is explicitly provided
        if pretrained:
            try:
                open_clip = importlib.import_module("open_clip")
                create_fn = getattr(open_clip, "create_model_and_transforms")
                tok_fn = getattr(open_clip, "get_tokenizer")

                model, _, preprocess = create_fn(
                    model_name,
                    pretrained=pretrained,
                    device=device
                )
                self.model = model
                self.preprocess = preprocess
                self.tokenizer = tok_fn(model_name)
                self.model.eval()
                self.available = True
            except Exception:
                self.available = False
                self.model = None
                self.preprocess = None
                self.tokenizer = None
        else:
            self.available = False

    def rank_candidate_concepts(self,
                                image_crops: List[Image.Image],
                                candidate_concepts: List[str]) -> List[Tuple[str, float]]:
        """
        Ranks candidate concept hypotheses according to average cosine similarity
        with the top-activating image crops.
        """
        if (
            not self.available
            or self.model is None
            or self.preprocess is None
            or self.tokenizer is None
            or len(image_crops) == 0
            or len(candidate_concepts) == 0
        ):
            # Fallback when CLIP is unavailable or inputs are empty
            return [(concept, 0.0) for concept in candidate_concepts]

        with torch.no_grad():
            text_tokens = self.tokenizer([f"a photo containing {c}" for c in candidate_concepts]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            img_tensors = torch.stack([self.preprocess(img) for img in image_crops]).to(self.device)
            image_features = self.model.encode_image(img_tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Similarity matrix: [N_crops, N_concepts]
            similarity = (image_features @ text_features.T).mean(dim=0).cpu().numpy()

        ranked = sorted(zip(candidate_concepts, similarity.tolist()), key=lambda x: x[1], reverse=True)
        return ranked
