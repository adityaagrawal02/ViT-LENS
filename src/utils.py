import os
import random
import json
import yaml
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from typing import List, Optional

_IMAGENET_LABELS_CACHE: Optional[List[str]] = None


def seed_everything(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto") -> torch.device:
    """Detect available compute device (CUDA GPU, Apple MPS, or CPU)."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: dict, filepath: str) -> None:
    """Save dictionary to formatted JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> dict:
    """Load dictionary from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_imagenet_labels() -> List[str]:
    """
    Returns standard 1,000 ImageNet-1K human-readable class names.
    Cached in memory after first retrieval.
    """
    global _IMAGENET_LABELS_CACHE
    if _IMAGENET_LABELS_CACHE is not None:
        return _IMAGENET_LABELS_CACHE

    # Attempt 1: torchvision ViT_B_16_Weights categories
    try:
        from torchvision.models import ViT_B_16_Weights
        categories = ViT_B_16_Weights.DEFAULT.meta.get("categories", [])
        if categories and len(categories) == 1000:
            _IMAGENET_LABELS_CACHE = [str(c) for c in categories]
            return _IMAGENET_LABELS_CACHE
    except Exception:
        pass

    # Attempt 2: local class mapping file if present
    mapping_path = "./data/imagenet_classes.json"
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) == 1000:
                    _IMAGENET_LABELS_CACHE = data
                    return _IMAGENET_LABELS_CACHE
        except Exception:
            pass

    # Fallback: indexed labels
    _IMAGENET_LABELS_CACHE = [f"Class #{i}" for i in range(1000)]
    return _IMAGENET_LABELS_CACHE


def format_class_label(class_idx: int, class_name: Optional[str] = None) -> str:
    """
    Returns human-readable name formatted with class index.
    Supports single argument: format_class_label(idx)
    or two arguments: format_class_label(idx, class_name)
    """
    if class_name is not None and str(class_name).strip():
        name = str(class_name).strip()
        if f"(#{class_idx})" in name:
            return name
        return f"{name} (#{class_idx})"

    labels = get_imagenet_labels()
    if 0 <= class_idx < len(labels):
        name = labels[class_idx]
        return f"{name} (#{class_idx})"
    return f"Class #{class_idx}"


def resolve_sae_checkpoint_path(explicit_path: Optional[str] = None) -> str:
    """
    Resolves the final SAE checkpoint path according to priority:
    1. explicit_path parameter
    2. VIT_LENS_SAE_CHECKPOINT environment variable
    3. config.yaml sae.checkpoint_path or sae.checkpoint_dir
    4. Google Colab default: /content/drive/MyDrive/ViT-LENS/checkpoints/final_sae_best.pt
    5. Local fallbacks: ./checkpoints/final_sae_best.pt or ./checkpoints/sae_checkpoint_latest.pt
    """
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    env_path = os.environ.get("VIT_LENS_SAE_CHECKPOINT")
    if env_path and os.path.exists(env_path):
        return env_path

    # Check candidates
    candidates = [
        env_path,
        "/content/drive/MyDrive/ViT-LENS/checkpoints/final_sae_best.pt",
        "./checkpoints/final_sae_best.pt",
        "./checkpoints/sae_checkpoint_latest.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c

    # Return env_path or default Colab path for informative error message
    return env_path or "/content/drive/MyDrive/ViT-LENS/checkpoints/final_sae_best.pt"


def denormalize_image(tensor: torch.Tensor,
                      mean: tuple = (0.485, 0.456, 0.406),
                      std: tuple = (0.229, 0.224, 0.225)) -> Image.Image:
    """Convert normalized PyTorch image tensor (C, H, W) back to PIL Image."""
    t = tensor.detach().cpu().clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    t = torch.clamp(t, 0.0, 1.0)
    img_np = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(img_np)


def overlay_heatmap_on_image(image: Image.Image,
                             heatmap_2d: np.ndarray,
                             colormap_name: str = "jet",
                             alpha: float = 0.55) -> Image.Image:
    """
    Overlay a normalized 2D heatmap on a PIL Image.

    Args:
        image: PIL Image (RGB)
        heatmap_2d: 2D numpy array [H, W], values typically in [0, 1]
        colormap_name: Matplotlib colormap name
        alpha: Blending weight for the heatmap overlay (0.0 = original, 1.0 = heatmap only)
    """
    w, h = image.size
    # Ensure heatmap is normalized between 0 and 1
    h_min, h_max = heatmap_2d.min(), heatmap_2d.max()
    if h_max > h_min:
        norm_map = (heatmap_2d - h_min) / (h_max - h_min)
    else:
        norm_map = np.zeros_like(heatmap_2d)

    # Resize heatmap to match image dimensions
    heatmap_img = Image.fromarray((norm_map * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
    norm_resized = np.array(heatmap_img) / 255.0

    # Colorize using colormap
    cmap = cm.get_cmap(colormap_name)
    colored_heatmap = cmap(norm_resized)[:, :, :3]  # Drop alpha
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

    # Blend with original image
    orig_np = np.array(image.convert("RGB"))
    blended = (alpha * colored_heatmap_uint8 + (1.0 - alpha) * orig_np).astype(np.uint8)
    return Image.fromarray(blended)


__all__ = [
    "seed_everything",
    "get_device",
    "load_config",
    "save_json",
    "load_json",
    "get_imagenet_labels",
    "format_class_label",
    "resolve_sae_checkpoint_path",
    "denormalize_image",
    "overlay_heatmap_on_image",
]
