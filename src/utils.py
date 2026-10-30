import os
import random
import json
import yaml
import torch
import numpy as np
from PIL import Image
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
                      mean: tuple = (0.5, 0.5, 0.5),
                      std: tuple = (0.5, 0.5, 0.5)) -> Image.Image:
    """
    Convert normalized PyTorch image tensor (C, H, W) back to PIL Image.
    Defaults to (0.5, 0.5, 0.5) matching timm's vit_base_patch16_224.augreg_in1k preprocessing.
    """
    t = tensor.detach().cpu().clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    t = torch.clamp(t, 0.0, 1.0)
    img_np = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(img_np)


def create_transparent_heatmap_overlay(image: Image.Image,
                                       heatmap_2d: np.ndarray,
                                       colormap_name: str = "inferno",
                                       smooth: bool = False,
                                       peak_row: Optional[int] = None,
                                       peak_col: Optional[int] = None) -> Image.Image:
    """
    Overlay a 2D activation heatmap onto an image with true per-pixel transparency:
    - Inactive patches (activation = 0) have alpha = 0 (100% transparent, underlying image is completely untinted).
    - Active patches have alpha proportional to activation strength, glowing over the image.
    - Optionally draws a high-contrast double box (black outer, yellow inner) around peak patch.

    Args:
        image: PIL Image (RGB)
        heatmap_2d: 2D numpy array [14, 14]
        colormap_name: Colormap name (default: "inferno")
        smooth: If False, uses nearest-neighbor (crisp 14x14 tiles); if True, uses bicubic interpolation.
        peak_row: Optional spatial row index of peak patch
        peak_col: Optional spatial column index of peak patch
    """
    from PIL import ImageDraw

    w, h = image.size
    orig_rgba = image.convert("RGBA")

    max_val = float(heatmap_2d.max())
    if max_val <= 0.0:
        # Inactive feature: return pristine image
        return orig_rgba.convert("RGB")

    norm_map = np.clip(heatmap_2d / max_val, 0.0, 1.0)
    resample_mode = Image.Resampling.BICUBIC if smooth else Image.Resampling.NEAREST
    heatmap_pil = Image.fromarray((norm_map * 255.0).astype(np.uint8)).resize((w, h), resample=resample_mode)
    norm_resized = np.array(heatmap_pil).astype(np.float32) / 255.0

    import matplotlib
    try:
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap_name)

    colored = cmap(norm_resized)  # [H, W, 4] with RGBA in [0, 1]

    # Transparency mask:
    # Where norm_resized == 0: alpha = 0.0 (completely transparent)
    # Where norm_resized > 0: alpha scales from 0.25 to 0.70
    alpha_mask = np.where(norm_resized > 0.0, 0.25 + 0.45 * norm_resized, 0.0)
    colored[:, :, 3] = alpha_mask

    overlay_layer = Image.fromarray((colored * 255.0).astype(np.uint8), mode="RGBA")
    composite = Image.alpha_composite(orig_rgba, overlay_layer)

    # Highlight peak patch box if specified
    if peak_row is not None and peak_col is not None and max_val > 0.0:
        grid_h, grid_w = heatmap_2d.shape
        patch_w = w // grid_w
        patch_h = h // grid_h
        x0 = peak_col * patch_w
        y0 = peak_row * patch_h
        x1 = (peak_col + 1) * patch_w
        y1 = (peak_row + 1) * patch_h

        draw = ImageDraw.Draw(composite)
        draw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(0, 0, 0), width=1)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 235, 59), width=2)

    return composite.convert("RGB")


def create_discrete_patch_grid(heatmap_2d: np.ndarray,
                               colormap_name: str = "inferno",
                               peak_row: Optional[int] = None,
                               peak_col: Optional[int] = None,
                               image_size: int = 224) -> Image.Image:
    """
    Renders a genuinely discrete 14x14 patch grid of exact dimensions (image_size, image_size)
    using nearest-neighbor scaling (each patch is a solid 16x16 tile), with subtle patch gridlines
    and a high-contrast yellow/black border outlining the peak argmax patch.
    """
    from PIL import ImageDraw
    import matplotlib

    grid_h, grid_w = heatmap_2d.shape
    max_val = float(heatmap_2d.max())
    norm_map = np.clip(heatmap_2d / max_val, 0.0, 1.0) if max_val > 0.0 else np.zeros_like(heatmap_2d)

    # Nearest-neighbor upscale to image_size ensures crisp, unsmoothed patch tiles
    img_nn = Image.fromarray((norm_map * 255.0).astype(np.uint8)).resize(
        (image_size, image_size), resample=Image.Resampling.NEAREST
    )
    norm_resized = np.array(img_nn).astype(np.float32) / 255.0

    try:
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap_name)

    colored = cmap(norm_resized)[:, :, :3]  # Drop alpha
    colored_uint8 = (colored * 255.0).astype(np.uint8)

    grid_img = Image.fromarray(colored_uint8, mode="RGB")
    draw = ImageDraw.Draw(grid_img)

    # Subtle grid lines separating individual 16x16 patches
    patch_w = image_size // grid_w
    patch_h = image_size // grid_h
    for c in range(1, grid_w):
        x = c * patch_w
        draw.line([(x, 0), (x, image_size)], fill=(70, 70, 70), width=1)
    for r in range(1, grid_h):
        y = r * patch_h
        draw.line([(0, y), (image_size, y)], fill=(70, 70, 70), width=1)

    # High-contrast double box on the peak patch (black outer, yellow inner)
    if peak_row is not None and peak_col is not None and max_val > 0.0:
        x0 = peak_col * patch_w
        y0 = peak_row * patch_h
        x1 = (peak_col + 1) * patch_w
        y1 = (peak_row + 1) * patch_h
        draw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(0, 0, 0), width=1)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 235, 59), width=2)

    return grid_img


def render_discrete_grid_with_colorbar(heatmap_2d: np.ndarray,
                                       peak_row: Optional[int] = None,
                                       peak_col: Optional[int] = None,
                                       colormap_name: str = "inferno",
                                       title: str = "Discrete 14×14 Patch Grid") -> Image.Image:
    """
    Renders a genuinely discrete 14×14 patch grid using nearest-neighbor interpolation,
    with visible patch gridlines, row/col coordinates (R1..R14, C1..C14), a high-contrast
    yellow/black box around the argmax patch, and a numeric colorbar in actual SAE activation units.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_val = float(heatmap_2d.max())
    vmax = max_val if max_val > 0.0 else 1.0

    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=100)
    im = ax.imshow(heatmap_2d, cmap=colormap_name, interpolation="nearest", vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SAE Activation (ReLU Units)", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    # Grid lines marking individual patch boundaries (14x14)
    ax.set_xticks(np.arange(-0.5, 14, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 14, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.35)
    ax.tick_params(which="minor", size=0)

    # Patch coordinate labels: 1..14
    ax.set_xticks(np.arange(0, 14, 2))
    ax.set_xticklabels([f"C{c+1}" for c in range(0, 14, 2)], fontsize=8)
    ax.set_yticks(np.arange(0, 14, 2))
    ax.set_yticklabels([f"R{r+1}" for r in range(0, 14, 2)], fontsize=8)

    # Draw peak bounding box
    if peak_row is not None and peak_col is not None and max_val > 0.0:
        rect_outer = plt.Rectangle((peak_col - 0.5, peak_row - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=3)
        rect_inner = plt.Rectangle((peak_col - 0.5, peak_row - 0.5), 1, 1, fill=False, edgecolor='yellow', linewidth=1.8)
        ax.add_patch(rect_outer)
        ax.add_patch(rect_inner)

    peak_text = f"Peak: {max_val:.2f} (R{peak_row+1}, C{peak_col+1})" if (peak_row is not None and max_val > 0) else "Inactive (0.00)"
    ax.set_title(f"{title}\n[{peak_text}]", fontsize=10, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_overlay_with_colorbar(overlay_pil: Image.Image,
                                 peak_val: float,
                                 colormap_name: str = "inferno",
                                 title: str = "Heatmap Overlay on Input") -> Image.Image:
    """
    Renders the composite transparent overlay alongside a matched numeric colorbar
    indicating actual SAE activation units [0.0, peak_val].
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = peak_val if peak_val > 0.0 else 1.0
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=100)
    ax.imshow(overlay_pil)
    ax.axis("off")

    try:
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = plt.get_cmap(colormap_name)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SAE Activation (ReLU Units)", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    peak_text = f"Peak: {peak_val:.2f}" if peak_val > 0.0 else "Inactive (0.00)"
    ax.set_title(f"{title}\n[{peak_text}]", fontsize=10, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def overlay_heatmap_on_image(image: Image.Image,
                             heatmap_2d: np.ndarray,
                             colormap_name: str = "inferno",
                             alpha: float = 0.55) -> Image.Image:
    """
    Legacy compatibility wrapper for overlaying a heatmap on a PIL image.
    Preserves transparent overlay behavior for inactive patches.
    """
    return create_transparent_heatmap_overlay(
        image=image,
        heatmap_2d=heatmap_2d,
        colormap_name=colormap_name,
        smooth=True
    )


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
    "create_discrete_patch_grid",
    "create_transparent_heatmap_overlay",
    "render_discrete_grid_with_colorbar",
    "render_overlay_with_colorbar",
    "overlay_heatmap_on_image",
]
