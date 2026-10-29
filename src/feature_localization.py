import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict

from src.utils import overlay_heatmap_on_image, denormalize_image


class FeatureLocalizer:
    """
    Computes spatial activation heatmaps for individual SAE features on input images,
    mapping 1D patch-token activations to 2D spatial overlays (14x14 -> 224x224).

    Important Interpretability Notice:
    - SAE feature heatmaps display where in the image plane a learned dictionary atom activated.
    - These are distinct from Transformer attention maps (which measure inter-token routing weights).
    - Peak patch activation indicates feature presence, but does NOT prove the exact causal pixels.
    """
    def __init__(self,
                 sae,
                 vit_wrapper,
                 grid_size: int = 14,
                 image_size: int = 224,
                 output_dir: str = "./outputs/heatmaps"):
        self.sae = sae
        self.vit = vit_wrapper
        self.grid_size = grid_size
        self.image_size = image_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def compute_feature_map(self,
                            image_tensor: torch.Tensor,
                            feature_id: int) -> Tuple[np.ndarray, float, float, int, int, int]:
        """
        Calculates 14x14 spatial activation matrix for given feature on an input image tensor.
        Args:
            image_tensor: [1, 3, 224, 224] or [3, 224, 224]
            feature_id: SAE feature index
        Returns:
            heatmap_14x14: 2D numpy array [14, 14]
            max_activation: highest activation strength across patches
            mean_activation: average activation across patches
            peak_patch_idx: flat index (0..195) of highest activation patch
            peak_row: spatial row index (0..13)
            peak_col: spatial col index (0..13)
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        self.vit.eval()
        self.sae.eval()

        with torch.no_grad():
            _, _, patch_acts, _ = self.vit(image_tensor)  # [1, 196, 768]
            flat_acts = patch_acts.squeeze(0)  # [196, 768]
            z_topk, _ = self.sae.encode(flat_acts)  # [196, d_sae]

            feat_acts = z_topk[:, feature_id].cpu().numpy()  # [196]
            max_act = float(np.max(feat_acts))
            mean_act = float(np.mean(feat_acts))
            peak_idx = int(np.argmax(feat_acts))
            peak_row = peak_idx // self.grid_size
            peak_col = peak_idx % self.grid_size

            heatmap_14x14 = feat_acts.reshape(self.grid_size, self.grid_size)

        return heatmap_14x14, max_act, mean_act, peak_idx, peak_row, peak_col

    def localize_and_visualize(self,
                               image_input,
                               feature_id: int,
                               candidate_label: str = "Unassigned Feature",
                               colormap_name: str = "jet",
                               alpha: float = 0.55) -> Dict:
        """
        Generates original image, raw heatmap, and composite blended overlay.
        Args:
            image_input: PIL Image or normalized PyTorch Tensor [3, 224, 224]
        Returns:
            Dictionary containing original PIL image, heatmap PIL, overlay PIL, and metadata.
        """
        from torchvision import transforms

        if isinstance(image_input, torch.Tensor):
            tensor_to_use = image_input if image_input.ndim == 4 else image_input.unsqueeze(0)
            orig_pil = denormalize_image(tensor_to_use[0])
        else:
            orig_pil = image_input.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensor_to_use = normalize(transforms.ToTensor()(orig_pil)).unsqueeze(0)

        heatmap_2d, max_act, mean_act, peak_idx, peak_row, peak_col = self.compute_feature_map(tensor_to_use, feature_id)
        overlay_pil = overlay_heatmap_on_image(orig_pil, heatmap_2d, colormap_name=colormap_name, alpha=alpha)

        # Build raw colored heatmap image
        blank = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        raw_heatmap_pil = overlay_heatmap_on_image(blank, heatmap_2d, colormap_name=colormap_name, alpha=1.0)

        active_patch_count = int(np.sum(heatmap_2d > 0))

        # Add crisp peak-patch highlight box if feature is active
        if max_act > 0.0:
            from PIL import ImageDraw
            patch_w = self.image_size // self.grid_size
            patch_h = self.image_size // self.grid_size
            x0 = peak_col * patch_w
            y0 = peak_row * patch_h
            x1 = (peak_col + 1) * patch_w
            y1 = (peak_row + 1) * patch_h

            draw_ov = ImageDraw.Draw(overlay_pil)
            draw_ov.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(0, 0, 0), width=1)
            draw_ov.rectangle([x0, y0, x1, y1], outline=(255, 235, 59), width=2)

            draw_raw = ImageDraw.Draw(raw_heatmap_pil)
            draw_raw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(0, 0, 0), width=1)
            draw_raw.rectangle([x0, y0, x1, y1], outline=(255, 235, 59), width=2)

            spatial_desc = (
                f"Peak Activation: {max_act:.4f} at Patch #{peak_idx} "
                f"(Row {peak_row + 1}/14, Col {peak_col + 1}/14, highlighted in yellow) | "
                f"Active Patches: {active_patch_count}/196 ({active_patch_count/196:.1%})"
            )
        else:
            spatial_desc = (
                f"Feature #{feature_id} is inactive on this image (Peak: 0.0000 across all 196 patches). "
                "Because activation f_i = 0, residual causal intervention produces exact zero delta."
            )

        return {
            "original_image": orig_pil,
            "raw_heatmap": raw_heatmap_pil,
            "overlay_image": overlay_pil,
            "heatmap_2d": heatmap_2d,
            "feature_id": feature_id,
            "candidate_label": candidate_label,
            "max_activation": max_act,
            "mean_activation": mean_act,
            "active_patch_count": active_patch_count,
            "peak_patch_idx": peak_idx,
            "peak_row": peak_row,
            "peak_col": peak_col,
            "has_activation": max_act > 0.0,
            "spatial_description": spatial_desc
        }
