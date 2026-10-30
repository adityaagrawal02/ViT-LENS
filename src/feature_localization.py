import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict

from src.utils import (
    denormalize_image,
    create_discrete_patch_grid,
    create_transparent_heatmap_overlay,
    render_discrete_grid_with_colorbar,
    render_overlay_with_colorbar,
)


class FeatureLocalizer:
    """
    Computes spatial activation heatmaps for individual SAE features on input images,
    mapping 1D patch-token activations to 2D spatial overlays (14x14 -> 224x224).

    Important Interpretability Notice:
    - SAE feature heatmaps display where in the image plane a learned dictionary atom activated.
    - These are distinct from Transformer attention maps (which measure inter-token routing weights).
    - Peak patch activation indicates feature presence, but does NOT prove the exact causal pixels.
    - Learned SAE features often respond to textures, background context, or edge boundaries;
      an activation on the background reflects the model's true representation, not a segmentation flaw.
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
            if max_act > 0.0:
                peak_idx = int(np.argmax(feat_acts))
                peak_row = peak_idx // self.grid_size
                peak_col = peak_idx % self.grid_size
            else:
                peak_idx = None
                peak_row = None
                peak_col = None

            heatmap_14x14 = feat_acts.reshape(self.grid_size, self.grid_size)

        return heatmap_14x14, max_act, mean_act, peak_idx, peak_row, peak_col

    def localize_and_visualize(self,
                               image_input,
                               feature_id: int,
                               candidate_label: str = "Unassigned Feature",
                               colormap_name: str = "inferno",
                               alpha: float = 0.55) -> Dict:
        """
        Generates original image, genuinely discrete 14x14 patch grid, and transparent overlay.
        Args:
            image_input: PIL Image or normalized PyTorch Tensor [3, 224, 224]
            feature_id: Index of the SAE feature
            candidate_label: Label or description of the feature
            colormap_name: Colormap name (default: "inferno")
            alpha: Unused legacy parameter; transparent overlays use per-patch activation weights.
        Returns:
            Dictionary containing original PIL image, discrete grid PIL, overlay PIL, and metadata.
        """
        if isinstance(image_input, torch.Tensor):
            tensor_to_use = image_input if image_input.ndim == 4 else image_input.unsqueeze(0)
            orig_pil = denormalize_image(tensor_to_use[0])
        else:
            # Use ViT evaluation transform for exact preprocessing alignment
            eval_tf = self.vit.get_eval_transform()
            tensor_to_use = eval_tf(image_input.convert("RGB")).unsqueeze(0)
            orig_pil = denormalize_image(tensor_to_use[0])

        heatmap_2d, max_act, mean_act, peak_idx, peak_row, peak_col = self.compute_feature_map(tensor_to_use, feature_id)
        active_patch_count = int(np.sum(heatmap_2d > 0))

        # 1. Genuinely discrete 14x14 patch grid (exact 224x224, nearest-neighbor tiles, gridlines, peak box)
        discrete_grid_pil = create_discrete_patch_grid(
            heatmap_2d=heatmap_2d,
            colormap_name=colormap_name,
            peak_row=peak_row if max_act > 0 else None,
            peak_col=peak_col if max_act > 0 else None,
            image_size=self.image_size
        )

        # 2. Transparent overlay on 224x224 input image:
        # Inactive patches have alpha=0 (100% transparent), original image remains completely visible.
        # Active patches glow with color proportional to activation.
        transparent_overlay_pil = create_transparent_heatmap_overlay(
            image=orig_pil,
            heatmap_2d=heatmap_2d,
            colormap_name=colormap_name,
            smooth=False,  # Crisp patch boundaries aligned with ViT tokens
            peak_row=peak_row if max_act > 0 else None,
            peak_col=peak_col if max_act > 0 else None
        )

        # 3. Figures with matched numeric colorbars for detailed quantitative analysis
        discrete_with_cbar_pil = render_discrete_grid_with_colorbar(
            heatmap_2d=heatmap_2d,
            peak_row=peak_row if max_act > 0 else None,
            peak_col=peak_col if max_act > 0 else None,
            colormap_name=colormap_name,
            title=f"Discrete 14×14 Patch Grid (Feature #{feature_id})"
        )
        overlay_with_cbar_pil = render_overlay_with_colorbar(
            overlay_pil=transparent_overlay_pil,
            peak_val=max_act,
            colormap_name=colormap_name,
            title=f"Transparent Overlay on Input (Feature #{feature_id})"
        )

        # 4. Smooth bicubic overlay (for comparison)
        smooth_overlay_pil = create_transparent_heatmap_overlay(
            image=orig_pil,
            heatmap_2d=heatmap_2d,
            colormap_name=colormap_name,
            smooth=True,
            peak_row=peak_row if max_act > 0 else None,
            peak_col=peak_col if max_act > 0 else None
        )

        if max_act > 0.0 and peak_row is not None and peak_col is not None and peak_idx is not None:
            spatial_desc = (
                f"Peak Activation: **{max_act:.4f}** at Patch #{peak_idx} "
                f"(Row {peak_row + 1}/14, Col {peak_col + 1}/14, yellow/black box) | "
                f"Active Patches: **{active_patch_count}/196** ({active_patch_count/196:.1%})\n\n"
                f"*Interpretability Note*: Learned SAE features capture dictionary representations that may "
                f"activate on contextual regions (e.g. background, ground, or edges) rather than full object silhouettes. "
                f"This reflects the model's true internal representations."
            )
        else:
            spatial_desc = (
                f"Feature inactive on this image. "
                f"Peak: **0.0000** across all 196 patches | Active Patches: **0/196** (0.0%). "
                f"No patch location is activated. No peak bounding box is displayed. "
                f"Because activation f_i = 0 across all patches, residual causal intervention produces an exact zero-delta no-op."
            )

        return {
            "original_image": orig_pil,
            "raw_heatmap": discrete_grid_pil,              # Genuinely discrete 14x14 grid (224, 224)
            "discrete_grid": discrete_grid_pil,            # Explicit alias (224, 224)
            "overlay_image": transparent_overlay_pil,      # Transparent overlay on input (224, 224)
            "overlay_plain": transparent_overlay_pil,      # Alias
            "discrete_with_cbar": discrete_with_cbar_pil,  # Standalone figure with colorbar
            "overlay_with_cbar": overlay_with_cbar_pil,    # Standalone figure with colorbar
            "smooth_overlay": smooth_overlay_pil,          # Smooth bicubic overlay (224, 224)
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
