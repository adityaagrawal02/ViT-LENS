import os
import torch
import warnings
from typing import Optional, Dict, Any, List
from PIL import Image

from src.utils import get_device, resolve_sae_checkpoint_path, format_class_label
from src.vit_model import ViTModelWrapper
from src.topk_sae import TopKSparseAutoencoder
from src.intervention import ResidualPreservingIntervention
from src.feature_localization import FeatureLocalizer
from src.evidence_service import EvidenceService

_MODEL_SERVICE_INSTANCE: Optional["ModelService"] = None


class ModelService:
    """
    Centralized, session-cached model service for ViT-LENS.
    Loads frozen timm ViT-B/16 and TopK SAE checkpoint once per application session.
    Provides device-agnostic execution, checkpoint validation, and inference APIs.
    """
    def __init__(self,
                 sae_checkpoint_path: Optional[str] = None,
                 vit_model_name: str = "vit_base_patch16_224.augreg_in1k",
                 target_layer_idx: int = 7,
                 device: Optional[torch.device] = None,
                 vit: Optional[ViTModelWrapper] = None,
                 sae: Optional[TopKSparseAutoencoder] = None):
        self.device = device or get_device("auto")
        self.vit_model_name = vit_model_name
        self.target_layer_idx = target_layer_idx

        # Issue realistic warning if executing on CPU
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if self.device.type == "cpu":
            warnings.warn(
                "[ViT-LENS] Compute device is CPU. Real-time ViT and SAE inference is functional, "
                "but will run slower than CUDA GPU execution in Google Colab. "
                "For faster interactive demonstration, run on a Colab GPU runtime.",
                UserWarning
            )

        # 1. Load Pretrained Frozen ViT
        if vit is not None:
            self.vit = vit
        else:
            self.vit = ViTModelWrapper(
                model_name=vit_model_name,
                pretrained=True,
                target_layer_idx=target_layer_idx,
                device=self.device
            )
        self.eval_transform = self.vit.get_eval_transform()

        # 2. Resolve & Load Final TopK SAE Checkpoint
        self.sae_checkpoint_path = resolve_sae_checkpoint_path(sae_checkpoint_path)
        self.sae: Optional[TopKSparseAutoencoder] = None
        self.sae_load_error: Optional[str] = None
        self.is_sae_loaded = False

        if sae is not None:
            self.sae = sae
            self.is_sae_loaded = True
        else:
            self._load_sae()

        # 3. Initialize downstream services
        if self.is_sae_loaded and self.sae is not None:
            self.intervenor = ResidualPreservingIntervention(self.vit, self.sae)
            self.localizer = FeatureLocalizer(self.sae, self.vit)
        else:
            self.intervenor = None
            self.localizer = None

        self.evidence_service = EvidenceService()

    def _load_sae(self):
        """Attempts to load and validate final SAE checkpoint."""
        if not os.path.exists(self.sae_checkpoint_path):
            self.sae_load_error = (
                f"Checkpoint file not found: '{self.sae_checkpoint_path}'.\n"
                "To mount your validated SAE in Google Colab, run:\n"
                "  from google.colab import drive\n"
                "  drive.mount('/content/drive')\n"
                "Or set the environment variable: VIT_LENS_SAE_CHECKPOINT=<path_to_checkpoint.pt>"
            )
            self.is_sae_loaded = False
            # Initialize an unweighted SAE architecture so structures remain valid
            self.sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32).to(self.device)
            self.sae.eval()
            return

        try:
            self.sae = TopKSparseAutoencoder.load_from_checkpoint(
                checkpoint_path=self.sae_checkpoint_path,
                device=self.device,
                expected_d_in=768,
                expected_d_sae=3072,
                expected_top_k=32
            )
            self.is_sae_loaded = True
            self.sae_load_error = None
        except Exception as e:
            self.sae_load_error = f"Failed to load checkpoint '{self.sae_checkpoint_path}': {str(e)}"
            self.is_sae_loaded = False
            self.sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32).to(self.device)
            self.sae.eval()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Applies exact timm evaluation transform."""
        img_rgb = image.convert("RGB")
        tensor = self.eval_transform(img_rgb)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def predict_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs clean forward pass through frozen ViT.
        Returns top-5 ImageNet predictions with names, probabilities, and logits.
        """
        tensor = self.preprocess_image(image)
        with torch.no_grad():
            logits, probs, patch_acts, cls_acts = self.vit(tensor)

        top5_probs, top5_indices = torch.topk(probs[0], k=5)
        top5_logits = logits[0, top5_indices]

        top5_predictions = []
        for i in range(5):
            idx = int(top5_indices[i].item())
            prob = float(top5_probs[i].item())
            logit = float(top5_logits[i].item())
            label = format_class_label(idx)
            top5_predictions.append({
                "class_idx": idx,
                "label": label,
                "probability": prob,
                "logit": logit
            })

        top1 = top5_predictions[0]

        return {
            "top1": top1,
            "top5": top5_predictions,
            "logits": logits[0].cpu().numpy(),
            "probs": probs[0].cpu().numpy(),
            "patch_activations": patch_acts[0],  # [196, 768]
            "cls_activation": cls_acts[0]       # [1, 768]
        }

    def get_active_features_on_image(self, image: Image.Image, top_n: int = 15) -> List[Dict[str, Any]]:
        """
        Encodes all 196 patch tokens through TopK SAE and returns the most strongly
        activating feature IDs across the image.
        """
        if self.sae is None:
            return []

        tensor = self.preprocess_image(image)
        with torch.no_grad():
            _, _, patch_acts, _ = self.vit(tensor)
            z_topk, _ = self.sae.encode(patch_acts[0])  # [196, d_sae]

            max_acts, _ = torch.max(z_topk, dim=0)  # [d_sae]
            patch_counts = (z_topk > 0).sum(dim=0)  # [d_sae]

            active_mask = max_acts > 0
            active_ids = torch.where(active_mask)[0].cpu().tolist()

        feature_summaries = []
        for fid in active_ids:
            fid = int(fid)
            max_val = float(max_acts[fid].item())
            p_count = int(patch_counts[fid].item())
            feat_info = self.evidence_service.get_feature_info(fid)
            feature_summaries.append({
                "feature_id": fid,
                "peak_activation": max_val,
                "active_patch_count": p_count,
                "label": feat_info["label"],
                "class_name": feat_info["class_name"],
                "notes": feat_info["notes"]
            })

        # Sort descending by peak activation
        feature_summaries.sort(key=lambda x: x["peak_activation"], reverse=True)
        return feature_summaries[:top_n]

    def localize_feature(self, image: Image.Image, feature_id: int) -> Dict[str, Any]:
        """
        Reshapes selected feature's 196 patch activations into 14x14 grid and generates overlays.
        """
        if self.localizer is None or self.sae is None:
            raise RuntimeError("SAE is not loaded. Cannot compute feature localization.")

        tensor = self.preprocess_image(image)
        feat_info = self.evidence_service.get_feature_info(feature_id)
        result = self.localizer.localize_and_visualize(
            tensor,
            feature_id=feature_id,
            candidate_label=feat_info["label"]
        )
        return result

    def run_causal_intervention(self,
                                image: Image.Image,
                                feature_id: int,
                                strength: float,
                                target_class_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes residual-preserving causal intervention using modified forward pass.
        """
        if self.intervenor is None or self.sae is None:
            raise RuntimeError("SAE is not loaded. Cannot execute causal intervention.")

        tensor = self.preprocess_image(image)
        result = self.intervenor.run_intervention(
            tensor,
            feature_id=feature_id,
            strength_multiplier=strength,
            target_class_idx=target_class_idx
        )
        return result


def get_model_service(sae_checkpoint_path: Optional[str] = None) -> ModelService:
    """Returns or creates the session-cached ModelService instance."""
    global _MODEL_SERVICE_INSTANCE
    if _MODEL_SERVICE_INSTANCE is None:
        _MODEL_SERVICE_INSTANCE = ModelService(sae_checkpoint_path=sae_checkpoint_path)
    return _MODEL_SERVICE_INSTANCE
