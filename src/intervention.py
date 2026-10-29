import torch
from typing import Dict, List, Optional


class ResidualPreservingIntervention:
    """
    Performs residual-preserving causal interventions directly on internal
    Vision Transformer representations using learned TopK SAE feature directions.

    Essential Validated Formula:
        x_edited = x_original + (alpha - 1) * f_i * d_i

    Where decoder direction d_i is calculated exactly as:
        d_i = sae.decode(one_hot_i) - sae.decode(zero_latent)

    Guarantees:
        - Modifies only the 196 patch tokens (tokens 1..196); strictly preserves CLS token (token 0)
        - Preserves the original residual / reconstruction error
        - Strict identity verification: at alpha=1.0, (alpha - 1) == 0 -> x_edited == x_original exactly
        - Features with zero activation (f_i == 0) produce zero modification
        - Uses safe hook registration and removal with try/finally to prevent hook accumulation
    """
    def __init__(self, vit_wrapper, sae):
        self.vit = vit_wrapper
        self.sae = sae

    def compute_decoder_direction(self, feature_id: int) -> torch.Tensor:
        """
        Calculates decoder direction exactly as:
            d_i = sae.decode(one_hot_i) - sae.decode(zero_latent)
        This formulation explicitly accounts for any decoder bias.
        """
        device = self.sae.W_dec.device
        dtype = self.sae.W_dec.dtype
        with torch.no_grad():
            one_hot = torch.zeros(1, self.sae.d_sae, device=device, dtype=dtype)
            one_hot[0, feature_id] = 1.0
            zero_latent = torch.zeros(1, self.sae.d_sae, device=device, dtype=dtype)
            d_i = self.sae.decode(one_hot) - self.sae.decode(zero_latent)
            return d_i.squeeze(0)  # [d_model]

    def apply_feature_edit(self,
                           activation_seq: torch.Tensor,
                           feature_id: int,
                           strength_multiplier: float,
                           patch_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Applies residual-preserving feature modification to sequence of activations.

        Args:
            activation_seq: Tensor of shape [B, N_tokens, d_model]
                            (Token 0 is CLS, tokens 1..196 are patches)
            feature_id: Index of SAE feature to intervene on (0 <= feature_id < d_sae)
            strength_multiplier: Multiplier alpha (0.0 = suppression, 1.0 = identity, 2.0 = amplification)
            patch_indices: Optional list of specific spatial patch token indices (0..195).
                           If None, applies across all 196 patch tokens.
        Returns:
            modified_seq: Tensor with targeted feature intervention applied.
        """
        # Strict numerical identity optimization
        if abs(strength_multiplier - 1.0) < 1e-9:
            return activation_seq.clone()

        modified_seq = activation_seq.clone()
        B, seq_len, d_model = activation_seq.shape
        num_patches = seq_len - 1

        # Decoder direction d_i
        d_i = self.compute_decoder_direction(feature_id).to(activation_seq.device)  # [d_model]

        # Patch activations: exclude CLS token 0
        patch_acts = activation_seq[:, 1:, :]  # [B, num_patches, d_model]
        flat_patch_acts = patch_acts.reshape(-1, d_model)

        # Compute feature activation f_i using TopK SAE
        with torch.no_grad():
            z_topk, _ = self.sae.encode(flat_patch_acts)
            f_i = z_topk[:, feature_id].reshape(B, num_patches)  # [B, num_patches]

        # Check if feature is active anywhere on target patches
        if patch_indices is not None:
            mask = torch.zeros(num_patches, device=activation_seq.device, dtype=torch.bool)
            mask[patch_indices] = True
            f_i = f_i * mask.unsqueeze(0)

        # If feature is completely inactive, delta is strictly zero
        if torch.all(f_i == 0.0):
            return modified_seq

        # Validated formula: delta = (alpha - 1.0) * f_i * d_i
        alpha_factor = strength_multiplier - 1.0
        delta = alpha_factor * f_i.unsqueeze(-1) * d_i.unsqueeze(0).unsqueeze(0)  # [B, num_patches, d_model]

        # Modify only the patch tokens; CLS token (token 0) remains unchanged
        modified_seq[:, 1:, :] = patch_acts + delta

        return modified_seq

    def run_intervention(self,
                         image_tensor: torch.Tensor,
                         feature_id: int,
                         strength_multiplier: float = 0.0,
                         patch_indices: Optional[List[int]] = None,
                         target_class_idx: Optional[int] = None) -> Dict:
        """
        Executes causal intervention experiment on an image.
        Runs:
        1. Clean forward pass (original logits, probabilities, class)
        2. Hooked forward pass with modified Block 7 activations
        3. Quantifies logit changes, probability changes, and prediction flips.
        Safely registers and removes hook using try/finally to prevent hook accumulation.
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.vit.device)

        # 1. Baseline pristine forward pass
        self.vit.clear_causal_intervention()
        orig_logits, orig_probs, patch_acts, _ = self.vit(image_tensor)

        # Check peak activation of feature on this image
        with torch.no_grad():
            z_topk, _ = self.sae.encode(patch_acts[0])  # [196, d_sae]
            feat_acts = z_topk[:, feature_id]
            max_feat_act = float(feat_acts.max().item())
            active_patch_count = int((feat_acts > 0).sum().item())

        orig_pred_class = torch.argmax(orig_probs, dim=-1).item()
        tracked_class = orig_pred_class if target_class_idx is None else target_class_idx

        orig_pred_prob = orig_probs[0, orig_pred_class].item()
        orig_target_prob = orig_probs[0, tracked_class].item()
        orig_target_logit = orig_logits[0, tracked_class].item()

        # 2. Causal intervention forward pass via hook with guaranteed cleanup
        def hook_fn(raw_act):
            return self.apply_feature_edit(
                raw_act,
                feature_id=feature_id,
                strength_multiplier=strength_multiplier,
                patch_indices=patch_indices
            )

        try:
            self.vit.set_causal_intervention(hook_fn)
            mod_logits, mod_probs, _, _ = self.vit(image_tensor)
        finally:
            self.vit.clear_causal_intervention()

        mod_pred_class = torch.argmax(mod_probs, dim=-1).item()
        mod_pred_prob = mod_probs[0, mod_pred_class].item()
        mod_target_prob = mod_probs[0, tracked_class].item()
        mod_target_logit = mod_logits[0, tracked_class].item()

        logit_diff = mod_target_logit - orig_target_logit
        prob_diff = mod_target_prob - orig_target_prob
        prediction_flipped = (orig_pred_class != mod_pred_class)

        return {
            "feature_id": feature_id,
            "strength": float(strength_multiplier),
            "max_feature_activation": max_feat_act,
            "active_patch_count": active_patch_count,
            "is_feature_active": max_feat_act > 0.0,
            "mode": "patch_specific" if patch_indices else "all_patches",
            "tracked_class": tracked_class,
            "orig_pred_class": orig_pred_class,
            "orig_pred_prob": float(orig_pred_prob),
            "orig_target_prob": float(orig_target_prob),
            "orig_target_logit": float(orig_target_logit),
            "mod_pred_class": mod_pred_class,
            "mod_pred_prob": float(mod_pred_prob),
            "mod_target_prob": float(mod_target_prob),
            "mod_target_logit": float(mod_target_logit),
            "logit_diff": float(logit_diff),
            "prob_diff": float(prob_diff),
            "prediction_flipped": bool(prediction_flipped),
            "orig_top5": torch.topk(orig_probs[0], k=5).indices.cpu().tolist(),
            "mod_top5": torch.topk(mod_probs[0], k=5).indices.cpu().tolist(),
            "orig_top5_probs": torch.topk(orig_probs[0], k=5).values.cpu().tolist(),
            "mod_top5_probs": torch.topk(mod_probs[0], k=5).values.cpu().tolist()
        }
