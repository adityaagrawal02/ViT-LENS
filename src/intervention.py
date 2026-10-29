import torch
from typing import Dict, List, Optional


class ResidualPreservingIntervention:
    """
    Performs residual-preserving causal interventions directly on internal
    Vision Transformer representations using learned TopK SAE feature directions.

    Formula:
        x_modified = x + (f'_i - f_i) * W_dec[i]

    Guarantees:
        - Modifies only the causal subspace defined by SAE feature direction i
        - Preserves the reconstruction residual: (x_modified - x_hat_modified) == (x - x_hat)
        - Strict identity verification: At alpha=1.0, f'_i = f_i -> x_modified == x exactly
        - Operates through PyTorch forward hooks without modifying ViT parameters
    """
    def __init__(self, vit_wrapper, sae):
        self.vit = vit_wrapper
        self.sae = sae

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
            patch_indices: Optional list of specific spatial patch token indices (1..196).
                           If None, applies across all 196 patch tokens.
        Returns:
            modified_seq: Tensor with targeted feature intervention applied.
        """
        # If multiplier is exactly 1.0 (identity), return untouched clone
        if abs(strength_multiplier - 1.0) < 1e-9:
            return activation_seq.clone()

        modified_seq = activation_seq.clone()
        B, seq_len, _ = activation_seq.shape

        # Identify target tokens (excluding CLS token 0 by default)
        if patch_indices is None:
            # All 196 spatial patch tokens (indices 1 to seq_len - 1)
            target_indices = list(range(1, seq_len))
        else:
            # Shift 0-indexed patch to 1-indexed token in ViT sequence
            target_indices = [idx + 1 if idx < seq_len - 1 else idx for idx in patch_indices]

        # Decoder feature direction: W_dec[feature_id], shape [d_model]
        w_dec_i = self.sae.W_dec[feature_id]  # [d_model]

        for b in range(B):
            for t_idx in target_indices:
                x_t = activation_seq[b, t_idx]  # [d_model]

                # Compute feature activation f_i using SAE encoder
                x_centered = x_t - self.sae.b_dec
                z_pre = torch.relu(x_centered @ self.sae.W_enc + self.sae.b_enc)  # [d_sae]

                # Check if feature is in Top-K
                topk_vals, topk_inds = torch.topk(z_pre, k=self.sae.top_k)
                if feature_id in topk_inds:
                    f_i = z_pre[feature_id]
                else:
                    f_i = torch.tensor(0.0, device=x_t.device)

                # Desired edited activation: f'_i = alpha * f_i
                f_prime_i = strength_multiplier * f_i
                delta_f = f_prime_i - f_i  # (alpha - 1.0) * f_i

                # Residual-preserving update: x_modified = x + delta_f * W_dec[i]
                delta_x = delta_f * w_dec_i
                modified_seq[b, t_idx] = x_t + delta_x

        return modified_seq

    def run_intervention(self,
                         image_tensor: torch.Tensor,
                         feature_id: int,
                         strength_multiplier: float = 0.0,
                         patch_indices: Optional[List[int]] = None) -> Dict:
        """
        Executes causal intervention experiment on an image.
        Runs:
        1. Clean forward pass (original logits, probabilities, class)
        2. Hooked forward pass with modified activations
        3. Quantifies logit changes, probability changes, and prediction flips.
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.vit.device)

        # 1. Baseline pristine forward pass
        self.vit.clear_causal_intervention()
        orig_logits, orig_probs, _, _ = self.vit(image_tensor)
        orig_pred_class = torch.argmax(orig_probs, dim=-1).item()
        orig_pred_prob = orig_probs[0, orig_pred_class].item()
        orig_target_logit = orig_logits[0, orig_pred_class].item()

        # 2. Causal intervention forward pass via hook
        def hook_fn(raw_act):
            return self.apply_feature_edit(
                raw_act,
                feature_id=feature_id,
                strength_multiplier=strength_multiplier,
                patch_indices=patch_indices
            )

        self.vit.set_causal_intervention(hook_fn)
        mod_logits, mod_probs, _, _ = self.vit(image_tensor)
        self.vit.clear_causal_intervention()

        mod_pred_class = torch.argmax(mod_probs, dim=-1).item()
        mod_pred_prob = mod_probs[0, mod_pred_class].item()
        mod_target_prob = mod_probs[0, orig_pred_class].item()
        mod_target_logit = mod_logits[0, orig_pred_class].item()

        logit_diff = mod_target_logit - orig_target_logit
        prob_diff = mod_target_prob - orig_pred_prob
        prediction_flipped = (orig_pred_class != mod_pred_class)

        return {
            "feature_id": feature_id,
            "strength": strength_multiplier,
            "mode": "patch_specific" if patch_indices else "all_patches",
            "orig_pred_class": orig_pred_class,
            "orig_pred_prob": float(orig_pred_prob),
            "orig_target_logit": float(orig_target_logit),
            "mod_pred_class": mod_pred_class,
            "mod_pred_prob": float(mod_pred_prob),
            "mod_target_prob": float(mod_target_prob),
            "mod_target_logit": float(mod_target_logit),
            "logit_diff": float(logit_diff),
            "prob_diff": float(prob_diff),
            "prediction_flipped": bool(prediction_flipped),
            "orig_top5": torch.topk(orig_probs[0], k=5).indices.cpu().tolist(),
            "mod_top5": torch.topk(mod_probs[0], k=5).indices.cpu().tolist()
        }
