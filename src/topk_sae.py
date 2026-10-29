import os
import torch
import torch.nn as nn
from typing import Tuple, Dict


class TopKSparseAutoencoder(nn.Module):
    """
    TopK Sparse Autoencoder (SAE) for learning interpretable visual dictionaries.

    Architecture:
        x in R^{d_in}
        x_centered = x - b_dec
        z_pre = ReLU(x_centered @ W_enc + b_enc)
        z_topk = TopK(z_pre, k)
        x_hat = z_topk @ W_dec + b_dec

    Properties:
        - Exact L0 sparsity guaranteed by hard TopK selection
        - Residual-preserving reconstruction: x_hat = x + residual
        - Decoder columns/features normalized to unit L2 norm
    """
    def __init__(self,
                 d_in: int = 768,
                 d_sae: int = 3072,
                 top_k: int = 32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.top_k = top_k

        # Encoder: maps d_in -> d_sae
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: maps d_sae -> d_in
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Dead feature tracking buffer (tracks number of steps since last non-zero activation)
        self.register_buffer("steps_since_activation", torch.zeros(d_sae, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize encoder and decoder weights with Kaiming uniform and decoder normalization."""
        nn.init.kaiming_uniform_(self.W_enc, a=0.2)
        nn.init.kaiming_uniform_(self.W_dec, a=0.2)
        # Transpose encoder weights initially as approximate inverse of decoder
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.t().clone()
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        """Constrain decoder feature vectors (rows of W_dec) to unit L2 norm."""
        norms = torch.norm(self.W_dec.data, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes centered inputs into TopK sparse activations.
        Returns:
            z_topk: [B, d_sae] sparse activations (at most top_k non-zeros per sample)
            topk_indices: [B, top_k] indices of active features
        """
        # Center input using pre-encoder decoder bias
        x_centered = x - self.b_dec
        z_pre = torch.relu(x_centered @ self.W_enc + self.b_enc)

        # Apply hard TopK selection
        topk_vals, topk_indices = torch.topk(z_pre, k=self.top_k, dim=-1)

        # Construct sparse tensor
        z_topk = torch.zeros_like(z_pre)
        z_topk.scatter_(-1, topk_indices, topk_vals)

        return z_topk, topk_indices

    def decode(self, z_topk: torch.Tensor) -> torch.Tensor:
        """Decodes sparse feature vector back to ViT activation space."""
        return z_topk @ self.W_dec + self.b_dec

    def get_decoder_direction(self, feature_id: int) -> torch.Tensor:
        """
        Calculates the decoder direction for feature i exactly as:
            d_i = sae.decode(one_hot_i) - sae.decode(zero_latent)
        This formulation explicitly handles any decoder bias.
        """
        device = self.W_dec.device
        dtype = self.W_dec.dtype
        with torch.no_grad():
            one_hot = torch.zeros(1, self.d_sae, device=device, dtype=dtype)
            one_hot[0, feature_id] = 1.0
            zero_latent = torch.zeros(1, self.d_sae, device=device, dtype=dtype)
            d_i = self.decode(one_hot) - self.decode(zero_latent)
            return d_i.squeeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through TopK SAE.
        Returns:
            x_hat: Reconstructed activation tensor [B, d_in]
            z_topk: Sparse latent feature tensor [B, d_sae]
            metrics: Dictionary of reconstruction metrics
        """
        z_topk, topk_indices = self.encode(x)
        x_hat = self.decode(z_topk)

        # Update dead feature tracker in training mode
        if self.training:
            with torch.no_grad():
                active_mask = (z_topk > 0).any(dim=0)  # [d_sae]
                self.steps_since_activation[active_mask] = 0
                self.steps_since_activation[~active_mask] += 1

        # Calculate reconstruction metrics
        mse_loss = nn.functional.mse_loss(x_hat, x)

        # Explained variance: 1 - Var(x - x_hat) / Var(x)
        var_x = torch.var(x, dim=0).sum().clamp(min=1e-8)
        var_err = torch.var(x - x_hat, dim=0).sum()
        explained_variance = 1.0 - (var_err / var_x)

        # L0 sparsity (average number of non-zero features per sample)
        l0_sparsity = (z_topk > 0).float().sum(dim=-1).mean()

        metrics = {
            "mse_loss": mse_loss,
            "explained_variance": explained_variance.clamp(min=0.0, max=1.0),
            "l0_sparsity": l0_sparsity
        }

        return x_hat, z_topk, metrics

    def get_dead_features(self, step_threshold: int = 500) -> torch.Tensor:
        """Returns boolean tensor of features that haven't fired for step_threshold steps."""
        return self.steps_since_activation >= step_threshold

    @torch.no_grad()
    def resample_dead_features(self, dead_mask: torch.Tensor, activations: torch.Tensor):
        """
        Resamples dead features from activations with highest current reconstruction error.
        """
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return

        x_hat, _, _ = self(activations)
        errors = torch.norm(activations - x_hat, p=2, dim=-1)
        _, top_err_indices = torch.topk(errors, k=min(num_dead, activations.shape[0]))

        dead_indices = torch.where(dead_mask)[0][:len(top_err_indices)]
        for i, dead_idx in enumerate(dead_indices):
            err_act = activations[top_err_indices[i]] - self.b_dec
            norm_err = err_act / torch.norm(err_act, p=2).clamp(min=1e-8)
            self.W_dec.data[dead_idx] = norm_err
            self.W_enc.data[:, dead_idx] = norm_err
            self.b_enc.data[dead_idx] = 0.0
            self.steps_since_activation[dead_idx] = 0

    @classmethod
    def load_from_checkpoint(cls,
                             checkpoint_path: str,
                             device: torch.device = torch.device("cpu"),
                             expected_d_in: int = 768,
                             expected_d_sae: int = 3072,
                             expected_top_k: int = 32) -> "TopKSparseAutoencoder":
        """
        Loads and validates final SAE weights from checkpoint dictionary.
        Supports 'model_state_dict', 'sae_state_dict', or root state_dict formats.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAE checkpoint not found at: {checkpoint_path}\n"
                "Please verify Google Drive is mounted or set VIT_LENS_SAE_CHECKPOINT."
            )

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Verify config if present
        cfg = ckpt.get("config", {})
        d_in = cfg.get("d_in", expected_d_in)
        d_sae = cfg.get("d_sae", expected_d_sae)
        top_k = cfg.get("top_k", expected_top_k)

        if d_in != expected_d_in or d_sae != expected_d_sae or top_k != expected_top_k:
            raise ValueError(
                f"Checkpoint config mismatch: got (d_in={d_in}, d_sae={d_sae}, top_k={top_k}), "
                f"expected (d_in={expected_d_in}, d_sae={expected_d_sae}, top_k={expected_top_k})"
            )

        sae = cls(d_in=d_in, d_sae=d_sae, top_k=top_k)

        # Extract state dict
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "sae_state_dict" in ckpt:
            state_dict = ckpt["sae_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Filter out any unexpected prefix
        clean_state = {}
        for k, v in state_dict.items():
            clean_k = k.replace("sae.", "").replace("model.", "")
            clean_state[clean_k] = v

        sae.load_state_dict(clean_state, strict=False)
        sae.to(device)
        sae.eval()
        sae.normalize_decoder()
        return sae
