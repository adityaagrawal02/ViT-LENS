import torch

from src.topk_sae import TopKSparseAutoencoder
from src.vit_model import ViTModelWrapper
from src.intervention import ResidualPreservingIntervention


def test_residual_preserving_identity():
    """
    Critical verification: When strength multiplier is 1.0 (identity),
    the modified activation MUST be mathematically identical to the original activation.
    """
    d_in = 768
    d_sae = 3072
    top_k = 32

    sae = TopKSparseAutoencoder(d_in=d_in, d_sae=d_sae, top_k=top_k)
    vit = ViTModelWrapper(pretrained=False, use_torchvision_fallback=True)
    intervenor = ResidualPreservingIntervention(vit, sae)

    # Sequence of 1 CLS + 196 patches
    dummy_seq = torch.randn(1, 197, d_in)
    feature_id = 42

    mod_seq = intervenor.apply_feature_edit(
        activation_seq=dummy_seq,
        feature_id=feature_id,
        strength_multiplier=1.0,
        patch_indices=None
    )

    diff = torch.abs(mod_seq - dummy_seq).max().item()
    assert diff < 1e-7, f"Identity intervention must yield 0 difference, got max diff: {diff}"


def test_residual_preservation_math():
    """
    Mathematical check:
    x_mod = x + (f'_i - f_i) * W_dec[i]
    Residual r = x - x_hat
    Residual r_mod = x_mod - x_hat_mod
    r_mod MUST equal r.
    """
    d_in = 768
    d_sae = 128
    top_k = 8
    sae = TopKSparseAutoencoder(d_in=d_in, d_sae=d_sae, top_k=top_k)

    x = torch.randn(1, d_in)
    z_topk, _ = sae.encode(x)
    x_hat = sae.decode(z_topk)
    r = x - x_hat

    # Choose an active feature
    active_indices = torch.where(z_topk[0] > 0)[0]
    if len(active_indices) == 0:
        feat_id = 0
    else:
        feat_id = active_indices[0].item()

    f_i = z_topk[0, feat_id]
    alpha = 2.0  # Amplification
    f_prime_i = alpha * f_i
    delta_f = f_prime_i - f_i

    # Apply residual-preserving modification
    x_mod = x + delta_f * sae.W_dec[feat_id]

    # Reconstruct modified activation
    z_mod = z_topk.clone()
    z_mod[0, feat_id] = f_prime_i
    x_hat_mod = sae.decode(z_mod)
    r_mod = x_mod - x_hat_mod

    # Residuals must match
    assert torch.allclose(r, r_mod, atol=1e-5), "Residual must be preserved under feature intervention"


def test_end_to_end_intervention_hook():
    """Test full forward pass causal intervention through hook mechanism."""
    d_in = 768
    d_sae = 128
    top_k = 8

    sae = TopKSparseAutoencoder(d_in=d_in, d_sae=d_sae, top_k=top_k)
    vit = ViTModelWrapper(pretrained=False, use_torchvision_fallback=True)
    intervenor = ResidualPreservingIntervention(vit, sae)

    dummy_image = torch.randn(1, 3, 224, 224)

    # 1. Identity intervention (1.0x) should produce exact same logits
    res_identity = intervenor.run_intervention(dummy_image, feature_id=5, strength_multiplier=1.0)
    assert abs(res_identity["logit_diff"]) < 1e-4, "Identity intervention must produce zero logit shift"
    assert abs(res_identity["prob_diff"]) < 1e-4, "Identity intervention must produce zero prob shift"

    # 2. Suppression intervention (0.0x)
    res_suppress = intervenor.run_intervention(dummy_image, feature_id=5, strength_multiplier=0.0)
    assert "orig_pred_class" in res_suppress
    assert "mod_pred_class" in res_suppress
    assert "logit_diff" in res_suppress


if __name__ == "__main__":
    test_residual_preserving_identity()
    test_residual_preservation_math()
    test_end_to_end_intervention_hook()
    print("All intervention tests passed successfully!")
