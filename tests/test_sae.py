import torch

from src.topk_sae import TopKSparseAutoencoder


def test_sae_initialization_and_dimensions():
    d_in = 768
    d_sae = 3072
    top_k = 32

    sae = TopKSparseAutoencoder(d_in=d_in, d_sae=d_sae, top_k=top_k)

    assert sae.W_enc.shape == (d_in, d_sae)
    assert sae.b_enc.shape == (d_sae,)
    assert sae.W_dec.shape == (d_sae, d_in)
    assert sae.b_dec.shape == (d_in,)


def test_decoder_unit_norm_constraint():
    d_in = 768
    d_sae = 3072
    top_k = 32

    sae = TopKSparseAutoencoder(d_in=d_in, d_sae=d_sae, top_k=top_k)
    sae.normalize_decoder()

    norms = torch.norm(sae.W_dec, p=2, dim=1)
    # Check that all feature vectors in W_dec have L2 norm close to 1.0
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Decoder weights must have unit norm"


def test_topk_sparsity_guarantee():
    d_in = 768
    d_sae = 3072
    top_k = 32
    batch_size = 8

    sae = TopKSparseAutoencoder(d_in=d_in, d_sae=d_sae, top_k=top_k)
    x = torch.randn(batch_size, d_in)

    z_topk, topk_indices = sae.encode(x)

    assert z_topk.shape == (batch_size, d_sae)
    assert topk_indices.shape == (batch_size, top_k)

    # Count non-zero elements per row
    non_zeros_per_sample = (z_topk > 0).sum(dim=-1)
    # Exactly top_k or fewer if fewer than k elements are positive
    assert (non_zeros_per_sample <= top_k).all()
    assert (non_zeros_per_sample == top_k).any()


def test_sae_forward_and_reconstruction_shapes():
    d_in = 768
    d_sae = 3072
    top_k = 32
    batch_size = 4

    sae = TopKSparseAutoencoder(d_in=d_in, d_sae=d_sae, top_k=top_k)
    x = torch.randn(batch_size, d_in)

    x_hat, z_topk, metrics = sae(x)

    assert x_hat.shape == (batch_size, d_in)
    assert "mse_loss" in metrics
    assert "explained_variance" in metrics
    assert "l0_sparsity" in metrics
    assert metrics["mse_loss"].item() >= 0.0


def test_dead_feature_tracking_and_resampling():
    d_in = 768
    d_sae = 128
    top_k = 8
    sae = TopKSparseAutoencoder(d_in=d_in, d_sae=d_sae, top_k=top_k)
    sae.train()

    # Artificially set steps for some features
    sae.steps_since_activation[0:10] = 600
    dead_mask = sae.get_dead_features(step_threshold=500)
    assert dead_mask[0].item() is True
    assert dead_mask[10].item() is False

    # Test resampling
    x = torch.randn(16, d_in)
    sae.resample_dead_features(dead_mask, x)
    assert sae.steps_since_activation[0].item() == 0


if __name__ == "__main__":
    test_sae_initialization_and_dimensions()
    test_decoder_unit_norm_constraint()
    test_topk_sparsity_guarantee()
    test_sae_forward_and_reconstruction_shapes()
    test_dead_feature_tracking_and_resampling()
    print("All SAE unit tests passed successfully!")
