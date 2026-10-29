import pytest
import torch

from src.topk_sae import TopKSparseAutoencoder
from src.vit_model import ViTModelWrapper
from src.intervention import ResidualPreservingIntervention
from src.feature_localization import FeatureLocalizer
from src.utils import get_imagenet_labels, format_class_label
from src.model_service import ModelService


@pytest.fixture
def mock_sae():
    """Create a standard test SAE matching our fixed architecture."""
    sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32)
    sae.eval()
    return sae


@pytest.fixture
def mock_vit():
    """Create a fast, non-pretrained ViT wrapper for integration testing."""
    vit = ViTModelWrapper(pretrained=False, use_torchvision_fallback=True)
    vit.model.eval()
    return vit


def test_block7_activation_shape(mock_vit):
    """Test 1: Output of Block 7 must have shape [B, 197, 768]."""
    dummy_input = torch.randn(2, 3, 224, 224)
    logits, probs, patch_acts, cls_act = mock_vit(dummy_input)

    raw_act = mock_vit.hook_manager.captured_activation
    assert raw_act is not None, "Block 7 output was not captured by ViTHookManager"
    assert raw_act.shape == (2, 197, 768), f"Expected raw shape [2, 197, 768], got {raw_act.shape}"
    assert patch_acts.shape == (2, 196, 768), f"Expected patch shape [2, 196, 768], got {patch_acts.shape}"
    assert cls_act.shape == (2, 1, 768), f"Expected cls shape [2, 1, 768], got {cls_act.shape}"


def test_cls_token_preservation(mock_vit, mock_sae):
    """Test 2: Token 0 (CLS) must remain strictly unmodified before and after intervention."""
    intervenor = ResidualPreservingIntervention(mock_vit, mock_sae)
    dummy_seq = torch.randn(1, 197, 768)
    original_cls = dummy_seq[:, 0:1, :].clone()

    # Apply aggressive amplification
    mod_seq = intervenor.apply_feature_edit(
        activation_seq=dummy_seq,
        feature_id=10,
        strength_multiplier=3.0
    )

    modified_cls = mod_seq[:, 0:1, :]
    assert torch.all(original_cls == modified_cls), "CLS token was modified during intervention!"


def test_spatial_grid_14x14_alignment(mock_vit, mock_sae):
    """Test 3: Feature localizer must correctly map 196 patch tokens to a 14x14 grid."""
    localizer = FeatureLocalizer(mock_sae, mock_vit)
    dummy_img = torch.randn(1, 3, 224, 224)

    heatmap, max_act, mean_act, peak_idx, peak_row, peak_col = localizer.compute_feature_map(
        dummy_img, feature_id=0
    )
    assert heatmap.shape == (14, 14), f"Expected 14x14 heatmap, got {heatmap.shape}"
    assert peak_row == peak_idx // 14, f"Row {peak_row} does not match peak_idx // 14 ({peak_idx // 14})"
    assert peak_col == peak_idx % 14, f"Col {peak_col} does not match peak_idx % 14 ({peak_idx % 14})"


def test_identity_intervention_exact_zero_diff(mock_vit, mock_sae):
    """Test 4: Identity intervention (alpha=1.0) must produce delta < 1e-7."""
    intervenor = ResidualPreservingIntervention(mock_vit, mock_sae)
    dummy_seq = torch.randn(1, 197, 768)

    mod_seq = intervenor.apply_feature_edit(
        activation_seq=dummy_seq,
        feature_id=123,
        strength_multiplier=1.0
    )

    diff = torch.abs(mod_seq - dummy_seq).max().item()
    assert diff < 1e-7, f"Identity intervention must yield < 1e-7 diff, got {diff}"


def test_hook_cleanup_on_exception(mock_vit, mock_sae):
    """Test 5: Hook manager must safely remove intervention hook even if an exception occurs."""
    intervenor = ResidualPreservingIntervention(mock_vit, mock_sae)
    dummy_img = torch.randn(1, 3, 224, 224)

    # Initial state
    assert mock_vit.hook_manager.intervention_fn is None

    # Simulate an error during hooked forward pass
    class CustomForwardError(Exception):
        pass

    original_forward = mock_vit.model.forward

    def faulty_forward(x):
        raise CustomForwardError("Simulated forward pass failure")

    mock_vit.model.forward = faulty_forward
    try:
        intervenor.run_intervention(dummy_img, feature_id=5, strength_multiplier=0.0)
    except CustomForwardError:
        pass
    finally:
        mock_vit.model.forward = original_forward

    # Intervention hook must be safely cleared
    assert mock_vit.hook_manager.intervention_fn is None, "Intervention hook was not cleaned up after error!"


def test_zero_activation_feature_causes_zero_delta(mock_vit, mock_sae):
    """Test 6: When feature activation f_i is 0 everywhere, intervention delta is zero."""
    intervenor = ResidualPreservingIntervention(mock_vit, mock_sae)

    # With zero activations, f_i will be zero everywhere
    dummy_seq = torch.zeros(1, 197, 768)

    mod_seq = intervenor.apply_feature_edit(
        activation_seq=dummy_seq,
        feature_id=999,
        strength_multiplier=0.0
    )

    diff = torch.abs(mod_seq - dummy_seq).max().item()
    assert diff < 1e-7, f"Inactive feature must yield 0 delta under suppression, got {diff}"


def test_checkpoint_validation_and_load(tmp_path):
    """Test 7: Checkpoint saving, resolving, and load_from_checkpoint validation."""
    sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32)
    ckpt_path = tmp_path / "test_sae_ckpt.pt"

    ckpt_payload = {
        "model_state_dict": sae.state_dict(),
        "config": {
            "d_in": 768,
            "d_sae": 3072,
            "top_k": 32
        }
    }
    torch.save(ckpt_payload, str(ckpt_path))

    # Test loading
    loaded_sae = TopKSparseAutoencoder.load_from_checkpoint(str(ckpt_path), device="cpu")
    assert loaded_sae.d_in == 768
    assert loaded_sae.d_sae == 3072
    assert loaded_sae.top_k == 32

    # Test config mismatch handling
    bad_ckpt_path = tmp_path / "bad_sae_ckpt.pt"
    bad_payload = {
        "model_state_dict": sae.state_dict(),
        "config": {
            "d_in": 512,  # wrong
            "d_sae": 1024,
            "top_k": 16
        }
    }
    torch.save(bad_payload, str(bad_ckpt_path))
    with pytest.raises(ValueError):
        TopKSparseAutoencoder.load_from_checkpoint(
            str(bad_ckpt_path),
            expected_d_in=768,
            device="cpu"
        )


def test_labels_and_formatting():
    """Test ImageNet label loader and formatting utility across all argument signatures."""
    labels = get_imagenet_labels()
    assert len(labels) == 1000
    assert "tench" in labels[0].lower() or "goldfish" in labels[1].lower()

    # Single-argument signature
    formatted = format_class_label(0)
    assert "(#0)" in formatted

    # Two-argument signature with custom label
    formatted_custom = format_class_label(588, "tench")
    assert formatted_custom == "tench (#588)"

    # Pre-formatted string should not get duplicate (#588)
    formatted_redundant = format_class_label(588, "tench (#588)")
    assert formatted_redundant == "tench (#588)"

    # Out of bounds index
    formatted_oob = format_class_label(9999)
    assert formatted_oob == "Class #9999"


def test_gradio_app_module_import():
    """Requirement 4: Ensure build_app can be directly imported from app.gradio_app."""
    from app.gradio_app import build_app, build_gradio_app, ViTLensApp, ViTLensGradioApp
    assert callable(build_app)
    assert callable(build_gradio_app)
    assert ViTLensApp is ViTLensGradioApp


def test_model_service_smoke(mock_vit, mock_sae):
    """Test ModelService end-to-end integration without network dependency."""
    from PIL import Image
    service = ModelService(device="cpu", vit=mock_vit, sae=mock_sae)
    dummy_image = Image.new("RGB", (224, 224), color=(50, 100, 150))

    # 1. Forward predictions
    pred_data = service.predict_image(dummy_image)
    assert len(pred_data["top5"]) == 5
    assert pred_data["top5"][0]["probability"] >= pred_data["top5"][1]["probability"]

    # 2. Extract active features
    top_features = service.get_active_features_on_image(dummy_image, top_n=5)
    assert isinstance(top_features, list)

    # 3. Intervene
    res = service.run_causal_intervention(dummy_image, feature_id=5, strength=1.0)
    assert res["feature_id"] == 5
    assert abs(res["logit_diff"]) < 1e-4
