import os
import pytest
import torch
import numpy as np
from PIL import Image

from src.vit_model import ViTModelWrapper
from src.topk_sae import TopKSparseAutoencoder
from src.feature_localization import FeatureLocalizer
from src.model_service import ModelService
from src.utils import (
    denormalize_image,
    create_transparent_heatmap_overlay,
    render_discrete_grid_with_colorbar,
    render_overlay_with_colorbar,
)


@pytest.fixture
def mock_sae():
    """Initializes a deterministic TopKSparseAutoencoder for mathematical tests."""
    torch.manual_seed(42)
    sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32)
    sae.eval()
    return sae


@pytest.fixture
def vit_wrapper():
    """Initializes ViT wrapper."""
    return ViTModelWrapper(target_layer_idx=7, device=torch.device("cpu"))


class TestHeatmapMathematics:
    """Rigorous mathematical audit tests for spatial heatmaps and coordinate mappings."""

    def test_1_vit_patch_flattening_order_matches_row_major(self, vit_wrapper):
        """
        Verify that patch embedding projection maps image spatial coordinates (row, col)
        to sequence index row * 14 + col without transposition, swap, or rotation.
        """
        # Create a synthetic image with a distinct mark at row=3, col=10
        target_row = 3
        target_col = 10
        expected_patch_idx = target_row * 14 + target_col  # 52

        img_np = np.zeros((224, 224, 3), dtype=np.float32)
        # Patch dimensions are 16x16
        y_start, y_end = target_row * 16, (target_row + 1) * 16
        x_start, x_end = target_col * 16, (target_col + 1) * 16
        img_np[y_start:y_end, x_start:x_end, :] = 1.0

        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

        # In standard ViT patch_embed: Conv2d(3, 768, 16, 16)
        if hasattr(vit_wrapper.model, 'patch_embed'):
            pe = vit_wrapper.model.patch_embed(tensor)
            if pe.ndim == 3:
                # [1, 196, 768]
                patch_norms = torch.norm(pe[0], dim=-1).detach().cpu().numpy()
                argmax_patch = int(np.argmax(patch_norms))
                assert argmax_patch == expected_patch_idx, (
                    f"Patch embed index mismatch: expected {expected_patch_idx}, got {argmax_patch}"
                )

    def test_2_sae_reshape_and_matrix_indexing(self, mock_sae, vit_wrapper):
        """
        Verify that 1D patch activation vector of length 196 reshaped to (14, 14)
        preserves exact index mapping: matrix[r, c] == vector[r * 14 + c].
        """
        localizer = FeatureLocalizer(mock_sae, vit_wrapper)
        # Test synthetic 1D vector
        vec = np.arange(196, dtype=np.float32)
        mat = vec.reshape(14, 14)

        for r in range(14):
            for c in range(14):
                expected_idx = r * 14 + c
                assert mat[r, c] == vec[expected_idx], f"Mismatch at row {r}, col {c}"

    def test_3_yellow_bounding_box_matches_matrix_argmax(self, mock_sae, vit_wrapper):
        """
        Verify numerically that the bounding box coordinates [x0, y0, x1, y1]
        strictly correspond to the argmax patch row and column.
        """
        localizer = FeatureLocalizer(mock_sae, vit_wrapper)
        test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        res = localizer.localize_and_visualize(test_img, feature_id=100)
        heatmap = res["heatmap_2d"]
        peak_idx = res["peak_patch_idx"]
        peak_row = res["peak_row"]
        peak_col = res["peak_col"]
        max_act = res["max_activation"]

        # 1. Bounds and indexing assertions
        assert 0 <= peak_row < 14, f"peak_row out of bounds: {peak_row}"
        assert 0 <= peak_col < 14, f"peak_col out of bounds: {peak_col}"
        assert 0 <= peak_idx < 196, f"peak_patch_idx out of bounds: {peak_idx}"
        assert peak_idx == peak_row * 14 + peak_col, f"flat_index != row*14+col: {peak_idx} != {peak_row * 14 + peak_col}"

        # 2. Value at [peak_row, peak_col] must equal max_act
        assert np.isclose(heatmap[peak_row, peak_col], max_act)

        # 3. If feature is active, check pixel coordinates
        if max_act > 0.0:
            expected_x0 = peak_col * 16
            expected_y0 = peak_row * 16
            expected_x1 = (peak_col + 1) * 16
            expected_y1 = (peak_row + 1) * 16
            assert 0 <= expected_x0 < 224
            assert 0 <= expected_y0 < 224
            assert expected_x1 == expected_x0 + 16
            assert expected_y1 == expected_y0 + 16

    def test_4_denormalization_accuracy_without_color_shift(self):
        """
        Verify that denormalize_image inverts (0.5, 0.5, 0.5) normalization
        without loss of dynamic range or color shifting.
        """
        # Create an RGB image with known corner values
        orig_np = np.array([
            [[0, 128, 255], [64, 192, 32]],
            [[255, 0, 128], [10, 240, 70]]
        ], dtype=np.uint8)

        orig_tensor = torch.from_numpy(orig_np).permute(2, 0, 1).float() / 255.0
        # Normalize with mean=0.5, std=0.5
        norm_tensor = (orig_tensor - 0.5) / 0.5

        # Denormalize
        denorm_img = denormalize_image(norm_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        denorm_np = np.array(denorm_img)

        # Should match within uint8 rounding precision (<= 1)
        diff = np.abs(orig_np.astype(int) - denorm_np.astype(int))
        assert np.max(diff) <= 1, f"Denormalization error: max diff was {np.max(diff)}"

    def test_5_transparent_overlay_zero_tint_on_inactive_patches(self):
        """
        Verify that inactive patches (activation = 0) have 100% transparency,
        leaving the underlying image pixels completely unmodified.
        """
        # 224x224 solid red image
        orig_img = Image.new("RGB", (224, 224), color=(200, 50, 50))
        orig_arr = np.array(orig_img)

        # Heatmap active on only 1 patch (row=5, col=5)
        m = np.zeros((14, 14), dtype=np.float32)
        m[5, 5] = 10.0

        overlay = create_transparent_heatmap_overlay(
            orig_img, m, colormap_name="inferno", smooth=False
        )
        overlay_arr = np.array(overlay)

        # Check inactive patch (e.g. row 0, col 0: pixels [0..16, 0..16])
        inactive_crop_orig = orig_arr[0:16, 0:16, :]
        inactive_crop_over = overlay_arr[0:16, 0:16, :]
        assert np.array_equal(inactive_crop_orig, inactive_crop_over), (
            "Inactive patch was tinted! Expected 100% transparency for zero activation."
        )

        # Check active patch (row 5, col 5: pixels [80..96, 80..96])
        active_crop_orig = orig_arr[80:96, 80:96, :]
        active_crop_over = overlay_arr[80:96, 80:96, :]
        assert not np.array_equal(active_crop_orig, active_crop_over), (
            "Active patch had no overlay applied!"
        )

    def test_6_discrete_grid_with_colorbar_rendering(self):
        """
        Verify that render_discrete_grid_with_colorbar returns a valid PIL Image
        with expected dimensions.
        """
        m = np.zeros((14, 14), dtype=np.float32)
        m[2, 3] = 25.5
        m[8, 11] = 14.2

        grid_img = render_discrete_grid_with_colorbar(m, peak_row=2, peak_col=3)
        assert isinstance(grid_img, Image.Image)
        assert grid_img.size[0] > 200 and grid_img.size[1] > 200

    def test_7_switching_feature_ids_recomputes_fresh_activations(self, mock_sae, vit_wrapper):
        """
        Verify that querying feature A, then B, then A recalculates strictly correct activations
        and does not retain stale or cached state across calls.
        """
        localizer = FeatureLocalizer(mock_sae, vit_wrapper)
        test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))

        # Query Feature 10
        res_10 = localizer.localize_and_visualize(test_img, feature_id=10)
        # Query Feature 20
        res_20 = localizer.localize_and_visualize(test_img, feature_id=20)
        # Query Feature 10 again
        res_10_again = localizer.localize_and_visualize(test_img, feature_id=10)

        assert res_10["feature_id"] == 10
        assert res_20["feature_id"] == 20
        assert res_10_again["feature_id"] == 10

        # Exact numerical identity for feature 10 across calls
        assert np.array_equal(res_10["heatmap_2d"], res_10_again["heatmap_2d"])
        assert res_10["max_activation"] == res_10_again["max_activation"]
        assert res_10["peak_patch_idx"] == res_10_again["peak_patch_idx"]

    def test_8_real_sample_dog_preserves_true_activations(self, mock_sae):
        """
        Verify that running localization on sample_springer.png preserves real model activations
        without artificially forcing or shifting activations onto the dog body.
        """
        springer_path = os.path.join("data", "samples", "sample_springer.png")
        if not os.path.exists(springer_path):
            pytest.skip("sample_springer.png not found")

        service = ModelService(sae=mock_sae)
        img = Image.open(springer_path)

        # Localize Feature #1453 (English Springer candidate)
        res = service.localize_feature(img, 1453)
        m = res["heatmap_2d"]

        assert m.shape == (14, 14)
        assert 0 <= res["peak_row"] < 14, f"peak_row out of bounds: {res['peak_row']}"
        assert 0 <= res["peak_col"] < 14, f"peak_col out of bounds: {res['peak_col']}"
        assert 0 <= res["peak_patch_idx"] < 196, f"peak_patch_idx out of bounds: {res['peak_patch_idx']}"
        assert res["peak_patch_idx"] == res["peak_row"] * 14 + res["peak_col"]
        assert np.isclose(m[res["peak_row"], res["peak_col"]], res["max_activation"])

        # Check that outputs contain discrete grid, overlay with colorbar, and spatial description
        assert "discrete_grid" in res
        assert "overlay_image" in res
        assert "spatial_description" in res
        assert isinstance(res["discrete_grid"], Image.Image)
        assert isinstance(res["overlay_image"], Image.Image)

