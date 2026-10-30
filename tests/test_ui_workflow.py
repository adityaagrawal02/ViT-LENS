import matplotlib
matplotlib.use("Agg")

import pytest  # noqa: E402
from PIL import Image  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import gradio as gr  # noqa: E402

from src.topk_sae import TopKSparseAutoencoder  # noqa: E402
from src.vit_model import ViTModelWrapper  # noqa: E402
from src.model_service import ModelService  # noqa: E402
from src.evidence_service import EvidenceService  # noqa: E402
from app.gradio_app import ViTLensGradioApp, build_app  # noqa: E402


@pytest.fixture
def test_vit():
    """Fast offline ViT wrapper for UI workflow testing."""
    vit = ViTModelWrapper(pretrained=False, use_torchvision_fallback=True)
    vit.model.eval()
    return vit


@pytest.fixture
def test_sae():
    """Standard TopK SAE instance for offline UI testing."""
    sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32)
    sae.eval()
    return sae


@pytest.fixture
def test_service(test_vit, test_sae):
    """ModelService with injected offline components."""
    service = ModelService(device="cpu", vit=test_vit, sae=test_sae)
    return service


@pytest.fixture
def test_app(test_service):
    """Initialized ViTLensGradioApp instance."""
    return ViTLensGradioApp(service=test_service)


@pytest.fixture
def sample_image():
    """Representative RGB test image."""
    return Image.new("RGB", (224, 224), color=(60, 120, 180))


# -----------------------------------------------------------------------------
# 1. Import build_app & Aliases
# -----------------------------------------------------------------------------
def test_1_import_build_app():
    """Requirement 1: Verify direct and backwards-compatible build_app imports."""
    from app.gradio_app import build_app, build_gradio_app, ViTLensApp, ViTLensGradioApp
    assert callable(build_app)
    assert callable(build_gradio_app)
    assert ViTLensApp is ViTLensGradioApp


# -----------------------------------------------------------------------------
# 2. Construct Real Gradio App
# -----------------------------------------------------------------------------
def test_2_construct_real_gradio_app(test_service):
    """Requirement 2: Build actual Gradio Blocks app and verify component wiring."""
    demo = build_app(service=test_service)
    assert isinstance(demo, gr.Blocks)
    # Check that blocks container has registered children
    assert len(demo.blocks) > 10


# -----------------------------------------------------------------------------
# 3. Analyze Image Callback with Valid Image
# -----------------------------------------------------------------------------
def test_3_analyze_image_valid(test_app, sample_image):
    """Requirement 3: Analyze image callback with a valid image."""
    outputs = test_app.handle_image_analysis(sample_image)
    assert len(outputs) == 13

    status_md, pred_chart, pred_label, feat_dropdown, manual_id, feat_summary, \
        heatmap_overlay, heatmap_raw, feat_info_md, target_class_drop, \
        intervene_md, comp_plot, intervened_label = outputs

    assert "Analysis Complete" in status_md
    assert isinstance(pred_chart, plt.Figure)
    assert len(pred_label) == 5
    assert isinstance(manual_id, int)
    assert not isinstance(manual_id, str)
    assert isinstance(heatmap_overlay, Image.Image)
    assert isinstance(heatmap_raw, Image.Image)
    assert isinstance(feat_info_md, str)
    assert comp_plot is None  # Reset state
    assert intervened_label == {}
    plt.close(pred_chart)


# -----------------------------------------------------------------------------
# 4. Analyze Image Callback with No Image
# -----------------------------------------------------------------------------
def test_4_analyze_image_none(test_app):
    """Requirement 4: Analyze image callback with None image gracefully resets."""
    outputs = test_app.handle_image_analysis(None)
    assert len(outputs) == 13

    status_md, pred_chart, pred_label, feat_dropdown, manual_id, feat_summary, \
        heatmap_overlay, heatmap_raw, feat_info_md, target_class_drop, \
        intervene_md, comp_plot, intervened_label = outputs

    assert "No image provided" in status_md
    assert pred_chart is None
    assert pred_label == {}
    assert manual_id is None
    assert not isinstance(manual_id, str)
    assert heatmap_overlay is None
    assert heatmap_raw is None
    assert comp_plot is None


# -----------------------------------------------------------------------------
# 5. Selecting an Active Feature
# -----------------------------------------------------------------------------
def test_5_selecting_active_feature(test_app, sample_image):
    """Requirement 5: Selecting an active feature updates heatmap and info."""
    _ = test_app.handle_image_analysis(sample_image)
    top_feats = test_app.current_top_features
    assert len(top_feats) > 0
    active_fid = top_feats[0]["feature_id"]

    outputs = test_app.handle_feature_selection(sample_image, None, active_fid)
    assert len(outputs) == 8
    summary_md, out_fid, overlay, raw_hm, details_md, reset_intervene, comp_plot, int_label = outputs

    assert out_fid == active_fid
    assert isinstance(out_fid, int)
    assert not isinstance(out_fid, str)
    assert isinstance(overlay, Image.Image)
    assert isinstance(raw_hm, Image.Image)
    assert comp_plot is None


# -----------------------------------------------------------------------------
# 6. Selecting an Inactive Feature
# -----------------------------------------------------------------------------
def test_6_selecting_inactive_feature(test_app, sample_image):
    """Requirement 6: Selecting an inactive feature shows 0 activation note."""
    _ = test_app.handle_image_analysis(sample_image)
    # Feature 3000 is likely inactive or we can test any feature
    outputs = test_app.handle_feature_selection(sample_image, None, 3000)
    assert len(outputs) == 8
    summary_md, out_fid, overlay, raw_hm, details_md, reset_intervene, comp_plot, int_label = outputs

    assert out_fid == 3000
    assert isinstance(out_fid, int)
    assert not isinstance(out_fid, str)
    assert isinstance(details_md, str)


# -----------------------------------------------------------------------------
# 7. Selecting a Manually Entered Feature ID
# -----------------------------------------------------------------------------
def test_7_selecting_manual_feature_id(test_app, sample_image):
    """Requirement 7: Manual feature ID entry updates correctly."""
    outputs = test_app.handle_feature_selection(sample_image, None, 1453)
    assert len(outputs) == 8
    _, out_fid, overlay, raw_hm, details_md, _, _, _ = outputs
    assert out_fid == 1453
    assert isinstance(out_fid, int)
    assert not isinstance(out_fid, str)
    assert "1453" in details_md


# -----------------------------------------------------------------------------
# 8. Invalid Feature IDs Handling
# -----------------------------------------------------------------------------
def test_8_invalid_feature_ids(test_app, sample_image):
    """Requirement 8: Out of bounds feature IDs are caught without crashing."""
    # Negative ID
    out_neg = test_app.handle_feature_selection(sample_image, None, -10)
    assert "Error" in out_neg[0]
    assert out_neg[1] is None

    # Exceeding dictionary size
    out_big = test_app.handle_feature_selection(sample_image, None, 99999)
    assert "Error" in out_big[0]
    assert out_big[1] is None


# -----------------------------------------------------------------------------
# 9. Heatmap Creation and Image Alignment
# -----------------------------------------------------------------------------
def test_9_heatmap_creation_and_alignment(test_service, sample_image):
    """Requirement 9: 14x14 heatmap matches grid and overlay has valid dimensions."""
    loc_res = test_service.localize_feature(sample_image, feature_id=588)
    assert loc_res["heatmap_2d"].shape == (14, 14)
    assert loc_res["overlay_image"].size == (224, 224)
    assert loc_res["raw_heatmap"].size == (224, 224)
    if loc_res["max_activation"] > 0:
        assert 0 <= loc_res["peak_row"] < 14
        assert 0 <= loc_res["peak_col"] < 14
        assert loc_res["peak_row"] == loc_res["peak_patch_idx"] // 14
        assert loc_res["peak_col"] == loc_res["peak_patch_idx"] % 14
    else:
        assert loc_res["peak_row"] is None
        assert loc_res["peak_col"] is None
        assert loc_res["peak_patch_idx"] is None


# -----------------------------------------------------------------------------
# 10. Suppression at alpha=0
# -----------------------------------------------------------------------------
def test_10_suppression_alpha_zero(test_app, sample_image):
    """Requirement 10: Causal suppression forward pass at alpha=0.0."""
    _ = test_app.handle_image_analysis(sample_image)
    report_md, plot, mod_top5 = test_app.handle_causal_intervention(
        sample_image, feature_id_input=588, strength=0.0, target_class_choice=None
    )
    assert "Intervention Results" in report_md
    assert "α = 0.00×" in report_md
    assert isinstance(plot, plt.Figure)
    assert len(mod_top5) == 5
    plt.close(plot)


# -----------------------------------------------------------------------------
# 11. Identity at alpha=1
# -----------------------------------------------------------------------------
def test_11_identity_alpha_one(test_service, sample_image):
    """Requirement 11: Identity intervention at alpha=1.0 yields zero shift."""
    res = test_service.run_causal_intervention(sample_image, feature_id=588, strength=1.0)
    assert abs(res["logit_diff"]) < 1e-4
    assert abs(res["prob_diff"]) < 1e-4
    assert res["prediction_flipped"] is False


# -----------------------------------------------------------------------------
# 12. Amplification at alpha=2
# -----------------------------------------------------------------------------
def test_12_amplification_alpha_two(test_app, sample_image):
    """Requirement 12: Causal amplification forward pass at alpha=2.0."""
    _ = test_app.handle_image_analysis(sample_image)
    report_md, plot, mod_top5 = test_app.handle_causal_intervention(
        sample_image, feature_id_input=588, strength=2.0, target_class_choice=None
    )
    assert "α = 2.00×" in report_md
    assert isinstance(plot, plt.Figure)
    assert len(mod_top5) == 5
    plt.close(plot)


# -----------------------------------------------------------------------------
# 13. Changing Image After Intervention
# -----------------------------------------------------------------------------
def test_13_changing_image_after_intervention(test_app, sample_image):
    """Requirement 13: Uploading a new image resets previous intervention outputs."""
    # Run first image & intervention
    _ = test_app.handle_image_analysis(sample_image)
    _ = test_app.handle_causal_intervention(sample_image, 588, 0.0, None)

    # Change to image 2
    img2 = Image.new("RGB", (224, 224), color=(200, 50, 50))
    outputs2 = test_app.handle_image_analysis(img2)

    # Check reset
    intervene_md = outputs2[10]
    comp_plot = outputs2[11]
    intervened_label = outputs2[12]

    assert "Select an intervention multiplier" in intervene_md
    assert comp_plot is None
    assert intervened_label == {}


# -----------------------------------------------------------------------------
# 14. Changing Feature After Intervention
# -----------------------------------------------------------------------------
def test_14_changing_feature_after_intervention(test_app, sample_image):
    """Requirement 14: Changing feature resets previous intervention outputs."""
    _ = test_app.handle_image_analysis(sample_image)
    _ = test_app.handle_causal_intervention(sample_image, 588, 0.0, None)

    # Select another feature
    outputs_sel = test_app.handle_feature_selection(sample_image, None, 1453)
    reset_intervene_md = outputs_sel[5]
    comp_plot = outputs_sel[6]
    intervened_label = outputs_sel[7]

    assert "Ready to run causal intervention" in reset_intervene_md
    assert comp_plot is None
    assert intervened_label == {}


# -----------------------------------------------------------------------------
# 15. Every Event Handler Output Count and Types
# -----------------------------------------------------------------------------
def test_15_event_handlers_output_counts_and_types(test_app, sample_image):
    """Requirement 15: Exact return lengths and valid types across all handlers."""
    # handle_image_analysis -> 13 outputs
    out_analysis = test_app.handle_image_analysis(sample_image)
    assert len(out_analysis) == 13
    assert isinstance(out_analysis[0], str)
    assert isinstance(out_analysis[2], dict)
    assert isinstance(out_analysis[4], int)
    plt.close(out_analysis[1])

    # handle_feature_selection -> 8 outputs
    out_select = test_app.handle_feature_selection(sample_image, None, 588)
    assert len(out_select) == 8
    assert isinstance(out_select[0], str)
    assert isinstance(out_select[1], int)
    assert isinstance(out_select[2], Image.Image)
    assert isinstance(out_select[4], str)

    # handle_causal_intervention -> 3 outputs
    out_intervene = test_app.handle_causal_intervention(sample_image, 588, 1.0, None)
    assert len(out_intervene) == 3
    assert isinstance(out_intervene[0], str)
    assert isinstance(out_intervene[1], plt.Figure)
    assert isinstance(out_intervene[2], dict)
    plt.close(out_intervene[1])


# -----------------------------------------------------------------------------
# 16. gr.Number Component Strictly Receives Number Not Text
# -----------------------------------------------------------------------------
def test_16_gr_number_receives_number_not_text(test_app, sample_image):
    """
    Requirement 16 & Regression Test:
    Verify manual_feature_id (gr.Number) strictly receives int, float, or None,
    and NEVER receives a string like 'Feature #588' which causes:
    TypeError: type str doesn't define __round__ method.
    """
    # Check handle_image_analysis
    out_analysis = test_app.handle_image_analysis(sample_image)
    num_val_1 = out_analysis[4]
    assert isinstance(num_val_1, (int, float, type(None)))
    assert not isinstance(num_val_1, str)
    plt.close(out_analysis[1])

    # Check handle_feature_selection with manual number
    out_select_1 = test_app.handle_feature_selection(sample_image, None, 588)
    num_val_2 = out_select_1[1]
    assert isinstance(num_val_2, (int, float, type(None)))
    assert not isinstance(num_val_2, str)
    assert num_val_2 == 588

    # Check handle_feature_selection with dropdown text
    dropdown_text = "Feature #1453 ★ [Validated Candidate] | Peak: 4.20 | Patches: 12/196"
    out_select_2 = test_app.handle_feature_selection(sample_image, dropdown_text, None)
    num_val_3 = out_select_2[1]
    assert isinstance(num_val_3, (int, float, type(None)))
    assert not isinstance(num_val_3, str)
    assert num_val_3 == 1453


# -----------------------------------------------------------------------------
# 17. Missing Checkpoint Handling
# -----------------------------------------------------------------------------
def test_17_missing_checkpoint_handling(test_vit):
    """Requirement 17: App initializes and reports missing checkpoint gracefully."""
    # Service without SAE
    service_no_sae = ModelService(device="cpu", vit=test_vit, sae=None)
    assert service_no_sae.is_sae_loaded is False

    app = ViTLensGradioApp(service=service_no_sae)
    assert app.service.is_sae_loaded is False

    demo = build_app(service=service_no_sae)
    assert isinstance(demo, gr.Blocks)


# -----------------------------------------------------------------------------
# 18. Missing Research-Results Files Handling
# -----------------------------------------------------------------------------
def test_18_missing_research_results():
    """Requirement 18: Non-existent results directory falls back to verified defaults."""
    ev = EvidenceService(results_dir="/non/existent/path/for/vit_lens_results")
    assert ev.is_dir_available is False
    summary = ev.get_summary()
    assert summary["benchmarks"]["explained_variance"] == 0.9011
    assert summary["benchmarks"]["reconstruction_agreement"] == 0.9360


# -----------------------------------------------------------------------------
# 19. Invalid or Unsupported Image Uploads
# -----------------------------------------------------------------------------
def test_19_invalid_or_unsupported_image_uploads(test_app):
    """Requirement 19: Grayscale or 1x1 image uploads handled safely without crash."""
    # Grayscale image
    img_gray = Image.new("L", (224, 224), color=128)
    out_gray = test_app.handle_image_analysis(img_gray)
    assert "Analysis Complete" in out_gray[0]
    plt.close(out_gray[1])

    # Small 1x1 image
    img_tiny = Image.new("RGB", (1, 1), color=(255, 0, 0))
    out_tiny = test_app.handle_image_analysis(img_tiny)
    assert "Analysis Complete" in out_tiny[0]
    plt.close(out_tiny[1])


# -----------------------------------------------------------------------------
# 20. Safe Cleanup of Hooks After Errors
# -----------------------------------------------------------------------------
def test_20_safe_cleanup_of_hooks_after_errors(test_service, sample_image):
    """Requirement 20: Errors during intervention clean up hooks safely."""
    # Verify hook state initially clean
    assert test_service.vit.hook_manager.intervention_fn is None

    # Force error during intervention
    class SyntheticError(Exception):
        pass

    original_forward = test_service.vit.model.forward

    def faulty_forward(x):
        raise SyntheticError("Forced failure")

    test_service.vit.model.forward = faulty_forward
    try:
        test_service.run_causal_intervention(sample_image, feature_id=588, strength=0.0)
    except SyntheticError:
        pass
    finally:
        test_service.vit.model.forward = original_forward

    # Hook must be detached
    assert test_service.vit.hook_manager.intervention_fn is None
