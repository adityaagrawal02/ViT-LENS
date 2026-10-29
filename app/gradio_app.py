import os
import sys
import argparse
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt

# Ensure repository root directory is placed first on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
elif sys.path[0] != _repo_root:
    sys.path.remove(_repo_root)
    sys.path.insert(0, _repo_root)

try:
    from src.utils import format_class_label  # noqa: E402
except (ImportError, AttributeError):
    try:
        from src import format_class_label  # noqa: E402
    except (ImportError, AttributeError):
        def format_class_label(class_idx: int, class_name: Optional[str] = None) -> str:  # noqa: E402
            if class_name:
                return f"{class_name} (#{class_idx})"
            return f"Class #{class_idx}"

from src.model_service import get_model_service, ModelService  # noqa: E402


def create_probability_comparison_plot(orig_label: str,
                                       orig_prob: float,
                                       mod_prob: float,
                                       strength: float,
                                       feature_id: int):
    """Generates a clean comparison bar plot of original vs intervened probability."""
    fig, ax = plt.subplots(figsize=(6, 2.8), dpi=150)
    categories = ["Original (1.0x)", f"Intervened ({strength:.1f}x)"]
    probs = [orig_prob, mod_prob]
    colors = ["#1f77b4", "#ff7f0e" if strength > 1.0 else "#d62728"]

    bars = ax.bar(categories, probs, color=colors, width=0.45)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Target Class Probability Shift | Feature #{feature_id}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, p in zip(bars, probs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{p:.2%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


class ViTLensGradioApp:
    """Encapsulates backend model calls and UI event handlers for ViT-LENS."""
    def __init__(self, service: Optional[ModelService] = None):
        self.service = service or get_model_service()
        self.evidence = self.service.evidence_service.get_summary()

        # Cache last inference results for interactive UI responsiveness
        self.last_image: Optional[Image.Image] = None
        self.last_pred_data: Optional[Dict[str, Any]] = None
        self.last_top5_indices: list = []

    def handle_image_input(self, image: Optional[Image.Image]) -> Tuple[str, dict, list, str]:
        """Runs base frozen ViT inference on input image."""
        if image is None:
            return "Please provide an image.", {}, gr.Dropdown(choices=[]), "No image provided."

        self.last_image = image
        pred_data = self.service.predict_image(image)
        self.last_pred_data = pred_data

        top1 = pred_data["top1"]
        top5 = pred_data["top5"]
        self.last_top5_indices = [p["class_idx"] for p in top5]

        # Format label dictionary for gr.Label
        probs_dict = {p["label"]: p["probability"] for p in top5}

        status_text = (
            f"**Predicted Top-1 Class**: {top1['label']}\n"
            f"Confidence: {top1['probability']:.2%} | Raw Logit: {top1['logit']:.3f}\n"
            f"Evaluated across all 1,000 ImageNet-1K output classes."
        )

        # Build dropdown options for class inspection
        class_choices = [f"{p['label']} (Logit: {p['logit']:.2f})" for p in top5]
        default_class = class_choices[0]

        # Scan active features on this image
        active_features = self.service.get_active_features_on_image(image, top_n=12)
        if active_features:
            act_text = "Top active features on this image: " + ", ".join(
                [f"#{f['feature_id']} ({f['peak_activation']:.1f})" for f in active_features[:6]]
            )
        else:
            act_text = "No strongly active SAE features detected on image."

        return status_text, probs_dict, gr.Dropdown(choices=class_choices, value=default_class), act_text

    def handle_feature_selection(self,
                                 image: Optional[Image.Image],
                                 feature_id_input: Any) -> Tuple[Optional[Image.Image], Optional[Image.Image], str, str]:
        """Computes spatial localization heatmap and metadata for selected feature."""
        if image is None:
            return None, None, "Please upload or select an image first.", ""

        try:
            # Parse feature ID whether provided as integer, string, or dropdown label
            if isinstance(feature_id_input, str):
                cleaned = feature_id_input.split()[0].replace("#", "").replace("Feature", "").strip()
                feature_id = int(cleaned)
            else:
                feature_id = int(feature_id_input)
        except Exception:
            feature_id = 588

        feat_info = self.service.evidence_service.get_feature_info(feature_id)

        try:
            loc_res = self.service.localize_feature(image, feature_id)
            heatmap = loc_res["raw_heatmap"]
            overlay = loc_res["overlay_image"]
            max_act = loc_res["max_activation"]
            p_count = loc_res["active_patch_count"]
            peak_desc = loc_res["spatial_description"]

            if max_act > 0:
                act_status = f"Active ({p_count}/196 patches) | Peak Strength: {max_act:.4f}"
            else:
                act_status = "Inactive on this image (Peak Strength: 0.0000; 0/196 patches)"

            details = (
                f"### Feature #{feature_id} Details\n"
                f"- **Tentative Hypothesis**: {feat_info['label']}\n"
                f"- **Activation on Image**: {act_status}\n"
                f"- **Localization**: {peak_desc}\n"
                f"- **Benchmark Causal Finding**: {feat_info['causal_effect_status']}\n"
                f"- **Notes**: {feat_info['notes']}\n\n"
                f"*Notice: SAE patch heatmaps reflect sparse dictionary activations in Block 7, "
                f"which are distinct from Transformer attention routing maps.*"
            )
            return heatmap, overlay, details, f"Feature #{feature_id}"
        except Exception as e:
            return None, None, f"Error computing localization: {str(e)}", f"Feature #{feature_id}"

    def handle_causal_intervention(self,
                                   image: Optional[Image.Image],
                                   feature_id_input: Any,
                                   strength: float,
                                   class_selection: Optional[str]) -> Tuple[str, Any, dict]:
        """Executes actual residual-preserving causal intervention forward pass."""
        if image is None:
            return "Please provide an image.", None, {}

        try:
            if isinstance(feature_id_input, str):
                cleaned = feature_id_input.split()[0].replace("#", "").replace("Feature", "").strip()
                feature_id = int(cleaned)
            else:
                feature_id = int(feature_id_input)
        except Exception:
            feature_id = 588

        # Determine target class index to inspect
        target_class_idx = None
        if class_selection and "(#" in class_selection:
            try:
                # Extract number between '(#' and ')'
                target_class_idx = int(class_selection.split("(#")[1].split(")")[0])
            except Exception:
                target_class_idx = None

        if target_class_idx is None and self.last_top5_indices:
            target_class_idx = self.last_top5_indices[0]

        try:
            res = self.service.run_causal_intervention(
                image=image,
                feature_id=feature_id,
                strength=strength,
                target_class_idx=target_class_idx
            )
        except Exception as e:
            return f"Intervention failed: {str(e)}", None, {}

        tracked_name = format_class_label(res["tracked_class"])
        orig_top1_name = format_class_label(res["orig_pred_class"])
        mod_top1_name = format_class_label(res["mod_pred_class"])

        is_active = res["is_feature_active"]
        inactive_warning = ""
        if not is_active:
            inactive_warning = (
                "\n> **Note**: Feature #" + str(feature_id) + " had **zero activation** on all 196 patch tokens for this image. "
                "Per the formula $x_{edited} = x + (\\alpha - 1) \\cdot f_i \\cdot d_i$, with $f_i = 0$, the intervention "
                "correctly produces zero modification ($x_{edited} \\equiv x_{original}$).\n"
            )

        report = (
            f"### Causal Intervention Results (Feature #{feature_id} @ {strength:.1f}x)\n"
            f"{inactive_warning}\n"
            f"- **Target Inspected Class**: {tracked_name}\n"
            f"- **Original Logit**: {res['orig_target_logit']:.4f} → **Modified Logit**: {res['mod_target_logit']:.4f} "
            f"(Delta: **{res['logit_diff']:+.4f}**)\n"
            f"- **Original Probability**: {res['orig_target_prob']:.2%} → **Modified Probability**: {res['mod_target_prob']:.2%} "
            f"(Delta: **{res['prob_diff']:+.2%}**)\n"
            f"- **Top-1 Prediction**: {orig_top1_name} → {mod_top1_name}\n"
            f"- **Top-1 Flip Occurred**: {'🚨 **YES**' if res['prediction_flipped'] else '✅ **NO**'}\n"
            f"- **Formula Used**: $x_{{edited}} = x + (\\alpha - 1) \\cdot f_i \\cdot d_i$ (Residual Preserved)\n"
            f"- **ViT Forward Pass**: Executed through frozen model with Block 7 hook."
        )

        # Build modified top-5 dictionary for label display
        mod_top5_dict = {}
        for c_idx, prob in zip(res["mod_top5"], res["mod_top5_probs"]):
            mod_top5_dict[format_class_label(c_idx)] = float(prob)

        # Build comparison plot
        plot_fig = create_probability_comparison_plot(
            orig_label=tracked_name,
            orig_prob=res["orig_target_prob"],
            mod_prob=res["mod_target_prob"],
            strength=strength,
            feature_id=feature_id
        )

        return report, plot_fig, mod_top5_dict


def build_app(service: Optional[ModelService] = None) -> gr.Blocks:
    """Builds the comprehensive ViT-LENS Gradio Web Application."""
    app_backend = ViTLensGradioApp(service=service)

    # Device & Status indicators
    device_name = app_backend.service.device.type.upper()
    sae_status_badge = (
        "🟢 SAE Checkpoint Loaded" if app_backend.service.is_sae_loaded
        else f"⚠️ Checkpoint Warning: {app_backend.service.sae_load_error or 'Not found'}"
    )

    with gr.Blocks(title="ViT-LENS Explorer") as demo:
        gr.Markdown(
            f"""
            # 🔬 ViT-LENS Explorer
            ### Discovering and Causally Testing Interpretable Features in Vision Transformers Using TopK Sparse Autoencoders
            **Model**: `vit_base_patch16_224.augreg_in1k` | **Intervention Layer**: Block 7 Residual Stream | **Device**: `{device_name}`
            *{sae_status_badge}*
            """
        )

        # -------------------------------------------------------------
        # Section 1: Image Input & Original Classification
        # -------------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Image Input")
                input_image = gr.Image(type="pil", label="Input Image (224x224 RGB)", height=280)

                with gr.Row():
                    sample_fish_btn = gr.Button("🐟 Sample: Fish (#0 / #1)", size="sm")
                    sample_springer_btn = gr.Button("🐕 Sample: Springer (#217)", size="sm")
                    sample_retriever_btn = gr.Button("🦮 Sample: Retriever (#207)", size="sm")

                classify_btn = gr.Button("🔍 Run ViT Inference", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 2. Original Pretrained ViT Prediction")
                base_status = gr.Markdown("Upload an image or select a sample above to begin.")
                base_probs_label = gr.Label(label="Top-5 ImageNet-1K Predictions", num_top_classes=5)
                active_features_text = gr.Markdown("")

        gr.Markdown("---")

        # -------------------------------------------------------------
        # Section 2: Feature Exploration & Spatial Localization
        # -------------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 3. SAE Feature Selection & Hypothesis")
                gr.Markdown(
                    "Select a researched feature from our validated Colab experiments or enter any feature index (0–3071):"
                )
                with gr.Row():
                    btn_feat_588 = gr.Button("Feature #588 (Fish-associated)", size="sm")
                    btn_feat_1453 = gr.Button("Feature #1453 (Springer-associated)", size="sm")
                    btn_feat_2077 = gr.Button("Feature #2077 (Fish-associated)", size="sm")

                feature_id_input = gr.Number(value=588, label="SAE Feature ID (0 to 3071)", precision=0)
                localize_btn = gr.Button("📍 Compute Spatial Heatmap", variant="secondary")
                feature_details_md = gr.Markdown("Click a feature preset or enter an ID to inspect.")

            with gr.Column(scale=1):
                gr.Markdown("### 4. 14×14 Patch Activation Heatmap")
                with gr.Row():
                    raw_heatmap_out = gr.Image(label="14×14 Activation Grid", height=220)
                    overlay_out = gr.Image(label="Bicubic Overlay on Input", height=220)

                gr.Markdown(
                    "> **Interpretability Note**: Heatmaps display spatial activation of sparse dictionary atoms across "
                    "the 196 patch tokens of Block 7. This is distinct from self-attention routing, "
                    "and does not prove the exact causal pixels."
                )

        gr.Markdown("---")

        # -------------------------------------------------------------
        # Section 3: Residual-Preserving Causal Intervention
        # -------------------------------------------------------------
        gr.Markdown("### 5. Essential Residual-Preserving Causal Intervention")
        gr.Markdown(
            """
            Intervene directly inside the frozen ViT residual stream at Block 7 output:
            $$x_{edited} = x_{original} + (\\alpha - 1) \\cdot f_i \\cdot d_i \\quad
            \\text{where } d_i = \\text{decode}(e_i) - \\text{decode}(0)$$
            Modifies **all 196 patch tokens**; preserves the CLS token and unexplained residual $\\mathbf{r} = x - \\hat{x}$.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    btn_suppress = gr.Button("0.0x (Suppress Feature)", variant="stop", size="sm")
                    btn_identity = gr.Button("1.0x (Identity Baseline)", size="sm")
                    btn_amplify = gr.Button("2.0x (Amplify Feature)", variant="primary", size="sm")

                strength_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.25, value=0.0,
                    label="Intervention Strength Multiplier (alpha)"
                )
                target_class_dropdown = gr.Dropdown(
                    label="Inspect Specific Class Logit & Probability",
                    choices=[],
                    value=None
                )
                intervene_btn = gr.Button("⚡ Execute Causal Intervention", variant="primary")

            with gr.Column(scale=1):
                intervene_report_md = gr.Markdown("Execute an intervention to observe modified logits and prediction changes.")
                comparison_plot = gr.Plot(label="Target Probability Before vs After")
                mod_probs_label = gr.Label(label="Intervened Top-5 Predictions", num_top_classes=5)

        gr.Markdown("---")

        # -------------------------------------------------------------
        # Section 4: Validated Research Findings & Limitations
        # -------------------------------------------------------------
        with gr.Accordion("📊 Research Evidence & Experimental Validation Summary", open=False):
            benchmarks = app_backend.evidence['benchmarks']
            mounted_str = "Yes" if app_backend.evidence['dir_mounted'] else "No, showing benchmark report"
            gr.Markdown(
                f"""
                ### Established Empirical Results (Validated in Google Colab)
                - **Final SAE Architecture**: $d_{{in}} = 768 \\to d_{{sae}} = 3072$ ($4\\times$ expansion), TopK $k = 32$.
                - **SAE Validation Explained Variance**: **{benchmarks['explained_variance']:.2%}**
                - **Downstream Prediction Agreement**: **{benchmarks['reconstruction_agreement']:.2%}** (evaluated across 500 images).
                - **Independent 200-Image Class Association & Control Experiments**:
                  - **Feature #588** (fish-associated): Class AUC = **0.9397**; causal suppression reduced fish logits vs controls.
                  - **Feature #1453** (English Springer-associated): Class AUC = **0.9958**;
                    causal suppression flipped predictions to alternative dog breeds.
                  - **Feature #2077** (fish-associated): Class AUC = **0.9956**; matched-control intervention was **inconclusive**.
                - **Scientific Caveat**: High class-association AUC does not prove a monosemantic concept detector. Controls are required.
                - **Results Directory**: `{app_backend.evidence['results_dir']}` (Mounted: `{mounted_str}`).
                """
            )

        # -------------------------------------------------------------
        # UI Callback Wiring
        # -------------------------------------------------------------
        # Sample image loading
        def load_sample(path: str):
            if os.path.exists(path):
                img = Image.open(path)
                return img
            return None

        sample_fish_btn.click(
            fn=lambda: load_sample("./data/samples/sample_fish.png"),
            outputs=[input_image]
        )
        sample_springer_btn.click(
            fn=lambda: load_sample("./data/samples/sample_springer.png"),
            outputs=[input_image]
        )
        sample_retriever_btn.click(
            fn=lambda: load_sample("./data/samples/sample_retriever.png"),
            outputs=[input_image]
        )

        # Classification
        classify_btn.click(
            fn=app_backend.handle_image_input,
            inputs=[input_image],
            outputs=[base_status, base_probs_label, target_class_dropdown, active_features_text]
        )

        # Feature Presets
        btn_feat_588.click(fn=lambda: 588, outputs=[feature_id_input])
        btn_feat_1453.click(fn=lambda: 1453, outputs=[feature_id_input])
        btn_feat_2077.click(fn=lambda: 2077, outputs=[feature_id_input])

        # Localization
        localize_btn.click(
            fn=app_backend.handle_feature_selection,
            inputs=[input_image, feature_id_input],
            outputs=[raw_heatmap_out, overlay_out, feature_details_md, feature_id_input]
        )

        # Intervention Strength Presets
        btn_suppress.click(fn=lambda: 0.0, outputs=[strength_slider])
        btn_identity.click(fn=lambda: 1.0, outputs=[strength_slider])
        btn_amplify.click(fn=lambda: 2.0, outputs=[strength_slider])

        # Execute Causal Intervention
        intervene_btn.click(
            fn=app_backend.handle_causal_intervention,
            inputs=[input_image, feature_id_input, strength_slider, target_class_dropdown],
            outputs=[intervene_report_md, comparison_plot, mod_probs_label]
        )

    return demo


# Backwards-compatible aliases for Colab notebooks and external scripts
build_gradio_app = build_app
ViTLensApp = ViTLensGradioApp


def main():
    parser = argparse.ArgumentParser(description="ViT-LENS Gradio Explorer")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", default=False, help="Enable temporary public Gradio share link")
    args = parser.parse_args()

    # Allow environment variable VIT_LENS_SHARE=true to enable sharing in Colab
    enable_share = args.share or (os.environ.get("VIT_LENS_SHARE", "false").lower() in ["1", "true", "yes"])

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=enable_share)


if __name__ == "__main__":
    main()
