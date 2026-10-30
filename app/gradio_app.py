import os
import sys
import re
import argparse
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import gradio as gr  # noqa: E402

# Ensure repository root directory is placed first on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
elif sys.path[0] != _repo_root:
    sys.path.remove(_repo_root)
    sys.path.insert(0, _repo_root)

try:
    from src.utils import format_class_label, get_imagenet_labels  # noqa: E402
except (ImportError, AttributeError):
    try:
        from src import format_class_label, get_imagenet_labels  # noqa: E402
    except (ImportError, AttributeError):
        def format_class_label(class_idx: int, class_name: Optional[str] = None) -> str:  # noqa: E402
            if class_name:
                return f"{class_name} (#{class_idx})"
            return f"Class #{class_idx}"

        def get_imagenet_labels() -> List[str]:  # noqa: E402
            return [f"Class #{i}" for i in range(1000)]

from src.model_service import get_model_service, ModelService  # noqa: E402


def create_prediction_bar_chart(top5: list) -> Optional[plt.Figure]:
    """Generates a clean horizontal bar chart for top-5 predictions."""
    if not top5:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=140)
    labels = [p["label"] for p in reversed(top5)]
    probs = [p["probability"] for p in reversed(top5)]
    logits = [p["logit"] for p in reversed(top5)]

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, probs, color="#2563eb", height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, max(1.0, max(probs) * 1.15))
    ax.set_xlabel("Confidence Probability", fontsize=9)
    ax.set_title("Top-5 ImageNet-1K Predictions", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, prob, logit in zip(bars, probs, logits):
        w = bar.get_width()
        ax.text(w + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%} (logit: {logit:.2f})",
                va="center", ha="left", fontsize=8, fontweight="bold", color="#1e293b")

    plt.tight_layout()
    return fig


def create_intervention_comparison_plot(
    tracked_name: str,
    orig_prob: float,
    mod_prob: float,
    orig_logit: float,
    mod_logit: float,
    strength: float,
    feature_id: int
) -> Optional[plt.Figure]:
    """Generates a 2-panel comparison chart: Probability shift (left) and Logit shift (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.8), dpi=140)

    cats = ["Original (1.0x)", f"Edited ({strength:.1f}x)"]
    probs = [orig_prob, mod_prob]
    logits = [orig_logit, mod_logit]
    bar_colors = ["#3b82f6", "#10b981" if strength > 1.0 else ("#ef4444" if strength < 1.0 else "#3b82f6")]

    # Panel 1: Probability (%)
    bars1 = ax1.bar(cats, probs, color=bar_colors, width=0.45)
    ax1.set_ylabel("Probability", fontsize=9)
    ax1.set_ylim(0.0, max(1.05, max(probs) * 1.15))
    ax1.set_title("Probability Shift", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    for b, p in zip(bars1, probs):
        ax1.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.02,
                 f"{p:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: Raw Logit Shift
    bars2 = ax2.bar(cats, logits, color=bar_colors, width=0.45)
    ax2.set_ylabel("Logit Value", fontsize=9)
    y_min = min(0.0, min(logits) * 1.15 if min(logits) < 0 else 0.0)
    y_max = max(1.0, max(logits) * 1.25)
    ax2.set_ylim(y_min, y_max)
    logit_diff = mod_logit - orig_logit
    diff_sign = "+" if logit_diff >= 0 else ""
    ax2.set_title(f"Logit Shift ({diff_sign}{logit_diff:.2f})", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    for b, l in zip(bars2, logits):
        ypos = b.get_height()
        va = "bottom" if ypos >= 0 else "top"
        offset = 0.02 * (y_max - y_min) if ypos >= 0 else -0.04 * (y_max - y_min)
        ax2.text(b.get_x() + b.get_width() / 2.0, ypos + offset,
                 f"{l:.2f}", ha="center", va=va, fontsize=9, fontweight="bold")

    fig.suptitle(f"{tracked_name} | Feature #{feature_id} (alpha={strength:.1f}x)",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


class ViTLensGradioApp:
    """Encapsulates backend model calls and UI event handlers for ViT-LENS."""
    def __init__(self, service: Optional[ModelService] = None):
        self.service = service or get_model_service()
        self.evidence = self.service.evidence_service.get_summary()

        # Session state
        self.current_image: Optional[Image.Image] = None
        self.current_pred_data: Optional[Dict[str, Any]] = None
        self.current_top_features: List[Dict[str, Any]] = []
        self.current_feature_id: Optional[int] = None
        self.current_tracked_class_idx: Optional[int] = None

    def handle_image_analysis(self, image: Optional[Image.Image]) -> Tuple:
        """
        Step 1 & 2: Runs frozen ViT inference, computes ImageNet-1k predictions,
        encodes all 196 patch tokens with TopK SAE, ranks active features,
        and automatically selects and visualizes the highest-activating feature.
        """
        if image is None:
            status_md = "⚠️ **No image provided.** Please upload an image or choose one of the sample photographs above."
            return (
                status_md,                            # prediction_status
                None,                                 # prediction_chart
                {},                                   # prediction_label
                gr.Dropdown(choices=[], value=None),  # discovered_features_dropdown
                None,                                 # manual_feature_id (int or None)
                "No image analyzed.",                 # feature_selection_summary
                None,                                 # heatmap_overlay
                None,                                 # heatmap_raw
                "Upload and analyze an image to inspect patch-token feature activations.",  # feature_info_md
                gr.Dropdown(choices=[], value=None),  # target_class_dropdown
                "Analyze an image to enable causal interventions.",  # intervention_metrics_md
                None,                                 # comparison_plot
                {}                                    # intervened_top5_label
            )

        self.current_image = image
        pred_data = self.service.predict_image(image)
        self.current_pred_data = pred_data

        top1 = pred_data["top1"]
        top5 = pred_data["top5"]
        self.current_tracked_class_idx = top1["class_idx"]

        # 1. Generate prediction visualization
        pred_chart = create_prediction_bar_chart(top5)
        pred_label_dict = {p["label"]: p["probability"] for p in top5}
        status_md = (
            f"✅ **Analysis Complete** | Top-1: **{top1['label']}** "
            f"(Confidence: **{top1['probability']:.1%}**, Raw Logit: **{top1['logit']:.2f}**)\n\n"
            f"Evaluated across all 1,000 ImageNet-1K output categories via frozen `vit_base_patch16_224.augreg_in1k`."
        )

        # 2. Populate target class dropdown choices
        class_choices = [p["label"] for p in top5]
        # Include benchmark categories if not already in top 5
        benchmark_classes = [0, 217, 207]
        for b_idx in benchmark_classes:
            b_lbl = format_class_label(b_idx)
            if b_lbl not in class_choices:
                class_choices.append(b_lbl)

        # 3. Discover active SAE features across all 196 patch tokens
        top_features = self.service.get_active_features_on_image(image, top_n=15)
        self.current_top_features = top_features

        if top_features:
            best_feat = top_features[0]
            best_fid = int(best_feat["feature_id"])
            self.current_feature_id = best_fid

            feat_choices = [f["display_text"] for f in top_features]
            feat_dropdown_val = best_feat["display_text"]

            selection_summary = (
                f"🎯 **Auto-Selected Top Discovered Feature #{best_fid}**: "
                f"Demonstrated the highest peak activation on this image "
                f"({best_feat['peak_activation']:.2f}, active on {best_feat['active_patch_count']}/196 patches). "
                f"Label: *{best_feat['label']}*."
            )

            # Automatically compute heatmap for the top active feature
            loc_res = self.service.localize_feature(image, best_fid)
            heatmap_overlay = loc_res["overlay_image"]
            heatmap_raw = loc_res["raw_heatmap"]
            feature_info_md = self._format_feature_details(best_fid, loc_res)
            manual_fid_out = best_fid
        else:
            feat_choices = []
            feat_dropdown_val = None
            self.current_feature_id = None
            selection_summary = "ℹ️ No features activated above threshold on this image."
            heatmap_overlay = None
            heatmap_raw = None
            feature_info_md = "No active features detected."
            manual_fid_out = None

        # Reset downstream intervention state
        reset_intervene_md = (
            "Select an intervention multiplier below (0.0x suppress, 1.0x identity, 2.0x amplify) "
            "and click **⚡ Execute Causal Intervention** to test causal necessity/sufficiency."
        )

        return (
            status_md,
            pred_chart,
            pred_label_dict,
            gr.Dropdown(choices=feat_choices, value=feat_dropdown_val),
            manual_fid_out,
            selection_summary,
            heatmap_overlay,
            heatmap_raw,
            feature_info_md,
            gr.Dropdown(choices=class_choices, value=top1["label"]),
            reset_intervene_md,
            None,
            {}
        )

    def handle_feature_selection(self,
                                 image: Optional[Image.Image],
                                 dropdown_choice: Optional[str],
                                 manual_id_input: Any) -> Tuple:
        """
        Step 3: Handles user selection of an SAE feature (via dropdown, preset, or manual ID).
        Computes spatial activation matrix across all 196 patch tokens and generates 14x14 heatmap.
        Guarantees manual_id_input receives a valid integer or None, never a string.
        """
        if image is None:
            return (
                "⚠️ Please upload and analyze an image first.",
                None,
                None,
                None,
                "No image provided.",
                "Upload an image to enable causal interventions.",
                None,
                {}
            )

        # Parse feature ID with priority: manual number > dropdown selection > fallback
        feature_id: Optional[int] = None

        if manual_id_input is not None and str(manual_id_input).strip():
            try:
                feature_id = int(float(str(manual_id_input).strip()))
            except Exception:
                feature_id = None

        if feature_id is None and dropdown_choice:
            match = re.search(r"#(\d+)", dropdown_choice)
            if match:
                feature_id = int(match.group(1))

        if feature_id is None:
            feature_id = self.current_feature_id or 588

        # Validate feature ID bounds
        d_sae = self.service.sae.d_sae if self.service.sae else 3072
        if not (0 <= feature_id < d_sae):
            err_msg = f"❌ **Error**: Feature ID #{feature_id} is out of bounds. Valid range: 0 to {d_sae - 1}."
            return (
                err_msg,
                None,
                None,
                None,
                err_msg,
                "Invalid feature ID selected.",
                None,
                {}
            )

        self.current_feature_id = feature_id

        try:
            loc_res = self.service.localize_feature(image, feature_id)
            heatmap_overlay = loc_res["overlay_image"]
            heatmap_raw = loc_res["raw_heatmap"]
            feature_info_md = self._format_feature_details(feature_id, loc_res)

            max_act = loc_res["max_activation"]
            p_count = loc_res["active_patch_count"]
            curated_str = " (Validated Research Candidate)" if loc_res["candidate_label"] != "Unlabeled SAE feature" else ""

            if max_act > 0:
                summary_md = (
                    f"🔍 **Selected Feature #{feature_id}**{curated_str} | "
                    f"Peak: **{max_act:.2f}** | Active Patches: **{p_count}/196** | "
                    f"Label: *{loc_res['candidate_label']}*"
                )
            else:
                summary_md = (
                    f"⚠️ **Feature inactive on this image** (Feature #{feature_id}: Peak 0.0000 across all 196 patches; 0/196 active patches). "
                    f"No patches activated. Residual causal intervention produces a zero-delta no-op."
                )

            reset_intervene_md = f"Ready to run causal intervention on **Feature #{feature_id}**."

            return (
                summary_md,
                feature_id,  # Numeric int, never string
                heatmap_overlay,
                heatmap_raw,
                feature_info_md,
                reset_intervene_md,
                None,
                {}
            )
        except Exception as e:
            err_md = f"❌ **Error computing localization for Feature #{feature_id}**: {str(e)}"
            return (
                err_md,
                feature_id,
                None,
                None,
                err_md,
                "Error during localization.",
                None,
                {}
            )

    def handle_causal_intervention(self,
                                   image: Optional[Image.Image],
                                   feature_id_input: Any,
                                   strength: float,
                                   target_class_choice: Optional[str]) -> Tuple:
        """
        Step 4: Executes actual residual-preserving causal intervention inside Block 7:
        x_edited = x_original + (alpha - 1) * f_i * d_i
        Preserves CLS token and unexplained residual r. Modifies only patch tokens.
        """
        if image is None:
            return "⚠️ Please provide an image first.", None, {}

        # Resolve feature ID
        try:
            if feature_id_input is not None and str(feature_id_input).strip():
                feature_id = int(float(str(feature_id_input).strip()))
            elif self.current_feature_id is not None:
                feature_id = self.current_feature_id
            else:
                feature_id = 588
        except Exception:
            feature_id = 588

        # Resolve target class index
        target_class_idx: Optional[int] = None
        if target_class_choice:
            match = re.search(r"#(\d+)", target_class_choice)
            if match:
                target_class_idx = int(match.group(1))

        if target_class_idx is None:
            target_class_idx = self.current_tracked_class_idx

        try:
            res = self.service.run_causal_intervention(
                image=image,
                feature_id=feature_id,
                strength=strength,
                target_class_idx=target_class_idx
            )
        except Exception as e:
            return f"❌ **Intervention failed**: {str(e)}", None, {}

        tracked_label = format_class_label(res["tracked_class"])
        orig_top1_label = format_class_label(res["orig_pred_class"])
        mod_top1_label = format_class_label(res["mod_pred_class"])

        logit_diff = res["logit_diff"]
        prob_diff = res["prob_diff"]
        is_active = res["is_feature_active"]
        flipped = res["prediction_flipped"]

        # Generate 2-panel comparison chart (Probability + Logit)
        comp_plot = create_intervention_comparison_plot(
            tracked_name=tracked_label,
            orig_prob=res["orig_target_prob"],
            mod_prob=res["mod_target_prob"],
            orig_logit=res["orig_target_logit"],
            mod_logit=res["mod_target_logit"],
            strength=strength,
            feature_id=feature_id
        )

        # Build modified top-5 dictionary for gr.Label
        mod_top5_dict = {}
        for c_idx, prob in zip(res["mod_top5"], res["mod_top5_probs"]):
            mod_top5_dict[format_class_label(c_idx)] = float(prob)

        # Format readable markdown metrics report
        flip_badge = "🚨 **YES — PREDICTION FLIPPED!**" if flipped else "🔒 **No (Top-1 Class Preserved)**"
        diff_sign = "+" if logit_diff >= 0 else ""
        prob_sign = "+" if prob_diff >= 0 else ""

        report_md = (
            f"### ⚡ Causal Intervention Results | Feature #{feature_id} (α = {strength:.2f}×)\n\n"
            f"| Metric | Original Forward (1.0×) | Intervened ({strength:.1f}×) | Net Causal Shift |\n"
            f"|---|---|---|---|\n"
            f"| **Top-1 Predicted Class** | `{orig_top1_label}` | `{mod_top1_label}` | {flip_badge} |\n"
            f"| **Top-1 Confidence** | {res['orig_pred_prob']:.2%} | {res['mod_pred_prob']:.2%} | "
            f"{res['mod_pred_prob'] - res['orig_pred_prob']:+.2%} |\n"
            f"| **Inspected Logit** (`{tracked_label}`) | {res['orig_target_logit']:.3f} | {res['mod_target_logit']:.3f} | "
            f"**`{diff_sign}{logit_diff:.3f}`** |\n"
            f"| **Inspected Prob** | {res['orig_target_prob']:.2%} | {res['mod_target_prob']:.2%} | "
            f"**`{prob_sign}{prob_diff:.2%}`** |\n\n"
            f"**Intervention Mode**: Residual-preserving edit applied to **all 196 patch tokens** of Block 7; "
            f"token 0 (CLS) strictly preserved. Feature active strength on image: `{res['max_feature_activation']:.4f}`."
        )

        if not is_active:
            report_md += (
                f"\n\n> ℹ️ **Notice**: Feature #{feature_id} is inactive on this image ($f_i = 0$). "
                f"Under the formula $x_{{edited}} = x + (\\alpha - 1) f_i d_i$, zero activation guarantees zero perturbation."
            )

        return report_md, comp_plot, mod_top5_dict

    def _format_feature_details(self, feature_id: int, loc_res: Dict[str, Any]) -> str:
        """Formats comprehensive interpretability and localization markdown note."""
        feat_info = self.service.evidence_service.get_feature_info(feature_id)
        max_act = loc_res["max_activation"]
        p_count = loc_res["active_patch_count"]
        mean_act = loc_res.get("mean_activation", 0.0)
        peak_row = loc_res.get("peak_row")
        peak_col = loc_res.get("peak_col")
        peak_idx = loc_res.get("peak_patch_idx")

        if max_act > 0 and peak_row is not None and peak_col is not None and peak_idx is not None:
            act_str = (
                f"Peak: **{max_act:.4f}** | Mean: **{mean_act:.4f}** | "
                f"Active on **{p_count}/196 patches** ({p_count/196:.1%}) | "
                f"Argmax: **Patch #{peak_idx}** (Row {peak_row + 1}/14, Col {peak_col + 1}/14)"
            )
        else:
            act_str = "Feature inactive on this image (Peak Strength: 0.0000 across all 196 patches; 0/196 active patches)"

        curated_tag = "⭐ Validated Research Candidate" if feat_info.get("is_curated") else "Discovered Dictionary Atom"

        return (
            f"#### Feature #{feature_id} Quantitative Inspection ({curated_tag})\n"
            f"- **Hypothesis / Semantic Label**: **{feat_info['label']}**\n"
            f"- **Associated ImageNet Category**: `{feat_info.get('class_name', 'N/A')}`\n"
            f"- **Activation Statistics**: {act_str}\n"
            f"- **Spatial Localization**: {loc_res['spatial_description']}\n"
            f"- **Empirical Benchmark Finding**: {feat_info.get('causal_effect_status', 'Untested in cohort benchmark')}\n"
            f"- **Interpretability Notes**: {feat_info.get('notes', '')}\n\n"
            f"> 💡 **Scientific Grounding Notice**: SAE features are learned unsupervised dictionary atoms capturing "
            f"Block 7 residual directions. Many features naturally activate on background context (e.g. grass, sky, ground, "
            f"or indoor wallpaper) or high-frequency textures rather than object boundaries. This is expected model behavior, "
            f"not an object segmentation error."
        )


def build_app(service: Optional[ModelService] = None) -> gr.Blocks:
    """Builds the complete polished ViT-LENS Gradio user interface."""
    app_backend = ViTLensGradioApp(service=service)
    device_name = str(app_backend.service.device).upper()
    is_cuda = "CUDA" in device_name

    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate"
    )

    with gr.Blocks(title="ViT-LENS Explorer", theme=theme) as demo:
        # -------------------------------------------------------------
        # Header & Compact Hardware / Checkpoint Status Bar
        # -------------------------------------------------------------
        gr.Markdown(
            """
            # 🔬 ViT-LENS Explorer
            ### Discover, Localize, and Causally Test Features in Vision Transformers Using TopK Sparse Autoencoders
            """
        )

        dev_str = f"{device_name} " + ('⚡' if is_cuda else '⚠️ (CPU)')
        ckpt_str = 'Mounted & Ready ✅' if app_backend.service.is_sae_loaded else 'Missing / Warning ❌'

        with gr.Row():
            gr.Markdown(
                f"""
                | ViT Backbone | Intervention Layer | TopK SAE Dictionary | Device | Checkpoint Status |
                |---|---|---|---|---|
                | `vit_base...` | **Block 7** | $d_{{sae}} = 3072, k = 32$ | **{dev_str}** | **{ckpt_str}** |
                """
            )

        if not app_backend.service.is_sae_loaded:
            gr.Warning(
                f"SAE Checkpoint not found at '{app_backend.service.sae_checkpoint_path}'. "
                "In Google Colab, mount Drive at /content/drive to access final_sae_best.pt."
            )

        gr.Markdown("---")

        # -------------------------------------------------------------
        # Step 1: Analyze Image
        # -------------------------------------------------------------
        gr.Markdown("## Step 1 — Analyze Image (Frozen ViT-B/16 Inference)")

        with gr.Row():
            with gr.Column(scale=5):
                input_image = gr.Image(type="pil", label="Input Image (224×224)", height=260)

                gr.Markdown("**Quick Research Demonstration Samples:**")
                with gr.Row():
                    btn_sample_fish = gr.Button("🐟 Tench (Class #0)", size="sm")
                    btn_sample_springer = gr.Button("🐶 English Springer (Class #217)", size="sm")
                    btn_sample_retriever = gr.Button("🐕 Golden Retriever (Class #207)", size="sm")

                analyze_btn = gr.Button("🔬 Analyze Image & Discover SAE Features", variant="primary", size="lg")

            with gr.Column(scale=7):
                prediction_status = gr.Markdown("Upload an image or pick a sample photo, then click **Analyze Image**.")
                prediction_chart = gr.Plot(label="Top-5 Classification Predictions", visible=True)
                prediction_label = gr.Label(label="Top-5 Probabilities", num_top_classes=5)

        gr.Markdown("---")

        # -------------------------------------------------------------
        # Step 2: Discover SAE Features
        # -------------------------------------------------------------
        gr.Markdown("## Step 2 — Discover SAE Features (Patch-Level Dictionary Decomposition)")

        with gr.Row():
            with gr.Column(scale=7):
                discovered_features_dropdown = gr.Dropdown(
                    label="Discovered Active Features on this Image (Ranked by Peak Activation)",
                    choices=[],
                    value=None,
                    interactive=True
                )
                feature_selection_summary = gr.Markdown(
                    "After image analysis, active dictionary features discovered by TopK SAE will be listed here."
                )

            with gr.Column(scale=5):
                manual_feature_id = gr.Number(
                    label="Or Enter Any SAE Feature ID (0–3071)",
                    value=None,
                    minimum=0,
                    maximum=3071,
                    step=1,
                    precision=0,
                    interactive=True
                )
                inspect_feature_btn = gr.Button("🔍 Inspect Feature & Update Heatmap", variant="secondary")

                with gr.Accordion("⭐ Validated Research Candidates (Historical Benchmarks)", open=False):
                    gr.Markdown("Click a researched candidate from our published experiments:")
                    with gr.Row():
                        btn_feat_588 = gr.Button("Feature #588 (Fish)", size="sm")
                        btn_feat_1453 = gr.Button("Feature #1453 (Springer)", size="sm")
                        btn_feat_2077 = gr.Button("Feature #2077 (Fish / Inconclusive)", size="sm")

        gr.Markdown("---")

        # -------------------------------------------------------------
        # Step 3: Understand Where the Feature Activates
        # -------------------------------------------------------------
        gr.Markdown("## Step 3 — Understand Where the Feature Activates (14×14 Spatial Heatmap)")
        gr.Markdown(
            "Every token in ViT Block 7 corresponds to an exact 16×16 image patch (14×14 grid = 196 tokens). "
            "The **left panel** displays the transparent overlay on the 224×224 input with an activation colorbar in real SAE units; "
            "the **right panel** displays the genuinely discrete 14×14 patch grid with nearest-neighbor interpolation, gridlines, and numeric colorbar."
        )

        with gr.Row():
            with gr.Column(scale=6):
                heatmap_overlay = gr.Image(
                    label="Transparent Overlay on 224×224 Input (with Numeric Colorbar & Peak Box)",
                    height=300
                )
            with gr.Column(scale=6):
                heatmap_raw = gr.Image(
                    label="Genuinely Discrete 14×14 Patch Grid (Nearest-Neighbor, Ticks & Colorbar)",
                    height=300
                )

        feature_info_md = gr.Markdown(
            "Select a discovered feature above to inspect its spatial localization across the 196 patch tokens."
        )

        gr.Markdown("---")

        # -------------------------------------------------------------
        # Step 4: Test Whether the Feature Matters (Causal Intervention)
        # -------------------------------------------------------------
        gr.Markdown("## Step 4 — Test Whether the Feature Matters (Residual-Preserving Causal Intervention)")

        gr.Markdown(
            """
            Intervene directly inside the frozen ViT residual stream at Block 7 output:
            $$x_{edited} = x_{original} + (\\alpha - 1) \\cdot f_i \\cdot d_i \\quad
            \\text{where } d_i = \\text{decode}(e_i) - \\text{decode}(\\mathbf{0})$$
            Modifies **all 196 patch tokens**; strictly preserves the CLS token and unexplained residual $\\mathbf{r} = x - \\hat{x}$.
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("**Quick Strength Presets:**")
                with gr.Row():
                    btn_suppress = gr.Button("🔴 0.0× (Suppress)", variant="stop", size="sm")
                    btn_identity = gr.Button("⚪ 1.0× (Identity Baseline)", size="sm")
                    btn_amplify = gr.Button("🟢 2.0× (Amplify)", variant="primary", size="sm")

                strength_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.25, value=0.0,
                    label="Intervention Multiplier (alpha)",
                    interactive=True
                )
                target_class_dropdown = gr.Dropdown(
                    label="Class to Track / Inspect Logit Shift",
                    choices=[],
                    value=None,
                    interactive=True
                )
                intervene_btn = gr.Button("⚡ Execute Causal Intervention", variant="primary", size="lg")

            with gr.Column(scale=7):
                intervention_metrics_md = gr.Markdown("Execute an intervention to observe modified logits and prediction changes.")
                comparison_plot = gr.Plot(label="Target Probability & Logit Shift", visible=True)
                intervened_top5_label = gr.Label(label="Intervened Top-5 Predictions", num_top_classes=5)

        gr.Markdown("---")

        # -------------------------------------------------------------
        # Step 5: Validated Research Findings & Limitations
        # -------------------------------------------------------------
        with gr.Accordion("📊 Research Evidence & Experimental Validation Summary (Google Colab Benchmarks)", open=False):
            benchmarks = app_backend.evidence["benchmarks"]
            mounted_str = "Yes ✅" if app_backend.evidence["dir_mounted"] else "No, showing verified report"
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
        def load_sample(path: str):
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
            return None

        analysis_outputs = [
            prediction_status,
            prediction_chart,
            prediction_label,
            discovered_features_dropdown,
            manual_feature_id,
            feature_selection_summary,
            heatmap_overlay,
            heatmap_raw,
            feature_info_md,
            target_class_dropdown,
            intervention_metrics_md,
            comparison_plot,
            intervened_top5_label
        ]

        feature_select_outputs = [
            feature_selection_summary,
            manual_feature_id,
            heatmap_overlay,
            heatmap_raw,
            feature_info_md,
            intervention_metrics_md,
            comparison_plot,
            intervened_top5_label
        ]

        # 1. Sample loading with automatic image analysis trigger
        btn_sample_fish.click(
            fn=lambda: load_sample("./data/samples/sample_fish.png"),
            outputs=[input_image]
        ).then(
            fn=app_backend.handle_image_analysis,
            inputs=[input_image],
            outputs=analysis_outputs
        )

        btn_sample_springer.click(
            fn=lambda: load_sample("./data/samples/sample_springer.png"),
            outputs=[input_image]
        ).then(
            fn=app_backend.handle_image_analysis,
            inputs=[input_image],
            outputs=analysis_outputs
        )

        btn_sample_retriever.click(
            fn=lambda: load_sample("./data/samples/sample_retriever.png"),
            outputs=[input_image]
        ).then(
            fn=app_backend.handle_image_analysis,
            inputs=[input_image],
            outputs=analysis_outputs
        )

        # 2. Primary Image Analysis
        analyze_btn.click(
            fn=app_backend.handle_image_analysis,
            inputs=[input_image],
            outputs=analysis_outputs
        )

        # 3. Feature Selection & Inspection
        discovered_features_dropdown.change(
            fn=lambda img, drop: app_backend.handle_feature_selection(img, drop, None),
            inputs=[input_image, discovered_features_dropdown],
            outputs=feature_select_outputs
        )

        inspect_feature_btn.click(
            fn=app_backend.handle_feature_selection,
            inputs=[input_image, discovered_features_dropdown, manual_feature_id],
            outputs=feature_select_outputs
        )

        # 4. Researched Candidate Feature Buttons
        btn_feat_588.click(
            fn=lambda: 588,
            outputs=[manual_feature_id]
        ).then(
            fn=lambda img: app_backend.handle_feature_selection(img, None, 588),
            inputs=[input_image],
            outputs=feature_select_outputs
        )

        btn_feat_1453.click(
            fn=lambda: 1453,
            outputs=[manual_feature_id]
        ).then(
            fn=lambda img: app_backend.handle_feature_selection(img, None, 1453),
            inputs=[input_image],
            outputs=feature_select_outputs
        )

        btn_feat_2077.click(
            fn=lambda: 2077,
            outputs=[manual_feature_id]
        ).then(
            fn=lambda img: app_backend.handle_feature_selection(img, None, 2077),
            inputs=[input_image],
            outputs=feature_select_outputs
        )

        # 5. Intervention Strength Presets
        btn_suppress.click(fn=lambda: 0.0, outputs=[strength_slider])
        btn_identity.click(fn=lambda: 1.0, outputs=[strength_slider])
        btn_amplify.click(fn=lambda: 2.0, outputs=[strength_slider])

        # 6. Execute Causal Intervention
        intervene_btn.click(
            fn=app_backend.handle_causal_intervention,
            inputs=[input_image, manual_feature_id, strength_slider, target_class_dropdown],
            outputs=[intervention_metrics_md, comparison_plot, intervened_top5_label]
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

    enable_share = args.share or (os.environ.get("VIT_LENS_SHARE", "false").lower() in ["1", "true", "yes"])

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=enable_share)


if __name__ == "__main__":
    main()
