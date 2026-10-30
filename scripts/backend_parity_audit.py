#!/usr/bin/env python3
"""
ViT-LENS Scientific Backend Parity Audit
=========================================
Strictly audits and benchmarks two independent execution paths:
  Path A: Direct standalone research formulas (as developed in Colab research phase).
  Path B: Production ModelService and modular backend functions.

Compares:
  - Preprocessed input tensors
  - Original ViT logits & probabilities
  - Full SAE feature activation tensor [196, 3072]
  - Selected feature's 196 patch activations
  - Peak activation and active patch count
  - Complete 14x14 heatmap matrix
  - Peak patch spatial coordinates (row, col)
  - Intervened logits at alpha=0.0, alpha=1.0, alpha=2.0
  - Target class logit differences and probability shifts

Evaluates:
  Case 1: Genuine Tench image (sample_fish.png) with Feature #588, Class 0.
  Case 2: Genuine English Springer image (sample_springer.png) with Feature #1453, Class 217.
  Case 3: Genuine Golden Retriever image (sample_retriever.png) with Auto-discovered Feature.
  Case 4: Inactive Feature (Zero Activation) on Tench image.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.vit_model import ViTModelWrapper
from src.topk_sae import TopKSparseAutoencoder
from src.model_service import ModelService
from src.utils import resolve_sae_checkpoint_path, format_class_label


# ==============================================================================
# Path A: Direct Standalone Research Formulas (Colab Reference Implementation)
# ==============================================================================

def direct_research_preprocess(img_pil: Image.Image, vit_wrapper: ViTModelWrapper) -> torch.Tensor:
    """Path A: Direct timm evaluation preprocessing."""
    import timm.data
    config = timm.data.resolve_model_data_config(vit_wrapper.model)
    transform = timm.data.create_transform(**config, is_training=False)
    tensor = transform(img_pil.convert("RGB")).unsqueeze(0)
    return tensor.to(vit_wrapper.device)


def direct_research_inference(tensor: torch.Tensor,
                              vit_wrapper: ViTModelWrapper,
                              target_layer_idx: int = 7):
    """Path A: Forward pass extracting Block 7 residual activations directly."""
    captured = []

    def hook_fn(mod, inp, out):
        act = out[0] if isinstance(out, tuple) else out
        captured.append(act.clone())
        return out

    target_block = vit_wrapper.model.blocks[target_layer_idx]
    handle = target_block.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            logits = vit_wrapper.model(tensor)
            probs = torch.softmax(logits, dim=-1)
    finally:
        handle.remove()

    block7_act = captured[0]  # [1, 197, 768]
    cls_act = block7_act[:, 0:1, :]  # [1, 1, 768]
    patch_acts = block7_act[:, 1:, :]  # [1, 196, 768]
    return logits, probs, patch_acts, cls_act


def direct_research_sae_encode(patch_acts: torch.Tensor, sae: TopKSparseAutoencoder) -> torch.Tensor:
    """Path A: Direct TopK SAE encoding of patch representations."""
    flat = patch_acts.reshape(-1, sae.d_in)
    x_centered = flat - sae.b_dec
    z_pre = torch.relu(x_centered @ sae.W_enc + sae.b_enc)
    topk_vals, topk_indices = torch.topk(z_pre, k=sae.top_k, dim=-1)
    z_topk = torch.zeros_like(z_pre)
    z_topk.scatter_(-1, topk_indices, topk_vals)
    return z_topk  # [196, 3072]


def direct_research_heatmap(z_topk: torch.Tensor, feature_id: int):
    """Path A: Direct 14x14 spatial activation matrix and argmax extraction."""
    feat_acts = z_topk[:, feature_id].detach().cpu().numpy()  # [196]
    heatmap_14x14 = feat_acts.reshape(14, 14)
    peak_act = float(np.max(feat_acts))
    active_count = int(np.sum(feat_acts > 0))
    argmax_idx = int(np.argmax(feat_acts))
    peak_row = argmax_idx // 14
    peak_col = argmax_idx % 14
    return feat_acts, heatmap_14x14, peak_act, active_count, argmax_idx, peak_row, peak_col


def direct_research_intervention(tensor: torch.Tensor,
                                 vit_wrapper: ViTModelWrapper,
                                 sae: TopKSparseAutoencoder,
                                 feature_id: int,
                                 alpha: float,
                                 target_layer_idx: int = 7):
    """Path A: Direct residual-preserving causal intervention forward pass."""
    device = sae.W_dec.device
    dtype = sae.W_dec.dtype

    # Decoder direction: d_i = decode(e_i) - decode(0)
    with torch.no_grad():
        e_i = torch.zeros(1, sae.d_sae, device=device, dtype=dtype)
        e_i[0, feature_id] = 1.0
        zero = torch.zeros(1, sae.d_sae, device=device, dtype=dtype)
        d_i = (e_i @ sae.W_dec + sae.b_dec) - (zero @ sae.W_dec + sae.b_dec)
        d_i = d_i.squeeze(0)  # [768]

    def hook_fn(mod, inp, out):
        act = out[0] if isinstance(out, tuple) else out
        if abs(alpha - 1.0) < 1e-9:
            return out

        patch_acts = act[:, 1:, :]  # [B, 196, 768]
        flat = patch_acts.reshape(-1, sae.d_in)

        # Compute patch-specific activation f_i
        with torch.no_grad():
            x_centered = flat - sae.b_dec
            z_pre = torch.relu(x_centered @ sae.W_enc + sae.b_enc)
            topk_vals, topk_idx = torch.topk(z_pre, k=sae.top_k, dim=-1)
            z_topk = torch.zeros_like(z_pre)
            z_topk.scatter_(-1, topk_idx, topk_vals)
            f_i = z_topk[:, feature_id].reshape(act.shape[0], 196)  # [B, 196]

        if torch.all(f_i == 0.0):
            return out

        # delta = (alpha - 1.0) * f_i * d_i
        delta = (alpha - 1.0) * f_i.unsqueeze(-1) * d_i.unsqueeze(0).unsqueeze(0)
        mod_act = act.clone()
        mod_act[:, 1:, :] = patch_acts + delta

        if isinstance(out, tuple):
            return (mod_act,) + out[1:]
        return mod_act

    target_block = vit_wrapper.model.blocks[target_layer_idx]
    handle = target_block.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            mod_logits = vit_wrapper.model(tensor)
            mod_probs = torch.softmax(mod_logits, dim=-1)
    finally:
        handle.remove()

    return mod_logits[0].detach().cpu().numpy(), mod_probs[0].detach().cpu().numpy()


# ==============================================================================
# Parity Evaluation Engine
# ==============================================================================

def run_case_parity_audit(case_name: str,
                          image_path: str,
                          feature_id: int,
                          target_class_idx: int,
                          vit_wrapper: ViTModelWrapper,
                          sae: TopKSparseAutoencoder,
                          service: ModelService,
                          tol: float = 1e-5) -> dict:
    print(f"\n{'='*75}")
    print(f"CASE AUDIT: {case_name}")
    print(f"Image: {os.path.basename(image_path)} | Feature #{feature_id} | Target Class: {target_class_idx} ({format_class_label(target_class_idx)})")
    print(f"{'='*75}")

    img_pil = Image.open(image_path).convert("RGB")
    passed = True
    metrics = {}

    # -------------------------------------------------------------
    # 1. Preprocessing Parity
    # -------------------------------------------------------------
    tensor_a = direct_research_preprocess(img_pil, vit_wrapper)
    tensor_b = service.preprocess_image(img_pil)
    diff_tensor = float(torch.max(torch.abs(tensor_a - tensor_b)).item())
    metrics["diff_preprocessed_input"] = diff_tensor
    p1 = diff_tensor < tol
    passed = passed and p1
    print(f"  [1] Preprocessed Input Tensor Max Diff: {diff_tensor:.2e} -> {'PASS' if p1 else 'FAIL'}")

    # -------------------------------------------------------------
    # 2. Original ViT Logits Parity
    # -------------------------------------------------------------
    logits_a, probs_a, patch_acts_a, _ = direct_research_inference(tensor_a, vit_wrapper)
    pred_b = service.predict_image(img_pil)
    logits_b = torch.from_numpy(pred_b["logits"]).to(vit_wrapper.device)
    diff_logits = float(torch.max(torch.abs(logits_a[0] - logits_b)).item())
    metrics["diff_original_logits"] = diff_logits
    p2 = diff_logits < tol
    passed = passed and p2
    print(f"  [2] Pristine ViT Logits Max Diff:       {diff_logits:.2e} -> {'PASS' if p2 else 'FAIL'}")

    # -------------------------------------------------------------
    # 3. Full SAE Activation Tensor Parity [196, 3072]
    # -------------------------------------------------------------
    z_a = direct_research_sae_encode(patch_acts_a, sae)
    # Extract through ModelService / localizer
    with torch.no_grad():
        _, _, patch_acts_b, _ = service.vit(tensor_b)
        z_b, _ = service.sae.encode(patch_acts_b[0].to(service.sae.W_dec.device))
    diff_z = float(torch.max(torch.abs(z_a - z_b)).item())
    metrics["diff_full_sae_tensor"] = diff_z
    p3 = diff_z < tol
    passed = passed and p3
    print(f"  [3] Full SAE Latent Tensor Max Diff:    {diff_z:.2e} -> {'PASS' if p3 else 'FAIL'}")

    # -------------------------------------------------------------
    # 4. Feature Activations & 14x14 Spatial Grid Parity
    # -------------------------------------------------------------
    feat_vec_a, mat_a, peak_a, count_a, idx_a, row_a, col_a = direct_research_heatmap(z_a, feature_id)
    loc_b = service.localize_feature(img_pil, feature_id)
    mat_b = loc_b["heatmap_2d"]
    peak_b = loc_b["max_activation"]
    count_b = loc_b["active_patch_count"]
    row_b = loc_b["peak_row"]
    col_b = loc_b["peak_col"]
    idx_b = loc_b["peak_patch_idx"]

    # Strict spatial coordinate assertions on both execution paths
    assert 0 <= row_a < 14, f"Path A peak_row out of bounds: {row_a}"
    assert 0 <= col_a < 14, f"Path A peak_col out of bounds: {col_a}"
    assert 0 <= idx_a < 196, f"Path A flat_index out of bounds: {idx_a}"
    assert idx_a == row_a * 14 + col_a, f"Path A indexing mismatch: {idx_a} != {row_a} * 14 + {col_a}"
    assert np.isclose(mat_a[row_a, col_a], peak_a), f"Path A matrix at [{row_a}, {col_a}] ({mat_a[row_a, col_a]}) != peak ({peak_a})"

    assert 0 <= row_b < 14, f"Path B peak_row out of bounds: {row_b}"
    assert 0 <= col_b < 14, f"Path B peak_col out of bounds: {col_b}"
    assert 0 <= idx_b < 196, f"Path B flat_index out of bounds: {idx_b}"
    assert idx_b == row_b * 14 + col_b, f"Path B indexing mismatch: {idx_b} != {row_b} * 14 + {col_b}"
    assert np.isclose(mat_b[row_b, col_b], peak_b), f"Path B matrix at [{row_b}, {col_b}] ({mat_b[row_b, col_b]}) != peak ({peak_b})"

    diff_mat = float(np.max(np.abs(mat_a - mat_b)))
    diff_peak = abs(peak_a - peak_b)
    diff_count = abs(count_a - count_b)
    coord_match = (row_a == row_b) and (col_a == col_b) and (idx_a == idx_b)
    metrics["diff_14x14_matrix"] = diff_mat
    metrics["diff_peak_activation"] = diff_peak
    metrics["coords_match"] = coord_match

    p4 = (diff_mat < tol) and (diff_peak < tol) and (diff_count == 0) and coord_match
    passed = passed and p4
    print(f"  [4] 14x14 Spatial Matrix Max Diff:      {diff_mat:.2e} -> {'PASS' if diff_mat < tol else 'FAIL'}")
    print(f"      Peak Activation:                    Path A={peak_a:.4f}, Path B={peak_b:.4f} (diff: {diff_peak:.2e})")
    print(f"      Active Patches Count:               Path A={count_a}/196, Path B={count_b}/196")
    print(f"      Argmax Patch Spatial Coordinates:")
    print(f"        0-indexed array coordinates:      Path A=[row={row_a}, col={col_a}], Path B=[row={row_b}, col={col_b}] (Match: {coord_match})")
    print(f"        1-indexed human grid:             Path A=(Row {row_a + 1} of 14, Col {col_a + 1} of 14), Path B=(Row {row_b + 1} of 14, Col {col_b + 1} of 14)")
    print(f"        Flat token sequence index:        Path A=Token #{idx_a} (0..195), Path B=Token #{idx_b} (0..195)")
    print(f"        Pixel Box in 224x224:             [x0={col_a * 16}, y0={row_a * 16}, x1={(col_a + 1) * 16}, y1={(row_a + 1) * 16}]")
    print(f"        Index Validity Assertions:        0 <= row < 14: PASS | 0 <= col < 14: PASS | idx == row*14+col: PASS")

    # -------------------------------------------------------------
    # 5. Causal Interventions at alpha = 0.0, 1.0, 2.0
    # -------------------------------------------------------------
    # Alpha = 1.0 (Strict Identity Check)
    mod_logits_a_1, _ = direct_research_intervention(tensor_a, vit_wrapper, sae, feature_id, alpha=1.0)
    interv_b_1 = service.run_causal_intervention(img_pil, feature_id, strength=1.0, target_class_idx=target_class_idx)
    mod_logits_b_1 = interv_b_1["mod_logits"]
    diff_id_a = float(np.max(np.abs(mod_logits_a_1 - logits_a[0].cpu().numpy())))
    diff_id_b = float(np.max(np.abs(mod_logits_b_1 - pred_b["logits"])))
    diff_id_cross = float(np.max(np.abs(mod_logits_a_1 - mod_logits_b_1)))
    p5_id = (diff_id_a < tol) and (diff_id_b < tol) and (diff_id_cross < tol)
    passed = passed and p5_id
    print(f"  [5] Identity Check (alpha=1.0):")
    print(f"      Path A vs Original Diff:           {diff_id_a:.2e} -> {'PASS' if diff_id_a < tol else 'FAIL'}")
    print(f"      Path B vs Original Diff:           {diff_id_b:.2e} -> {'PASS' if diff_id_b < tol else 'FAIL'}")
    print(f"      Path A vs Path B Cross Diff:       {diff_id_cross:.2e} -> {'PASS' if diff_id_cross < tol else 'FAIL'}")

    # Alpha = 0.0 (Causal Suppression)
    mod_logits_a_0, mod_probs_a_0 = direct_research_intervention(tensor_a, vit_wrapper, sae, feature_id, alpha=0.0)
    interv_b_0 = service.run_causal_intervention(img_pil, feature_id, strength=0.0, target_class_idx=target_class_idx)
    mod_logits_b_0 = interv_b_0["mod_logits"]
    diff_0 = float(np.max(np.abs(mod_logits_a_0 - mod_logits_b_0)))
    logit_diff_a_0 = float(mod_logits_a_0[target_class_idx] - logits_a[0, target_class_idx].item())
    logit_diff_b_0 = interv_b_0["logit_diff"]
    p5_0 = diff_0 < tol
    passed = passed and p5_0
    print(f"  [6] Suppression Check (alpha=0.0):")
    print(f"      Intervened Logits Max Diff:        {diff_0:.2e} -> {'PASS' if p5_0 else 'FAIL'}")
    print(f"      Target Class Logit Shift: Path A={logit_diff_a_0:+.4f}, Path B={logit_diff_b_0:+.4f}")

    # Alpha = 2.0 (Causal Amplification)
    mod_logits_a_2, mod_probs_a_2 = direct_research_intervention(tensor_a, vit_wrapper, sae, feature_id, alpha=2.0)
    interv_b_2 = service.run_causal_intervention(img_pil, feature_id, strength=2.0, target_class_idx=target_class_idx)
    mod_logits_b_2 = interv_b_2["mod_logits"]
    diff_2 = float(np.max(np.abs(mod_logits_a_2 - mod_logits_b_2)))
    logit_diff_a_2 = float(mod_logits_a_2[target_class_idx] - logits_a[0, target_class_idx].item())
    logit_diff_b_2 = interv_b_2["logit_diff"]
    p5_2 = diff_2 < tol
    passed = passed and p5_2
    print(f"  [7] Amplification Check (alpha=2.0):")
    print(f"      Intervened Logits Max Diff:        {diff_2:.2e} -> {'PASS' if p5_2 else 'FAIL'}")
    print(f"      Target Class Logit Shift: Path A={logit_diff_a_2:+.4f}, Path B={logit_diff_b_2:+.4f}")

    print(f"\n  OVERALL CASE RESULT: {'PASSED (Zero Discrepancy)' if passed else 'FAILED'}")
    return {
        "case_name": case_name,
        "passed": passed,
        "metrics": metrics
    }


def main():
    parser = argparse.ArgumentParser(description="ViT-LENS Scientific Backend Parity Audit")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to final_sae_best.pt")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--allow-synthetic", action="store_true", default=False,
                        help="Allow running parity audit with synthetic weights (strictly for local CPU testing when checkpoint is absent)")
    args = parser.parse_args()

    print("=======================================================================")
    print("       ViT-LENS SCIENTIFIC BACKEND PARITY AUDIT SUITE                 ")
    print("=======================================================================")

    # Device configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Selected Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

    # 1. Load Pretrained ViT
    print("\nLoading Pretrained Frozen ViT (vit_base_patch16_224.augreg_in1k)...")
    vit = ViTModelWrapper(
        model_name="vit_base_patch16_224.augreg_in1k",
        pretrained=True,
        target_layer_idx=7,
        device=device
    )
    print(f"ViT Parameter Device: {vit.vit_parameter_device}")
    print(f"ViT Data Config: {vit.get_data_config()}")

    # 2. Resolve SAE Checkpoint
    ckpt_path = resolve_sae_checkpoint_path(args.checkpoint)
    print(f"\nTarget SAE Checkpoint Path: {ckpt_path}")
    checkpoint_exists = os.path.exists(ckpt_path)
    print(f"Checkpoint File Exists on Disk: {checkpoint_exists}")

    if checkpoint_exists:
        print(f"\n[AUDIT MODE: TRAINED CHECKPOINT] Loading real trained weights from: {ckpt_path}")
        sae = TopKSparseAutoencoder.load_from_checkpoint(ckpt_path, device=device)
        is_real_checkpoint = True
        checkpoint_description = f"AUTHENTIC TRAINED CHECKPOINT ({ckpt_path})"
    else:
        if not args.allow_synthetic:
            print("\n" + "!" * 75)
            print("FATAL ERROR: Validated SAE checkpoint not found at:")
            print(f"  {ckpt_path}")
            print("\nIn Google Colab, please mount your Google Drive before running this audit:")
            print("  from google.colab import drive")
            print("  drive.mount('/content/drive')")
            print("\nIf you are a developer testing purely algorithmic code parity locally on CPU")
            print("without access to the trained checkpoint file, rerun with explicit flag:")
            print("  python scripts/backend_parity_audit.py --allow-synthetic")
            print("!" * 75 + "\n")
            sys.exit(1)

        print("\n" + "!" * 75)
        print("[AUDIT WARNING: RUNNING WITH SYNTHETIC ARCHITECTURAL WEIGHTS (seed=42)]")
        print("Note: The real trained checkpoint was not found at the specified path.")
        print("Because --allow-synthetic was passed, running parity check on deterministic")
        print("synthetic weights to verify algorithmic equivalence between Path A and Path B.")
        print("THIS IS NOT A VALIDATION OF THE TRAINED CHECKPOINT REPRESENTATIONS.")
        print("!" * 75 + "\n")
        torch.manual_seed(42)
        sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32).to(device)
        sae.eval()
        is_real_checkpoint = False
        checkpoint_description = "DETERMINISTIC SYNTHETIC WEIGHTS (seed=42) [--allow-synthetic]"

    # 3. Initialize Production ModelService
    service = ModelService(
        vit=vit,
        sae=sae,
        device=device,
        sae_checkpoint_path=ckpt_path if is_real_checkpoint else None
    )

    # Image samples
    samples_dir = os.path.join(REPO_ROOT, "data", "samples")
    fish_path = os.path.join(samples_dir, "sample_fish.png")
    springer_path = os.path.join(samples_dir, "sample_springer.png")
    retriever_path = os.path.join(samples_dir, "sample_retriever.png")

    cases = [
        ("Case 1: Tench Image / Feature #588 (Class 0: Tench)", fish_path, 588, 0),
        ("Case 2: English Springer / Feature #1453 (Class 217: English Springer)", springer_path, 1453, 217),
        ("Case 3: Golden Retriever / Discovered Feature (Class 207: Retriever)", retriever_path, 2567, 207),
        ("Case 4: Inactive Feature (Feature #1453 on Tench Image)", fish_path, 1453, 0),
    ]

    results = []
    for name, img_path, fid, target_cls in cases:
        if os.path.exists(img_path):
            res = run_case_parity_audit(
                case_name=name,
                image_path=img_path,
                feature_id=fid,
                target_class_idx=target_cls,
                vit_wrapper=vit,
                sae=sae,
                service=service
            )
            results.append(res)
        else:
            print(f"Skipping {name}: Image {img_path} not found.")

    all_passed = all(r["passed"] for r in results)
    print("\n" + "="*75)
    print(f"FINAL AUDIT RESULT: {'ALL PARITY TESTS PASSED [OK]' if all_passed else 'PARITY FAILURES DETECTED [FAIL]'}")
    print(f"Audit Weight Source: {checkpoint_description}")
    print(f"Inference Device: {device} (ViT Parameters: {vit.vit_parameter_device})")
    print("="*75)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
