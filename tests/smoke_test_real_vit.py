import sys
import os
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vit_model import ViTModelWrapper  # noqa: E402
from src.topk_sae import TopKSparseAutoencoder  # noqa: E402
from src.intervention import ResidualPreservingIntervention  # noqa: E402
from src.feature_localization import FeatureLocalizer  # noqa: E402


def run_real_vit_smoke_test():
    print("=" * 60)
    print("ViT-LENS: Real Pretrained ViT-B/16 End-to-End Smoke Test")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"Compute Device: {device}")

    # 1. Load Real Pretrained ViT-B/16
    print("\n[Step 1/5] Loading Pretrained ViT-B/16...")
    vit = ViTModelWrapper(model_name="vit_base_patch16_224", pretrained=True, target_layer_idx=7, device=device)
    print(f"  Model Type Loaded: {vit.model_type}")
    print(f"  Target Layer Hook: Block #{vit.target_layer_idx}")
    assert vit.model_type in ["timm", "torchvision"], "Must use real pretrained weights"

    # 2. Prepare genuine image tensor
    print("\n[Step 2/5] Preparing Real Image Input...")
    img_path = "./data/samples/sample_retriever.png"
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB").resize((224, 224))
    else:
        img = Image.new("RGB", (224, 224), (200, 180, 120))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(transforms.ToTensor()(img)).unsqueeze(0)

    # 3. Execute Real Forward Pass & Verify Shapes
    print("\n[Step 3/5] Executing Real Forward Inference Pass...")
    with torch.no_grad():
        logits, probs, patch_acts, cls_acts = vit(img_tensor)

    top5_probs, top5_classes = torch.topk(probs[0], k=5)
    print(f"  Prediction Logits Shape: {logits.shape} (Expected [1, 1000])")
    print(f"  Patch Activations Shape: {patch_acts.shape} (Expected [1, 196, 768])")
    print(f"  CLS Token Shape: {cls_acts.shape} (Expected [1, 1, 768])")
    print(f"  Top-1 Class: #{top5_classes[0].item()} (Confidence: {top5_probs[0].item():.2%})")

    assert patch_acts.shape == (1, 196, 768), "Patch activation shape mismatch"
    assert cls_acts.shape == (1, 1, 768), "CLS token shape mismatch"

    # 4. TopK SAE Sparse Encoding & Localization
    print("\n[Step 4/5] Testing TopK SAE Encoding & Feature Localization...")
    sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32).to(device)
    localizer = FeatureLocalizer(sae, vit)

    heatmap_14x14, max_act, mean_act = localizer.compute_feature_map(img_tensor, feature_id=127)
    print(f"  Heatmap Shape: {heatmap_14x14.shape} (Expected [14, 14])")
    print(f"  Feature #127 Max Activation: {max_act:.4f} | Mean: {mean_act:.4f}")
    assert heatmap_14x14.shape == (14, 14), "Heatmap grid must be 14x14"

    # 5. Residual-Preserving Causal Intervention on Real ViT
    print("\n[Step 5/5] Executing Residual-Preserving Causal Intervention...")
    intervenor = ResidualPreservingIntervention(vit, sae)

    # 5a. Identity check at 1.0x
    res_identity = intervenor.run_intervention(img_tensor, feature_id=127, strength_multiplier=1.0)
    print(f"  Identity (1.0x) Logit Diff: {res_identity['logit_diff']:.8f} (Must be < 1e-4)")
    print(f"  Identity (1.0x) Prob Diff:  {res_identity['prob_diff']:.8f} (Must be < 1e-4)")
    assert abs(res_identity["logit_diff"]) < 1e-4, "Identity intervention must produce zero change"

    # 5b. Suppression check at 0.0x
    res_supp = intervenor.run_intervention(img_tensor, feature_id=127, strength_multiplier=0.0)
    print(f"  Suppression (0.0x) Prob Diff: {res_supp['prob_diff']:+.6f}")
    print(f"  Suppression (0.0x) Logit Diff: {res_supp['logit_diff']:+.6f}")

    # 5c. Amplification check at 2.0x
    res_amp = intervenor.run_intervention(img_tensor, feature_id=127, strength_multiplier=2.0)
    print(f"  Amplification (2.0x) Prob Diff: {res_amp['prob_diff']:+.6f}")
    print(f"  Amplification (2.0x) Logit Diff: {res_amp['logit_diff']:+.6f}")

    print("\n" + "=" * 60)
    print("ALL REAL PRETRAINED ViT SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_real_vit_smoke_test()
