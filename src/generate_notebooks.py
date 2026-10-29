import os
import json

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "accelerator": "GPU",
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

def markdown_cell(source):
    if isinstance(source, list):
        src = [s if s.endswith("\n") else s + "\n" for s in source]
    else:
        src = [s + "\n" for s in source.split("\n")]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src
    }

def code_cell(source):
    if isinstance(source, list):
        src = [s if s.endswith("\n") else s + "\n" for s in source]
    else:
        src = [s + "\n" for s in source.split("\n")]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src
    }

def generate_all_notebooks():
    os.makedirs("./notebooks", exist_ok=True)

    # -------------------------------------------------------------
    # Notebook 01: Setup and Dataset
    # -------------------------------------------------------------
    nb01_cells = [
        markdown_cell([
            "# ViT-LENS: Notebook 01 — Setup & Dataset Preparation",
            "**Research Goal**: Environment configuration, Google Drive mounting, dependency installation, and dataset pipeline initialization for ViT-LENS."
        ]),
        code_cell([
            "# Step 1: Detect execution environment & mount Google Drive (if on Google Colab)",
            "import os, sys",
            "try:",
            "    from google.colab import drive",
            "    drive.mount('/content/drive')",
            "    project_root = '/content/drive/MyDrive/ViT-LENS'",
            "    os.makedirs(project_root, exist_ok=True)",
            "    print(f'Mounted Google Drive at {project_root}')",
            "except Exception:",
            "    project_root = os.getcwd()",
            "    print(f'Running locally / standard environment at {project_root}')",
            "",
            "if project_root not in sys.path:",
            "    sys.path.append(project_root)"
        ]),
        code_cell([
            "# Step 2: Install dependencies",
            "!pip install -q timm torchvision pyyaml matplotlib pandas tqdm gradio"
        ]),
        code_cell([
            "# Step 3: Device detection & configuration verification",
            "import torch",
            "from src.utils import get_device, load_config, seed_everything",
            "",
            "seed_everything(42)",
            "device = get_device('auto')",
            "print(f'Target Compute Device: {device}')",
            "if device.type == 'cuda':",
            "    print(f'GPU Name: {torch.cuda.get_device_name(0)}')",
            "    print(f'VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')",
            "else:",
            "    print('Running on CPU. For main SAE training, please switch Colab runtime to GPU (T4).')",
            "",
            "config = load_config('config.yaml')",
            "print('Configuration successfully loaded.')"
        ]),
        markdown_cell([
            "### Dataset Loading & Class Mapping",
            "ViT-LENS supports arbitrary image classification datasets with ImageFolder structure (`root/class_name/*.jpg`).",
            "When using ImageNet-100 with an ImageNet-1K pretrained ViT, we correctly map the 100 subset classes to their true ImageNet-1K target indices."
        ]),
        code_cell([
            "# Step 4: Prepare dataset (Generate prototype dataset for immediate execution or load custom)",
            "from src.dataset import create_synthetic_prototype_dataset, build_dataloaders",
            "",
            "data_dir = config['dataset']['data_dir'] + '/prototype'",
            "print(f'Setting up dataset at: {data_dir}')",
            "",
            "# Generate prototype samples if folder does not exist",
            "create_synthetic_prototype_dataset(target_dir=data_dir, num_classes=5, images_per_class=40)",
            "",
            "train_loader, val_loader, dataset = build_dataloaders(",
            "    data_dir=data_dir,",
            "    batch_size=config['dataset']['batch_size'],",
            "    val_split=config['dataset']['val_split'],",
            "    image_size=config['dataset']['image_size'],",
            "    num_workers=0",
            ")",
            "",
            "print(f'Dataset size: {len(dataset)} images across {len(dataset.classes)} classes.')",
            "print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')"
        ])
    ]

    with open("./notebooks/01_setup_and_dataset.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb01_cells), f, indent=2)

    # -------------------------------------------------------------
    # Notebook 02: ViT Activation Extraction
    # -------------------------------------------------------------
    nb02_cells = [
        markdown_cell([
            "# ViT-LENS: Notebook 02 — Vision Transformer Activation Extraction",
            "**Research Goal**: Load pretrained frozen ViT-B/16, attach PyTorch forward hooks to target intermediate Transformer Block (Block 7), and stream patch activations into manageable chunk files (`chunk_*.pt`)."
        ]),
        code_cell([
            "# Step 1: Initialize Frozen Pretrained ViT-B/16",
            "import torch",
            "from src.utils import get_device, load_config",
            "from src.vit_model import ViTModelWrapper",
            "from src.dataset import build_dataloaders",
            "from src.activation_extractor import ActivationExtractor",
            "",
            "device = get_device('auto')",
            "config = load_config('config.yaml')",
            "",
            "vit = ViTModelWrapper(",
            "    model_name=config['vit']['model_name'],",
            "    pretrained=True,",
            "    target_layer_idx=config['vit']['target_layer_idx'],",
            "    device=device",
            ")",
            "print(f'Loaded ViT model ({vit.model_type}) with target hook on Block #{vit.target_layer_idx}')"
        ]),
        code_cell([
            "# Step 2: Verify activation shapes on test batch",
            "dummy_x = torch.randn(2, 3, 224, 224).to(device)",
            "logits, probs, patch_acts, cls_acts = vit(dummy_x)",
            "",
            "print(f'Logits shape: {logits.shape} (Expected [2, 1000])')",
            "print(f'Patch activations shape: {patch_acts.shape} (Expected [2, 196, 768])')",
            "print(f'CLS token activation shape: {cls_acts.shape} (Expected [2, 1, 768])')",
            "assert patch_acts.shape == (2, 196, 768), 'Shape mismatch on patch tokens!'"
        ]),
        code_cell([
            "# Step 3: Run chunked extraction pipeline over dataset",
            "data_dir = config['dataset']['data_dir'] + '/prototype'",
            "train_loader, _, _ = build_dataloaders(data_dir, batch_size=16, val_split=0.0, num_workers=0)",
            "",
            "extractor = ActivationExtractor(",
            "    vit_wrapper=vit,",
            "    output_dir=config['extraction']['output_dir'],",
            "    chunk_size=config['extraction']['chunk_size'],",
            "    exclude_cls=not config['vit']['include_cls_token']",
            ")",
            "",
            "saved_chunks = extractor.extract_and_save(train_loader, max_images=config['dataset']['max_samples'])",
            "print(f'Successfully extracted and stored {len(saved_chunks)} activation chunks.')"
        ])
    ]

    with open("./notebooks/02_vit_activation_extraction.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb02_cells), f, indent=2)

    # -------------------------------------------------------------
    # Notebook 03: Train TopK SAE
    # -------------------------------------------------------------
    nb03_cells = [
        markdown_cell([
            "# ViT-LENS: Notebook 03 — TopK Sparse Autoencoder (SAE) Training",
            "**Research Goal**: Train custom TopK SAE ($d_{in}=768, d_{sae}=3072, k=32$) on extracted ViT residual-stream activations with decoder column normalization and resumable checkpointing."
        ]),
        code_cell([
            "# Step 1: Initialize TopK SAE Architecture",
            "import torch",
            "from torch.utils.data import DataLoader",
            "from src.utils import get_device, load_config",
            "from src.topk_sae import TopKSparseAutoencoder",
            "from src.sae_trainer import SAETrainer",
            "from src.activation_extractor import ChunkedActivationDataset",
            "",
            "device = get_device('auto')",
            "config = load_config('config.yaml')",
            "",
            "sae = TopKSparseAutoencoder(",
            "    d_in=config['sae']['d_in'],",
            "    d_sae=config['sae']['d_sae'],",
            "    top_k=config['sae']['top_k']",
            ").to(device)",
            "",
            "print(f'Initialized TopK SAE: {sae.d_in} -> {sae.d_sae} (Expansion {sae.d_sae/sae.d_in:.1f}x) | k = {sae.top_k}')"
        ]),
        code_cell([
            "# Step 2: Initialize Chunked Activation DataLoader",
            "act_dataset = ChunkedActivationDataset(config['extraction']['output_dir'])",
            "print(f'Total patch activation vectors in dataset: {len(act_dataset):,}')",
            "",
            "act_loader = DataLoader(",
            "    act_dataset,",
            "    batch_size=config['sae']['batch_size'],",
            "    shuffle=True,",
            "    num_workers=0",
            ")"
        ]),
        code_cell([
            "# Step 3: Run SAE Training Loop with Checkpointing",
            "trainer = SAETrainer(",
            "    sae=sae,",
            "    lr=config['sae']['lr'],",
            "    weight_decay=config['sae']['weight_decay'],",
            "    device=device,",
            "    checkpoint_dir=config['sae']['checkpoint_dir'],",
            "    dead_feature_threshold_steps=config['sae']['dead_feature_threshold_steps'],",
            "    resample_dead_features=config['sae']['resample_dead_features']",
            ")",
            "",
            "history = trainer.fit(",
            "    dataloader=act_loader,",
            "    num_epochs=config['sae']['num_epochs'],",
            "    save_every_epochs=config['sae']['save_every_epochs']",
            ")",
            "print('Training complete! Final checkpoint saved to checkpoints/sae_checkpoint_latest.pt')"
        ])
    ]

    with open("./notebooks/03_train_topk_sae.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb03_cells), f, indent=2)

    # -------------------------------------------------------------
    # Notebook 04: Feature Discovery and Localization
    # -------------------------------------------------------------
    nb04_cells = [
        markdown_cell([
            "# ViT-LENS: Notebook 04 — Feature Discovery & Spatial Localization",
            "**Research Goal**: Discover active dictionary features across dataset images, extract top-activating image patches, generate 14x14 spatial heatmaps, and formulate candidate semantic hypotheses."
        ]),
        code_cell([
            "# Step 1: Load Trained SAE & Pretrained ViT",
            "import os, torch",
            "from PIL import Image",
            "import matplotlib.pyplot as plt",
            "from src.utils import get_device, load_config",
            "from src.vit_model import ViTModelWrapper",
            "from src.topk_sae import TopKSparseAutoencoder",
            "from src.dataset import build_dataloaders",
            "from src.feature_explorer import FeatureExplorer",
            "from src.feature_localization import FeatureLocalizer",
            "",
            "device = get_device('auto')",
            "config = load_config('config.yaml')",
            "",
            "vit = ViTModelWrapper(pretrained=True, target_layer_idx=7, device=device)",
            "sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32).to(device)",
            "",
            "ckpt_path = os.path.join(config['sae']['checkpoint_dir'], 'sae_checkpoint_latest.pt')",
            "if os.path.exists(ckpt_path):",
            "    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)",
            "    sae.load_state_dict(ckpt['sae_state_dict'])",
            "    print('Loaded trained SAE checkpoint.')",
            "else:",
            "    print('Using initialized SAE weights for demonstration.')"
        ]),
        code_cell([
            "# Step 2: Scan Dataset for Top Activating Image Patches",
            "data_dir = config['dataset']['data_dir'] + '/prototype'",
            "eval_loader, _, _ = build_dataloaders(data_dir, batch_size=8, val_split=0.0, num_workers=0)",
            "",
            "explorer = FeatureExplorer(sae=sae, vit_wrapper=vit, output_dir=config['feature_discovery']['output_dir'])",
            "feature_records = explorer.scan_dataset_activations(eval_loader, max_samples=40)",
            "print(f'Scanned activations. Found {len(feature_records)} active features.')"
        ]),
        code_cell([
            "# Step 3: Generate Visual Patch Grids for Discovered Features",
            "target_features = list(feature_records.keys())[:3] if feature_records else [127, 42, 88]",
            "for feat_id in target_features:",
            "    if feat_id in feature_records:",
            "        montage, path = explorer.create_feature_visual_grid(",
            "            feature_id=feat_id,",
            "            records=feature_records[feat_id],",
            "            top_k=8,",
            "            candidate_label=f'Feature #{feat_id}'",
            "        )",
            "        print(f'Saved visual montage for Feature #{feat_id} -> {path}')"
        ]),
        code_cell([
            "# Step 4: Spatial Localization Heatmap (14x14 -> 224x224 Overlay)",
            "sample_path = './data/samples/sample_retriever.png'",
            "if os.path.exists(sample_path):",
            "    sample_img = Image.open(sample_path)",
            "else:",
            "    sample_img = Image.new('RGB', (224, 224), (200, 150, 100))",
            "",
            "localizer = FeatureLocalizer(sae=sae, vit_wrapper=vit)",
            "res = localizer.localize_and_visualize(sample_img, feature_id=127, candidate_label='Fur Texture')",
            "",
            "fig, axes = plt.subplots(1, 3, figsize=(12, 4))",
            "axes[0].imshow(res['original_image'])",
            "axes[0].set_title('Original Input')",
            "axes[1].imshow(res['raw_heatmap'])",
            "axes[1].set_title('14x14 Upsampled Heatmap')",
            "axes[2].imshow(res['overlay_image'])",
            "axes[2].set_title(f'Alpha Overlay (Feat #127: {res[\"candidate_label\"]})')",
            "for ax in axes: ax.axis('off')",
            "plt.tight_layout()",
            "plt.show()"
        ])
    ]

    with open("./notebooks/04_feature_discovery.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb04_cells), f, indent=2)

    # -------------------------------------------------------------
    # Notebook 05: Causal Intervention
    # -------------------------------------------------------------
    nb05_cells = [
        markdown_cell([
            "# ViT-LENS: Notebook 05 — Residual-Preserving Causal Intervention",
            "**Research Goal**: Intervene directly inside the frozen ViT residual stream using learned TopK SAE feature directions:",
            "$$x_{modified} = x + (f'_i - f_i) \\cdot W_{dec}[i]$$",
            "Test feature necessity via Suppression ($0\\times$) and sufficiency via Amplification ($2\\times$) with random feature control experiments."
        ]),
        code_cell([
            "# Step 1: Initialize Causal Intervention System",
            "import torch",
            "from PIL import Image",
            "from src.utils import get_device, load_config",
            "from src.vit_model import ViTModelWrapper",
            "from src.topk_sae import TopKSparseAutoencoder",
            "from src.intervention import ResidualPreservingIntervention",
            "",
            "device = get_device('auto')",
            "config = load_config('config.yaml')",
            "",
            "vit = ViTModelWrapper(pretrained=True, target_layer_idx=7, device=device)",
            "sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32).to(device)",
            "intervenor = ResidualPreservingIntervention(vit_wrapper=vit, sae=sae)"
        ]),
        code_cell([
            "# Step 2: Strict Numerical Verification of Identity Intervention (alpha = 1.0)",
            "dummy_x = torch.randn(1, 3, 224, 224).to(device)",
            "res_identity = intervenor.run_intervention(dummy_x, feature_id=127, strength_multiplier=1.0)",
            "",
            "print(f'Logit Difference: {res_identity[\"logit_diff\"]:.8f}')",
            "print(f'Probability Difference: {res_identity[\"prob_diff\"]:.8f}')",
            "assert abs(res_identity['logit_diff']) < 1e-4, 'Identity intervention MUST yield zero difference!'"
        ]),
        code_cell([
            "# Step 3: Execute Suppression (0x) and Amplification (2x) Interventions",
            "for strength in [0.0, 0.5, 1.0, 1.5, 2.0]:",
            "    res = intervenor.run_intervention(dummy_x, feature_id=127, strength_multiplier=strength)",
            "    print(f'Strength {strength:.1f}x -> Target Prob: {res[\"mod_target_prob\"]:.4f} | '",
            "          f'Prob Shift: {res[\"prob_diff\"]:+.4f} | Flipped: {res[\"prediction_flipped\"]}')"
        ])
    ]

    with open("./notebooks/05_causal_intervention.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb05_cells), f, indent=2)

    # -------------------------------------------------------------
    # Notebook 06: Evaluation and Demo Dashboard
    # -------------------------------------------------------------
    nb06_cells = [
        markdown_cell([
            "# ViT-LENS: Notebook 06 — Evaluation & ViT-LENS Explorer Dashboard",
            "**Research Goal**: Run controlled causal evaluation experiments against unrelated control features, generate comparative scientific visualizations, and launch the interactive Gradio web application."
        ]),
        code_cell([
            "# Step 1: Run Cohort Evaluation with Random-Feature Controls",
            "import os, torch",
            "from src.utils import get_device, load_config",
            "from src.vit_model import ViTModelWrapper",
            "from src.topk_sae import TopKSparseAutoencoder",
            "from src.dataset import build_dataloaders",
            "from src.evaluation import CausalEvaluator",
            "",
            "device = get_device('auto')",
            "config = load_config('config.yaml')",
            "",
            "vit = ViTModelWrapper(pretrained=True, target_layer_idx=7, device=device)",
            "sae = TopKSparseAutoencoder(d_in=768, d_sae=3072, top_k=32).to(device)",
            "evaluator = CausalEvaluator(vit, sae)",
            "",
            "data_dir = config['dataset']['data_dir'] + '/prototype'",
            "eval_loader, _, _ = build_dataloaders(data_dir, batch_size=8, val_split=0.0, num_workers=0)",
            "",
            "results = evaluator.evaluate_feature_causality(",
            "    target_feature_id=127,",
            "    eval_dataloader=eval_loader,",
            "    num_images=16,",
            "    num_control_features=4,",
            "    strengths=(0.0, 2.0)",
            ")",
            "print('Evaluation complete. Summary metrics:')",
            "print(results['summary'])"
        ]),
        code_cell([
            "# Step 2: Launch Interactive Faculty Demonstration Web App (ViT-LENS Explorer)",
            "from app.gradio_app import ViTLensApp, build_gradio_app",
            "",
            "app = ViTLensApp()",
            "demo = build_gradio_app(app)",
            "demo.launch(share=True, debug=False)"
        ])
    ]

    with open("./notebooks/06_evaluation_and_demo.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb06_cells), f, indent=2)

    print("All 6 executable Google Colab notebooks generated successfully in ./notebooks/")

if __name__ == "__main__":
    generate_all_notebooks()

