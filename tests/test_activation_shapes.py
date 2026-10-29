import os
import shutil
import torch

from src.vit_model import ViTModelWrapper
from src.dataset import create_synthetic_prototype_dataset, build_dataloaders
from src.activation_extractor import ActivationExtractor, ChunkedActivationDataset


def test_vit_activation_shapes():
    # Use mock/torchvision ViT for fast CPU testing
    vit = ViTModelWrapper(pretrained=False, use_torchvision_fallback=True)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    logits, probs, patch_acts, cls_acts = vit(dummy_input)

    assert logits.shape == (batch_size, 1000), f"Expected [2, 1000], got {logits.shape}"
    assert probs.shape == (batch_size, 1000), f"Expected [2, 1000], got {probs.shape}"
    assert patch_acts.shape == (batch_size, 196, 768), f"Expected [2, 196, 768], got {patch_acts.shape}"
    assert cls_acts.shape == (batch_size, 1, 768), f"Expected [2, 1, 768], got {cls_acts.shape}"


def test_activation_extractor_chunking():
    temp_dir = "./data/test_temp_data"
    temp_act_dir = "./data/test_temp_activations"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_act_dir, exist_ok=True)

    try:
        # Create small synthetic dataset (2 classes, 4 images each = 8 images)
        create_synthetic_prototype_dataset(target_dir=temp_dir, num_classes=2, images_per_class=4)
        train_loader, _, _ = build_dataloaders(temp_dir, batch_size=4, val_split=0.25, num_workers=0)

        vit = ViTModelWrapper(pretrained=False, use_torchvision_fallback=True)
        extractor = ActivationExtractor(vit, output_dir=temp_act_dir, chunk_size=4)

        chunk_paths = extractor.extract_and_save(train_loader, progress_bar=False)
        assert len(chunk_paths) >= 1, "At least one activation chunk should be saved"

        first_chunk = torch.load(chunk_paths[0], weights_only=False)
        assert "activations" in first_chunk
        assert "metadata" in first_chunk
        assert first_chunk["activations"].ndim == 3  # [N_images, 196, 768]
        assert first_chunk["activations"].shape[1] == 196
        assert first_chunk["activations"].shape[2] == 768

        # Test ChunkedActivationDataset
        dataset = ChunkedActivationDataset(temp_act_dir)
        assert len(dataset) > 0
        sample_vec = dataset[0]
        assert sample_vec.shape == (768,)
    finally:
        # Clean up temporary test data
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(temp_act_dir):
            shutil.rmtree(temp_act_dir)


if __name__ == "__main__":
    test_vit_activation_shapes()
    test_activation_extractor_chunking()
    print("All activation and shape tests passed successfully!")
