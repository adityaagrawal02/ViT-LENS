import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Optional


class ActivationExtractor:
    """
    Extracts and stores intermediate transformer layer residual activations
    in manageable, chunked files to support low-RAM execution and resumability.
    """
    def __init__(self,
                 vit_wrapper,
                 output_dir: str = "./data/activations",
                 chunk_size: int = 100,
                 exclude_cls: bool = True):
        self.vit = vit_wrapper
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.exclude_cls = exclude_cls
        os.makedirs(output_dir, exist_ok=True)

    def get_existing_chunks(self) -> List[str]:
        """Returns sorted list of already extracted chunk file paths."""
        return sorted(glob.glob(os.path.join(self.output_dir, "chunk_*.pt")))

    def extract_and_save(self,
                         dataloader: DataLoader,
                         max_images: Optional[int] = None,
                         progress_bar: bool = True) -> List[str]:
        """
        Runs ViT inference over dataloader and persists activations in chunked files.
        Resumes automatically if chunks already exist.
        """
        existing_chunks = self.get_existing_chunks()
        start_chunk_idx = len(existing_chunks)
        saved_chunk_paths = list(existing_chunks)

        current_activations = []
        current_metadata = []
        images_processed = start_chunk_idx * self.chunk_size
        chunk_idx = start_chunk_idx

        pbar = tqdm(dataloader, desc="Extracting Activations", disable=not progress_bar)
        self.vit.eval()

        with torch.no_grad():
            for batch in pbar:
                if max_images and images_processed >= max_images:
                    break

                images = batch["image"]
                logits, probs, patch_acts, cls_acts = self.vit(images)

                # Target activations: patch tokens [B, 196, 768]
                acts = patch_acts if self.exclude_cls else torch.cat([cls_acts, patch_acts], dim=1)
                acts = acts.detach().cpu().to(torch.float32)

                batch_size = images.shape[0]
                for i in range(batch_size):
                    if max_images and images_processed >= max_images:
                        break

                    current_activations.append(acts[i])  # Shape: [196, 768]
                    current_metadata.append({
                        "image_id": batch["id"][i].item() if isinstance(batch["id"][i], torch.Tensor) else batch["id"][i],
                        "path": batch["path"][i],
                        "label": batch["label"][i].item() if isinstance(batch["label"][i], torch.Tensor) else batch["label"][i],
                        "class_name": batch["class_name"][i],
                        "pred_class": torch.argmax(probs[i]).item(),
                        "pred_prob": torch.max(probs[i]).item()
                    })
                    images_processed += 1

                    # Write out chunk when buffer reaches chunk_size
                    if len(current_activations) >= self.chunk_size:
                        chunk_path = self._save_chunk(chunk_idx, current_activations, current_metadata)
                        saved_chunk_paths.append(chunk_path)
                        chunk_idx += 1
                        current_activations = []
                        current_metadata = []

        # Flush any remaining activations
        if len(current_activations) > 0:
            chunk_path = self._save_chunk(chunk_idx, current_activations, current_metadata)
            saved_chunk_paths.append(chunk_path)

        return saved_chunk_paths

    def _save_chunk(self, chunk_idx: int, activations: List[torch.Tensor], metadata: List[Dict]) -> str:
        """Saves a single chunk of activations and metadata."""
        chunk_path = os.path.join(self.output_dir, f"chunk_{chunk_idx:05d}.pt")
        stacked_acts = torch.stack(activations, dim=0)  # [chunk_size, num_patches, d_model]
        torch.save({
            "activations": stacked_acts,
            "metadata": metadata,
            "shape": list(stacked_acts.shape)
        }, chunk_path)
        return chunk_path


class ChunkedActivationDataset(Dataset):
    """
    PyTorch Dataset that loads patch activation vectors across chunked files
    without loading the entire dataset into memory simultaneously.
    """
    def __init__(self, chunk_dir: str):
        self.chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.pt")))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No activation chunks found in {chunk_dir}")

        # Index the chunk offsets
        self.chunk_offsets = [0]
        self.chunk_sizes = []
        total_vectors = 0

        for p in self.chunk_paths:
            header = torch.load(p, map_location="cpu", weights_only=False)
            shape = header["shape"]  # [N_images, 196, 768]
            n_vectors = shape[0] * shape[1]
            self.chunk_sizes.append(n_vectors)
            total_vectors += n_vectors
            self.chunk_offsets.append(total_vectors)

        self.total_vectors = total_vectors
        self._cached_chunk_idx = None
        self._cached_chunk_data = None

    def __len__(self):
        return self.total_vectors

    def _load_chunk_for_vector(self, global_idx: int):
        """Binary search or scan to find which chunk contains global_idx."""
        # Find chunk index
        for i, offset in enumerate(self.chunk_offsets[:-1]):
            if self.chunk_offsets[i] <= global_idx < self.chunk_offsets[i + 1]:
                chunk_idx = i
                break
        else:
            chunk_idx = len(self.chunk_paths) - 1

        if self._cached_chunk_idx != chunk_idx:
            chunk = torch.load(self.chunk_paths[chunk_idx], map_location="cpu", weights_only=False)
            # Flatten images x patches -> [N_img * 196, 768]
            acts = chunk["activations"].reshape(-1, chunk["activations"].shape[-1])
            self._cached_chunk_data = acts
            self._cached_chunk_idx = chunk_idx

        local_idx = global_idx - self.chunk_offsets[chunk_idx]
        return self._cached_chunk_data[local_idx]

    def __getitem__(self, idx: int):
        return self._load_chunk_for_vector(idx)
