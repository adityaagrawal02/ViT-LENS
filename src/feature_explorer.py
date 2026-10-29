import os
import json
import torch
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple
from tqdm import tqdm


class FeatureExplorer:
    """
    Analyzes TopK SAE feature activations across dataset images to discover
    monosemantic or interpretable visual concepts.
    Extracts top-activating patches, creates visual patch grids, and manages candidate semantic hypotheses.
    """
    def __init__(self,
                 sae,
                 vit_wrapper,
                 output_dir: str = "./outputs/features",
                 patch_size: int = 16,
                 grid_size: int = 14,
                 context_padding: int = 16):
        self.sae = sae
        self.vit = vit_wrapper
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.context_padding = context_padding
        os.makedirs(output_dir, exist_ok=True)

    def scan_dataset_activations(self,
                                 dataloader,
                                 max_samples: int = 500,
                                 top_k_features_to_track: int = 30) -> Dict[int, List[Dict]]:
        """
        Scans evaluation images through ViT and SAE, recording maximum activation strengths
        and spatial patch coordinates for features.
        Returns:
            feature_records: dict mapping feature_id -> list of top activation records
        """
        self.sae.eval()
        self.vit.eval()

        feature_records: Dict[int, List[Dict]] = {}
        samples_processed = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Scanning for Feature Activations"):
                if samples_processed >= max_samples:
                    break

                images = batch["image"]
                _, _, patch_acts, _ = self.vit(images)  # [B, 196, 768]
                B, N, D = patch_acts.shape

                # Encode patch activations through SAE
                flat_acts = patch_acts.reshape(-1, D)
                z_topk, _ = self.sae.encode(flat_acts)  # [B * 196, d_sae]
                z_reshaped = z_topk.reshape(B, N, -1)  # [B, 196, d_sae]

                for b in range(B):
                    if samples_processed >= max_samples:
                        break

                    img_path = batch["path"][b]
                    img_id = batch["id"][b].item() if isinstance(batch["id"][b], torch.Tensor) else batch["id"][b]
                    cls_name = batch["class_name"][b]

                    # For each patch, find active features
                    sample_acts = z_reshaped[b]  # [196, d_sae]
                    max_per_feature, best_patch_idx = torch.max(sample_acts, dim=0)  # [d_sae]

                    # Filter features with non-zero activations
                    active_feature_ids = torch.where(max_per_feature > 0)[0].cpu().numpy()

                    for feat_id in active_feature_ids:
                        feat_id = int(feat_id)
                        strength = max_per_feature[feat_id].item()
                        patch_idx = best_patch_idx[feat_id].item()
                        row = patch_idx // self.grid_size
                        col = patch_idx % self.grid_size

                        if feat_id not in feature_records:
                            feature_records[feat_id] = []

                        feature_records[feat_id].append({
                            "image_id": img_id,
                            "image_path": img_path,
                            "class_name": cls_name,
                            "feature_id": feat_id,
                            "activation_strength": float(strength),
                            "patch_idx": int(patch_idx),
                            "patch_row": int(row),
                            "patch_col": int(col)
                        })

                    samples_processed += 1

        # Sort each feature's activations in descending order
        for feat_id in feature_records:
            feature_records[feat_id] = sorted(
                feature_records[feat_id],
                key=lambda x: x["activation_strength"],
                reverse=True
            )

        return feature_records

    def extract_patch_crop(self,
                           image_path: str,
                           patch_row: int,
                           patch_col: int,
                           include_context: bool = True) -> Image.Image:
        """
        Extracts the image patch (and optional context padding) corresponding to (row, col)
        in the 14x14 ViT patch grid.
        """
        img = Image.open(image_path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        x0 = patch_col * self.patch_size
        y0 = patch_row * self.patch_size
        x1 = x0 + self.patch_size
        y1 = y0 + self.patch_size

        if include_context:
            pad = self.context_padding
            w, h = img.size
            crop_box = (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(w, x1 + pad),
                min(h, y1 + pad)
            )
            crop = img.crop(crop_box)
        else:
            crop = img.crop((x0, y0, x1, y1))

        return crop

    def create_feature_visual_grid(self,
                                   feature_id: int,
                                   records: List[Dict],
                                   top_k: int = 16,
                                   candidate_label: str = "Unassigned Hypothesis") -> Tuple[Image.Image, str]:
        """
        Builds a visual montage of the top-activating image patches for a given feature.
        """
        top_records = records[:top_k]
        grid_cols = 4
        grid_rows = (len(top_records) + grid_cols - 1) // grid_cols
        cell_size = 80  # Display cell size in pixels
        header_height = 40

        montage_w = grid_cols * cell_size
        montage_h = grid_rows * cell_size + header_height

        montage = Image.new("RGB", (montage_w, montage_h), color=(245, 245, 248))
        draw = ImageDraw.Draw(montage)

        # Draw header
        header_text = f"Feature #{feature_id} | Hypothesis: {candidate_label}"
        draw.text((8, 12), header_text, fill=(20, 20, 30))

        for idx, rec in enumerate(top_records):
            r = idx // grid_cols
            c = idx % grid_cols
            try:
                crop = self.extract_patch_crop(rec["image_path"], rec["patch_row"], rec["patch_col"], include_context=True)
                crop_resized = crop.resize((cell_size - 6, cell_size - 6), Image.Resampling.BICUBIC)
                x = c * cell_size + 3
                y = header_height + r * cell_size + 3
                montage.paste(crop_resized, (x, y))
                # Add activation strength tag
                draw.text((x + 2, y + 2), f"{rec['activation_strength']:.1f}", fill=(255, 255, 0))
            except Exception:
                pass

        out_path = os.path.join(self.output_dir, f"feature_{feature_id:04d}_grid.png")
        montage.save(out_path)
        return montage, out_path

    def save_candidate_hypotheses(self, labels_dict: Dict[int, Dict], output_path: str = "./outputs/features/candidate_labels.json"):
        """
        Saves candidate semantic hypotheses to JSON.
        Format:
        {
          "127": {
             "hypothesis": "Fur Texture",
             "description": "Consistently activates on canine and feline body fur",
             "confidence": "Medium",
             "annotator": "Research Team"
          }
        }
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(labels_dict, f, indent=2)
