import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from tqdm import tqdm

from src.topk_sae import TopKSparseAutoencoder
from src.utils import save_json


class SAETrainer:
    """
    Trainer for TopK Sparse Autoencoder.
    Handles memory-efficient training over chunked activations, decoder column normalization,
    dead feature tracking, resumable checkpointing, and metric logging.
    """
    def __init__(self,
                 sae: TopKSparseAutoencoder,
                 lr: float = 3e-4,
                 weight_decay: float = 0.0,
                 device: torch.device = torch.device("cpu"),
                 checkpoint_dir: str = "./checkpoints",
                 dead_feature_threshold_steps: int = 500,
                 resample_dead_features: bool = False):
        self.sae = sae.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.dead_feature_threshold_steps = dead_feature_threshold_steps
        self.resample_dead_features = resample_dead_features
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        self.global_step = 0
        self.current_epoch = 0
        self.history: List[Dict] = []

    def save_checkpoint(self, filename: str = "sae_checkpoint_latest.pt"):
        """Saves complete trainer state for seamless resumption in Google Colab."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "sae_state_dict": self.sae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": {
                "d_in": self.sae.d_in,
                "d_sae": self.sae.d_sae,
                "top_k": self.sae.top_k
            }
        }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str):
        """Loads trainer state from checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.sae.load_state_dict(checkpoint["sae_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.history = checkpoint.get("history", [])

    def train_epoch(self, dataloader: DataLoader, progress_bar: bool = True) -> Dict[str, float]:
        """Trains the TopK SAE for one epoch."""
        self.sae.train()
        total_loss = 0.0
        total_exp_var = 0.0
        total_l0 = 0.0
        steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}", disable=not progress_bar)

        for batch_acts in pbar:
            batch_acts = batch_acts.to(self.device)

            self.optimizer.zero_grad()
            _, _, metrics = self.sae(batch_acts)

            loss = metrics["mse_loss"]
            loss.backward()

            # Prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Constrain decoder vectors to unit norm
            self.sae.normalize_decoder()

            # Optional dead feature resampling
            if self.resample_dead_features and (self.global_step % 250 == 0) and (self.global_step > 0):
                dead_mask = self.sae.get_dead_features(self.dead_feature_threshold_steps)
                if dead_mask.sum() > 0:
                    self.sae.resample_dead_features(dead_mask, batch_acts)

            total_loss += loss.item()
            total_exp_var += metrics["explained_variance"].item()
            total_l0 += metrics["l0_sparsity"].item()
            steps += 1
            self.global_step += 1

            if progress_bar and steps % 20 == 0:
                dead_pct = (self.sae.get_dead_features(self.dead_feature_threshold_steps).float().mean() * 100.0).item()
                pbar.set_postfix({
                    "MSE": f"{loss.item():.5f}",
                    "ExpVar": f"{metrics['explained_variance'].item():.3f}",
                    "Dead%": f"{dead_pct:.1f}%"
                })

        avg_loss = total_loss / max(steps, 1)
        avg_exp_var = total_exp_var / max(steps, 1)
        avg_l0 = total_l0 / max(steps, 1)
        dead_pct = (self.sae.get_dead_features(self.dead_feature_threshold_steps).float().mean() * 100.0).item()

        epoch_metrics = {
            "epoch": self.current_epoch + 1,
            "global_step": self.global_step,
            "loss": avg_loss,
            "explained_variance": avg_exp_var,
            "l0_sparsity": avg_l0,
            "dead_feature_pct": dead_pct
        }

        self.history.append(epoch_metrics)
        self.current_epoch += 1
        return epoch_metrics

    def fit(self,
            dataloader: DataLoader,
            num_epochs: int = 10,
            save_every_epochs: int = 1,
            metrics_path: Optional[str] = "./outputs/sae_training_history.json") -> List[Dict]:
        """Runs complete training loop with periodic checkpointing and metric export."""
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, num_epochs):
            self.train_epoch(dataloader)

            if (epoch + 1) % save_every_epochs == 0:
                ckpt_name = f"sae_checkpoint_epoch_{epoch + 1:02d}.pt"
                self.save_checkpoint(ckpt_name)
                self.save_checkpoint("sae_checkpoint_latest.pt")

            if metrics_path:
                save_json({"history": self.history}, metrics_path)

        return self.history
