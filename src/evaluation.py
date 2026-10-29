import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from src.intervention import ResidualPreservingIntervention
from src.utils import save_json


class CausalEvaluator:
    """
    Orchestrates causal evaluation experiments and control benchmarks.
    Quantifies causal efficacy of target SAE features versus randomized or unrelated
    control features across evaluation image sets.
    """
    def __init__(self,
                 vit_wrapper,
                 sae,
                 output_dir: str = "./outputs/evaluation"):
        self.vit = vit_wrapper
        self.sae = sae
        self.intervenor = ResidualPreservingIntervention(vit_wrapper, sae)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def select_control_features(self,
                                target_feature_id: int,
                                num_controls: int = 8,
                                candidate_feature_ids: Optional[List[int]] = None) -> List[int]:
        """
        Selects control features unrelated to target feature.
        Picks random or non-overlapping active features from SAE latent space.
        """
        total_sae_features = self.sae.d_sae
        if candidate_feature_ids is None:
            # Sample uniformly from full dictionary, excluding target
            all_indices = [i for i in range(total_sae_features) if i != target_feature_id]
            np.random.seed(42)
            controls = np.random.choice(all_indices, size=num_controls, replace=False).tolist()
        else:
            filtered = [i for i in candidate_feature_ids if i != target_feature_id]
            controls = filtered[:num_controls]

        return [int(c) for c in controls]

    def evaluate_feature_causality(self,
                                   target_feature_id: int,
                                   eval_dataloader,
                                   num_images: int = 50,
                                   num_control_features: int = 6,
                                   strengths: Tuple[float, ...] = (0.0, 2.0)) -> Dict:
        """
        Executes controlled causal intervention across a cohort of evaluation images.
        Compares target feature suppression/amplification against control features.
        """
        control_features = self.select_control_features(target_feature_id, num_controls=num_control_features)

        results = {
            "target_feature_id": target_feature_id,
            "control_features": control_features,
            "strengths": list(strengths),
            "target_effects": {str(s): [] for s in strengths},
            "control_effects": {str(s): [] for s in strengths},
            "summary": {}
        }

        images_tested = 0

        for batch in tqdm(eval_dataloader, desc=f"Evaluating Feature #{target_feature_id}"):
            if images_tested >= num_images:
                break

            images = batch["image"]
            B = images.shape[0]

            for b in range(B):
                if images_tested >= num_images:
                    break

                img = images[b:b+1]

                for strength in strengths:
                    # Target feature intervention
                    target_res = self.intervenor.run_intervention(img, target_feature_id, strength_multiplier=strength)
                    results["target_effects"][str(strength)].append(target_res)

                    # Control features intervention
                    ctrl_log_diffs = []
                    ctrl_prob_diffs = []
                    ctrl_flips = []

                    for ctrl_id in control_features:
                        c_res = self.intervenor.run_intervention(img, ctrl_id, strength_multiplier=strength)
                        ctrl_log_diffs.append(c_res["logit_diff"])
                        ctrl_prob_diffs.append(c_res["prob_diff"])
                        ctrl_flips.append(c_res["prediction_flipped"])

                    results["control_effects"][str(strength)].append({
                        "mean_logit_diff": float(np.mean(ctrl_log_diffs)),
                        "mean_prob_diff": float(np.mean(ctrl_prob_diffs)),
                        "any_flip": any(ctrl_flips)
                    })

                images_tested += 1

        # Calculate summary statistics
        for s in strengths:
            s_str = str(s)
            tgt_list = results["target_effects"][s_str]
            ctrl_list = results["control_effects"][s_str]

            tgt_prob_diffs = [r["prob_diff"] for r in tgt_list]
            tgt_logit_diffs = [r["logit_diff"] for r in tgt_list]
            tgt_flips = sum(1 for r in tgt_list if r["prediction_flipped"])

            ctrl_prob_diffs = [r["mean_prob_diff"] for r in ctrl_list]
            ctrl_logit_diffs = [r["mean_logit_diff"] for r in ctrl_list]
            ctrl_flips = sum(1 for r in ctrl_list if r["any_flip"])

            n = len(tgt_list)
            results["summary"][s_str] = {
                "target_mean_prob_diff": float(np.mean(tgt_prob_diffs)),
                "target_std_prob_diff": float(np.std(tgt_prob_diffs)),
                "target_mean_logit_diff": float(np.mean(tgt_logit_diffs)),
                "target_flip_rate": float(tgt_flips / max(n, 1)),
                "control_mean_prob_diff": float(np.mean(ctrl_prob_diffs)),
                "control_std_prob_diff": float(np.std(ctrl_prob_diffs)),
                "control_mean_logit_diff": float(np.mean(ctrl_logit_diffs)),
                "control_flip_rate": float(ctrl_flips / max(n, 1)),
                "effect_size_diff": float(np.mean(tgt_prob_diffs) - np.mean(ctrl_prob_diffs))
            }

        # Save results to JSON
        json_path = os.path.join(self.output_dir, f"causal_eval_feature_{target_feature_id:04d}.json")
        save_json(results, json_path)

        # Generate summary plot
        self.plot_intervention_comparison(results, target_feature_id)

        return results

    def plot_intervention_comparison(self, results: Dict, feature_id: int) -> str:
        """Generates scientific bar chart comparing Target vs Control effects."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        strengths = results["strengths"]
        categories = [f"Suppression ({strengths[0]}x)", f"Amplification ({strengths[1]}x)"]

        # 1. Probability shift comparison
        tgt_prob_shifts = [results["summary"][str(s)]["target_mean_prob_diff"] for s in strengths]
        ctrl_prob_shifts = [results["summary"][str(s)]["control_mean_prob_diff"] for s in strengths]

        x = np.arange(len(categories))
        width = 0.35

        axes[0].bar(x - width/2, tgt_prob_shifts, width, label=f"Target Feature #{feature_id}", color="#1f77b4")
        axes[0].bar(x + width/2, ctrl_prob_shifts, width, label="Control Features (Avg)", color="#7f7f7f")
        axes[0].set_ylabel("Mean Target Class Probability Shift")
        axes[0].set_title("Probability Change (Target vs Control)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[0].legend()
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)

        # 2. Prediction flip rate comparison
        tgt_flips = [results["summary"][str(s)]["target_flip_rate"] * 100 for s in strengths]
        ctrl_flips = [results["summary"][str(s)]["control_flip_rate"] * 100 for s in strengths]

        axes[1].bar(x - width/2, tgt_flips, width, label=f"Target Feature #{feature_id}", color="#d62728")
        axes[1].bar(x + width/2, ctrl_flips, width, label="Control Features", color="#c7c7c7")
        axes[1].set_ylabel("Prediction Flip Rate (%)")
        axes[1].set_title("Top-1 Prediction Flip Rate")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].legend()
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"causal_comparison_feature_{feature_id:04d}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        return plot_path
