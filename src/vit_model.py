import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable


class ViTHookManager:
    """
    Manages PyTorch forward hooks on ViT transformer blocks for both:
    1. Read-only activation extraction
    2. Causal intervention injection (replacing/modifying residual stream activations)
    """
    def __init__(self):
        self.hook_handle = None
        self.captured_activation = None
        self.intervention_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def register(self, module: nn.Module):
        """Register forward hook on the target transformer block."""
        self.remove()

        def hook(mod, inputs, output):
            # Output of a transformer block is typically a tensor of shape [B, N, D]
            # or tuple (x,) depending on implementation.
            if isinstance(output, tuple):
                act = output[0]
                is_tuple = True
            else:
                act = output
                is_tuple = False

            # Cache the pristine activation
            self.captured_activation = act.clone()

            # If a causal intervention function is attached, execute it
            if self.intervention_fn is not None:
                modified_act = self.intervention_fn(act)
                if is_tuple:
                    return (modified_act,) + output[1:]
                return modified_act
            return output

        self.hook_handle = module.register_forward_hook(hook)

    def set_intervention(self, fn: Optional[Callable[[torch.Tensor], torch.Tensor]]):
        """Set or clear the causal intervention function."""
        self.intervention_fn = fn

    def remove(self):
        """Remove active hook handle."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        self.captured_activation = None
        self.intervention_fn = None


class ViTModelWrapper(nn.Module):
    """
    Wrapper around pretrained Vision Transformer (ViT-B/16).
    Freezes model weights, extracts logits, probabilities, and intermediate activations,
    and supports causal feature intervention via residual stream hooks.
    """
    def __init__(self,
                 model_name: str = "vit_base_patch16_224",
                 pretrained: bool = True,
                 target_layer_idx: int = 7,
                 device: torch.device = torch.device("cpu"),
                 use_torchvision_fallback: bool = True):
        super().__init__()
        self.model_name = model_name
        self.target_layer_idx = target_layer_idx
        self.device = device
        self.hook_manager = ViTHookManager()
        self.d_model = 768
        self.num_patches = 196
        self.patch_size = 16

        self.model = self._load_model(model_name, pretrained, use_torchvision_fallback)
        self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self._attach_hook()

    def _load_model(self, model_name: str, pretrained: bool, use_torchvision: bool):
        """Loads ViT model from torchvision or timm with reliable fallback."""
        # Prefer torchvision when requested or for standard ViT-B/16 checkpoints
        if model_name.lower() in ["vit_b_16", "torchvision", "vit_base_patch16_224"]:
            if use_torchvision:
                try:
                    import torchvision.models as models
                    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
                    model = models.vit_b_16(weights=weights)
                    self.model_type = "torchvision"
                    return model
                except Exception:
                    pass

        try:
            import timm
            model = timm.create_model(model_name, pretrained=pretrained)
            self.model_type = "timm"
            return model
        except Exception as e:
            if use_torchvision:
                try:
                    import torchvision.models as models
                    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
                    model = models.vit_b_16(weights=weights)
                    self.model_type = "torchvision"
                    return model
                except Exception:
                    return self._create_mock_vit()
            raise e

    def _create_mock_vit(self):
        """Creates a mock ViT architecture matching ViT-B/16 dimensions for offline testing."""
        self.model_type = "mock"

        class MockBlock(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )

            def forward(self, x):
                return x + self.mlp(self.norm(x))

        class MockViT(nn.Module):
            def __init__(self, d_model=768, num_classes=1000, num_blocks=12):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
                self.pos_embed = nn.Parameter(torch.zeros(1, 197, d_model))
                self.blocks = nn.ModuleList([MockBlock(d_model) for _ in range(num_blocks)])
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, num_classes)

            def forward(self, x):
                B = x.shape[0]
                x = self.patch_embed(x).flatten(2).transpose(1, 2)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                cls_out = x[:, 0]
                return self.head(cls_out)

        return MockViT()

    def _get_target_block(self) -> nn.Module:
        """Retrieve target transformer block module based on architecture."""
        if self.model_type == "timm":
            # In timm ViT: model.blocks[i]
            return self.model.blocks[self.target_layer_idx]
        elif self.model_type == "torchvision":
            # In torchvision ViT: model.encoder.layers[f'encoder_layer_{i}']
            return self.model.encoder.layers[self.target_layer_idx]
        elif self.model_type == "mock":
            return self.model.blocks[self.target_layer_idx]
        else:
            raise ValueError(f"Unknown model type {self.model_type}")

    def _attach_hook(self):
        """Attaches the forward hook to target layer."""
        target_block = self._get_target_block()
        self.hook_manager.register(target_block)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes forward pass.
        Returns:
            logits: [B, num_classes]
            probabilities: [B, num_classes]
            patch_activations: [B, 196, d_model] (CLS excluded)
            cls_activation: [B, 1, d_model]
        """
        x = x.to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)

        raw_activation = self.hook_manager.captured_activation
        if raw_activation is None:
            raise RuntimeError("Activation hook did not capture any output.")

        # ViT sequence layout: token 0 is CLS token, tokens 1..196 are spatial patch tokens
        cls_activation = raw_activation[:, 0:1, :]
        patch_activations = raw_activation[:, 1:, :]

        assert patch_activations.shape[1] == self.num_patches, (
            f"Expected {self.num_patches} patches, got {patch_activations.shape[1]}"
        )
        assert patch_activations.shape[2] == self.d_model, (
            f"Expected hidden dimension {self.d_model}, got {patch_activations.shape[2]}"
        )

        return logits, probs, patch_activations, cls_activation

    def set_causal_intervention(self, intervention_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]):
        """Attach causal intervention hook."""
        self.hook_manager.set_intervention(intervention_fn)

    def clear_causal_intervention(self):
        """Clear causal intervention hook."""
        self.hook_manager.set_intervention(None)
