"""
models_transformer.py
=====================
TAS Journal Suite — Vision Transformer (ViT) Activation Extractor

Extracts CLS token, mean token, and full hidden states from the
last Transformer encoder layer for TAS analysis.

Author  : Kang, In-Su
License : MIT
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class TASViT(nn.Module):

    def __init__(self, num_classes: int = 10, pretrained: bool = False,
                 image_size: int = 224):
        super().__init__()
        self._activations = {}

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights, image_size=image_size)
        in_feats = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_feats, num_classes)

        # Hook last encoder layer
        self.backbone.encoder.layers[-1].register_forward_hook(
            self._hook("last_encoder")
        )

    def _hook(self, name: str):
        def h(module, inp, out):
            self._activations[name] = out.detach().cpu().numpy()
        return h

    def forward(self, x):
        self._activations.clear()
        return self.backbone(x)

    def get_hidden(self) -> np.ndarray:
        """Full hidden states: shape (B, num_tokens, 768)."""
        return self._activations.get("last_encoder")


def extract_cls_token(model: TASViT) -> np.ndarray:
    """CLS token: shape (B, 768) — primary TAS extraction point."""
    h = model.get_hidden()
    return h[:, 0, :] if h is not None else None


def extract_mean_token(model: TASViT) -> np.ndarray:
    """Mean over all tokens: shape (B, 768)."""
    h = model.get_hidden()
    return h.mean(axis=1) if h is not None else None


class ActivationBuffer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._buf = []

    def append(self, activation: np.ndarray):
        self._buf.append(activation.mean(axis=0))

    def ready(self) -> bool:
        return len(self._buf) >= self.window_size

    def get(self) -> list:
        return self._buf[-self.window_size:]

    def clear(self):
        self._buf.clear()
