"""
models_cnn.py
=============
TAS Journal Suite — CNN Activation Extractor

Lightweight CNN for MNIST / Fashion-MNIST with
forward hooks to extract intermediate activations
for TAS analysis.

Author  : Kang, In-Su
License : MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CNN Model ─────────────────────────────────────────────────

class TASCNN(nn.Module):
    """
    3-layer CNN with hook-based activation extraction.
    Suitable for MNIST / Fashion-MNIST (1-channel, 28x28).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self._activations = {}

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 256)
        self.fc2   = nn.Linear(256, num_classes)
        self.drop  = nn.Dropout(0.25)

        # Register hooks
        self.fc1.register_forward_hook(self._make_hook("fc1"))
        self.conv2.register_forward_hook(self._make_hook("conv2"))

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._activations[name] = (
                output.detach().cpu().numpy()
            )
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._activations.clear()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.drop(x)))
        return self.fc2(x)

    def get_activation(self, layer: str) -> np.ndarray:
        return self._activations.get(layer, None)


# ── Feature Extraction ────────────────────────────────────────

def extract_tas_feature(model: TASCNN,
                        layer: str = "fc1") -> np.ndarray:
    """Return batch activation for the specified layer."""
    act = model.get_activation(layer)
    if act is None:
        return None
    if act.ndim > 2:
        act = act.reshape(act.shape[0], -1)
    return act


# ── Sliding Window Buffer ─────────────────────────────────────

class ActivationBuffer:
    """Collects per-batch mean activations into a sliding window."""

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
