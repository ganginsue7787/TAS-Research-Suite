"""
models_resnet.py
================
TAS Journal Suite — ResNet18 Activation Extractor

Hooks into layer4 (abstract residual features, 512-D)
and avgpool for TAS analysis.

Recommended extraction point: avgpool (512-D, richest
topological information without over-compression).

Author  : Kang, In-Su
License : MIT
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TASResNet18(nn.Module):

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        self._activations = {}

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        self.backbone.layer4.register_forward_hook(self._hook("layer4"))
        self.backbone.avgpool.register_forward_hook(self._hook("avgpool"))

    def _hook(self, name: str):
        def h(module, inp, out):
            self._activations[name] = out.detach().cpu().numpy()
        return h

    def forward(self, x):
        self._activations.clear()
        return self.backbone(x)

    def get_activation(self, layer: str = "avgpool") -> np.ndarray:
        return self._activations.get(layer)


def extract_tas_feature(model: TASResNet18,
                        layer: str = "avgpool") -> np.ndarray:
    act = model.get_activation(layer)
    if act is None:
        return None
    return act.reshape(act.shape[0], -1)


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
