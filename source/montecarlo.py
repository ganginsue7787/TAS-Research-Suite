"""
montecarlo.py
=============
TAS Journal Suite — Monte Carlo Experiment Engine

Runs grid:
    {CNN, ResNet18, ViT}
  x {MNIST, FashionMNIST, CIFAR10, CIFAR100}
  x alpha in {1.0, 2.0, 2.5, 3.0, 5.0}
  x n_runs >= 50 Monte Carlo repetitions

Outputs montecarlo_results.csv with columns:
    run, model, dataset, alpha,
    delta_tas, delta_hp, t_gc, t_tas, t_hp

Author  : Kang, In-Su
License : MIT
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from tas_core import (compute_tas_metrics, compute_collapse_rate,
                      detect_precursor_state)


# ── Dataset Loader ────────────────────────────────────────────

_DATASETS = {
    "MNIST":        (torchvision.datasets.MNIST,        T.Compose([T.ToTensor()])),
    "FashionMNIST": (torchvision.datasets.FashionMNIST, T.Compose([T.ToTensor()])),
    "CIFAR10":      (torchvision.datasets.CIFAR10,      T.Compose([T.ToTensor(),
                        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])),
    "CIFAR100":     (torchvision.datasets.CIFAR100,     T.Compose([T.ToTensor(),
                        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])),
}

def get_loader(dataset_name: str, batch_size: int = 32,
               root: str = "./data") -> DataLoader:
    cls, tfm = _DATASETS[dataset_name]
    ds = cls(root=root, train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ── Model Factory ─────────────────────────────────────────────

def build_model(model_name: str, dataset_name: str) -> nn.Module:
    n_cls = 100 if dataset_name == "CIFAR100" else 10
    in_ch = 1 if dataset_name in ("MNIST", "FashionMNIST") else 3

    from models_cnn import TASCNN
    from models_resnet import TASResNet18
    from models_transformer import TASViT

    if model_name == "CNN":
        return TASCNN(num_classes=n_cls, in_channels=in_ch)
    elif model_name == "ResNet18":
        m = TASResNet18(num_classes=n_cls)
        if in_ch == 1:
            m.backbone.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        return m
    elif model_name == "ViT":
        return TASViT(num_classes=n_cls, image_size=32)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Extractor ─────────────────────────────────────────────────

def extract_features(model, model_name: str) -> np.ndarray:
    if model_name == "CNN":
        from models_cnn import extract_tas_feature
        return extract_tas_feature(model, "fc1")
    elif model_name == "ResNet18":
        from models_resnet import extract_tas_feature
        return extract_tas_feature(model, "avgpool")
    elif model_name == "ViT":
        from models_transformer import extract_cls_token
        return extract_cls_token(model)
    return None


# ── Early Warning Detection ───────────────────────────────────

def _first_threshold_crossing(series: list, sigma: float = 3.0) -> int:
    arr = np.array(series)
    if len(arr) < 20:
        return None
    mu, sd = arr[:20].mean(), arr[:20].std()
    thr = mu + sigma * sd
    idx = np.where(arr > thr)[0]
    return int(idx[0]) if len(idx) > 0 else None


def _gradient_collapse_step(grad_series: list) -> int:
    arr = np.array(grad_series)
    if len(arr) < 20:
        return None
    baseline = arr[:20].mean()
    idx = np.where(arr < 0.1 * baseline)[0]
    return int(idx[0]) if len(idx) > 0 else None


# ── Single Run ────────────────────────────────────────────────

def run_single(model_name: str, dataset_name: str,
               alpha: float, window_size: int = 20,
               max_batches: int = 200,
               device: str = "cpu") -> dict:

    model  = build_model(model_name, dataset_name).to(device)
    loader = get_loader(dataset_name)
    crit   = nn.CrossEntropyLoss()
    opt    = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    hp_series, tas_series, grad_series = [], [], []
    buf = []
    model.train()

    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out  = model(x)
        loss = crit(out, y)
        loss.backward()

        # Gradient norm
        gn = sum(p.grad.norm(2).item() ** 2
                 for p in model.parameters()
                 if p.grad is not None) ** 0.5
        opt.step()

        act = extract_features(model, model_name)
        if act is None:
            continue
        buf.append(act.mean(axis=0))

        if len(buf) >= window_size:
            history = np.array(buf[-window_size:])
            res = compute_tas_metrics(history, alpha=alpha)
            hp_series.append(res["hp_total"])
            tas_series.append(res["tas_total"])
            grad_series.append(gn)

    t_hp  = _first_threshold_crossing(hp_series)
    t_tas = _first_threshold_crossing(tas_series)
    t_gc  = _gradient_collapse_step(grad_series)

    def _delta(t_a, t_b):
        return (t_b - t_a) if (t_a is not None and t_b is not None) else np.nan

    return {
        "t_hp": t_hp, "t_tas": t_tas, "t_gc": t_gc,
        "delta_hp":  _delta(t_hp,  t_gc),
        "delta_tas": _delta(t_tas, t_gc),
        "hp_series":  hp_series,
        "tas_series": tas_series,
        "grad_series": grad_series,
    }


# ── Monte Carlo Engine ────────────────────────────────────────

class MonteCarloEngine:

    def __init__(self,
                 models:   list = None,
                 datasets: list = None,
                 alphas:   list = None,
                 n_runs:   int  = 50,
                 device:   str  = "cpu"):
        self.models   = models   or ["CNN", "ResNet18", "ViT"]
        self.datasets = datasets or ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]
        self.alphas   = alphas   or [1.0, 2.0, 2.5, 3.0, 5.0]
        self.n_runs   = n_runs
        self.device   = device

    def run(self, output_csv: str = "montecarlo_results.csv") -> pd.DataFrame:
        records = []
        total = len(self.models) * len(self.datasets) * len(self.alphas) * self.n_runs
        done  = 0

        for model_name in self.models:
            for dataset_name in self.datasets:
                for alpha in self.alphas:
                    for run in range(self.n_runs):
                        done += 1
                        print(f"[{done}/{total}] {model_name} | {dataset_name} "
                              f"| alpha={alpha} | run={run+1}")
                        try:
                            result = run_single(model_name, dataset_name,
                                                alpha, device=self.device)
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            result = {"t_hp": None, "t_tas": None, "t_gc": None,
                                      "delta_hp": np.nan, "delta_tas": np.nan}

                        records.append({
                            "run": run, "model": model_name,
                            "dataset": dataset_name, "alpha": alpha,
                            "delta_tas": result["delta_tas"],
                            "delta_hp":  result["delta_hp"],
                            "t_gc":  result["t_gc"],
                            "t_tas": result["t_tas"],
                            "t_hp":  result["t_hp"],
                        })

        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(df)} records to {output_csv}")
        return df


# ── Quick Run (single model) ──────────────────────────────────

if __name__ == "__main__":
    engine = MonteCarloEngine(
        models=["CNN"], datasets=["MNIST"], alphas=[2.5], n_runs=3
    )
    df = engine.run("montecarlo_quick.csv")
    print(df)
