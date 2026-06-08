"""
MNIST_Demo.py
=============
TAS Research Suite — MNIST Demo Notebook Script

This script is structured as a Jupyter-compatible sequential demo.
Convert to notebook: jupytext --to notebook MNIST_Demo.py

Sections:
  1.  Imports & Setup
  2.  Load MNIST Dataset
  3.  Define CNN Model with TAS Hooks
  4.  Training Loop with Real-time TAS Monitoring
  5.  Threshold Detection & Alert
  6.  Visualisation
  7.  Statistical Summary

Run:
    cd TAS_GitHub
    python notebooks/MNIST_Demo.py
"""

# %% [1] Imports & Setup
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models_cnn import TASCNN, extract_tas_feature, ActivationBuffer
from tas_core   import compute_tas_metrics, compute_tewi, detect_precursor_state

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE     = 'cpu'
LR         = 0.05       # High LR → Sigmoid saturation → controlled collapse
BATCH_SIZE = 32
MAX_BATCH  = 200
WINDOW     = 20
ALPHA      = 2.5

print("Setup complete. Device:", DEVICE)

# %% [2] Load MNIST Dataset
transform = T.ToTensor()
dataset   = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Dataset: MNIST  |  Batches available: {len(loader)}")

# %% [3] Define CNN with TAS Hooks
model = TASCNN(num_classes=10, in_channels=1).to(DEVICE)
crit  = nn.CrossEntropyLoss()
opt   = optim.SGD(model.parameters(), lr=LR)
buf   = ActivationBuffer(window_size=WINDOW)

print(f"Model: TASCNN (Sigmoid, lr={LR})")
print(f"TAS extraction layer: fc1  |  Window: {WINDOW}  |  α: {ALPHA}")

# %% [4] Training Loop with Real-time TAS Monitoring
hp_series   = []
tas_series  = []
tewi_series = []
grad_series = []
loss_series = []
steps       = []

model.train()
print("\n--- Training Loop ---")

for bi, (x, y) in enumerate(loader):
    if bi >= MAX_BATCH:
        break

    x, y = x.to(DEVICE), y.to(DEVICE)
    opt.zero_grad()
    out  = model(x)
    loss = crit(out, y)
    loss.backward()

    # Gradient norm
    gn = sum(p.grad.norm(2).item()**2
             for p in model.parameters() if p.grad is not None) ** 0.5
    opt.step()

    # Extract activations
    act = extract_tas_feature(model, 'fc1')
    if act is None:
        continue
    buf.append(act)

    if buf.ready():
        history = np.array(buf.get())
        res     = compute_tas_metrics(history, alpha=ALPHA)
        tewi    = compute_tewi(res['hp_h0'], res['tas_h0'], res['hp_h1'],
                               w1=0.45, w2=0.25, w3=0.30)

        hp_series.append(res['hp_total'])
        tas_series.append(res['tas_total'])
        tewi_series.append(tewi)
        grad_series.append(gn)
        loss_series.append(loss.item())
        steps.append(bi)

        if (bi + 1) % 30 == 0:
            print(f"  step {bi+1:3d}  loss={loss.item():.4f}  "
                  f"grad={gn:.4f}  Hp={res['hp_total']:.3f}  TAS={res['tas_total']:.3f}")

# %% [5] Threshold Detection & Alert
def _crossing(series, n=20, sigma=3.0):
    arr = np.array(series)
    if len(arr) < n + 1: return None
    mu, sd = arr[:n].mean(), arr[:n].std()
    idx = np.where(arr > mu + sigma * sd)[0]
    return int(idx[0]) if len(idx) > 0 else None

def _gc(series, n=20):
    arr = np.array(series)
    if len(arr) < n + 1: return None
    baseline = arr[:n].mean()
    idx = np.where(arr < 0.1 * baseline)[0]
    return int(idx[0]) if len(idx) > 0 else None

t_pe  = _crossing(hp_series)
t_tas = _crossing(tas_series)
t_gc  = _gc(grad_series)
pc    = detect_precursor_state(hp_series, tas_series)

print("\n=== Alert Summary ===")
print(f"  Stage 1 — Precursor Alert (Hₚ): step {t_pe}")
print(f"  Stage 2 — Danger Alert (TAS)  : step {t_tas}")
print(f"  Gradient Collapse (t_GC)       : step {t_gc}")
if t_gc and t_pe:  print(f"  Δt_PE  = {t_gc - t_pe} steps lead time")
if t_gc and t_tas: print(f"  Δt_TAS = {t_gc - t_tas} steps lead time")
print(f"  Precursor ordering confirmed   : {pc['precursor']}")

# %% [6] Visualisation
fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

ax = axes[0]
ax.plot(steps, hp_series, color='crimson', lw=2, label='Hₚ (Persistent Entropy)')
if t_pe:  ax.axvline(steps[t_pe],  color='crimson', ls='--', label=f'Precursor Alert (t={steps[t_pe]})')
if t_gc:  ax.axvline(steps[t_gc],  color='black',   ls=':',  lw=2, label=f'Gradient Collapse (t={steps[t_gc]})')
ax.set_ylabel('Hₚ'); ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.4)
ax.set_title('Stage 1 — Persistent Entropy Hₚ (Precursor Alert)', fontweight='bold')

ax = axes[1]
ax.plot(steps, tas_series, color='steelblue', lw=2, label='TAS')
if t_tas: ax.axvline(steps[t_tas], color='steelblue', ls='--', label=f'Danger Alert (t={steps[t_tas]})')
if t_gc:  ax.axvline(steps[t_gc],  color='black', ls=':', lw=2)
ax.set_ylabel('TAS'); ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.4)
ax.set_title('Stage 2 — Topological Anomaly Score (Danger Alert)', fontweight='bold')

ax = axes[2]
ax.plot(steps, tewi_series, color='darkorange', lw=2, label='TEWI')
if t_gc:  ax.axvline(steps[t_gc], color='black', ls=':', lw=2)
ax.set_ylabel('TEWI'); ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.4)
ax.set_title('TEWI = 0.45·Hₚ(H₀) + 0.25·TAS + 0.30·Hₚ(H₁)', fontweight='bold')

ax = axes[3]
ax.plot(steps, grad_series, color='darkgreen', lw=2, label='Gradient Norm')
if t_gc:  ax.axvline(steps[t_gc], color='black', ls=':', lw=2, label=f't_GC={steps[t_gc]}')
ax.set_ylabel('‖∇L‖₂'); ax.set_xlabel('Training Batch')
ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.4)
ax.set_title('Gradient Norm (Reference — Reactive)', fontweight='bold')

plt.suptitle('TAS Demo: CNN + MNIST — Early Warning of Gradient Collapse', fontsize=12, fontweight='bold')
plt.tight_layout()
out = 'notebooks/MNIST_Demo_output.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {out}")

# %% [7] Statistical Summary
print("\n=== Statistical Summary ===")
print(f"  Hₚ max              : {max(hp_series):.4f}")
print(f"  TAS max             : {max(tas_series):.4f}")
print(f"  TEWI max            : {max(tewi_series):.4f}")
print(f"  Grad norm at t_GC   : {grad_series[t_gc]:.4f}" if t_gc else "  Grad collapse: not detected")
print(f"\nMNIST Demo complete ✓")
