"""
mnist_cnn_demo.py
=================
TAS Research Suite — CNN + MNIST Early Warning Demo

Trains a Sigmoid CNN on MNIST with a deliberately high learning rate
to induce gradient collapse, monitoring TAS and Hₚ in real time.

Run:
    cd TAS_GitHub
    python examples/mnist_cnn_demo.py

Requirements: torch, torchvision, gudhi, numpy, matplotlib
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models_cnn import TASCNN, extract_tas_feature, ActivationBuffer
from tas_core   import compute_tas_metrics, compute_tewi, detect_precursor_state


def get_mnist_loader(batch_size=32, root='./data'):
    ds = torchvision.datasets.MNIST(
        root=root, train=True, download=True,
        transform=T.ToTensor()
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _first_crossing(series, n_baseline=20, sigma=3.0):
    arr = np.array(series)
    if len(arr) < n_baseline + 1:
        return None
    mu, sd = arr[:n_baseline].mean(), arr[:n_baseline].std()
    idx = np.where(arr > mu + sigma * sd)[0]
    return int(idx[0]) if len(idx) > 0 else None


def _grad_collapse(series, n_baseline=20):
    arr = np.array(series)
    if len(arr) < n_baseline + 1:
        return None
    baseline = arr[:n_baseline].mean()
    idx = np.where(arr < 0.1 * baseline)[0]
    return int(idx[0]) if len(idx) > 0 else None


def main():
    DEVICE     = 'cpu'
    LR         = 0.05          # High LR to induce Sigmoid saturation
    MAX_BATCH  = 150
    WINDOW     = 20
    ALPHA      = 2.5

    print('=' * 55)
    print('  TAS Demo: CNN + MNIST (Gradient Collapse)')
    print('=' * 55)

    model  = TASCNN(num_classes=10, in_channels=1).to(DEVICE)
    loader = get_mnist_loader()
    crit   = nn.CrossEntropyLoss()
    opt    = optim.SGD(model.parameters(), lr=LR)
    buf    = ActivationBuffer(window_size=WINDOW)

    hp_list, tas_list, grad_list, loss_list = [], [], [], []
    model.train()

    for bi, (x, y) in enumerate(loader):
        if bi >= MAX_BATCH:
            break

        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out  = model(x)
        loss = crit(out, y)
        loss.backward()

        gn = sum(p.grad.norm(2).item() ** 2
                 for p in model.parameters() if p.grad is not None) ** 0.5
        opt.step()

        act = extract_tas_feature(model, 'fc1')
        if act is None:
            continue
        buf.append(act)

        if buf.ready():
            history = np.array(buf.get())
            res  = compute_tas_metrics(history, alpha=ALPHA)
            tewi = compute_tewi(res['hp_h0'], res['tas_h0'], res['hp_h1'],
                                w1=0.45, w2=0.25, w3=0.30)
            hp_list.append(res['hp_total'])
            tas_list.append(res['tas_total'])
            grad_list.append(gn)
            loss_list.append(loss.item())

        if (bi + 1) % 20 == 0:
            print(f'  batch {bi+1:3d}/{MAX_BATCH}  '
                  f'loss={loss.item():.4f}  grad={gn:.4f}  '
                  f'Hp={hp_list[-1]:.3f}  TAS={tas_list[-1]:.3f}')

    # Crossings
    t_pe  = _first_crossing(hp_list)
    t_tas = _first_crossing(tas_list)
    t_gc  = _grad_collapse(grad_list)
    pc    = detect_precursor_state(hp_list, tas_list)

    print(f'\n=== Results ===')
    print(f'  t_PE  = {t_pe}   (Hₚ alert step)')
    print(f'  t_TAS = {t_tas}  (TAS alert step)')
    print(f'  t_GC  = {t_gc}   (gradient collapse step)')
    if t_gc and t_pe:
        print(f'  Δt_PE  = {t_gc - t_pe} steps lead time')
    if t_gc and t_tas:
        print(f'  Δt_TAS = {t_gc - t_tas} steps lead time')
    print(f'  Precursor state confirmed: {pc["precursor"]}')

    # Plot
    steps = list(range(len(hp_list)))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(steps, hp_list, color='crimson', lw=2)
    axes[0].set_ylabel('Hₚ')
    axes[0].set_title('Persistent Entropy Hₚ (Precursor Alert)', fontweight='bold')
    if t_pe:  axes[0].axvline(t_pe,  color='crimson', ls='--', label=f't_PE={t_pe}')
    if t_gc:  axes[0].axvline(t_gc,  color='black',   ls=':',  label=f't_GC={t_gc}')
    axes[0].legend(); axes[0].grid(ls='--', alpha=0.5)

    axes[1].plot(steps, tas_list, color='steelblue', lw=2)
    axes[1].set_ylabel('TAS')
    axes[1].set_title('Topological Anomaly Score (Danger Alert)', fontweight='bold')
    if t_tas: axes[1].axvline(t_tas, color='steelblue', ls='--', label=f't_TAS={t_tas}')
    if t_gc:  axes[1].axvline(t_gc,  color='black',     ls=':',  label=f't_GC={t_gc}')
    axes[1].legend(); axes[1].grid(ls='--', alpha=0.5)

    axes[2].plot(steps, grad_list, color='darkgreen', lw=2)
    axes[2].set_ylabel('Grad Norm')
    axes[2].set_xlabel('Sliding Window Step')
    axes[2].set_title('Gradient Norm (Reference)', fontweight='bold')
    if t_gc:  axes[2].axvline(t_gc, color='black', ls=':', label=f't_GC={t_gc}')
    axes[2].legend(); axes[2].grid(ls='--', alpha=0.5)

    plt.suptitle('TAS Demo: CNN + MNIST Early Warning of Gradient Collapse', fontsize=12)
    plt.tight_layout()
    out = 'examples/mnist_cnn_demo_output.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'\nPlot saved: {out}  ✓')


if __name__ == '__main__':
    main()
