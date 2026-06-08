"""
ResNet18_Experiment.py
======================
TAS Research Suite — ResNet18 + CIFAR-10 Experiment

Demonstrates TAS monitoring on ResNet18 (avgpool, 512-D)
with a realistic collapse scenario: cosine LR without warmup.

Run:  python notebooks/ResNet18_Experiment.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models_resnet import TASResNet18, extract_tas_feature, ActivationBuffer
from tas_core import compute_tas_metrics, compute_tewi

DEVICE, LR, MAX_BATCH, WINDOW, ALPHA = 'cpu', 0.1, 150, 20, 2.5
transform = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
ds     = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
model  = TASResNet18(num_classes=10).to(DEVICE)
# 1-channel patch for CIFAR (not needed — CIFAR is 3-ch)
crit   = nn.CrossEntropyLoss()
opt    = optim.SGD(model.parameters(), lr=LR)
sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_BATCH)
buf    = ActivationBuffer(window_size=WINDOW)

hp_s, tas_s, grad_s = [], [], []
model.train()
print("ResNet18 + CIFAR-10 (cosine LR, no warmup)")
for bi, (x, y) in enumerate(loader):
    if bi >= MAX_BATCH: break
    x, y = x.to(DEVICE), y.to(DEVICE)
    opt.zero_grad()
    out  = model(x)
    loss = crit(out, y)
    loss.backward()
    gn   = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
    opt.step(); sched.step()
    act = extract_tas_feature(model, 'avgpool')
    if act is None: continue
    buf.append(act)
    if buf.ready():
        res = compute_tas_metrics(np.array(buf.get()), alpha=ALPHA)
        hp_s.append(res['hp_total']); tas_s.append(res['tas_total']); grad_s.append(gn)
    if (bi+1) % 30 == 0:
        print(f"  batch {bi+1}  grad={gn:.4f}  Hp={hp_s[-1]:.3f}  TAS={tas_s[-1]:.3f}")

steps = list(range(len(hp_s)))
fig, ax = plt.subplots(2,1,figsize=(10,6),sharex=True)
ax[0].plot(steps, hp_s,   color='crimson',  lw=2, label='Hₚ'); ax[0].set_ylabel('Hₚ')
ax[0].plot(steps, tas_s,  color='steelblue',lw=2, label='TAS'); ax[0].legend(); ax[0].grid(ls='--',alpha=0.4)
ax[0].set_title('ResNet18 avgpool — Hₚ & TAS (CIFAR-10, cosine LR)', fontweight='bold')
ax[1].plot(steps, grad_s, color='darkgreen',lw=2, label='Grad Norm'); ax[1].set_ylabel('‖∇L‖₂')
ax[1].set_xlabel('Sliding Window Step'); ax[1].legend(); ax[1].grid(ls='--',alpha=0.4)
ax[1].set_title('Gradient Norm', fontweight='bold')
plt.tight_layout()
out = 'notebooks/ResNet18_Experiment_output.png'
plt.savefig(out, dpi=150); plt.close()
print(f"Saved {out}  ✓")
