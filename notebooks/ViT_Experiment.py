"""
ViT_Experiment.py
=================
TAS Research Suite — Vision Transformer + MNIST Demo

Extracts CLS token (768-D) from last ViT encoder layer.
Note: ViT-B/16 requires image_size=224 by default; we use image_size=32
      for speed in this demo (not pretrained).

Run:  python notebooks/ViT_Experiment.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models_transformer import TASViT, extract_cls_token, ActivationBuffer
from tas_core import compute_tas_metrics, compute_tewi

DEVICE, LR, MAX_BATCH, WINDOW, ALPHA = 'cpu', 0.01, 80, 20, 2.5
transform = T.Compose([T.Resize(32), T.ToTensor()])
ds     = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

# ViT expects 3-channel input; convert 1-ch MNIST
class GrayTo3Ch(nn.Module):
    def forward(self, x): return x.repeat(1,3,1,1)

model    = TASViT(num_classes=10, image_size=32).to(DEVICE)
gray2rgb = GrayTo3Ch()
crit     = nn.CrossEntropyLoss()
opt      = optim.AdamW(model.parameters(), lr=LR)
buf      = ActivationBuffer(window_size=WINDOW)

hp_s, tas_s, grad_s = [], [], []
model.train()
print("ViT-B/16 (32px, random init) + MNIST — CLS token TAS")
for bi, (x, y) in enumerate(loader):
    if bi >= MAX_BATCH: break
    x  = gray2rgb(x).to(DEVICE); y = y.to(DEVICE)
    opt.zero_grad()
    out  = model(x)
    loss = crit(out, y)
    loss.backward()
    gn   = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
    opt.step()
    cls = extract_cls_token(model)
    if cls is None: continue
    buf.append(cls)
    if buf.ready():
        res = compute_tas_metrics(np.array(buf.get()), alpha=ALPHA)
        hp_s.append(res['hp_total']); tas_s.append(res['tas_total']); grad_s.append(gn)
    if (bi+1) % 20 == 0:
        print(f"  batch {bi+1}  grad={gn:.4f}  Hp={hp_s[-1]:.3f}  TAS={tas_s[-1]:.3f}")

steps = list(range(len(hp_s)))
fig, ax = plt.subplots(2,1,figsize=(10,6),sharex=True)
ax[0].plot(steps, hp_s,  color='crimson',  lw=2, label='Hₚ')
ax[0].plot(steps, tas_s, color='steelblue',lw=2, label='TAS')
ax[0].set_ylabel('Value'); ax[0].legend(); ax[0].grid(ls='--',alpha=0.4)
ax[0].set_title('ViT CLS Token — Hₚ & TAS', fontweight='bold')
ax[1].plot(steps, grad_s, color='darkgreen', lw=2)
ax[1].set_ylabel('‖∇L‖₂'); ax[1].set_xlabel('Step')
ax[1].set_title('Gradient Norm', fontweight='bold'); ax[1].grid(ls='--',alpha=0.4)
plt.suptitle('ViT + MNIST — TAS CLS Token Monitoring', fontsize=11, fontweight='bold')
plt.tight_layout()
out = 'notebooks/ViT_Experiment_output.png'
plt.savefig(out, dpi=150); plt.close()
print(f"Saved {out}  ✓")
