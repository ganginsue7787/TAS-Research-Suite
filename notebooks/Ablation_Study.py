"""
Ablation_Study.py
=================
TAS Research Suite — Ablation Study Notebook

Systematically evaluates:
  1. Effect of anisotropic weight α  (α ∈ {1.0, 2.0, 2.5, 3.0, 5.0})
  2. Effect of TEWI weight combinations
  3. Effect of window size W          (W ∈ {10, 15, 20, 30})

Run:
    cd TAS_GitHub
    python notebooks/Ablation_Study.py

Output: notebooks/Ablation_Study_output.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tas_core import (
    build_state_space,
    anisotropic_distance_matrix,
    compute_persistence,
    persistent_entropy,
    topological_anomaly_score,
    compute_tewi,
)


# ── Synthetic trajectory generator ───────────────────────────

def make_trajectory(T=60, D=32, collapse_at=45, seed=42):
    rng = np.random.default_rng(seed)
    stable   = rng.normal(0, 0.08, (collapse_at, D))
    drift    = np.linspace(0, 1, T - collapse_at)[:, None]
    unstable = rng.normal(0, 0.5, (T - collapse_at, D)) * (1 + drift * 4)
    return np.vstack([stable, unstable]).astype(np.float32)


def compute_series(trajectory, alpha=2.5, window=20):
    """Return Hₚ and TAS series from sliding window."""
    hp_s, tas_s = [], []
    for t in range(window, len(trajectory)):
        win = trajectory[t - window:t]
        ss  = build_state_space(win)
        D   = anisotropic_distance_matrix(ss, alpha=alpha)
        st  = compute_persistence(D)

        from tas_core import extract_lifetimes
        L0  = extract_lifetimes(st.persistence_intervals_in_dimension(0))
        hp_s.append(persistent_entropy(L0))
        tas_s.append(topological_anomaly_score(L0))
    return hp_s, tas_s


def first_cross(series, n=20, sigma=3.0):
    arr = np.array(series)
    if len(arr) < n + 1: return None
    mu, sd = arr[:n].mean(), arr[:n].std()
    idx = np.where(arr > mu + sigma * sd)[0]
    return int(idx[0]) if len(idx) > 0 else None


# ── Run ablation ─────────────────────────────────────────────

traj       = make_trajectory(T=60, D=32, collapse_at=45)
ALPHAS     = [1.0, 2.0, 2.5, 3.0, 5.0]
WINDOWS    = [10, 15, 20, 30]
TEWI_OPTS  = {
    'Equal (1/3)':          (1/3, 1/3, 1/3),
    'Optimised':            (0.45, 0.25, 0.30),
    'Hₚ(H₀) only':         (1.0, 0.0, 0.0),
    'TAS only':             (0.0, 1.0, 0.0),
}

print("=== Ablation Study ===\n")

# ── α ablation ────────────────────────────────────────────────
print("1. α ablation (W=20)")
alpha_results = {}
for alpha in ALPHAS:
    hp_s, tas_s = compute_series(traj, alpha=alpha, window=20)
    t_pe  = first_cross(hp_s)
    t_tas = first_cross(tas_s)
    dt_pe  = (45 - 20 - t_pe)  if t_pe  is not None else None
    dt_tas = (45 - 20 - t_tas) if t_tas is not None else None
    alpha_results[alpha] = {'hp': hp_s, 'tas': tas_s, 'dt_pe': dt_pe, 'dt_tas': dt_tas}
    print(f"  α={alpha:.1f}  Δt_PE={dt_pe}  Δt_TAS={dt_tas}")

# ── Window ablation ───────────────────────────────────────────
print("\n2. Window size ablation (α=2.5)")
window_results = {}
for W in WINDOWS:
    if W >= len(traj) - 5: continue
    hp_s, tas_s = compute_series(traj, alpha=2.5, window=W)
    t_pe  = first_cross(hp_s)
    dt_pe = (45 - W - t_pe) if t_pe is not None else None
    window_results[W] = {'hp': hp_s, 'dt_pe': dt_pe}
    print(f"  W={W:2d}  Δt_PE={dt_pe}")

# ── TEWI weight ablation ──────────────────────────────────────
print("\n3. TEWI weight ablation (α=2.5, W=20)")
hp_s, tas_s = compute_series(traj, alpha=2.5, window=20)
# Simulate H1 as slightly earlier version of Hp
hp1_s = [h * 0.9 for h in hp_s]
tewi_variants = {}
for name, (w1, w2, w3) in TEWI_OPTS.items():
    series = [compute_tewi(h, t, h1, w1, w2, w3)
              for h, t, h1 in zip(hp_s, tas_s, hp1_s)]
    t_cross = first_cross(series)
    dt = (45 - 20 - t_cross) if t_cross is not None else None
    tewi_variants[name] = {'series': series, 'dt': dt}
    print(f"  {name:20s}  Δt={dt}")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: α effect on Hₚ
ax = axes[0]
colors = ['#aec6cf', '#6baed6', '#2171b5', '#08519c', '#08306b']
for (alpha, res), col in zip(alpha_results.items(), colors):
    ax.plot(res['hp'], color=col, lw=2, label=f'α={alpha}')
ax.axvline(45 - 20, color='black', ls=':', lw=2, label='Collapse')
ax.set_title('Effect of α on Persistent Entropy Hₚ', fontweight='bold')
ax.set_xlabel('Sliding Window Step')
ax.set_ylabel('Hₚ')
ax.legend(fontsize=9)
ax.grid(ls='--', alpha=0.4)

# Panel 2: Window size effect
ax = axes[1]
colors2 = ['#fd8d3c', '#e6550d', '#a63603', '#7f2704']
for (W, res), col in zip(window_results.items(), colors2):
    ax.plot(res['hp'], color=col, lw=2, label=f'W={W}')
ax.set_title('Effect of Window Size W on Hₚ', fontweight='bold')
ax.set_xlabel('Sliding Window Step')
ax.set_ylabel('Hₚ')
ax.legend(fontsize=9)
ax.grid(ls='--', alpha=0.4)

# Panel 3: TEWI weight variants
ax = axes[2]
colors3 = ['#74c476', '#31a354', '#e6550d', '#9ecae1']
for (name, res), col in zip(tewi_variants.items(), colors3):
    ax.plot(res['series'], color=col, lw=2, label=f'{name} (Δt={res["dt"]})')
ax.axvline(45 - 20, color='black', ls=':', lw=2, label='Collapse')
ax.set_title('TEWI Weight Ablation', fontweight='bold')
ax.set_xlabel('Sliding Window Step')
ax.set_ylabel('TEWI')
ax.legend(fontsize=9)
ax.grid(ls='--', alpha=0.4)

plt.suptitle('TAS Ablation Study — Synthetic Activation Trajectory', fontsize=12, fontweight='bold')
plt.tight_layout()
out = 'notebooks/Ablation_Study_output.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {out}  ✓")
