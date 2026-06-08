"""
simple_tas_demo.py
==================
TAS Research Suite — Minimal Working Example

Demonstrates the core TAS pipeline on synthetic data:
  1. Generate synthetic activation trajectory (stable → unstable)
  2. Compute TAS, Hₚ, TEWI via anisotropic persistent homology
  3. Detect precursor state (t_PE < t_TAS)
  4. Plot results

Run:
    cd TAS_GitHub
    python examples/simple_tas_demo.py

Requirements: numpy, gudhi, matplotlib
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tas_core import (
    compute_tas_metrics,
    compute_tewi,
    compute_collapse_rate,
    detect_precursor_state,
)


# ── 1. Generate synthetic trajectory ─────────────────────────────────────────

def make_synthetic_trajectory(T: int = 80, D: int = 64,
                               collapse_at: int = 60,
                               seed: int = 42) -> np.ndarray:
    """
    Simulate activation trajectory with progressive instability.

    - Steps 0..collapse_at-1 : stable (low variance, smooth)
    - Steps collapse_at..T   : unstable (high variance, fragmented)
    """
    rng = np.random.default_rng(seed)

    stable   = rng.normal(0.0, 0.1, size=(collapse_at, D))
    # Gradual drift to simulate manifold fragmentation
    drift = np.linspace(0, 1, T - collapse_at)[:, None]
    unstable = rng.normal(0.0, 0.5, size=(T - collapse_at, D)) * (1 + drift * 3)

    trajectory = np.concatenate([stable, unstable], axis=0)
    return trajectory.astype(np.float32)


# ── 2. Sliding-window TAS/Hₚ computation ─────────────────────────────────────

def run_sliding_window(trajectory: np.ndarray,
                        window_size: int = 20,
                        alpha: float = 2.5) -> dict:
    """Compute TAS, Hₚ, TEWI for each sliding window."""
    T = len(trajectory)
    hp_series   = []
    tas_series  = []
    tewi_series = []
    steps       = []

    for t in range(window_size, T):
        window = trajectory[t - window_size : t]
        res = compute_tas_metrics(window, alpha=alpha)

        hp_series.append(res['hp_total'])
        tas_series.append(res['tas_total'])
        tewi_series.append(compute_tewi(
            res['hp_h0'], res['tas_h0'], res['hp_h1'],
            w1=0.45, w2=0.25, w3=0.30
        ))
        steps.append(t)

    collapse_rate = compute_collapse_rate(tas_series)

    return {
        'steps':         steps,
        'hp':            hp_series,
        'tas':           tas_series,
        'tewi':          tewi_series,
        'collapse_rate': collapse_rate,
    }


# ── 3. Threshold detection ────────────────────────────────────────────────────

def detect_crossings(series: list, baseline_n: int = 20,
                     sigma: float = 3.0) -> int:
    arr = np.array(series)
    mu  = arr[:baseline_n].mean()
    sd  = arr[:baseline_n].std()
    thr = mu + sigma * sd
    idx = np.where(arr > thr)[0]
    return int(idx[0]) if len(idx) > 0 else None


# ── 4. Plot ───────────────────────────────────────────────────────────────────

def plot_results(result: dict, collapse_at: int,
                 t_pe: int, t_tas: int,
                 output: str = 'examples/simple_tas_demo_output.png'):

    steps = result['steps']
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Panel 1: Hₚ
    ax = axes[0]
    ax.plot(steps, result['hp'], color='crimson', lw=2, label='Persistent Entropy Hₚ')
    ax.set_ylabel('Hₚ', color='crimson')
    if t_pe is not None:
        ax.axvline(steps[t_pe], color='crimson', ls='--', lw=1.5,
                   label=f'Precursor alert (step {steps[t_pe]})')
    ax.axvline(collapse_at, color='black', ls=':', lw=2, label=f'Collapse (step {collapse_at})')
    ax.legend(fontsize=9)
    ax.set_title('Stage 1 — Precursor Alert: Persistent Entropy Hₚ', fontweight='bold')
    ax.grid(ls='--', alpha=0.5)

    # Panel 2: TAS
    ax = axes[1]
    ax.plot(steps, result['tas'], color='steelblue', lw=2, label='TAS')
    ax.set_ylabel('TAS', color='steelblue')
    if t_tas is not None:
        ax.axvline(steps[t_tas], color='steelblue', ls='--', lw=1.5,
                   label=f'Danger alert (step {steps[t_tas]})')
    ax.axvline(collapse_at, color='black', ls=':', lw=2)
    ax.legend(fontsize=9)
    ax.set_title('Stage 2 — Danger Alert: Topological Anomaly Score (TAS)', fontweight='bold')
    ax.grid(ls='--', alpha=0.5)

    # Panel 3: TEWI
    ax = axes[2]
    ax.plot(steps, result['tewi'], color='darkorange', lw=2, label='TEWI')
    ax.axvline(collapse_at, color='black', ls=':', lw=2, label=f'Collapse (step {collapse_at})')
    ax.set_ylabel('TEWI')
    ax.set_xlabel('Training Step')
    ax.legend(fontsize=9)
    ax.set_title('TEWI = 0.45·Hₚ(H₀) + 0.25·TAS + 0.30·Hₚ(H₁)', fontweight='bold')
    ax.grid(ls='--', alpha=0.5)

    # Annotate lead time
    if t_pe is not None and t_tas is not None:
        delta_pe  = collapse_at - steps[t_pe]
        delta_tas = collapse_at - steps[t_tas]
        fig.text(0.5, 0.01,
                 f'Δt_PE = {delta_pe} steps  |  Δt_TAS = {delta_tas} steps  |  '
                 f'Precursor ordering confirmed: {steps[t_pe]} < {steps[t_tas]} < {collapse_at}',
                 ha='center', fontsize=10, color='darkgreen', fontweight='bold')

    plt.suptitle('TAS Early Warning Demo — Synthetic Activation Trajectory', fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()
    print(f'Plot saved: {output}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print('  TAS Research Suite — Simple Demo')
    print('=' * 55)

    COLLAPSE_AT  = 60
    WINDOW_SIZE  = 20
    ALPHA        = 2.5
    T            = 80

    # Generate trajectory
    print(f'\n1. Generating synthetic trajectory (T={T}, D=64, collapse@{COLLAPSE_AT})')
    traj = make_synthetic_trajectory(T=T, collapse_at=COLLAPSE_AT)

    # Sliding-window TAS/Hₚ
    print(f'2. Running sliding-window TDA (W={WINDOW_SIZE}, α={ALPHA})')
    result = run_sliding_window(traj, window_size=WINDOW_SIZE, alpha=ALPHA)

    # Detect crossings
    t_pe  = detect_crossings(result['hp'])
    t_tas = detect_crossings(result['tas'])

    # Precursor state
    pc = detect_precursor_state(result['hp'], result['tas'])

    # Report
    print('\n=== Results ===')
    if t_pe is not None:
        print(f'  Hₚ threshold crossing : step {result["steps"][t_pe]}  '
              f'(Δt = {COLLAPSE_AT - result["steps"][t_pe]} steps before collapse)')
    else:
        print('  Hₚ: no threshold crossing detected')

    if t_tas is not None:
        print(f'  TAS threshold crossing: step {result["steps"][t_tas]}  '
              f'(Δt = {COLLAPSE_AT - result["steps"][t_tas]} steps before collapse)')
    else:
        print('  TAS: no threshold crossing detected')

    print(f'  Precursor state      : {pc["precursor"]}  '
          f'(t_PE={pc["t_PE"]}, t_TAS={pc["t_TAS"]})')
    print(f'  Max TEWI             : {max(result["tewi"]):.4f}')

    # Plot
    print('\n3. Saving plot ...')
    plot_results(result, COLLAPSE_AT, t_pe, t_tas)

    print('\nDemo complete. ✓')


if __name__ == '__main__':
    main()
