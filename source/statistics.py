"""
statistics.py
=============
TAS Journal Suite — Statistical Validation Module

Computes from Monte Carlo CSV:
  - Basic statistics (n, mean, median, std, min, max)
  - Standard error & 95% confidence interval (t-distribution)
  - One-sample t-test  (H0: Δt = 0)
  - Wilcoxon signed-rank test (non-parametric)
  - Cohen's d  (effect size)
  - Hedges' g  (small-sample correction)
  - Bootstrap 95% CI  (5000 resamples)
  - Effect size label (Negligible / Small / Medium / Large / Very Large)

Author  : Kang, In-Su
License : MIT
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon, t as t_dist


# ── Basic Statistics ──────────────────────────────────────────

def basic_statistics(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return {
        "n":      len(x),
        "mean":   float(np.mean(x)),
        "median": float(np.median(x)),
        "std":    float(np.std(x, ddof=1)),
        "min":    float(np.min(x)),
        "max":    float(np.max(x)),
    }


def standard_error(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def confidence_interval_95(x: np.ndarray) -> tuple:
    x  = np.asarray(x)
    se = standard_error(x)
    return t_dist.interval(0.95, df=len(x)-1, loc=np.mean(x), scale=se)


# ── Hypothesis Tests ──────────────────────────────────────────

def one_sample_ttest(x: np.ndarray) -> dict:
    """H0: mean(Δt) = 0  vs  H1: mean(Δt) > 0  (one-sided via 2-sided / 2)."""
    stat, p = ttest_1samp(x, 0.0)
    return {"t": float(stat), "p": float(p / 2)}   # one-sided


def wilcoxon_test(x: np.ndarray) -> dict:
    """Wilcoxon signed-rank test (non-parametric, H0: median = 0)."""
    stat, p = wilcoxon(x)
    return {"W": float(stat), "p": float(p)}


# ── Effect Sizes ──────────────────────────────────────────────

def cohens_d(x: np.ndarray) -> float:
    """d = mean / std  (one-sample, comparing to 0)."""
    x = np.asarray(x)
    return float(np.mean(x) / (np.std(x, ddof=1) + 1e-12))


def hedges_g(x: np.ndarray) -> float:
    """Hedges' g with Hedges correction for small samples."""
    d = cohens_d(x)
    n = len(x)
    return float(d * (1.0 - 3.0 / (4.0 * n - 9.0)))


def effect_size_label(d: float) -> str:
    d = abs(d)
    if   d < 0.20: return "Negligible"
    elif d < 0.50: return "Small"
    elif d < 0.80: return "Medium"
    elif d < 1.20: return "Large"
    else:          return "Very Large"


# ── Bootstrap CI ──────────────────────────────────────────────

def bootstrap_ci(x: np.ndarray, n_boot: int = 5000,
                 alpha: float = 0.05) -> tuple:
    """Non-parametric bootstrap confidence interval for the mean."""
    x = np.asarray(x)
    means = [np.mean(np.random.choice(x, size=len(x), replace=True))
             for _ in range(n_boot)]
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


# ── Full Analysis ─────────────────────────────────────────────

def analyze_delta_t(delta_t) -> dict:
    """
    Complete statistical analysis of an array of Δt values.

    Returns a dict suitable for Table 4 (Statistical Significance Test).
    """
    x = np.asarray(delta_t, dtype=float)
    x = x[~np.isnan(x)]

    stats  = basic_statistics(x)
    ci95   = confidence_interval_95(x)
    bci    = bootstrap_ci(x)
    tt     = one_sample_ttest(x)
    wt     = wilcoxon_test(x)
    d      = cohens_d(x)
    g      = hedges_g(x)

    return {
        **stats,
        "SE":           standard_error(x),
        "CI95":         ci95,
        "BootstrapCI":  bci,
        "t":            tt["t"],
        "t_p":          tt["p"],
        "W":            wt["W"],
        "wilcoxon_p":   wt["p"],
        "cohens_d":     d,
        "hedges_g":     g,
        "effect":       effect_size_label(d),
    }


# ── CSV Reader ────────────────────────────────────────────────

def analyze_csv(csv_path: str) -> dict:
    """Load montecarlo_results.csv and analyze Δt_TAS and Δt_Hp."""
    df = pd.read_csv(csv_path)
    return {
        "TAS": analyze_delta_t(df["delta_tas"].dropna()),
        "HP":  analyze_delta_t(df["delta_hp"].dropna()),
    }


def print_report(results: dict):
    for key, res in results.items():
        print(f"\n=== {key} Δt Analysis ===")
        print(f"  N            = {res['n']}")
        print(f"  Mean Δt      = {res['mean']:.3f}")
        print(f"  Median Δt    = {res['median']:.3f}")
        print(f"  Std          = {res['std']:.3f}")
        print(f"  SE           = {res['SE']:.3f}")
        print(f"  95% CI       = [{res['CI95'][0]:.3f}, {res['CI95'][1]:.3f}]")
        print(f"  Bootstrap CI = [{res['BootstrapCI'][0]:.3f}, {res['BootstrapCI'][1]:.3f}]")
        print(f"  t-stat       = {res['t']:.3f}   p = {res['t_p']:.2e}")
        print(f"  Wilcoxon W   = {res['W']:.1f}   p = {res['wilcoxon_p']:.2e}")
        print(f"  Cohen's d    = {res['cohens_d']:.3f}  ({res['effect']})")
        print(f"  Hedges' g    = {res['hedges_g']:.3f}")


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "montecarlo_results.csv"
    results = analyze_csv(csv)
    print_report(results)
