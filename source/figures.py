"""
figures.py
==========
TAS Journal Suite — IEEE/Elsevier Figure Generator

Produces 6 publication-quality figures (300 DPI):
  Fig 1 — TAS vs Loss vs Gradient Norm   (★★★★★)
  Fig 2 — Early Warning Detection         (★★★★★)
  Fig 3 — Δt Distribution Histogram       (★★★★★)
  Fig 4 — Model Comparison Boxplot        (★★★★★)
  Fig 5 — Ablation Study on α             (★★★★★)
  Fig 6 — Dataset Comparison Bar          (★★★★★)

Author  : Kang, In-Su
License : MIT
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DPI      = 300
FIG_SIZE = (8, 5)
FONT     = {"family": "serif", "size": 12}
matplotlib.rc("font", **FONT)
matplotlib.rc("axes", titlesize=13, labelsize=12)
matplotlib.rc("legend", fontsize=11)


# ── Figure 1 — TAS / Loss / Gradient ─────────────────────────

def figure1_tas_loss_grad(tas, loss, grad,
                           output="Figure1_TAS_Loss_Gradient.png",
                           t_pe=None, t_gc=None):
    steps = np.arange(len(tas))
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)

    ax1.plot(steps, tas,  label="TAS",  color="crimson",  lw=2)
    ax1.plot(steps, loss, label="Loss", color="steelblue", lw=2)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("TAS / Loss")

    ax2 = ax1.twinx()
    ax2.plot(steps, grad, label="Grad Norm", color="darkorange", lw=1.5, ls="--")
    ax2.set_ylabel("Gradient Norm")

    if t_pe is not None:
        ax1.axvline(t_pe, color="black", ls=":", lw=1.5, label=f"t_PE={t_pe}")
    if t_gc is not None:
        ax1.axvline(t_gc, color="purple", ls="--", lw=1.5, label=f"t_GC={t_gc}")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")
    plt.title("TAS vs Loss vs Gradient Norm")
    fig.tight_layout()
    plt.savefig(output, dpi=DPI)
    plt.close()
    print(f"Saved {output}")


# ── Figure 2 — Early Warning Detection ───────────────────────

def figure2_early_warning(tas, t_tas, t_gc,
                            output="Figure2_EarlyWarning.png"):
    plt.figure(figsize=FIG_SIZE)
    plt.plot(tas, lw=2, color="crimson", label="TAS")
    if t_tas is not None:
        plt.axvline(t_tas, ls="--", color="navy",
                    label=f"TAS Warning (t={t_tas})")
    if t_gc is not None:
        plt.axvline(t_gc, ls=":", color="purple",
                    label=f"Gradient Collapse (t={t_gc})")
    if t_tas is not None and t_gc is not None:
        plt.annotate("", xy=(t_gc, max(tas) * 0.7),
                     xytext=(t_tas, max(tas) * 0.7),
                     arrowprops=dict(arrowstyle="<->", color="green", lw=2))
        plt.text((t_tas + t_gc) / 2, max(tas) * 0.72,
                 f"Δt={t_gc-t_tas}", ha="center", color="green")
    plt.legend()
    plt.title("Early Warning Detection")
    plt.xlabel("Sliding Window Step")
    plt.ylabel("TAS")
    plt.tight_layout()
    plt.savefig(output, dpi=DPI)
    plt.close()
    print(f"Saved {output}")


# ── Figure 3 — Δt Histogram ───────────────────────────────────

def figure3_delta_histogram(delta_t, output="Figure3_DeltaT_Histogram.png"):
    plt.figure(figsize=FIG_SIZE)
    plt.hist(delta_t, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
    plt.axvline(np.mean(delta_t), color="red",   ls="--", lw=2,
                label=f"Mean={np.mean(delta_t):.1f}")
    plt.axvline(np.median(delta_t), color="orange", ls=":",  lw=2,
                label=f"Median={np.median(delta_t):.1f}")
    plt.xlabel("Δt (Early Warning Lead Time, steps)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Early Warning Time (Δt)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=DPI)
    plt.close()
    print(f"Saved {output}")


# ── Figure 4 — Model Boxplot ──────────────────────────────────

def figure4_model_boxplot(df: pd.DataFrame,
                           output="Figure4_ModelComparison.png"):
    models = sorted(df["model"].unique())
    data   = [df.loc[df["model"] == m, "delta_tas"].dropna().values
              for m in models]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, labels=models, patch_artist=True, notch=True)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Early Warning Lead Time: CNN vs ResNet18 vs ViT")
    ax.set_ylabel("Δt (steps)")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output, dpi=DPI)
    plt.close()
    print(f"Saved {output}")


# ── Figure 5 — Ablation Study on α ───────────────────────────

def figure5_ablation_alpha(df: pd.DataFrame,
                            output="Figure5_AblationAlpha.png"):
    grouped = df.groupby("alpha")["delta_tas"].agg(["mean", "std"])
    plt.figure(figsize=FIG_SIZE)
    plt.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                 marker="o", lw=2, capsize=5, color="steelblue")
    plt.xlabel("Anisotropic Weight α")
    plt.ylabel("Mean Δt (steps)")
    plt.title("Ablation Study: Effect of α on Early Warning Lead Time")
    plt.grid(ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output, dpi=DPI)
    plt.close()
    print(f"Saved {output}")


# ── Figure 6 — Dataset Comparison ────────────────────────────

def figure6_dataset_comparison(df: pd.DataFrame,
                                output="Figure6_DatasetComparison.png"):
    grouped = df.groupby("dataset")["delta_tas"].agg(["mean", "std"])
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(grouped.index, grouped["mean"], color=colors, alpha=0.8,
                  yerr=grouped["std"], capsize=5)
    ax.set_ylabel("Mean Δt (steps)")
    ax.set_title("Early Warning Lead Time Across Datasets")
    ax.grid(axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output, dpi=DPI)
    plt.close()
    print(f"Saved {output}")


# ── Batch Generator ───────────────────────────────────────────

def generate_all_figures(experiment_result: dict,
                          montecarlo_df: pd.DataFrame,
                          out_dir: str = "figures"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    tas  = experiment_result["tas_series"]
    loss = experiment_result.get("loss_series", [0] * len(tas))
    grad = experiment_result.get("grad_series", [1] * len(tas))

    figure1_tas_loss_grad(tas, loss, grad,
                          output=f"{out_dir}/Figure1_TAS_Loss_Gradient.png",
                          t_pe=experiment_result.get("t_hp"),
                          t_gc=experiment_result.get("t_gc"))
    figure2_early_warning(tas,
                          experiment_result.get("t_tas"),
                          experiment_result.get("t_gc"),
                          output=f"{out_dir}/Figure2_EarlyWarning.png")
    figure3_delta_histogram(montecarlo_df["delta_tas"].dropna(),
                            output=f"{out_dir}/Figure3_DeltaT_Histogram.png")
    figure4_model_boxplot(montecarlo_df,
                          output=f"{out_dir}/Figure4_ModelComparison.png")
    figure5_ablation_alpha(montecarlo_df,
                           output=f"{out_dir}/Figure5_AblationAlpha.png")
    figure6_dataset_comparison(montecarlo_df,
                               output=f"{out_dir}/Figure6_DatasetComparison.png")
    print(f"\nAll figures saved to {out_dir}/")
