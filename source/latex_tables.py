"""
latex_tables.py
===============
TAS Journal Suite — IEEE / Elsevier / Springer LaTeX Table Generator

Generates 4 publication-ready LaTeX tables from Monte Carlo CSV:
  Table 1 — Model comparison      (CNN / ResNet18 / ViT)
  Table 2 — Dataset comparison    (MNIST / Fashion / CIFAR10 / CIFAR100)
  Table 3 — Ablation study        (alpha sweep)
  Table 4 — Statistical tests     (t-test, Wilcoxon, Cohen's d, Hedges' g)

Author  : Kang, In-Su
License : MIT
"""

import pandas as pd
import numpy as np
from statistics import analyze_delta_t


# ── Helper ────────────────────────────────────────────────────

def _save(text: str, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {filename}")


def _fmt(v, decimals=3):
    if isinstance(v, float) and np.isnan(v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


# ── Table 1 — Model Comparison ────────────────────────────────

def table1_model_comparison(df: pd.DataFrame,
                             filename: str = "Table1_ModelComparison.tex") -> str:
    rows = []
    for model in sorted(df["model"].unique()):
        sub = df.loc[df["model"] == model, "delta_tas"].dropna()
        rows.append({
            "Model":  model,
            "N":      len(sub),
            r"Mean $\Delta t$": f"{sub.mean():.2f}",
            r"Std":             f"{sub.std():.2f}",
            r"Median":          f"{sub.median():.2f}",
        })
    tdf = pd.DataFrame(rows)

    latex = (
        "\\begin{table}[!t]\n"
        "\\caption{Early Warning Lead Time $\\Delta t$ by Model Architecture}\n"
        "\\label{tab:model}\n"
        "\\centering\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Model & $N$ & Mean $\\Delta t$ & Std & Median \\\\\n"
        "\\midrule\n"
    )
    for _, row in tdf.iterrows():
        latex += (f"{row['Model']} & {row['N']} & "
                  f"{row[r'Mean $\\Delta t$']} & "
                  f"{row[r'Std']} & {row[r'Median']} \\\\\n")
    latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _save(latex, filename)
    return latex


# ── Table 2 — Dataset Comparison ─────────────────────────────

def table2_dataset_comparison(df: pd.DataFrame,
                               filename: str = "Table2_DatasetComparison.tex") -> str:
    rows = []
    for ds in sorted(df["dataset"].unique()):
        sub = df.loc[df["dataset"] == ds, "delta_tas"].dropna()
        rows.append({
            "Dataset": ds,
            "N":       len(sub),
            "Mean":    f"{sub.mean():.2f}",
            "Std":     f"{sub.std():.2f}",
        })

    latex = (
        "\\begin{table}[!t]\n"
        "\\caption{Early Warning Lead Time $\\Delta t$ by Dataset}\n"
        "\\label{tab:dataset}\n"
        "\\centering\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Dataset & $N$ & Mean $\\Delta t$ & Std \\\\\n"
        "\\midrule\n"
    )
    for row in rows:
        latex += f"{row['Dataset']} & {row['N']} & {row['Mean']} & {row['Std']} \\\\\n"
    latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _save(latex, filename)
    return latex


# ── Table 3 — Ablation Study (α) ──────────────────────────────

def table3_ablation_alpha(df: pd.DataFrame,
                           filename: str = "Table3_AblationAlpha.tex") -> str:
    rows = []
    for alpha in sorted(df["alpha"].unique()):
        sub = df.loc[df["alpha"] == alpha, "delta_tas"].dropna()
        rows.append({
            r"$\alpha$": alpha,
            "N":         len(sub),
            "Mean":      f"{sub.mean():.2f}",
            "Std":       f"{sub.std():.2f}",
        })

    latex = (
        "\\begin{table}[!t]\n"
        "\\caption{Ablation Study: Effect of Anisotropic Weight $\\alpha$}\n"
        "\\label{tab:ablation}\n"
        "\\centering\n"
        "\\begin{tabular}{rrrr}\n"
        "\\toprule\n"
        "$\\alpha$ & $N$ & Mean $\\Delta t$ & Std \\\\\n"
        "\\midrule\n"
    )
    for row in rows:
        latex += (f"{row[r'$\\alpha$']} & {row['N']} & "
                  f"{row['Mean']} & {row['Std']} \\\\\n")
    latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _save(latex, filename)
    return latex


# ── Table 4 — Statistical Tests ───────────────────────────────

def table4_statistics(stats_dict: dict,
                       filename: str = "Table4_Statistics.tex") -> str:
    """
    stats_dict: output of analyze_delta_t() from statistics.py
    """
    ci    = stats_dict["CI95"]
    bci   = stats_dict["BootstrapCI"]
    rows  = [
        ("$N$",                    str(stats_dict["n"])),
        ("Mean $\\Delta t$",       f"{stats_dict['mean']:.3f}"),
        ("Median $\\Delta t$",     f"{stats_dict['median']:.3f}"),
        ("Std",                    f"{stats_dict['std']:.3f}"),
        ("SE",                     f"{stats_dict['SE']:.3f}"),
        ("95\\% CI",               f"[{ci[0]:.3f},\\;{ci[1]:.3f}]"),
        ("Bootstrap 95\\% CI",     f"[{bci[0]:.3f},\\;{bci[1]:.3f}]"),
        ("$t$-statistic",          f"{stats_dict['t']:.3f}"),
        ("$t$-test $p$",           f"{stats_dict['t_p']:.2e}"),
        ("Wilcoxon $W$",           f"{stats_dict['W']:.0f}"),
        ("Wilcoxon $p$",           f"{stats_dict['wilcoxon_p']:.2e}"),
        ("Cohen's $d$",            f"{stats_dict['cohens_d']:.3f}"),
        ("Hedges' $g$",            f"{stats_dict['hedges_g']:.3f}"),
        ("Effect Size",            stats_dict["effect"]),
    ]

    latex = (
        "\\begin{table}[!t]\n"
        "\\caption{Statistical Significance of Early Warning Lead Time $\\Delta t$}\n"
        "\\label{tab:statistics}\n"
        "\\centering\n"
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "Metric & Value \\\\\n"
        "\\midrule\n"
    )
    for metric, value in rows:
        latex += f"{metric} & {value} \\\\\n"
    latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _save(latex, filename)
    return latex


# ── Batch Generator ───────────────────────────────────────────

def generate_all_tables(montecarlo_df: pd.DataFrame,
                         stats_result: dict,
                         out_dir: str = "tables"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    table1_model_comparison(montecarlo_df,  f"{out_dir}/Table1_ModelComparison.tex")
    table2_dataset_comparison(montecarlo_df, f"{out_dir}/Table2_DatasetComparison.tex")
    table3_ablation_alpha(montecarlo_df,     f"{out_dir}/Table3_AblationAlpha.tex")
    table4_statistics(stats_result,          f"{out_dir}/Table4_Statistics.tex")
    print(f"\nAll tables saved to {out_dir}/")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    csv = sys.argv[1] if len(sys.argv) > 1 else "montecarlo_results.csv"
    import pandas as pd
    from statistics import analyze_csv
    df  = pd.read_csv(csv)
    res = analyze_csv(csv)
    generate_all_tables(df, res["TAS"])
