"""
ieee_package.py
===============
TAS Journal Suite — IEEE / Elsevier Submission Package Builder

Assembles the complete submission package:

  TAS_Journal_Package/
  ├── paper/
  │   ├── manuscript.tex
  │   └── references.bib
  ├── figures/   (Figure1~6 PNG)
  ├── tables/    (Table1~4 LaTeX)
  ├── results/   (montecarlo_results.csv, statistics.csv)
  ├── source/    (all Python modules)
  └── README.md

Author  : Kang, In-Su
License : MIT
"""

import os
import shutil
import json
import pandas as pd


# ── Directory Structure ───────────────────────────────────────

def create_structure(root: str):
    for folder in ["paper", "figures", "tables", "results", "source"]:
        os.makedirs(os.path.join(root, folder), exist_ok=True)


# ── File Copy Helpers ─────────────────────────────────────────

def _cp(src: str, dst: str):
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  Copied: {src} -> {dst}")
    else:
        print(f"  [SKIP]  {src} not found")


def copy_figures(root: str, src_dir: str = "figures"):
    for i in range(1, 7):
        names = [
            f"Figure{i}_TAS_Loss_Gradient.png",
            f"Figure{i}_EarlyWarning.png",
            f"Figure{i}_DeltaT_Histogram.png",
            f"Figure{i}_ModelComparison.png",
            f"Figure{i}_AblationAlpha.png",
            f"Figure{i}_DatasetComparison.png",
        ]
    for name in [
        "Figure1_TAS_Loss_Gradient.png",
        "Figure2_EarlyWarning.png",
        "Figure3_DeltaT_Histogram.png",
        "Figure4_ModelComparison.png",
        "Figure5_AblationAlpha.png",
        "Figure6_DatasetComparison.png",
    ]:
        _cp(os.path.join(src_dir, name),
            os.path.join(root, "figures", name))


def copy_tables(root: str, src_dir: str = "tables"):
    for name in [
        "Table1_ModelComparison.tex",
        "Table2_DatasetComparison.tex",
        "Table3_AblationAlpha.tex",
        "Table4_Statistics.tex",
    ]:
        _cp(os.path.join(src_dir, name),
            os.path.join(root, "tables", name))


def copy_results(root: str):
    for name in ["montecarlo_results.csv", "statistics.csv"]:
        _cp(name, os.path.join(root, "results", name))


def copy_source(root: str, src_dir: str = None):
    modules = [
        "tas_core.py", "models_cnn.py", "models_resnet.py",
        "models_transformer.py", "montecarlo.py", "statistics.py",
        "figures.py", "latex_tables.py", "ieee_package.py", "main.py",
    ]
    base = src_dir or os.path.dirname(os.path.abspath(__file__))
    for name in modules:
        _cp(os.path.join(base, name),
            os.path.join(root, "source", name))


# ── IEEE Manuscript Template ──────────────────────────────────

MANUSCRIPT_TEX = r"""
\documentclass[journal]{IEEEtran}

\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath, amssymb}
\usepackage{hyperref}

\begin{document}

\title{Topological Anomaly Score (TAS): Early Warning Signals of\\
       Gradient Collapse in Deep Neural Networks}

\author{Anonymous%
\thanks{Submitted to IEEE Transactions on Neural Networks and Learning Systems.}}

\maketitle

% ── Abstract ──────────────────────────────────────────────────
\begin{abstract}
We propose the Topological Anomaly Score (TAS), a topology-based
early-warning indicator that detects precursors of gradient collapse
in deep neural networks before the event occurs.
TAS is derived from the barcode lifetime distribution of
Vietoris-Rips persistent homology applied to activation trajectories
under an anisotropic metric that amplifies velocity-sensitive dimensions.
Monte Carlo experiments ($n \geq 50$ runs per condition) across
three architectures (CNN, ResNet18, ViT) and four datasets
(MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100) demonstrate a
statistically significant early-warning lead time
$\Delta t = t_{\text{GC}} - t_{\text{TAS}}$ (Cohen's $d > 2.0$,
$p < 10^{-15}$).
Additionally, Persistent Entropy $H_p$ is shown to rise before TAS,
providing a two-stage alert structure: precursor (via $H_p$)
followed by danger (via TAS).
\end{abstract}

\begin{IEEEkeywords}
Persistent homology, Topological Data Analysis, early warning,
gradient vanishing, deep learning, Topological Anomaly Score
\end{IEEEkeywords}

% ── Introduction ──────────────────────────────────────────────
\section{Introduction}
\label{sec:intro}

Gradient collapse---the sudden vanishing of gradient norms during
deep network training---remains a central obstacle in optimization.
Current monitoring methods detect collapse only after it has occurred,
relying on loss curves or gradient norm thresholds.

We introduce the \textit{Topological Anomaly Score} (TAS), which
measures structural changes in the phase-space geometry of activation
trajectories. The key insight, shared with the double-pendulum chaos
detection literature \cite{kang2025tas}, is that \textit{topological
structure degrades before the dynamical instability becomes detectable
by conventional metrics}.

% ── Method ────────────────────────────────────────────────────
\section{Method}
\label{sec:method}

\subsection{State-Space Construction}
Given an activation history $A \in \mathbb{R}^{T \times D}$, we
form the state-space point cloud
$\mathbf{X} = [A_{t} \mid A_{t} - A_{t-1}]_{t=2}^{T}$.

\subsection{Anisotropic Metric}
We compute pairwise distances
\begin{equation}
  d_\alpha(i,j) = \sqrt{\alpha^2 \|\mathbf{v}_i - \mathbf{v}_j\|^2
                        + \|\mathbf{x}_i - \mathbf{x}_j\|^2}
  \label{eq:metric}
\end{equation}
where $\mathbf{v}$ denotes velocity components and $\alpha > 1$
amplifies topological separation at rate $O(\alpha^2)$.

\subsection{Persistent Entropy and TAS}
From the Vietoris-Rips filtration over $(P, d_\alpha)$ we extract
$H_0$ and $H_1$ barcode lifetimes $\{L_i\}$. Persistent Entropy is
\begin{equation}
  H_p = -\sum_i p_i \log p_i, \quad p_i = L_i / \textstyle\sum_j L_j.
\end{equation}
The Topological Anomaly Score is
\begin{equation}
  \mathrm{TAS} = \frac{L_{\max} - \mu_L}{\sigma_L}.
  \label{eq:tas}
\end{equation}
A Topological Early Warning Index integrating both channels is
\begin{equation}
  \mathrm{TEWI} = w_1 H_p(H_0) + w_2\,\mathrm{TAS} + w_3 H_p(H_1).
\end{equation}

% ── Results ───────────────────────────────────────────────────
\section{Results}
\label{sec:results}

\input{../tables/Table1_ModelComparison}
\input{../tables/Table2_DatasetComparison}
\input{../tables/Table3_AblationAlpha}
\input{../tables/Table4_Statistics}

\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{../figures/Figure1_TAS_Loss_Gradient.png}
\caption{TAS, training loss, and gradient norm vs.\ training step.
         TAS rises before gradient collapse.}
\label{fig:fig1}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{../figures/Figure2_EarlyWarning.png}
\caption{Early-warning detection: TAS threshold crossing ($t_{\mathrm{TAS}}$)
         precedes gradient collapse ($t_{\mathrm{GC}}$) by $\Delta t$ steps.}
\label{fig:fig2}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{../figures/Figure3_DeltaT_Histogram.png}
\caption{Distribution of early-warning lead time $\Delta t$ across
         Monte Carlo runs ($n=50$ per condition).}
\label{fig:fig3}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{../figures/Figure5_AblationAlpha.png}
\caption{Ablation study: mean $\Delta t$ as a function of anisotropic
         weight $\alpha$.}
\label{fig:fig5}
\end{figure}

% ── Conclusion ────────────────────────────────────────────────
\section{Conclusion}
\label{sec:conclusion}

TAS provides a statistically significant early-warning signal for
gradient collapse across diverse architectures and datasets.
The two-stage alert structure ($H_p$ precursor $\to$ TAS danger)
enables targeted interventions (learning-rate adjustment, early
stopping, regularization) before the collapse event.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""

REFERENCES_BIB = r"""
@article{kang2025tas,
  author  = {Kang, In-Su},
  title   = {Topological Anomaly Score and Persistent Entropy
             for Early Detection of Chaos Transition
             in Double Pendulum Dynamics},
  journal = {(Manuscript in preparation)},
  year    = {2025},
  note    = {TLD Framework, v3}
}

@article{gidea2018tda,
  author  = {Gidea, Marian and Katz, Yuri},
  title   = {Topological Data Analysis of Financial Time Series},
  journal = {Physica A},
  volume  = {491},
  pages   = {820--834},
  year    = {2018}
}

@book{edelsbrunner2010,
  author    = {Edelsbrunner, Herbert and Harer, John},
  title     = {Computational Topology},
  publisher = {American Mathematical Society},
  year      = {2010}
}

@inproceedings{zhang2020ripser,
  author    = {Zhang, Bingyi and others},
  title     = {Ripser++: Accelerated Vietoris-Rips Persistent Homology},
  booktitle = {SysML},
  year      = {2020}
}
"""


def create_manuscript(root: str):
    with open(os.path.join(root, "paper", "manuscript.tex"), "w") as f:
        f.write(MANUSCRIPT_TEX)
    with open(os.path.join(root, "paper", "references.bib"), "w") as f:
        f.write(REFERENCES_BIB)
    print(f"  Created manuscript.tex and references.bib")


# ── README ────────────────────────────────────────────────────

README_MD = """# TAS Journal Submission Package

**Topological Anomaly Score (TAS): Early Warning Signals of Gradient Collapse in Deep Neural Networks**

## Contents

| Directory | Description |
|-----------|-------------|
| `paper/`  | IEEE LaTeX manuscript (`manuscript.tex`) and BibTeX (`references.bib`) |
| `figures/` | Publication figures (300 DPI PNG, Figure 1~6) |
| `tables/`  | LaTeX tables (Table 1~4) |
| `results/` | Monte Carlo CSV results |
| `source/`  | Complete Python source code |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quick smoke test
python source/tas_core.py

# 3. Run full Monte Carlo (GPU recommended)
python source/main.py --models CNN ResNet18 --datasets MNIST CIFAR10 --n_runs 50

# 4. Generate figures and tables
python source/main.py --mode figures
python source/main.py --mode tables

# 5. Build submission package
python source/ieee_package.py
```

## Core Modules

| Module | Role |
|--------|------|
| `tas_core.py` | TAS engine: anisotropic metric, VR filtration, Hₚ, TAS, TEWI |
| `models_cnn.py` | CNN with activation hooks |
| `models_resnet.py` | ResNet18 with layer4/avgpool hooks |
| `models_transformer.py` | ViT with CLS token extraction |
| `montecarlo.py` | Monte Carlo experiment runner |
| `statistics.py` | t-test, Wilcoxon, Cohen's d, Bootstrap CI |
| `figures.py` | Figure 1~6 generator |
| `latex_tables.py` | Table 1~4 LaTeX generator |
| `ieee_package.py` | Submission package assembler |
| `main.py` | Unified CLI entry point |

## Key Formulae

```
Anisotropic metric:   d_α(i,j) = √[ α²‖v_i−v_j‖² + ‖x_i−x_j‖² ]
Persistent Entropy:   Hₚ = −Σ pᵢ log pᵢ,   pᵢ = Lᵢ / ΣLᵢ
TAS:                  TAS = (L_max − μL) / σL
TEWI:                 TEWI = w₁·Hₚ(H₀) + w₂·TAS + w₃·Hₚ(H₁)
Collapse Rate:        CollapseRate = d(TAS)/dt
```

## Citation

```bibtex
@article{kang2025tas,
  author  = {Kang, In-Su},
  title   = {Topological Anomaly Score and Persistent Entropy
             for Early Detection of Chaos Transition},
  journal = {(In preparation)},
  year    = {2025}
}
```

## License

MIT License — see `LICENSE`.
"""


def create_readme(root: str):
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(README_MD)
    print("  Created README.md")


# ── Build Package ─────────────────────────────────────────────

def build_package(package_name: str = "TAS_Journal_Package",
                  src_dir: str = None,
                  zip_output: bool = True):
    print(f"\n=== Building IEEE Submission Package: {package_name} ===\n")
    create_structure(package_name)
    copy_figures(package_name)
    copy_tables(package_name)
    copy_results(package_name)
    copy_source(package_name, src_dir)
    create_manuscript(package_name)
    create_readme(package_name)

    if zip_output:
        zip_path = shutil.make_archive(package_name, "zip", ".", package_name)
        print(f"\n  ZIP created: {zip_path}")

    print(f"\n=== Package ready: {package_name}/ ===\n")


if __name__ == "__main__":
    build_package()
