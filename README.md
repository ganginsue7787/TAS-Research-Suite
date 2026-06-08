# TAS Journal Suite

**Topological Anomaly Score (TAS): Early Warning Signals of Gradient Collapse in Deep Neural Networks**

> *TLD (Topological Langevin Dynamics) Framework — v3*
> Author: Kang, In-Su  

[![CI](https://github.com/your-username/TAS-Journal-Suite/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/TAS-Journal-Suite/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

---

## Overview

TAS detects **topological precursors of gradient collapse** in deep neural networks before the event occurs, by analyzing the phase-space geometry of activation trajectories using Persistent Homology.

| Indicator | Role | Timing |
|-----------|------|--------|
| **Hₚ (Persistent Entropy)** | Precursor / early-warning alert | t_PE — earliest |
| **TAS (Topological Anomaly Score)** | Danger / severity alert | t_TAS — after Hₚ |
| **TEWI** | Integrated index | w₁Hₚ(H₀) + w₂TAS + w₃Hₚ(H₁) |
| **CollapseRate** | Acceleration alert | d(TAS)/dt |

---

## Project Structure

```
TAS_GitHub/
├── source/
│   ├── tas_core.py           # Core TAS engine (VR filtration, Hₚ, TAS, TEWI)
│   ├── models_cnn.py         # CNN + activation hooks
│   ├── models_resnet.py      # ResNet18 + layer4/avgpool hooks
│   ├── models_transformer.py # ViT + CLS token extraction
│   ├── montecarlo.py         # Monte Carlo runner (CNN/ResNet/ViT × 4 datasets × α × 50 runs)
│   ├── statistics.py         # t-test, Wilcoxon, Cohen's d, Bootstrap CI
│   ├── figures.py            # Figure 1~6 (300 DPI PNG)
│   ├── latex_tables.py       # Table 1~4 (IEEE LaTeX)
│   ├── ieee_package.py       # Submission package assembler
│   └── main.py               # Unified CLI
├── paper/                    # IEEE LaTeX manuscript template
├── figures/                  # Generated figures
├── tables/                   # Generated LaTeX tables
├── results/                  # Monte Carlo CSV outputs
├── requirements.txt
├── .gitignore
├── LICENSE
└── .github/workflows/ci.yml  # GitHub Actions CI
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/your-username/TAS-Journal-Suite.git
cd TAS-Journal-Suite

# Install
pip install -r requirements.txt

# 1. Self-test (no GPU needed)
python source/main.py --mode test

# 2. Monte Carlo (CNN × MNIST × 3 runs — quick smoke test)
python source/main.py --mode mc \
    --models CNN --datasets MNIST --alphas 2.5 --n_runs 3

# 3. Full experiment (GPU recommended)
python source/main.py --mode mc \
    --models CNN ResNet18 ViT \
    --datasets MNIST FashionMNIST CIFAR10 CIFAR100 \
    --alphas 1.0 2.0 2.5 3.0 5.0 \
    --n_runs 50 --device cuda

# 4. Generate figures
python source/main.py --mode figures --csv results/montecarlo_results.csv

# 5. Generate LaTeX tables
python source/main.py --mode tables --csv results/montecarlo_results.csv

# 6. Statistical report
python source/main.py --mode stats --csv results/montecarlo_results.csv

# 7. Build IEEE submission package
python source/main.py --mode package
```

---

## Key Formulae

```
Anisotropic metric:    d_α(i,j) = √[ α²‖v_i−v_j‖² + ‖x_i−x_j‖² ]
Persistent Entropy:    Hₚ = −Σᵢ pᵢ log pᵢ,   pᵢ = Lᵢ / ΣLᵢ
TAS:                   TAS = (L_max − μL) / σL
TEWI:                  TEWI = w₁·Hₚ(H₀) + w₂·TAS + w₃·Hₚ(H₁)
Collapse Rate:         CollapseRate = ΔTAS / Δt
Precursor condition:   t_PE < t_TAS  → Precursor State
```

---

## Expected Results (per paper)

| Metric | Expected Value |
|--------|---------------|
| Mean Δt | ~12 steps |
| Cohen's d | > 2.0 |
| t-test p | < 10⁻¹⁵ |
| Wilcoxon p | < 10⁻¹² |
| Effect Size | Very Large |

---

## Citation

```bibtex
@article{kang2025tas,
  author  = {Kang, In-Su},
  title   = {Topological Anomaly Score and Persistent Entropy
             for Early Detection of Chaos Transition
             in Double Pendulum Dynamics},
  journal = {(Manuscript in preparation — TLD Framework v3)},
  year    = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
