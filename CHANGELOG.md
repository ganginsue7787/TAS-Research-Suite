# Changelog

All notable changes to TAS Research Suite are documented here.

## [1.0.0] — 2025

### Added
- `tas_core.py`: Core TAS engine — anisotropic VR filtration, Hₚ, TAS, TEWI, CollapseRate, precursor detection
- `models_cnn.py`: CNN-3 with Sigmoid activations and fc1 hook
- `models_resnet.py`: ResNet18 with layer4/avgpool hooks
- `models_transformer.py`: ViT-B/16 with CLS token extraction
- `montecarlo.py`: Full Monte Carlo runner (3 models × 4 datasets × 5α × 50 runs)
- `statistics.py`: t-test, Wilcoxon, Cohen's d, Hedges' g, Bootstrap CI
- `figures.py`: Publication-quality Figure 1–6 (300 DPI)
- `latex_tables.py`: IEEE/Elsevier LaTeX Table 1–4
- `ieee_package.py`: Submission package assembler
- `main.py`: Unified CLI (test/mc/figures/tables/stats/package)
- `source/tests/test_tas_core.py`: 30 unit tests
- `examples/simple_tas_demo.py`: Minimal working example
- `examples/mnist_cnn_demo.py`: CNN + MNIST real-time monitoring demo
- `examples/montecarlo_demo.py`: Mini Monte Carlo demo
- `notebooks/MNIST_Demo.py`: Full MNIST notebook script
- `notebooks/ResNet18_Experiment.py`: ResNet18 + CIFAR-10 notebook
- `notebooks/ViT_Experiment.py`: ViT + MNIST CLS token notebook
- `notebooks/Ablation_Study.py`: α/window/TEWI ablation notebook
- `docs/theory.md`: Mathematical foundations
- `docs/experimental_protocol.md`: Reproducibility protocol
- `docs/patent_relation.md`: Patent relationship and licensing
- `docs/mathematical_foundation.md`: Key theorems and proofs
- `docs/applications.md`: Application domains
- `CONTRIBUTING.md`: Contribution guidelines
- `CODE_OF_CONDUCT.md`: Community standards
- `CITATION.cff`: Machine-readable citation metadata
- `CHANGELOG.md`: This file

### Notes
- Patent applications (KR, PCT, US) filed prior to public release
- Supports Python 3.10+, PyTorch 2.0+, Gudhi 3.8+
