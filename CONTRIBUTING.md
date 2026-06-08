# Contributing to TAS Research Suite

Thank you for your interest in contributing to the **TAS (Topological Anomaly Score) Research Suite** — the open-source implementation of the Topological Langevin Dynamics (TLD) framework.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Research Contributions](#research-contributions)
- [Reproducibility Policy](#reproducibility-policy)
- [Patent Notice](#patent-notice)

---

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold these standards.

---

## How to Contribute

### Reporting Issues

Use [GitHub Issues](https://github.com/kang-insu/TAS-Research-Suite/issues) to report:

- 🐛 **Bugs** — unexpected behaviour, errors, crashes
- 📖 **Documentation problems** — missing docs, incorrect formulas
- 🔬 **Reproducibility issues** — results you cannot replicate
- 💡 **Feature requests** — new metrics, new applications, new architectures

When reporting a bug, please include:
```
- Python version
- PyTorch / Gudhi versions
- Minimal reproducible example
- Expected vs. actual output
- Traceback (if applicable)
```

### Pull Requests

1. **Fork** the repository
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding standards below
4. **Add tests** in `source/tests/`
5. **Run tests**
   ```bash
   pytest source/tests/ -v
   ```
6. **Commit with a descriptive message**
   ```bash
   git commit -m "feat: add H2 barcode support to tas_core.py"
   ```
7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Open a Pull Request** against `main` with:
   - Description of the change
   - Reference to the issue it addresses
   - Any new experimental results

---

## Development Setup

```bash
# Clone
git clone https://github.com/kang-insu/TAS-Research-Suite.git
cd TAS-Research-Suite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install pytest pytest-cov black isort

# Run self-test
python source/main.py --mode test

# Run tests
pytest source/tests/ -v --cov=source
```

---

## Coding Standards

| Standard | Tool | Command |
|----------|------|---------|
| Style (PEP8) | `black` | `black source/` |
| Import order | `isort` | `isort source/` |
| Type hints | — | Add to new functions |
| Docstrings | NumPy style | See `tas_core.py` for examples |

### Function Documentation Template

```python
def my_function(x: np.ndarray, alpha: float = 2.5) -> dict:
    """
    One-line summary.

    Longer description if needed.

    Parameters
    ----------
    x     : ndarray, shape (T, D)
        Description of x.
    alpha : float
        Anisotropic weight, must be > 1.

    Returns
    -------
    result : dict
        Keys: 'hp', 'tas', 'lifetimes'.

    References
    ----------
    [1] Kang, I.-S. (2025). TAS Framework. Manuscript.
    """
```

---

## Research Contributions

We especially welcome contributions in the following areas:

### New TDA Metrics
- Higher homology orders (H₂, H₃)
- Persistent Cohomology
- Mapper-based topology
- Euler Characteristic Curves

### Persistent Homology Improvements
- Faster filtration algorithms
- GPU-native implementations (Ripser++ integration)
- Approximate PH for large-scale point clouds

### Deep Learning Benchmarks
- LLM fine-tuning collapse (LLaMA, Mistral, Qwen)
- Diffusion model training instability
- Reinforcement learning reward collapse
- Multi-modal model training dynamics

### Nonlinear Dynamical System Applications
- Lorenz, Rössler, Duffing attractor validation
- Climate tipping point detection
- Epileptic seizure prediction (EEG)
- Financial market crash early warning

### Power System Applications
- Voltage collapse early warning
- Frequency instability detection
- Distributed Energy Resource (DER) anomaly detection
- AMI-based load forecasting instability

### Medical Applications
- Alzheimer's biomarker trajectory topology
- Cardiac arrhythmia onset detection

---

## Reproducibility Policy

All contributions that claim experimental results must:

1. **Provide a script** that exactly reproduces the results from scratch
2. **Fix random seeds** for all stochastic operations
3. **Specify hardware** (CPU/GPU model, memory) and software versions
4. **Include CSV/JSON output** of raw results
5. **State the runtime** on the specified hardware

Contributions with non-reproducible results will not be merged.

---

## Patent Notice

Core methods in this repository (anisotropic persistent homology, adaptive α-tuning, Topological Fingerprint Library, TEWI) are subject to patent applications (KR, PCT, US) filed by the author. The source code is released under MIT License for **research use**. Commercial use may require a patent license — please contact the author for details.

---

## Contact

- Issues: [GitHub Issues](https://github.com/kang-insu/TAS-Research-Suite/issues)
- Email: (institutional email)
- Paper: See `paper/manuscript.tex` and associated references

Thank you for helping improve TAS Research Suite! 🔬
