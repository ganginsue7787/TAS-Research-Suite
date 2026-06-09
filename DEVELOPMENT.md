# Development Guide

Welcome to the TAS-Research-Suite development guide! This document provides comprehensive instructions for setting up your development environment, contributing to the project, and maintaining code quality.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** (check with `python --version`)
- **Git** configured with your GitHub account
- **pip** or **conda** for package management (recommended: `conda` for CUDA support)
- **GPU support** (optional but recommended for full experiments)
  - NVIDIA CUDA 11.8+ (for GPU acceleration)
  - cuDNN 8.0+ (for PyTorch optimization)

---

## Setting Up Your Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/ganginsue7787/TAS-Research-Suite.git
cd TAS-Research-Suite
```

### 2. Create a Virtual Environment

**Using venv (CPU):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda (recommended for GPU):**
```bash
conda create -n tas-dev python=3.10
conda activate tas-dev
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

### 3. Install in Development Mode

```bash
# Install with all development dependencies
pip install -e ".[dev,jupyter]"

# For GPU support with ripser++
pip install -e ".[dev,jupyter,gpu]"
```

### 4. Verify Installation

```bash
# Run the self-test
python source/main.py --mode test

# Check imports
python -c "import source; import torch; import gudhi; print('All imports successful!')"
```

### 5. Set Up Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

This will automatically run linting and formatting checks before each commit.

---

## Project Structure

```
TAS-Research-Suite/
├── source/
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # CLI entry point
│   ├── tas_core.py              # Core TAS engine (VR, Hp, TAS, TEWI)
│   ├── models_cnn.py            # CNN architecture + hooks
│   ├── models_resnet.py         # ResNet18 + layer4/avgpool
│   ├── models_transformer.py    # ViT + CLS token extraction
│   ├── montecarlo.py            # Monte Carlo runner
│   ├── statistics.py            # Statistical tests
│   ├── figures.py               # Visualization (Figures 1-6)
│   ├── latex_tables.py          # LaTeX table generation
│   └── ieee_package.py          # IEEE submission assembler
├── notebooks/
│   ├── 01_quickstart.ipynb                    # Getting started guide
│   ├── 02_tas_introduction.ipynb              # TAS algorithm walkthrough
│   ├── 03_visualization.ipynb                 # Results visualization
│   └── 04_analysis_workflow.ipynb             # Full analysis pipeline
├── tests/
│   ├── __init__.py
│   ├── test_tas_core.py         # Core algorithm tests
│   ├── test_models.py           # Model architecture tests
│   ├── test_montecarlo.py       # Monte Carlo runner tests
│   └── test_statistics.py       # Statistical function tests
├── docs/
│   ├── API.md                   # API reference
│   ├── ALGORITHM.md             # Algorithm explanation
│   └── EXAMPLES.md              # Usage examples
├── results/                     # Monte Carlo outputs (auto-generated)
├── figures/                     # Generated figures
├── tables/                      # Generated LaTeX tables
├── .github/
│   ├── workflows/
│   │   ├── ci.yml               # Tests and linting
│   │   ├── tests.yml            # Comprehensive testing
│   │   ├── lint.yml             # Code quality checks
│   │   └── release.yml          # Automated releases
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md        # Bug report template
│       ├── feature_request.md   # Feature request template
│       └── question.md          # Question template
├── pyproject.toml               # Modern Python packaging config
├── setup.py                     # Legacy setup script
├── requirements.txt             # Direct pip requirements
├── .pre-commit-config.yaml      # Pre-commit hooks config
├── .gitignore
├── LICENSE
└── README.md
```

---

## Development Workflow

### 1. Creating a Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create a feature branch
git checkout -b feature/your-feature-name

# For bug fixes:
git checkout -b fix/bug-description

# For documentation:
git checkout -b docs/doc-improvement
```

### 2. Making Changes

- Make small, focused commits
- Write clear commit messages following [Conventional Commits](https://www.conventionalcommits.org/)

**Commit message examples:**
```
feat: add support for EfficientNet architecture
fix: resolve NaN values in persistent entropy calculation
docs: update API documentation for tas_core module
refactor: simplify vietoris-rips complex computation
test: add integration tests for montecarlo runner
```

### 3. Running Tests Locally

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tas_core.py -v

# Run with coverage report
pytest --cov=source --cov-report=html

# Run quick smoke test (skip slow tests)
pytest -m "not slow"
```

### 4. Code Quality Checks

```bash
# Format code with black
black source/ tests/

# Sort imports with isort
isort source/ tests/

# Lint with flake8
flake8 source/ tests/ --max-line-length=100

# Type checking with mypy
mypy source/
```

Or run all checks at once:
```bash
# Using the pre-commit hook (if installed)
pre-commit run --all-files

# Or manually via pytest plugin
pytest --cov=source --cov-report=term-missing --lf
```

### 5. Pushing and Creating a Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Go to GitHub and create a Pull Request
# Link any related issues with "Closes #123" in the PR description
```

**PR Checklist:**
- [ ] Tests pass locally (`pytest`)
- [ ] Code is formatted (`black`)
- [ ] Imports are sorted (`isort`)
- [ ] No linting issues (`flake8`)
- [ ] Type hints are correct (`mypy`)
- [ ] Docstrings are added/updated
- [ ] CHANGELOG.md is updated
- [ ] Tests cover new functionality (>80% coverage)

---

## Code Quality Standards

### Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these additions:

- **Line length:** 100 characters (enforced by `black`)
- **Type hints:** Required for public functions and methods
- **Docstrings:** Google-style docstrings for all modules, classes, and functions

### Docstring Example

```python
def calculate_persistent_entropy(births: np.ndarray, deaths: np.ndarray) -> float:
    """Calculate persistent entropy from homology data.
    
    This function computes the persistent entropy as described in [Reference],
    which measures the complexity of topological features.
    
    Args:
        births: Array of birth times from Vietoris-Rips filtration.
        deaths: Array of death times from Vietoris-Rips filtration.
        
    Returns:
        Persistent entropy value (non-negative float).
        
    Raises:
        ValueError: If births and deaths have different lengths.
        
    Example:
        >>> births = np.array([0.0, 0.1, 0.2])
        >>> deaths = np.array([0.5, 0.8, 1.0])
        >>> Hp = calculate_persistent_entropy(births, deaths)
    """
    if len(births) != len(deaths):
        raise ValueError("births and deaths must have equal length")
    
    # Implementation...
```

### Type Hints Example

```python
from typing import Optional, Tuple, List
import numpy as np
import torch

def process_activation(
    activation: torch.Tensor,
    alpha: float,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process neural activation with optional normalization.
    
    Args:
        activation: Input tensor of shape (batch_size, features)
        alpha: Anisotropy scaling parameter
        normalize: Whether to apply min-max normalization
        
    Returns:
        Tuple of (normalized_array, distances_array)
    """
    # Implementation...
```

---

## Testing

### Writing Tests

Place tests in the `tests/` directory with the naming convention `test_*.py`:

```python
# tests/test_tas_core.py
import pytest
import numpy as np
from source.tas_core import calculate_persistent_entropy


class TestPersistentEntropy:
    """Test suite for persistent entropy calculation."""
    
    def test_basic_calculation(self):
        """Test basic entropy calculation with known values."""
        births = np.array([0.0, 0.1, 0.2])
        deaths = np.array([0.5, 0.8, 1.0])
        
        Hp = calculate_persistent_entropy(births, deaths)
        
        assert isinstance(Hp, float)
        assert Hp >= 0.0
        assert not np.isnan(Hp)
    
    def test_empty_input(self):
        """Test handling of empty input arrays."""
        births = np.array([])
        deaths = np.array([])
        
        Hp = calculate_persistent_entropy(births, deaths)
        
        assert Hp == 0.0
    
    @pytest.mark.parametrize("length", [10, 100, 1000])
    def test_scaling(self, length: int):
        """Test entropy calculation scales correctly with input size."""
        births = np.linspace(0.0, 1.0, length)
        deaths = np.linspace(0.5, 1.5, length)
        
        Hp = calculate_persistent_entropy(births, deaths)
        
        assert Hp >= 0.0
        assert not np.isnan(Hp)
```

### Test Markers

Use pytest markers to organize tests:

```python
@pytest.mark.slow
def test_full_montecarlo():
    """Slow test - skipped with 'pytest -m "not slow"'."""
    pass

@pytest.mark.gpu
def test_cuda_acceleration():
    """GPU test - only runs if CUDA available."""
    pass
```

### Running Coverage Report

```bash
# Generate coverage report
pytest --cov=source --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## Documentation

### Adding Docstrings

All public modules, classes, and functions should have docstrings:

```python
"""tas_core.py

Core TAS engine implementing Vietoris-Rips filtration, persistent homology,
and topological anomaly score calculation.

This module provides the main computational pipeline for detecting topological
precursors of gradient collapse in neural networks.

Classes:
    VRComplex: Vietoris-Rips complex construction and filtration
    PersistentHomology: Persistent homology computation
    TASCalculator: Complete TAS pipeline
"""
```

### Updating CHANGELOG.md

When adding features or fixes, update `CHANGELOG.md`:

```markdown
## [Unreleased]

### Added
- Support for EfficientNet architecture
- GPU acceleration for Vietoris-Rips computation
- Jupyter notebook examples

### Fixed
- NaN handling in persistent entropy calculation
- Memory leak in activation hook

### Changed
- Improved Monte Carlo runner performance by 2x
- Updated documentation structure
```

---

## Troubleshooting

### Common Issues

**Issue: `ImportError: No module named 'source'`**

Solution: Install in development mode:
```bash
pip install -e .
```

**Issue: CUDA out of memory errors**

Solution: Reduce batch size or number of Monte Carlo runs:
```bash
python source/main.py --mode mc --batch_size 32 --n_runs 10
```

**Issue: Tests fail with `torch.cuda.OutOfMemoryError`**

Solution: Run tests on CPU:
```bash
pytest --cpu
# Or set environment variable
CUDA_VISIBLE_DEVICES="" pytest
```

**Issue: `gudhi` installation fails**

Solution: Use conda for easier installation:
```bash
conda install -c conda-forge gudhi
# Then install remaining packages
pip install -e ".[dev,jupyter]"
```

**Issue: Pre-commit hook slows down commits**

Solution: Temporarily skip hooks:
```bash
git commit --no-verify
```

Or configure pre-commit to skip certain checks:
```yaml
# In .pre-commit-config.yaml
- repo: https://github.com/psf/black
  args: ['--line-length=100']
```

### Getting Help

- **Documentation:** Check `docs/` directory
- **API Reference:** See `docs/API.md`
- **Examples:** Review `notebooks/` directory
- **Issues:** Search [GitHub Issues](https://github.com/ganginsue7787/TAS-Research-Suite/issues)
- **Discussions:** Use [GitHub Discussions](https://github.com/ganginsue7787/TAS-Research-Suite/discussions)

---

## Additional Resources

- [Python Development Guide](https://devguide.python.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Type Hints in Python](https://docs.python.org/3/library/typing.html)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## Questions?

Feel free to:
1. Check existing [Issues](https://github.com/ganginsue7787/TAS-Research-Suite/issues)
2. Start a [Discussion](https://github.com/ganginsue7787/TAS-Research-Suite/discussions)
3. Open a [New Issue](https://github.com/ganginsue7787/TAS-Research-Suite/issues/new)

Happy coding! 🚀
