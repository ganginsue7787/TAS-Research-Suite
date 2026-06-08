# Experimental Protocol

## Standard Monte Carlo Setup

### Datasets
| Dataset | Classes | Train Size | Input |
|---------|---------|-----------|-------|
| MNIST | 10 | 60,000 | 1×28×28 |
| Fashion-MNIST | 10 | 60,000 | 1×28×28 |
| CIFAR-10 | 10 | 50,000 | 3×32×32 |
| CIFAR-100 | 100 | 50,000 | 3×32×32 |

### Models and Extraction Points
| Model | Layer | Dimension |
|-------|-------|-----------|
| CNN-3 (Sigmoid) | fc1 | 256 |
| ResNet18 | avgpool | 512 |
| ViT-B/16 | CLS token (last encoder) | 768 |

### TDA Parameters
```
Window size W        : 20 batches
Anisotropic weight α : 2.5 (auto-tuned)
Homology orders      : H₀, H₁
Threshold            : μ ± 3σ (first 20 steps as baseline)
```

### Monte Carlo
- n = 50 runs per condition
- Random seed: varies per run (seed = run_index)
- Output: run_{001..050}.csv per condition

### Collapse Detection
Gradient collapse at first step where:
```
‖∇L‖₂ < 0.1 × mean(‖∇L‖₂ over first 20 steps)
```

### Metrics Computed Per Run
- `t_PE` : Hₚ first threshold crossing
- `t_TAS`: TAS first threshold crossing  
- `t_GC` : Gradient collapse step
- `delta_tas = t_GC - t_TAS`
- `delta_hp  = t_GC - t_PE`
- `tewi_series`: TEWI time series

## Baseline Comparison Protocol

All baselines use identical window size W=20, threshold μ±3σ, and evaluation metric (lead time Δt vs gradient collapse).

### Neural Persistence
Computed from weight matrix filtration (not activations). Serves as topological baseline.

### Jacobian Spectral Norm
σ_max(∂f/∂x) via power iteration at each batch.

### Hessian Sharpness
λ_max(∇²L) via Lanczos iteration (Hessian-vector products only).

### Fisher Information (diagonal)
E[‖∂ log p/∂θ‖²] approximated via batch gradients.

### Gradient Norm (reference baseline)
‖∇L‖₂ per batch — defines t_GC.
