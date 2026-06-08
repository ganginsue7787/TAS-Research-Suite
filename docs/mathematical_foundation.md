# Mathematical Foundation

## Key Theorems and Results

### Theorem T3 — Anisotropic Sensitivity (Kang 2025a)

For α > 1 and chaotic trajectory X_c vs periodic trajectory X_p at equal energy:

```
L_max(X_c; d_α) / L_max(X_p; d_α)  >  L_max(X_c; d₁) / L_max(X_p; d₁)
```

The ratio grows as O(α²) in the velocity-dominated regime.

### Theorem T2 — Verified Precursor Ordering (Kang 2025a)

Empirically confirmed: t_PE < t_LLE in the double pendulum chaotic regime.
Δt = 1.8 seconds lead time.

### Conjecture 1 — Jacobian-Topology Bound (Kang 2025a)

```
L_max(t) ≤ C · exp[ ∫₀ᵗ λ_max(J(s)) ds ]
```

Proof in preparation. Experimental boundary condition confirmed numerically.

### Theorem 1 — Manifold Fragmentation (Kang 2025c, this work)

As activation manifold M_t approaches fragmentation:
- Hₚ(t) → log(dim H₀(M_t))    [maximum entropy as components equalise]
- TAS(t) → ∞                    [L_max diverges from population mean]

Proof sketch provided in manuscript. Full proof in preparation.

## Stability of Persistent Homology

**Cohen-Steiner et al. (2007)**: For persistence diagrams D_f, D_g computed from
functions f, g on a topological space:

```
d_bottleneck(D_f, D_g) ≤ ‖f − g‖_∞
```

Applied to point clouds via the Hausdorff distance, this guarantees that small
activation perturbations (noise, batch variation) cause bounded changes in
TAS and Hₚ — ensuring noise robustness of the early-warning signals.

## Bayesian Optimisation for α

The α-tuning objective is:
```
α* = argmax_α D(α)
D(α) = |Hₚ(unstable window; d_α) − Hₚ(stable window; d_α)| / σ_noise
```

Lazy Filtration update rule: rescale edge weights by α² without rebuilding
the simplicial complex. Complexity: O(n²) per α update vs O(n³) for full rebuild.
