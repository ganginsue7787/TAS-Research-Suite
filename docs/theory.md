# TAS Theory — Mathematical Foundation

## Overview

The TAS (Topological Anomaly Score) framework is built on three mathematical pillars:
**Persistent Homology**, **Anisotropic Metric Design**, and **Statistical Anomaly Detection**.

---

## 1. Persistent Homology

Given a metric space (P, d), the **Vietoris-Rips filtration** {VR(P,r)}_{r≥0} builds
a nested sequence of simplicial complexes. As r increases, topological features appear
and disappear. Each feature is recorded as a barcode [birth_i, death_i) with lifetime:

```
L_i = death_i - birth_i
L_max = max_i L_i
```

**Stability Theorem** (Cohen-Steiner et al. 2007): Small perturbations of the point cloud
cause bounded changes in the persistence diagram (bottleneck distance ≤ Hausdorff distance).
This ensures TAS/Hₚ are noise-robust.

---

## 2. Anisotropic Metric

```
d_α(i,j) = √[ α²‖v_i − v_j‖² + ‖x_i − x_j‖² ]
```

- **x** = position components (activation values)
- **v** = velocity components (frame-to-frame differences)  
- **α > 1** amplifies velocity sensitivity

**Theorem T3** (Kang 2025): The discrimination ratio between unstable and stable trajectories
grows as O(α²) under d_α versus the isotropic metric d₁.

---

## 3. Persistent Entropy

```
Hₚ = −∑ᵢ pᵢ log(pᵢ + ε),   pᵢ = Lᵢ / ∑ⱼ Lⱼ
```

- **Low Hₚ**: one barcode dominates (simple topology, stable regime)
- **High Hₚ**: many comparable-lifetime barcodes (complex topology, unstable regime)
- **Empirical finding**: Hₚ rises before TAS, which rises before instability onset

---

## 4. Topological Anomaly Score

```
TAS = (L_max − μ_L) / σ_L
```

Standardised z-score of the maximum barcode lifetime. Large TAS indicates
anomalous isolated topological structure — a later but sharper precursor than Hₚ.

---

## 5. TEWI — Topological Early Warning Index

```
TEWI = w₁·Hₚ(H₀) + w₂·TAS + w₃·Hₚ(H₁)
```

Integrates three topological channels. H₁ loops (w₃) capture cycle structure that
collapses earliest; H₀ components (w₁) capture connectivity; TAS (w₂) captures
isolated anomalies.

---

## 6. Precursor Ordering (Empirical)

```
t_PE(H₁) < t_PE(H₀) < t_TAS < t_LC
```

Confirmed in >94% of collapse-detected runs across all experimental conditions.

---

## 7. Connections to TLD Programme

| Paper | System | Indicator | Lead Time |
|-------|---------|-----------|-----------|
| TLD-1 (Kang 2025a) | Double pendulum | Hₚ vs LLE | 1.8 s |
| TLD-2 (Kang 2025b) | Power grid | Local TAS | Minutes |
| TLD-3 (this work) | Neural network | Hₚ, TAS, TEWI | 11–20 steps |

**Unifying principle**: Topological complexity of the trajectory-space point cloud
is a leading indicator of dynamical instability, domain-agnostically.
