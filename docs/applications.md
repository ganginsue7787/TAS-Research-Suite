# Applications of TAS Framework

## 1. Deep Learning — Learning Collapse Detection

**Problem**: Detect gradient collapse before it occurs.  
**Input**: Layer activation vectors during training.  
**Output**: Precursor alert (Hₚ) and Danger alert (TAS) with lead times of 11–20 steps.

```python
from source.tas_core import compute_tas_metrics

result = compute_tas_metrics(activation_history, alpha=2.5)
# result['hp_total'] : Persistent Entropy (early warning)
# result['tas_total']: TAS (danger alert)
```

**Architectures validated**: CNN, ResNet18, ViT-B/16  
**Datasets**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100

---

## 2. Power Distribution Systems — Voltage Collapse Early Warning

**Problem**: Detect power grid instability before outage occurs.  
**Input**: (V, I, P, Q, f, θ) from SCADA/ADMS/AMI/DAS.  
**Output**: Local TAS per feeder/transformer/switch + GIS visualization.

Key features:
- No additional sensors required — uses existing SCADA data
- GIS-based hazard visualization
- Preventive maintenance priority = Local TAS × criticality weight × customer impact

**Reference**: Kang (2025b). TAS-Based Early Warning of Voltage Collapse in Power Systems.

---

## 3. Chaotic Dynamical Systems — Chaos Onset Detection

**Problem**: Detect chaos transition before Lyapunov exponent becomes positive.  
**Input**: Phase-space trajectory (θ₁, ω₁, θ₂, ω₂) of double pendulum.  
**Output**: Hₚ-based precursor 1.8 s before LLE threshold crossing.

**Reference**: Kang (2025a). TAS & Persistent Entropy for Double Pendulum Chaos.

---

## 4. Planned Applications

| Application | Status | Key Challenge |
|-------------|--------|---------------|
| LLM fine-tuning collapse (LLaMA, Mistral) | In preparation | GPU scale |
| Epileptic seizure prediction (EEG) | Planned | Clinical data access |
| Climate tipping point detection | Planned | ERA5 reanalysis data |
| Lorenz/Rössler/Duffing validation | In preparation | Cross-system generality |
| Alzheimer biomarker trajectory topology | Planned | p-tau 217 time series |

---

## 5. Extension to Other Topological Indicators

| Indicator | Status | Use Case |
|-----------|--------|---------|
| H₂ voids | Experimental | 3D manifold topology |
| Euler Characteristic Curve | Planned | Alternative to Hₚ |
| Persistence Images | Planned | ML integration |
| Mapper graphs | Planned | High-dimensional visualization |
