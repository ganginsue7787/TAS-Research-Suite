"""
tas_core.py
===========
TAS Journal Suite — Core Engine

Implements:
  - State-space construction from activation trajectories
  - Anisotropic metric distance matrix  (Patent Claim 1b / Theorem T3)
  - Vietoris-Rips persistent homology   (H0, H1)
  - Persistent Entropy Hp
  - Topological Anomaly Score TAS = (Lmax - mu) / sigma
  - Topological Collapse Rate  CollapseRate = d(TAS)/dt
  - TEWI = w1*Hp(H0) + w2*TAS + w3*Hp(H1)

Reference:
  Kang, I.-S. (2025). Topological Anomaly Score and Persistent Entropy
  for Early Detection of Chaos Transition in Double Pendulum Dynamics.
  (TLD Framework, v3)

Author  : Kang, In-Su
License : MIT
"""

import numpy as np
import gudhi as gd


# ── 1. State-Space Construction ───────────────────────────────

def build_state_space(activation_history: np.ndarray) -> np.ndarray:
    """
    Build [position | velocity] state-space from activation trajectory.

    Parameters
    ----------
    activation_history : ndarray, shape (T, D)
        Time-ordered activation snapshots.

    Returns
    -------
    state_space : ndarray, shape (T-1, 2D)
    """
    current  = activation_history[1:]
    previous = activation_history[:-1]
    velocity = current - previous
    return np.concatenate([current, velocity], axis=1)


# ── 2. Anisotropic Distance Matrix ────────────────────────────

def anisotropic_distance_matrix(state_space: np.ndarray,
                                alpha: float = 2.5) -> np.ndarray:
    """
    d_alpha(i,j) = sqrt( alpha^2 * ||v_i - v_j||^2 + ||x_i - x_j||^2 )

    Velocity components are weighted by alpha > 1, amplifying
    topological separation between stable and unstable regimes
    at a rate proportional to alpha^2  (Theorem T3).

    Parameters
    ----------
    state_space : ndarray, shape (N, 2D)
    alpha       : float > 1, anisotropic weight

    Returns
    -------
    D : ndarray, shape (N, N), symmetric
    """
    n   = state_space.shape[0]
    mid = state_space.shape[1] // 2
    x   = state_space[:, :mid]
    v   = state_space[:, mid:]

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dv = np.sum((v[i] - v[j]) ** 2)
            dx = np.sum((x[i] - x[j]) ** 2)
            d  = np.sqrt(alpha ** 2 * dv + dx)
            D[i, j] = d
            D[j, i] = d
    return D


# ── 3. Persistent Homology ────────────────────────────────────

def compute_persistence(distance_matrix: np.ndarray,
                        max_dimension: int = 2,
                        max_edge_length: float = None):
    """
    Construct Vietoris-Rips filtration and compute H0, H1 persistence.

    Returns
    -------
    simplex_tree : gudhi.SimplexTree (with persistence computed)
    """
    if max_edge_length is None:
        max_edge_length = float(np.max(distance_matrix))
    rips = gd.RipsComplex(distance_matrix=distance_matrix,
                          max_edge_length=max_edge_length)
    st   = rips.create_simplex_tree(max_dimension=max_dimension)
    st.persistence()
    return st


# ── 4. Barcode Lifetimes ──────────────────────────────────────

def extract_lifetimes(intervals: np.ndarray) -> np.ndarray:
    """
    Convert barcode [birth, death] intervals to lifetime vector.
    Infinite deaths -> 1.2 x max_finite_death.
    """
    if len(intervals) == 0:
        return np.array([])
    finite    = intervals[np.isfinite(intervals[:, 1])]
    max_death = float(np.max(finite[:, 1])) if len(finite) > 0 else 1.0
    lifetimes = []
    for birth, death in intervals:
        if not np.isfinite(death):
            death = max_death * 1.2
        lifetimes.append(death - birth)
    return np.array(lifetimes)


# ── 5. Persistent Entropy ─────────────────────────────────────

def persistent_entropy(lifetimes: np.ndarray) -> float:
    """
    Hp = -sum_i  p_i * log(p_i),   p_i = L_i / sum(L)

    Early-warning indicator: Hp rises before TAS and LLE
    (Theorem T2 — empirically verified, t_PE < t_LLE by 1.8 s).
    """
    if len(lifetimes) == 0 or np.sum(lifetimes) <= 0:
        return 0.0
    p = lifetimes / np.sum(lifetimes)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


# ── 6. Topological Anomaly Score ──────────────────────────────

def topological_anomaly_score(lifetimes: np.ndarray) -> float:
    """
    TAS = (L_max - mu_L) / sigma_L

    Structural severity monitor: TAS detects sustained anomalies
    after instability has fully developed (t_TAS > t_LLE typically).

    This formula is one specific embodiment; the independent patent
    claim uses the broader 'statistical anomaly index of the barcode
    lifetime distribution'.
    """
    if len(lifetimes) < 2:
        return 0.0
    return float((np.max(lifetimes) - np.mean(lifetimes)) /
                 (np.std(lifetimes) + 1e-8))


# ── 7. Per-dimension Helpers ──────────────────────────────────

def _dim_metrics(simplex_tree, dim: int) -> dict:
    intervals = simplex_tree.persistence_intervals_in_dimension(dim)
    L = extract_lifetimes(intervals)
    return {"lifetimes": L, "hp": persistent_entropy(L),
            "tas": topological_anomaly_score(L)}


# ── 8. Main TAS Engine ────────────────────────────────────────

def compute_tas_metrics(activation_history: np.ndarray,
                        alpha: float = 2.5) -> dict:
    """
    Full TAS pipeline.

    Parameters
    ----------
    activation_history : ndarray, shape (T, D)
    alpha              : anisotropic weight alpha > 1

    Returns
    -------
    dict with keys:
        hp_total, tas_total,
        hp_h0, hp_h1, tas_h0, tas_h1,
        lifetimes_h0, lifetimes_h1
    """
    ss = build_state_space(activation_history)
    D  = anisotropic_distance_matrix(ss, alpha=alpha)
    st = compute_persistence(D)
    h0 = _dim_metrics(st, 0)
    h1 = _dim_metrics(st, 1)
    return {
        "hp_total":     h0["hp"]  + h1["hp"],
        "tas_total":    h0["tas"] + h1["tas"],
        "hp_h0":        h0["hp"],   "hp_h1":  h1["hp"],
        "tas_h0":       h0["tas"],  "tas_h1": h1["tas"],
        "lifetimes_h0": h0["lifetimes"],
        "lifetimes_h1": h1["lifetimes"],
    }


# ── 9. TEWI — Topological Early Warning Index ─────────────────

def compute_tewi(hp_h0: float, tas: float, hp_h1: float,
                 w1: float = 1.0, w2: float = 1.0, w3: float = 1.0) -> float:
    """
    TEWI = w1*Hp(H0) + w2*TAS + w3*Hp(H1)

    Weighted combination of three topological channels.
    Weights determined by training data or operator configuration.
    (Patent Claim 8, v3)
    """
    return w1 * hp_h0 + w2 * tas + w3 * hp_h1


# ── 10. Collapse Rate ─────────────────────────────────────────

def compute_collapse_rate(tas_series: list) -> list:
    """
    CollapseRate(t) = TAS(t) - TAS(t-1)   [discrete d/dt]

    Issues structural collapse acceleration alert when
    CollapseRate > threshold.  (Patent Claim 7, v3)
    """
    return list(np.diff(np.array(tas_series, dtype=float)))


# ── 11. Precursor State Detection ─────────────────────────────

def detect_precursor_state(hp_series: list, tas_series: list,
                            hp_thresh: float = None,
                            tas_thresh: float = None) -> dict:
    """
    Classify interval as 'Precursor State' when
        t_PE  (Hp threshold crossing) < t_TAS (TAS threshold crossing)

    Returns times and precursor flag.  (Patent Claim 6, v3)
    """
    hp  = np.array(hp_series)
    tas = np.array(tas_series)

    if hp_thresh is None:
        hp_thresh  = np.mean(hp[:20])  + 3 * np.std(hp[:20])
    if tas_thresh is None:
        tas_thresh = np.mean(tas[:20]) + 3 * np.std(tas[:20])

    hp_cross  = next((i for i, v in enumerate(hp)  if v > hp_thresh),  None)
    tas_cross = next((i for i, v in enumerate(tas) if v > tas_thresh), None)

    precursor = (hp_cross is not None and
                 tas_cross is not None and
                 hp_cross < tas_cross)
    return {"t_PE": hp_cross, "t_TAS": tas_cross, "precursor": precursor}


# ── Self-Test ─────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    history = np.random.randn(30, 128)
    res = compute_tas_metrics(history, alpha=2.5)
    print("=== TAS Core Self-Test ===")
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray(len={len(v)})")
        else:
            print(f"  {k}: {v:.4f}")
    tewi = compute_tewi(res["hp_h0"], res["tas_h0"], res["hp_h1"])
    print(f"  TEWI: {tewi:.4f}")
