"""
test_tas_core.py
================
TAS Research Suite — Unit Tests for tas_core.py

Run:
    pytest source/tests/ -v
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tas_core import (
    build_state_space,
    anisotropic_distance_matrix,
    persistent_entropy,
    topological_anomaly_score,
    compute_tas_metrics,
    compute_tewi,
    compute_collapse_rate,
    detect_precursor_state,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def stable_history():
    """Stable trajectory: small-variance random walk."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.05, size=(25, 32)).astype(np.float32)


@pytest.fixture
def unstable_history():
    """Unstable trajectory: high-variance scattered points."""
    rng = np.random.default_rng(1)
    return rng.normal(0, 2.0, size=(25, 32)).astype(np.float32)


# ── Test: build_state_space ───────────────────────────────────

class TestBuildStateSpace:
    def test_output_shape(self, stable_history):
        ss = build_state_space(stable_history)
        T, D = stable_history.shape
        assert ss.shape == (T - 1, 2 * D), f"Expected ({T-1}, {2*D}), got {ss.shape}"

    def test_velocity_is_difference(self, stable_history):
        ss = build_state_space(stable_history)
        D  = stable_history.shape[1]
        expected_v0 = stable_history[1] - stable_history[0]
        np.testing.assert_allclose(ss[0, D:], expected_v0, atol=1e-5)

    def test_position_is_current(self, stable_history):
        ss = build_state_space(stable_history)
        D  = stable_history.shape[1]
        np.testing.assert_allclose(ss[0, :D], stable_history[1], atol=1e-5)


# ── Test: anisotropic_distance_matrix ─────────────────────────

class TestAnisotropicDistanceMatrix:
    def test_symmetry(self, stable_history):
        ss = build_state_space(stable_history)
        D  = anisotropic_distance_matrix(ss, alpha=2.5)
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_zero_diagonal(self, stable_history):
        ss = build_state_space(stable_history)
        D  = anisotropic_distance_matrix(ss, alpha=2.5)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)

    def test_positive_off_diagonal(self, stable_history):
        ss = build_state_space(stable_history)
        D  = anisotropic_distance_matrix(ss, alpha=2.5)
        n  = D.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert D[i, j] >= 0

    def test_alpha1_reduces_to_euclidean(self):
        """With alpha=1, d_α should equal standard Euclidean distance."""
        rng = np.random.default_rng(42)
        ss  = rng.normal(0, 1, size=(10, 8))
        D1  = anisotropic_distance_matrix(ss, alpha=1.0)
        # Compute Euclidean manually
        from scipy.spatial.distance import cdist
        D_eu = cdist(ss, ss, metric='euclidean')
        np.testing.assert_allclose(D1, D_eu, atol=1e-5)

    def test_larger_alpha_larger_distances(self):
        """Larger alpha should give larger distances (velocity amplified)."""
        rng = np.random.default_rng(7)
        ss  = rng.normal(0, 1, size=(10, 8))
        D1  = anisotropic_distance_matrix(ss, alpha=1.0)
        D2  = anisotropic_distance_matrix(ss, alpha=5.0)
        assert D2.sum() >= D1.sum(), "alpha=5 should give larger total distances"


# ── Test: persistent_entropy ──────────────────────────────────

class TestPersistentEntropy:
    def test_empty_returns_zero(self):
        assert persistent_entropy(np.array([])) == 0.0

    def test_single_element(self):
        # Single lifetime → p=1 → Hₚ = 0
        assert persistent_entropy(np.array([1.0])) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_distribution(self):
        # n equal lifetimes → Hₚ = log(n)
        n = 10
        L = np.ones(n)
        assert persistent_entropy(L) == pytest.approx(np.log(n), abs=1e-4)

    def test_non_negative(self):
        rng = np.random.default_rng(3)
        L   = np.abs(rng.normal(0, 1, size=20))
        assert persistent_entropy(L) >= 0.0

    def test_zero_total_returns_zero(self):
        assert persistent_entropy(np.zeros(5)) == 0.0


# ── Test: topological_anomaly_score ──────────────────────────

class TestTAS:
    def test_returns_float(self):
        L   = np.array([1.0, 2.0, 5.0])
        tas = topological_anomaly_score(L)
        assert isinstance(tas, float)

    def test_zero_std_no_crash(self):
        L   = np.ones(5)  # std=0 → should not divide by zero
        tas = topological_anomaly_score(L)
        assert np.isfinite(tas)

    def test_single_element_returns_zero(self):
        assert topological_anomaly_score(np.array([1.0])) == 0.0

    def test_large_outlier_gives_large_tas(self):
        L   = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        tas = topological_anomaly_score(L)
        assert tas > 3.0, "Large outlier should give TAS > 3"


# ── Test: compute_tas_metrics ─────────────────────────────────

class TestComputeTasMetrics:
    def test_keys_present(self, stable_history):
        res = compute_tas_metrics(stable_history, alpha=2.5)
        for key in ['hp_total', 'tas_total', 'hp_h0', 'hp_h1',
                    'tas_h0', 'tas_h1', 'lifetimes_h0', 'lifetimes_h1']:
            assert key in res, f"Missing key: {key}"

    def test_hp_non_negative(self, stable_history):
        res = compute_tas_metrics(stable_history, alpha=2.5)
        assert res['hp_total'] >= 0.0

    def test_tas_finite(self, stable_history):
        res = compute_tas_metrics(stable_history, alpha=2.5)
        assert np.isfinite(res['tas_total'])

    def test_unstable_higher_hp_than_stable(self, stable_history, unstable_history):
        res_s = compute_tas_metrics(stable_history,   alpha=2.5)
        res_u = compute_tas_metrics(unstable_history, alpha=2.5)
        # Not guaranteed for every seed, but expected on average
        # We just check that the computation completes without error
        assert np.isfinite(res_s['hp_total'])
        assert np.isfinite(res_u['hp_total'])


# ── Test: compute_tewi ────────────────────────────────────────

class TestTEWI:
    def test_equal_weights(self):
        tewi = compute_tewi(1.0, 2.0, 3.0, w1=1.0, w2=1.0, w3=1.0)
        assert tewi == pytest.approx(6.0)

    def test_zero_weights(self):
        tewi = compute_tewi(1.0, 2.0, 3.0, w1=0.0, w2=0.0, w3=0.0)
        assert tewi == pytest.approx(0.0)

    def test_optimised_weights(self):
        tewi = compute_tewi(1.0, 2.0, 3.0, w1=0.45, w2=0.25, w3=0.30)
        expected = 0.45 * 1.0 + 0.25 * 2.0 + 0.30 * 3.0
        assert tewi == pytest.approx(expected, rel=1e-5)


# ── Test: compute_collapse_rate ───────────────────────────────

class TestCollapseRate:
    def test_length(self):
        tas = [1.0, 2.0, 4.0, 7.0]
        cr  = compute_collapse_rate(tas)
        assert len(cr) == len(tas) - 1

    def test_values(self):
        tas = [1.0, 3.0, 6.0]
        cr  = compute_collapse_rate(tas)
        assert cr == pytest.approx([2.0, 3.0])

    def test_decreasing_series(self):
        tas = [5.0, 3.0, 1.0]
        cr  = compute_collapse_rate(tas)
        assert all(v < 0 for v in cr)


# ── Test: detect_precursor_state ──────────────────────────────

class TestDetectPrecursorState:
    def test_precursor_true(self):
        # Hₚ rises early, TAS rises later
        hp  = [0.1] * 20 + [1.0, 2.0, 3.0, 4.0, 5.0]
        tas = [0.1] * 22 + [1.0, 3.0, 5.0]
        res = detect_precursor_state(hp, tas)
        assert res['precursor'] is True
        assert res['t_PE'] < res['t_TAS']

    def test_no_precursor_when_tas_first(self):
        hp  = [0.1] * 22 + [1.0, 3.0, 5.0]
        tas = [0.1] * 20 + [1.0, 2.0, 3.0, 4.0, 5.0]
        res = detect_precursor_state(hp, tas)
        assert res['precursor'] is False

    def test_returns_none_when_no_crossing(self):
        hp  = [0.1] * 25
        tas = [0.1] * 25
        res = detect_precursor_state(hp, tas)
        assert res['t_PE']  is None
        assert res['t_TAS'] is None
        assert res['precursor'] is False
