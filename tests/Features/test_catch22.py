import numpy as np
import pytest

from ncpi.Features import Features


# ======================================================================================
# Optional dependency handling
# ======================================================================================

pycatch22 = pytest.importorskip("pycatch22")


# ======================================================================================
# Helpers: simple, noisy inputs + parity pairing
# ======================================================================================

def _make_noisy_signal(n=2000, seed=123, fs=250.0):
    """Generate a reproducible 1D signal with structure + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = (
        0.8 * np.sin(2 * np.pi * 10 * t) +
        0.3 * np.sin(2 * np.pi * 3 * t + 0.2) +
        0.25 * rng.standard_normal(n)
    )
    return x.astype(float)


def _pair_features_vs_library(x):
    """
    Compute catch22 features using:
      - Features.catch22 (wrapper)
      - pycatch22.catch22_all (reference)

    Returns (out, ref) as numpy arrays.
    """
    feat = Features(method="catch22", params={"normalize": False})  # normalize irrelevant here
    out = feat.catch22(x)

    ref_dict = pycatch22.catch22_all(x)
    if not isinstance(ref_dict, dict) or "values" not in ref_dict:
        raise RuntimeError("Unexpected output format from pycatch22.catch22_all")
    ref = ref_dict["values"]

    return np.asarray(out, dtype=float), np.asarray(ref, dtype=float)


# ======================================================================================
# Wrapper-level behavior tests
# ======================================================================================

def test_catch22_returns_22_values():
    x = _make_noisy_signal(n=2048, seed=1)
    feat = Features(method="catch22", params={"normalize": False})
    out = feat.catch22(x)

    out = np.asarray(out)
    assert out.shape == (22,)


def test_catch22_rejects_non_1d_like_inputs():
    feat = Features(method="catch22", params={"normalize": False})
    x2d = np.zeros((10, 10))

    # Features.catch22 passes through to pycatch22; behavior depends on pycatch22,
    # but we at least assert it raises something.
    with pytest.raises(Exception):
        feat.catch22(x2d)


def test_catch22_is_deterministic_for_fixed_input():
    x = _make_noisy_signal(n=2000, seed=7)
    feat = Features(method="catch22", params={"normalize": False})

    a = np.asarray(feat.catch22(x), dtype=float)
    b = np.asarray(feat.catch22(x), dtype=float)

    assert a.shape == (22,)
    assert np.allclose(a, b, atol=0.0, rtol=0.0)


# ======================================================================================
# Parity test: Features.catch22 vs pycatch22.catch22_all
# ======================================================================================

def test_catch22_parity_against_pycatch22():
    x = _make_noisy_signal(n=4096, seed=42)

    out, ref = _pair_features_vs_library(x)

    assert out.shape == (22,)
    assert ref.shape == (22,)

    # For a thin wrapper, we expect exact equality (same library call, same input)
    # but allow tiny FP tolerance just in case of platform differences.
    assert np.allclose(out, ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 10])
def test_catch22_parity_multiple_noisy_inputs(seed):
    x = _make_noisy_signal(n=3000, seed=seed)
    out, ref = _pair_features_vs_library(x)
    assert np.allclose(out, ref, atol=1e-12, rtol=1e-12)
