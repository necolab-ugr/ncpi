import numpy as np
import pytest

from ncpi.Features import Features


# -------------------------
# Helpers
# -------------------------

def _make_samples(n_samples=6, n_points=512, seed=0):
    """Reproducible set of 1D samples with structure + noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_points, endpoint=False)
    samples = []
    for k in range(n_samples):
        x = (
            0.8 * np.sin(2 * np.pi * (5 + k) * t) +
            0.2 * np.cos(2 * np.pi * 2 * t + 0.1 * k) +
            0.15 * rng.standard_normal(n_points)
        )
        samples.append(x.astype(float))
    return samples


def _assert_specparam_schema(d):
    """Minimal schema check for specparam output dict."""
    assert isinstance(d, dict)
    for key in ["aperiodic_params", "peak_cf", "peak_pw", "peak_bw", "n_peaks",
                "selected_peaks", "all_peaks", "metrics"]:
        assert key in d


def _assert_dfa_schema(d):
    """Minimal schema check for dfa output dict."""
    assert isinstance(d, dict)
    for key in ["dfa", "window_sizes", "fluctuations", "dfa_intercept"]:
        assert key in d


def _assert_fei_schema(d):
    """Minimal schema check for fEI output dict."""
    assert isinstance(d, dict)
    for key in ["fEI_outliers_removed", "fEI_val", "num_outliers", "wAmp", "wDNF"]:
        assert key in d


# ======================================================================================
# Core compute_features behavior (method-agnostic)
# ======================================================================================

def test_compute_features_empty_returns_empty_list():
    f = Features(method="catch22", params={"normalize": False})
    out = f.compute_features([], n_jobs=1)
    assert out == []


def test_compute_features_preserves_order_for_catch22():
    pycatch22 = pytest.importorskip("pycatch22")

    samples = _make_samples(n_samples=5, n_points=400, seed=1)
    f = Features(method="catch22", params={"normalize": False})

    out = f.compute_features(samples, n_jobs=2, chunksize=1, start_method="spawn")

    assert isinstance(out, list)
    assert len(out) == len(samples)
    # Each element should be 22-length
    for row in out:
        arr = np.asarray(row)
        assert arr.shape == (22,)

    # Order check: compare to sequential reference using the same class method
    ref = [f.catch22(s) for s in samples]
    for a, b in zip(out, ref):
        assert np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-12, rtol=1e-12)


def test_compute_features_raises_on_non_1d_sample():
    pycatch22 = pytest.importorskip("pycatch22")

    good = np.random.default_rng(0).standard_normal(100)
    bad = np.zeros((10, 10))
    f = Features(method="catch22", params={"normalize": False})

    # The worker validates ndim and should raise ValueError
    with pytest.raises(ValueError):
        f.compute_features([good, bad], n_jobs=2, chunksize=1, start_method="spawn")


# ======================================================================================
# Normalization behavior (worker-side z-score + edge cases)
# ======================================================================================

def test_compute_features_constant_signal_nan_schema_catch22_when_normalize_true():
    pycatch22 = pytest.importorskip("pycatch22")

    const = np.ones(256, dtype=float)  # std == 0
    f = Features(method="catch22", params={"normalize": True})

    out = f.compute_features([const], n_jobs=1, chunksize=1, start_method="spawn")
    assert len(out) == 1
    arr = np.asarray(out[0], dtype=float)
    assert arr.shape == (22,)
    assert np.all(np.isnan(arr))


def test_compute_features_constant_signal_nan_schema_specparam_when_normalize_true():
    pytest.importorskip("specparam")

    const = np.ones(256, dtype=float)  # std == 0 -> worker returns NaN schema for specparam
    f = Features(method="specparam", params={"normalize": True, "fs": 250})

    out = f.compute_features([const], n_jobs=1, chunksize=1, start_method="spawn")
    assert len(out) == 1
    d = out[0]
    _assert_specparam_schema(d)
    assert np.allclose(d["aperiodic_params"], np.array([np.nan, np.nan]), equal_nan=True)
    assert np.isnan(d["peak_cf"])
    assert d["n_peaks"] == 0
    assert isinstance(d["metrics"], dict)


# ======================================================================================
# Specparam compute_features integration (basic sanity)
# ======================================================================================

def test_compute_features_specparam_outputs_list_of_dicts():
    pytest.importorskip("specparam")

    samples = _make_samples(n_samples=3, n_points=1024, seed=2)
    f = Features(method="specparam", params={"normalize": False, "fs": 250})

    out = f.compute_features(samples, n_jobs=2, chunksize=1, start_method="spawn")

    assert isinstance(out, list)
    assert len(out) == len(samples)
    for d in out:
        _assert_specparam_schema(d)
        # basic type sanity
        assert np.asarray(d["aperiodic_params"]).ndim == 1


def test_compute_features_dfa_outputs_list_of_dicts():
    samples = _make_samples(n_samples=2, n_points=512, seed=4)
    f = Features(
        method="dfa",
        params={
            "sampling_frequency": 100.0,
            "fit_interval": [1, 3],
            "compute_interval": [1, 3],
            "overlap": True,
            "runtime": "python",
        },
    )

    out = f.compute_features(samples, n_jobs=1, chunksize=1, start_method="spawn")

    assert isinstance(out, list)
    assert len(out) == len(samples)
    for d in out:
        _assert_dfa_schema(d)
        assert np.asarray(d["window_sizes"]).ndim == 1


def test_compute_features_fei_outputs_list_of_dicts():
    samples = _make_samples(n_samples=2, n_points=1000, seed=5)
    f = Features(
        method="fEI",
        params={
            "sampling_frequency": 100.0,
            "window_size_sec": 1.0,
            "window_overlap": 0.5,
            "runtime": "python",
            "dfa_value": 1.0,
        },
    )

    out = f.compute_features(samples, n_jobs=1, chunksize=1, start_method="spawn")

    assert isinstance(out, list)
    assert len(out) == len(samples)
    for d in out:
        _assert_fei_schema(d)
        assert np.asarray(d["wAmp"]).ndim == 1


# ======================================================================================
# Chunksize behavior (smoke test)
# ======================================================================================

@pytest.mark.parametrize("chunksize", [1, 2, 10])
def test_compute_features_chunksize_smoke_catch22(chunksize):
    pycatch22 = pytest.importorskip("pycatch22")

    samples = _make_samples(n_samples=7, n_points=300, seed=3)
    f = Features(method="catch22", params={"normalize": False})

    out = f.compute_features(samples, n_jobs=2, chunksize=chunksize, start_method="spawn")

    assert len(out) == len(samples)
    for row in out:
        assert np.asarray(row).shape == (22,)
