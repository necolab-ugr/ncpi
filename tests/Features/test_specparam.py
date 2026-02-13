import numpy as np
import pytest

import scipy.signal as scipy_signal

specparam = pytest.importorskip("specparam")
from specparam import SpectralModel
from specparam.sim import sim_power_spectrum
from specparam.sim.utils import set_random_seed

from ncpi.Features import Features


# ======================================================================================
# Helpers to make direct, apples-to-apples comparisons against specparam.SpectralModel
# ======================================================================================

def _sanitize_like_features(freqs, psd):
    """Mirror Features.specparam: squeeze -> filter finite & positive freqs and psd."""
    freqs = np.asarray(freqs).squeeze()
    psd = np.asarray(psd).squeeze()
    m = np.isfinite(freqs) & np.isfinite(psd) & (freqs > 0) & (psd > 0)
    freqs = freqs[m]
    psd = psd[m]
    return freqs, psd


def _clip_freq_range_like_features(freqs, freq_range):
    """Mirror Features.specparam: clip requested range to available freqs."""
    fmin, fmax = map(float, freq_range)
    if fmin >= fmax:
        raise ValueError("`freq_range` must be (fmin, fmax) with fmin < fmax.")
    fmin = max(fmin, float(np.min(freqs)))
    fmax = min(fmax, float(np.max(freqs)))
    if fmin >= fmax:
        raise ValueError("`freq_range` is outside the available frequency support.")
    return [fmin, fmax]


def _select_peaks_like_features(peaks, select_peak, fmin, fmax):
    """Mirror the exact selection logic in Features.specparam."""
    peak_cf = peak_pw = peak_bw = np.nan
    all_peaks = np.empty((0, 3), dtype=float)
    selected_peaks = np.empty((0, 3), dtype=float)

    peaks = np.asarray(peaks, dtype=float)
    if peaks.ndim == 1 and peaks.size == 3:
        peaks = peaks.reshape(1, 3)
    elif peaks.size == 0:
        peaks = np.empty((0, 3), dtype=float)

    n_peaks = peaks.shape[0]

    if n_peaks > 0:
        cfs = peaks[:, 0]
        pws = peaks[:, 1]

        if select_peak == "all":
            all_peaks = peaks

        elif select_peak == "max_pw":
            idx = int(np.nanargmax(pws))
            selected_peaks = peaks[idx:idx + 1]

        elif select_peak == "max_cf_in_range":
            mask = (cfs >= fmin) & (cfs <= fmax)
            if np.any(mask):
                idx = int(np.nanargmax(cfs[mask]))
                selected_peaks = peaks[mask][idx:idx + 1]

        else:
            raise ValueError("select_peak must be one of {'all', 'max_pw', 'max_cf_in_range'}")

    if selected_peaks.shape[0] == 1:
        peak_cf, peak_pw, peak_bw = map(float, selected_peaks[0])

    return peak_cf, peak_pw, peak_bw, selected_peaks, all_peaks


def _extract_reference_schema(freqs, psd, *, freq_range, model_kwargs, select_peak):
    """
    Fit SpectralModel directly and extract outputs in the SAME schema as Features.specparam.
    """
    freqs, psd = _sanitize_like_features(freqs, psd)
    fr = _clip_freq_range_like_features(freqs, freq_range)
    fmin, fmax = fr

    fm = SpectralModel(**model_kwargs)
    fm.fit(freqs, psd, fr)

    aperiodic_params = np.asarray(fm.results.params.aperiodic.params, dtype=float).squeeze()
    aperiodic_params = np.atleast_1d(aperiodic_params).astype(float)

    peaks = np.asarray(fm.results.params.periodic.params, dtype=float)
    try:
        n_peaks = int(fm.results.n_peaks)
    except Exception:
        n_peaks = int(peaks.shape[0])

    peak_cf, peak_pw, peak_bw, selected_peaks, all_peaks = _select_peaks_like_features(
        peaks=peaks, select_peak=select_peak, fmin=fmin, fmax=fmax
    )

    metric_values = fm.results.metrics.results
    if model_kwargs.get("metrics"):
        metrics_out = {m: float(metric_values.get(m, np.nan)) for m in model_kwargs["metrics"]}
    else:
        metrics_out = {"gof_rsquared": float(metric_values.get("gof_rsquared", np.nan))}

    return {
        "aperiodic_params": aperiodic_params,
        "peak_cf": peak_cf,
        "peak_pw": peak_pw,
        "peak_bw": peak_bw,
        "n_peaks": n_peaks,
        "selected_peaks": selected_peaks,
        "all_peaks": all_peaks,
        "metrics": metrics_out,
    }


def _assert_peaks_close(a, b, atol=(1e-2, 1e-2, 1e-2), rtol=1e-3):
    """Compare peak arrays after sorting by CF (order can differ)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    assert a.shape == b.shape
    if a.size == 0:
        return

    a = a[np.argsort(a[:, 0])]
    b = b[np.argsort(b[:, 0])]

    assert np.allclose(a[:, 0], b[:, 0], atol=atol[0], rtol=rtol)  # CF
    assert np.allclose(a[:, 1], b[:, 1], atol=atol[1], rtol=rtol)  # PW
    assert np.allclose(a[:, 2], b[:, 2], atol=atol[2], rtol=rtol)  # BW


def _assert_outputs_match(out, ref, *, select_peak):
    # Aperiodic params
    assert out["aperiodic_params"].shape == ref["aperiodic_params"].shape
    assert np.allclose(out["aperiodic_params"], ref["aperiodic_params"], atol=1e-2, rtol=1e-3)

    # Peak count
    assert out["n_peaks"] == ref["n_peaks"]

    # Peaks based on mode
    if select_peak == "all":
        _assert_peaks_close(out["all_peaks"], ref["all_peaks"])
        assert out["selected_peaks"].shape == (0, 3)
        assert np.isnan(out["peak_cf"]) and np.isnan(out["peak_pw"]) and np.isnan(out["peak_bw"])
    else:
        assert np.allclose(out["selected_peaks"], ref["selected_peaks"], atol=1e-2, rtol=1e-3) or (
            out["selected_peaks"].size == 0 and ref["selected_peaks"].size == 0
        )
        # scalar peak values should agree, or both be NaN
        for k in ["peak_cf", "peak_pw", "peak_bw"]:
            assert np.isclose(out[k], ref[k], atol=1e-2, rtol=1e-3) or (np.isnan(out[k]) and np.isnan(ref[k]))

    # Metrics dict keys match & values match
    assert set(out["metrics"].keys()) == set(ref["metrics"].keys())
    for mk in out["metrics"].keys():
        assert np.isclose(out["metrics"][mk], ref["metrics"][mk], atol=1e-6, rtol=1e-4) or (
            np.isnan(out["metrics"][mk]) and np.isnan(ref["metrics"][mk])
        )


# ======================================================================================
# Fixtures
# ======================================================================================

@pytest.fixture
def feat_default():
    # A conservative default that should fit typical simulated spectra
    return Features(
        method="specparam",
        params={
            "fs": 250.0,
            "freq_range": (3, 40),
            "specparam_model": {"min_peak_height": 0.05, "verbose": False},
        },
    )


# ======================================================================================
# Wrapper-level behavior tests (no direct parity required)
# ======================================================================================

def test_requires_either_sample_or_psd(feat_default):
    with pytest.raises(ValueError):
        feat_default.specparam()
    freqs = np.linspace(1, 40, 50)
    psd = np.ones_like(freqs)
    with pytest.raises(ValueError):
        feat_default.specparam(sample=np.random.randn(1000), freqs=freqs, power_spectrum=psd)


def test_sample_must_be_1d(feat_default):
    with pytest.raises(ValueError, match=r"1D"):
        feat_default.specparam(sample=np.zeros((10, 10)), fs=250.0)


def test_freqs_psd_must_be_1d(feat_default):
    freqs = np.linspace(1, 40, 50)
    psd_bad = np.ones((2, 50))
    with pytest.raises(ValueError, match=r"must be 1D"):
        feat_default.specparam(freqs=freqs, power_spectrum=psd_bad)


def test_freqs_psd_same_length(feat_default):
    freqs = np.linspace(1, 40, 100)
    psd = np.ones(99)
    with pytest.raises(ValueError, match=r"same length"):
        feat_default.specparam(freqs=freqs, power_spectrum=psd)


def test_freq_range_sanity(feat_default):
    freqs = np.linspace(1, 40, 100)
    psd = (1 / freqs) + 1e-3
    with pytest.raises(ValueError, match=r"fmin < fmax"):
        feat_default.specparam(freqs=freqs, power_spectrum=psd, freq_range=(40, 1))


def test_select_peak_invalid_raises():
    from specparam.sim import sim_power_spectrum
    from specparam.sim.utils import set_random_seed

    set_random_seed(21)

    # Make a spectrum with a clear oscillatory peak
    freqs, psd = sim_power_spectrum(
        [3, 40],
        {"fixed": [1, 1]},
        {"gaussian": [[10, 0.4, 2.0]]},  # strong peak
        nlv=0.001,
    )

    feat = Features(
        method="specparam",
        params={
            "fs": 250.0,
            "freq_range": (3, 40),
            "specparam_model": {"min_peak_height": 0.01, "verbose": False},
        },
    )

    # Now invalid select_peak must be evaluated because there will be peaks
    with pytest.raises(ValueError, match=r"select_peak must be one of"):
        feat.specparam(freqs=freqs, power_spectrum=psd, select_peak="not_a_mode")


def test_param_precedence_explicit_over_params():
    """
    Ensures explicit arguments override self.params (as in _resolve_param).
    """
    set_random_seed(21)
    freqs, psd = sim_power_spectrum([3, 40], {"fixed": [1, 1]}, {"gaussian": [[10, 0.2, 1.25]]})

    feat = Features(
        method="specparam",
        params={
            "fs": 250.0,
            "freq_range": (3, 40),
            "select_peak": "all",
            "specparam_model": {"min_peak_height": 0.05, "verbose": False},
        },
    )

    # Override select_peak explicitly
    out = feat.specparam(freqs=freqs, power_spectrum=psd, select_peak="max_pw")
    assert out["selected_peaks"].shape in [(0, 3), (1, 3)]
    assert out["all_peaks"].shape == (0, 3)


def test_freq_range_clips_to_support(feat_default):
    """
    Features.specparam clips freq_range to freqs support before passing to SpectralModel.
    """
    set_random_seed(21)
    freqs, psd = sim_power_spectrum([3, 40], {"fixed": [1, 1]}, {"gaussian": [[10, 0.2, 1.25]]})

    # Pass an intentionally too-wide range
    out = feat_default.specparam(freqs=freqs, power_spectrum=psd, freq_range=(0, 1000))
    assert isinstance(out["metrics"]["gof_rsquared"], float)


def test_no_peaks_behavior():
    """
    If you set min_peak_height super high, you should get 0 peaks and NaN peak scalars.
    """
    set_random_seed(21)
    freqs, psd = sim_power_spectrum([3, 40], {"fixed": [1, 1]}, {"gaussian": [[10, 0.2, 1.25]]})

    feat = Features(
        method="specparam",
        params={
            "fs": 250.0,
            "freq_range": (3, 40),
            "specparam_model": {"min_peak_height": 10.0, "verbose": False},  # effectively "no peaks"
        },
    )
    out = feat.specparam(freqs=freqs, power_spectrum=psd, select_peak="max_pw")
    assert out["n_peaks"] == 0
    assert out["selected_peaks"].shape == (0, 3)
    assert out["all_peaks"].shape == (0, 3)
    assert np.isnan(out["peak_cf"]) and np.isnan(out["peak_pw"]) and np.isnan(out["peak_bw"])


# ======================================================================================
# Parity tests: Features.specparam vs direct SpectralModel
# ======================================================================================

@pytest.mark.parametrize("select_peak", ["max_pw", "max_cf_in_range", "all"])
def test_parity_fixed_gaussian_basic(select_peak):
    set_random_seed(21)

    freqs, psd = sim_power_spectrum(
        [3, 40],
        {"fixed": [1, 1]},
        {"gaussian": [[10, 0.2, 1.25], [30, 0.15, 2.0]]},
    )
    freq_range = (3, 40)
    model_kwargs = {"min_peak_height": 0.05, "verbose": False}

    ref = _extract_reference_schema(
        freqs, psd, freq_range=freq_range, model_kwargs=model_kwargs, select_peak=select_peak
    )

    feat = Features(method="specparam", params={"fs": 250.0, "freq_range": freq_range, "specparam_model": model_kwargs})
    out = feat.specparam(freqs=freqs, power_spectrum=psd, select_peak=select_peak)

    _assert_outputs_match(out, ref, select_peak=select_peak)


def test_parity_knee_mode():
    """
    Knee-mode parity (aperiodic_mode='knee') as in tutorial patterns.
    """
    set_random_seed(21)

    freqs, psd = sim_power_spectrum(
        [1, 150],
        {"knee": [1, 125, 1.25]},
        {"gaussian": [[8, 0.15, 1.0], [30, 0.1, 2.0]]},
    )
    freq_range = (1, 150)
    model_kwargs = {"min_peak_height": 0.05, "aperiodic_mode": "knee", "verbose": False}

    ref = _extract_reference_schema(
        freqs, psd, freq_range=freq_range, model_kwargs=model_kwargs, select_peak="all"
    )

    feat = Features(method="specparam", params={"fs": 250.0, "freq_range": freq_range, "specparam_model": model_kwargs})
    out = feat.specparam(freqs=freqs, power_spectrum=psd, select_peak="all")

    _assert_outputs_match(out, ref, select_peak="all")


def test_parity_custom_metrics():
    """
    Custom metrics parity: Features returns exactly the metrics list when passed in model kwargs.
    """
    set_random_seed(21)

    freqs, psd = sim_power_spectrum([3, 40], {"fixed": [1, 1]}, {"gaussian": [[10, 0.2, 1.25]]})
    freq_range = (3, 40)

    metrics = ["error_mse", "gof_adjrsquared"]
    model_kwargs = {"min_peak_height": 0.05, "verbose": False, "metrics": metrics}

    ref = _extract_reference_schema(
        freqs, psd, freq_range=freq_range, model_kwargs=model_kwargs, select_peak="max_pw"
    )

    feat = Features(method="specparam", params={"fs": 250.0, "freq_range": freq_range, "specparam_model": model_kwargs})
    out = feat.specparam(freqs=freqs, power_spectrum=psd, select_peak="max_pw")

    _assert_outputs_match(out, ref, select_peak="max_pw")
    assert set(out["metrics"].keys()) == set(metrics)


def test_parity_max_cf_in_range_when_none_in_range():
    """
    If no peaks fall inside [fmin,fmax], Features returns empty selected_peaks and NaN scalars.
    Compare to reference extraction.
    """
    set_random_seed(21)

    # Put peaks at ~10 and ~30, then choose a narrow freq_range away from them
    freqs, psd = sim_power_spectrum(
        [3, 40],
        {"fixed": [1, 1]},
        {"gaussian": [[10, 0.2, 1.25], [30, 0.15, 2.0]]},
    )

    freq_range = (20, 25)  # excludes both peaks
    model_kwargs = {"min_peak_height": 0.05, "verbose": False}
    select_peak = "max_cf_in_range"

    ref = _extract_reference_schema(
        freqs, psd, freq_range=freq_range, model_kwargs=model_kwargs, select_peak=select_peak
    )

    feat = Features(method="specparam", params={"fs": 250.0, "freq_range": freq_range, "specparam_model": model_kwargs})
    out = feat.specparam(freqs=freqs, power_spectrum=psd, select_peak=select_peak)

    _assert_outputs_match(out, ref, select_peak=select_peak)
    assert out["selected_peaks"].shape == (0, 3)
    assert np.isnan(out["peak_cf"]) and np.isnan(out["peak_pw"]) and np.isnan(out["peak_bw"])


def test_parity_sample_path_welch_matches_manual_psd():
    """
    Test the 'sample' path: Features computes Welch internally.
    We compare:
      - Features(specparam(sample=...)) result
      - Reference SpectralModel fit on the *same Welch PSD*
    """
    set_random_seed(21)

    fs = 250.0
    dur_s = 4.0
    n = int(fs * dur_s)
    t = np.arange(n) / fs

    # signal with oscillation ~10 Hz + noise, so we get a reasonable peak
    x = np.sin(2 * np.pi * 10 * t) + 0.25 * np.random.randn(n)

    welch_kwargs = {"nperseg": int(0.5 * fs)}  # match Features default segment length
    freqs, psd = scipy_signal.welch(x, fs=fs, **welch_kwargs)

    freq_range = (3, 40)
    model_kwargs = {"min_peak_height": 0.05, "verbose": False}
    select_peak = "max_pw"

    # Reference on the Welch PSD
    ref = _extract_reference_schema(
        freqs, psd, freq_range=freq_range, model_kwargs=model_kwargs, select_peak=select_peak
    )

    # Features using sample path
    feat = Features(
        method="specparam",
        params={
            "fs": fs,
            "freq_range": freq_range,
            "welch_kwargs": welch_kwargs,
            "specparam_model": model_kwargs,
        },
    )
    out = feat.specparam(sample=x, select_peak=select_peak)

    _assert_outputs_match(out, ref, select_peak=select_peak)


def test_parity_freq_range_clipping_outside_support():
    """
    Mirrors Features behavior where freq_range is clipped before fitting.
    Compare to a reference extraction that does the same clip.
    """
    set_random_seed(21)

    freqs, psd = sim_power_spectrum([3, 40], {"fixed": [1, 1]}, {"gaussian": [[10, 0.2, 1.25]]})

    # intentionally outside support: should clip to [min(freqs), max(freqs)]
    freq_range = (0, 1000)
    model_kwargs = {"min_peak_height": 0.05, "verbose": False}
    select_peak = "max_pw"

    ref = _extract_reference_schema(freqs, psd, freq_range=freq_range, model_kwargs=model_kwargs, select_peak=select_peak)

    feat = Features(method="specparam", params={"fs": 250.0, "freq_range": freq_range, "specparam_model": model_kwargs})
    out = feat.specparam(freqs=freqs, power_spectrum=psd, select_peak=select_peak)

    _assert_outputs_match(out, ref, select_peak=select_peak)
