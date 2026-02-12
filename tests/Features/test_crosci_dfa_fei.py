import numpy as np
import numpy.ma as np_ma
from scipy.stats import t as scipy_t

from ncpi.Features import Features


def _create_window_indices(length_signal, length_window, window_offset):
    window_starts = np.arange(0, length_signal - length_window, window_offset)
    num_windows = len(window_starts)

    one_window_index = np.arange(0, length_window)
    all_window_index = np.tile(one_window_index, (num_windows, 1)).astype(int)

    all_window_index = all_window_index + np.tile(
        np.transpose(window_starts[np.newaxis, :]), (1, length_window)
    ).astype(int)

    return all_window_index


def _generalized_esd(x, max_ols, alpha=0.05, full_output=False, ubvar=False):
    if max_ols < 1:
        raise ValueError(
            "Maximum number of outliers, `max_ols`, must be > 1. "
            "Specify, e.g., max_ols = 2."
        )
    xm = np_ma.array(x)
    n = len(xm)

    r_vals = []
    l_vals = []
    max_indices = []
    for i in range(max_ols + 1):
        xmean = xm.mean()
        xstd = xm.std(ddof=int(ubvar))
        rr = np.abs((xm - xmean) / xstd)
        max_indices.append(np.argmax(rr))
        r_vals.append(rr[max_indices[-1]])
        if i >= 1:
            p = 1.0 - alpha / (2.0 * (n - i + 1))
            per_point = scipy_t.ppf(p, n - i - 1)
            l_vals.append(
                (n - i) * per_point / np.sqrt((n - i - 1 + per_point**2) * (n - i + 1))
            )
        xm[max_indices[-1]] = np_ma.masked

    r_vals.pop(-1)
    ofound = False
    for i in range(max_ols - 1, -1, -1):
        if r_vals[i] > l_vals[i]:
            ofound = True
            break

    if ofound:
        if not full_output:
            return i + 1, max_indices[0: i + 1]
        return i + 1, max_indices[0: i + 1], r_vals, l_vals, max_indices

    if not full_output:
        return 0, []
    return 0, [], r_vals, l_vals, max_indices


def _dfa_reference(sample, sampling_frequency, fit_interval, compute_interval, overlap):
    window_sizes = np.floor(np.logspace(-1, 3, 81) * sampling_frequency).astype(int)
    window_sizes = np.sort(np.unique(window_sizes))
    window_sizes = window_sizes[
        (window_sizes >= compute_interval[0] * sampling_frequency)
        & (window_sizes <= compute_interval[1] * sampling_frequency)
    ]

    fluctuations = np.full(window_sizes.shape, np.nan, dtype=float)

    window_overlap = 0.5 if overlap else 0.0
    window_offset = np.floor(window_sizes * (1 - window_overlap)).astype(int)
    signal_profile = np.cumsum(sample - np.mean(sample))

    for i_window_size, window_size in enumerate(window_sizes):
        offset = int(window_offset[i_window_size])
        if offset <= 0:
            continue
        all_window_index = _create_window_indices(sample.shape[0], int(window_size), offset)
        if all_window_index.size == 0:
            continue
        x_signal = signal_profile[all_window_index]
        _, fluc, _, _, _ = np.polyfit(
            np.arange(window_size), np.transpose(x_signal), deg=1, full=True
        )
        fluctuations[i_window_size] = np.mean(np.sqrt(fluc / window_size))

    fit_interval_first_window = np.argwhere(
        window_sizes >= fit_interval[0] * sampling_frequency
    )[0][0]
    fit_interval_last_window = np.argwhere(
        window_sizes <= fit_interval[1] * sampling_frequency
    )[-1][0]

    if fit_interval_first_window > 0:
        if (
            np.abs(
                window_sizes[fit_interval_first_window - 1] / sampling_frequency
                - fit_interval[0]
            )
            <= fit_interval[0] / 100
        ):
            if np.abs(
                window_sizes[fit_interval_first_window - 1] / sampling_frequency
                - fit_interval[0]
            ) < np.abs(
                window_sizes[fit_interval_first_window] / sampling_frequency
                - fit_interval[0]
            ):
                fit_interval_first_window = fit_interval_first_window - 1

    x_fit = np.log10(
        window_sizes[fit_interval_first_window: fit_interval_last_window + 1]
    )
    y_fit = np.log10(
        fluctuations[fit_interval_first_window: fit_interval_last_window + 1]
    )
    model = np.polyfit(x_fit, y_fit, 1)
    dfa_intercept = model[1]
    dfa_val = model[0]

    return dfa_val, window_sizes, fluctuations, dfa_intercept


def _fei_reference(
    sample,
    sampling_frequency,
    window_size_sec,
    window_overlap,
    dfa_value,
    dfa_threshold=0.6,
):
    window_size = int(window_size_sec * sampling_frequency)
    window_offset = int(np.floor(window_size * (1 - window_overlap)))
    all_window_index = _create_window_indices(sample.shape[0], window_size, window_offset)

    signal_profile = np.cumsum(sample - np.mean(sample))
    w_original_amp = np.mean(sample[all_window_index], axis=1)

    x_amp = np.tile(np.transpose(w_original_amp[np.newaxis, :]), (1, window_size))
    x_signal = signal_profile[all_window_index]
    x_signal = np.divide(x_signal, x_amp)

    _, fluc, _, _, _ = np.polyfit(
        np.arange(window_size), np.transpose(x_signal), deg=1, full=True
    )
    w_dnf = np.sqrt(fluc / window_size)

    fei_val = 1 - np.corrcoef(w_original_amp, w_dnf)[0, 1]

    gesd_alpha = 0.05
    max_outliers_percentage = 0.025
    max_num_outliers = max(int(np.round(max_outliers_percentage * len(w_original_amp))), 2)

    outlier_indexes_wamp = _generalized_esd(w_original_amp, max_num_outliers, gesd_alpha)[1]
    outlier_indexes_wdnf = _generalized_esd(w_dnf, max_num_outliers, gesd_alpha)[1]
    outlier_union = outlier_indexes_wamp + outlier_indexes_wdnf
    num_outliers = len(outlier_union)
    not_outlier_both = np.setdiff1d(np.arange(len(w_original_amp)), np.array(outlier_union))

    fei_outliers_removed = 1 - np.corrcoef(
        w_original_amp[not_outlier_both], w_dnf[not_outlier_both]
    )[0, 1]

    if dfa_value <= dfa_threshold:
        fei_val = np.nan
        fei_outliers_removed = np.nan

    return fei_outliers_removed, fei_val, num_outliers, w_original_amp, w_dnf


def _make_demo_signal(num_channels=2, num_seconds=40, sampling_frequency=250, seed=42):
    rng = np.random.default_rng(seed)
    n = int(num_seconds * sampling_frequency)
    return rng.random((num_channels, n))


def test_dfa_parity_against_reference_demo():
    sampling_frequency = 250
    fit_interval = [5, 30]
    compute_interval = [5, 30]
    overlap = True

    signal = _make_demo_signal(
        num_channels=2,
        num_seconds=40,
        sampling_frequency=sampling_frequency,
        seed=1,
    )
    amplitude_envelope = signal

    dfa_val, window_sizes, fluctuations, dfa_intercept = _dfa_reference(
        amplitude_envelope[0],
        sampling_frequency,
        fit_interval,
        compute_interval,
        overlap,
    )

    feat = Features(
        method="dfa",
        params={
            "sampling_frequency": sampling_frequency,
            "fit_interval": fit_interval,
            "compute_interval": compute_interval,
            "overlap": overlap,
            "runtime": "python",
        },
    )
    out = feat.dfa(amplitude_envelope[0])

    assert np.array_equal(out["window_sizes"], window_sizes)
    assert np.allclose(
        out["fluctuations"], fluctuations, rtol=1e-7, atol=1e-7, equal_nan=True
    )
    assert np.isclose(out["dfa"], dfa_val, rtol=1e-7, atol=1e-7, equal_nan=True)
    assert np.isclose(
        out["dfa_intercept"], dfa_intercept, rtol=1e-7, atol=1e-7, equal_nan=True
    )


def test_fei_parity_against_reference_demo():
    sampling_frequency = 250
    fit_interval = [5, 30]
    compute_interval = [5, 30]
    overlap = True
    window_size_sec = 5
    window_overlap = 0.8

    signal = _make_demo_signal(
        num_channels=2,
        num_seconds=40,
        sampling_frequency=sampling_frequency,
        seed=2,
    )
    amplitude_envelope = signal

    dfa_val, _, _, _ = _dfa_reference(
        amplitude_envelope[0],
        sampling_frequency,
        fit_interval,
        compute_interval,
        overlap,
    )

    fei_outliers_removed, fei_val, num_outliers, w_amp, w_dnf = _fei_reference(
        amplitude_envelope[0],
        sampling_frequency,
        window_size_sec,
        window_overlap,
        dfa_val,
    )

    feat = Features(
        method="fEI",
        params={
            "sampling_frequency": sampling_frequency,
            "window_size_sec": window_size_sec,
            "window_overlap": window_overlap,
            "runtime": "python",
            "dfa_value": float(dfa_val),
        },
    )
    out = feat.fEI(amplitude_envelope[0], dfa_value=float(dfa_val))

    assert np.allclose(out["wAmp"], w_amp, rtol=1e-7, atol=1e-7, equal_nan=True)
    assert np.allclose(out["wDNF"], w_dnf, rtol=1e-7, atol=1e-7, equal_nan=True)
    assert np.isclose(out["fEI_val"], fei_val, rtol=1e-7, atol=1e-7, equal_nan=True)
    assert np.isclose(
        out["fEI_outliers_removed"],
        fei_outliers_removed,
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
    )
    assert np.isclose(
        out["num_outliers"], num_outliers, rtol=1e-7, atol=1e-7, equal_nan=True
    )
