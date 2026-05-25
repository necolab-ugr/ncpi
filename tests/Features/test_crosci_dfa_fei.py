import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert

from ncpi.Features import Features


GLOBAL_SEED = int(os.environ.get("NCPI_CROSCI_PARITY_SEED", "0"))
PARITY_DURATION_SECONDS = float(
    os.environ.get("NCPI_CROSCI_PARITY_DURATION_SECONDS", "60")
)
PARITY_NUM_CHANNELS = int(os.environ.get("NCPI_CROSCI_PARITY_NUM_CHANNELS", "2"))

DFA_FIT_INTERVAL = [5, 30]
DFA_COMPUTE_INTERVAL = [5, 30]
DFA_OVERLAP = True
FEI_WINDOW_SIZE_SEC = 5
FEI_WINDOW_OVERLAP = 0.8

DFA_TOL = {"rtol": 1e-8, "atol": 1e-8}
FEI_TOL = {"rtol": 1e-7, "atol": 1e-7}

pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="crosci parity test is intended for local/manual execution only.",
)


@dataclass(frozen=True)
class ParityCase:
    name: str
    signal: np.ndarray
    sampling_frequency: int


def _log(message):
    print(f"[crosci parity] {message}", flush=True)


@pytest.fixture(scope="session")
def crosci_biomarkers(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("crosci_runtime")
    os.environ["HOME"] = str(tmp_dir)
    os.environ["MNE_HOME"] = str(tmp_dir / "mne")
    os.environ["NUMBA_CACHE_DIR"] = str(tmp_dir / "numba_cache")

    _log("Importing crosci biomarkers.")
    try:
        from crosci.biomarkers import DFA, fEI
    except Exception as exc:
        pytest.skip(f"crosci is not importable in this environment: {exc}")
    _log("crosci biomarkers imported successfully.")

    return DFA, fEI


def _test_duration_seconds():
    minimum = DFA_COMPUTE_INTERVAL[1] + 10
    return max(PARITY_DURATION_SECONDS, minimum)


def _rng(offset=0):
    return np.random.default_rng(GLOBAL_SEED + offset)


def _demo_signal(duration_seconds, num_channels):
    _log("Generating random envelope data from the crosci demo setup.")
    sampling_frequency = 250
    rng = _rng(0)
    signal = rng.random((num_channels, int(duration_seconds * sampling_frequency)))
    return ParityCase(
        name="crosci_demo_random_envelope",
        signal=signal,
        sampling_frequency=sampling_frequency,
    )


def _readme_lfp_signal(duration_seconds, num_channels):
    _log("Generating synthetic LFP envelope data from the README example setup.")
    rng = _rng(1)
    sampling_frequency = 1000
    dt = 1.0 / sampling_frequency
    n_samples = int(sampling_frequency * duration_seconds)

    rate_e_hz = 2.0
    rate_i_hz = 5.0
    n_e = 8000
    n_i = 2000
    v_rest_mv = -65.0
    e_ampa_mv = 0.0
    e_gabaa_mv = -80.0
    tau_rise_ampa_ms = 0.1
    tau_decay_ampa_ms = 2.0
    tau_rise_gabaa_ms = 0.5
    tau_decay_gabaa_ms = 10.0
    eps = 1e-12

    def conductance_kernel(tau_rise_ms, tau_decay_ms, support_ms=200.0):
        t = np.arange(0.0, support_ms / 1000.0, 1.0 / sampling_frequency)
        tau_r = tau_rise_ms / 1000.0
        tau_d = tau_decay_ms / 1000.0
        k = np.exp(-t / tau_d) - np.exp(-t / tau_r)
        k[k < 0.0] = 0.0
        return k / (np.sum(k) + eps)

    k_ampa = conductance_kernel(tau_rise_ampa_ms, tau_decay_ampa_ms)
    k_gabaa = conductance_kernel(tau_rise_gabaa_ms, tau_decay_gabaa_ms)

    def poisson_counts_from_isi(rate_hz):
        expected_spikes = max(1, int(rate_hz * duration_seconds))
        n_draws = max(32, int(expected_spikes + 8.0 * np.sqrt(expected_spikes) + 64))
        isi = rng.exponential(scale=1.0 / rate_hz, size=n_draws)
        spike_times = np.cumsum(isi)
        while spike_times[-1] < duration_seconds:
            extra = rng.exponential(scale=1.0 / rate_hz, size=n_draws)
            spike_times = np.concatenate(
                [spike_times, spike_times[-1] + np.cumsum(extra)]
            )
        spike_bins = (spike_times[spike_times < duration_seconds] / dt).astype(int)
        return np.bincount(spike_bins, minlength=n_samples).astype(float)

    def simulate_lfp(target_inh_over_exc):
        spikes_e = poisson_counts_from_isi(rate_e_hz * n_e)
        spikes_i = poisson_counts_from_isi(rate_i_hz * n_i)
        g_e = np.convolve(spikes_e, k_ampa, mode="same")
        g_i = np.convolve(spikes_i, k_gabaa, mode="same")
        g_i *= (target_inh_over_exc * np.mean(g_e)) / (np.mean(g_i) + eps)
        i_e = g_e * (v_rest_mv - e_ampa_mv)
        i_i = g_i * (v_rest_mv - e_gabaa_mv)
        lfp = i_e + i_i
        norm = np.sqrt(np.sum(np.abs(np.fft.rfft(lfp)) ** 2) + eps)
        return np.abs(lfp / norm)

    targets = np.linspace(2.0, 6.0, num_channels)
    signal = np.vstack([simulate_lfp(target) for target in targets])
    return ParityCase(
        name="readme_gao_2017_lfp_envelope",
        signal=signal,
        sampling_frequency=sampling_frequency,
    )


def _mne_sample_signal(duration_seconds, num_channels):
    _log("Preparing MNE sample alpha-envelope data.")
    mne = pytest.importorskip("mne")

    with tempfile.TemporaryDirectory(prefix="mne_sample_") as temp_dir:
        _log("Downloading MNE sample dataset into a temporary directory.")
        try:
            data_path = mne.datasets.sample.data_path(
                path=temp_dir,
                download=True,
                update_path=False,
                verbose=False,
            )
        except Exception as exc:
            pytest.skip(f"MNE sample dataset could not be downloaded: {exc}")

        raw_path = Path(data_path) / "MEG/sample/sample_audvis_raw.fif"
        _log("Loading MNE sample raw FIF file.")
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        raw.pick(picks="eeg", exclude="bads")
        if len(raw.ch_names) < num_channels:
            pytest.skip("MNE sample dataset does not have enough EEG channels")

        fs = float(raw.info["sfreq"])
        n_samples = int(fs * duration_seconds)
        if raw.n_times < n_samples:
            pytest.skip(
                f"MNE sample dataset is shorter than {duration_seconds:.1f} seconds"
            )

        signal = raw.get_data(picks=list(range(num_channels)))[:, :n_samples]

        band = [8.0, 13.0]
        _log("Filtering MNE EEG data in the alpha band.")
        filtered = mne.filter.filter_data(
            signal,
            fs,
            band[0],
            band[1],
            filter_length="auto",
            fir_window="hamming",
            phase="zero",
            fir_design="firwin",
            pad="reflect_limited",
            verbose=False,
        )

    trim = int(fs)
    filtered = filtered[:, trim:-trim]
    n_fft = mne.filter.next_fast_len(filtered.shape[1])
    _log("Computing the MNE alpha amplitude envelope.")
    analytic = hilbert(filtered, N=n_fft, axis=-1)[..., : filtered.shape[1]]
    amplitude_envelope = np.abs(analytic)

    return ParityCase(
        name="mne_sample_alpha_envelope",
        signal=amplitude_envelope,
        sampling_frequency=int(round(fs)),
    )


def _make_case(origin):
    duration_seconds = _test_duration_seconds()
    num_channels = PARITY_NUM_CHANNELS
    if origin == "crosci_demo":
        return _demo_signal(duration_seconds, num_channels)
    if origin == "readme":
        return _readme_lfp_signal(duration_seconds, num_channels)
    if origin == "mne_sample":
        return _mne_sample_signal(duration_seconds, num_channels)
    raise ValueError(f"Unknown parity origin: {origin}")


def _ncpi_dfa(signal, sampling_frequency):
    _log("Computing DFA with ncpi Features.")
    feat = Features(
        method="dfa",
        params={
            "sampling_frequency": sampling_frequency,
            "fit_interval": DFA_FIT_INTERVAL,
            "compute_interval": DFA_COMPUTE_INTERVAL,
            "overlap": DFA_OVERLAP,
        },
    )
    return feat.dfa(signal)


def _ncpi_fei_from_dfa(signal, sampling_frequency, dfa_array):
    _log("Computing fE/I with ncpi Features using ncpi DFA values.")
    feat = Features(
        method="fEI",
        params={
            "sampling_frequency": sampling_frequency,
            "window_size_sec": FEI_WINDOW_SIZE_SEC,
            "window_overlap": FEI_WINDOW_OVERLAP,
        },
    )

    out_by_channel = [
        feat.fEI(signal[ch_idx], dfa_value=float(dfa_array[ch_idx]))
        for ch_idx in range(signal.shape[0])
    ]

    return {
        "fEI_outliers_removed": np.asarray(
            [out["fEI_outliers_removed"] for out in out_by_channel],
            dtype=float,
        )[:, np.newaxis],
        "fEI_val": np.asarray(
            [out["fEI_val"] for out in out_by_channel],
            dtype=float,
        )[:, np.newaxis],
        "num_outliers": np.asarray(
            [out["num_outliers"] for out in out_by_channel],
            dtype=float,
        )[:, np.newaxis],
        "wAmp": np.vstack([out["wAmp"] for out in out_by_channel]),
        "wDNF": np.vstack([out["wDNF"] for out in out_by_channel]),
    }


def _assert_dfa_parity(ncpi_out, crosci_out):
    _log("Comparing DFA outputs from ncpi and crosci.")
    crosci_dfa, crosci_window_sizes, crosci_fluctuations, crosci_intercept = (
        crosci_out
    )

    assert np.array_equal(ncpi_out["window_sizes"], crosci_window_sizes)
    np.testing.assert_allclose(
        ncpi_out["DFA"], crosci_dfa, equal_nan=True, **DFA_TOL
    )
    np.testing.assert_allclose(
        ncpi_out["fluctuations"],
        crosci_fluctuations,
        equal_nan=True,
        **DFA_TOL,
    )
    np.testing.assert_allclose(
        ncpi_out["dfa_intercept"], crosci_intercept, equal_nan=True, **DFA_TOL
    )
    _log("DFA parity comparison passed.")


def _assert_fei_parity(ncpi_out, crosci_out):
    _log("Comparing fE/I outputs from ncpi and crosci.")
    crosci_fei_or, crosci_fei_val, crosci_num_outliers, crosci_w_amp, crosci_w_dnf = (
        crosci_out
    )

    np.testing.assert_allclose(
        ncpi_out["fEI_outliers_removed"],
        crosci_fei_or,
        equal_nan=True,
        **FEI_TOL,
    )
    np.testing.assert_allclose(
        ncpi_out["fEI_val"], crosci_fei_val, equal_nan=True, **FEI_TOL
    )
    np.testing.assert_allclose(
        ncpi_out["num_outliers"],
        crosci_num_outliers,
        equal_nan=True,
        **FEI_TOL,
    )
    np.testing.assert_allclose(
        ncpi_out["wAmp"], crosci_w_amp, equal_nan=True, **FEI_TOL
    )
    np.testing.assert_allclose(
        ncpi_out["wDNF"], crosci_w_dnf, equal_nan=True, **FEI_TOL
    )
    _log("fE/I parity comparison passed.")


@pytest.mark.parametrize("origin", ["crosci_demo", "readme", "mne_sample"])
def test_dfa_and_fei_parity_against_crosci(origin, crosci_biomarkers):
    crosci_dfa, crosci_fei = crosci_biomarkers
    _log(f"Starting parity test for origin: {origin}.")
    case = _make_case(origin)
    _log(
        f"Generated case '{case.name}' with shape {case.signal.shape} "
        f"and sampling frequency {case.sampling_frequency} Hz."
    )

    ncpi_dfa_out = _ncpi_dfa(case.signal, case.sampling_frequency)
    _log("Computing DFA with crosci.")
    crosci_dfa_out = crosci_dfa(
        case.signal,
        case.sampling_frequency,
        DFA_FIT_INTERVAL,
        DFA_COMPUTE_INTERVAL,
        overlap=DFA_OVERLAP,
        runtime="c",
    )
    _assert_dfa_parity(ncpi_dfa_out, crosci_dfa_out)

    ncpi_fei_out = _ncpi_fei_from_dfa(
        case.signal,
        case.sampling_frequency,
        ncpi_dfa_out["DFA"],
    )
    _log("Computing fE/I with crosci using crosci DFA values.")
    crosci_fei_out = crosci_fei(
        case.signal,
        case.sampling_frequency,
        FEI_WINDOW_SIZE_SEC,
        FEI_WINDOW_OVERLAP,
        crosci_dfa_out[0],
        runtime="c",
    )
    _assert_fei_parity(ncpi_fei_out, crosci_fei_out)
    _log(f"Finished parity test for origin: {origin}.")
