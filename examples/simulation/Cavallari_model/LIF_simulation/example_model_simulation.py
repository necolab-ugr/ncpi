"""Run and visualize the Cavallari LIF example pipeline.

The script follows the same high-level flow as the Hagen example pipeline:
configure model parameters, run or load simulations, compute a signal, plot
spikes/rates/signals/spectra, and compute summary features.
"""

import pickle
import inspect
import os
import sys
import time
from copy import deepcopy
from importlib import util
from pathlib import Path
import ncpi
import numpy as np
import pandas as pd
import scipy.signal as ss
from matplotlib import pyplot as plt


# Local folder layout used by ncpi.Simulation and cached outputs.
ROOT_DIR = Path(__file__).resolve().parent
PARAMS_DIR = ROOT_DIR / "params"
PYTHON_DIR = ROOT_DIR / "python"
OUTPUT_DIR = ROOT_DIR / "output"
DATA_DIR = ROOT_DIR / "data"

# Multicompartment analysis parameters used only by the kernel/CDM path.
MC_ANALYSIS_PARAMS = ROOT_DIR.parent / "MC_simulation" / "analysis_params.py"

# Shared plotting and feature-extraction settings.
DECIMATION_FACTOR = 10
PLOT_WINDOW_MS = [4000, 4100]
FEATURE_METHODS = ["catch22", "power_spectrum_parameterization"]
PSD_FREQ_RANGE = (5.0, 200.0)
TRANSIENT_MS = 500.0

sys.path.append(str(PARAMS_DIR))

# Choose "proxy" to keep the original LFP-proxy workflow, or "kernel" to compute CDM from kernels.
signal_method = "proxy"

# Set to True to run new simulations of the Cavallari LIF network model, or False to load
# precomputed results from pickle files located in the local 'data' folder.
compute_new_sim = True

# Choose to either download files and precomputed outputs used in simulations of
# the multicompartment neuron network model (True) or load them from a local
# path (False). Only used when signal_method == "kernel".
zenodo_dw_mult = False

# Zenodo URL that contains the data (used if zenodo_dw_mult is True)
zenodo_URL_mult = "https://zenodo.org/api/records/15429373"

# Zenodo directory where the data is stored. This must be an absolute path to
# correctly load morphologies in NEURON.
zenodo_dir = Path(
    os.path.expandvars(
        os.path.expanduser(os.path.join("$HOME", "multicompartment_neuron_network"))
    )
)

# Number of repetitions of each simulation
trials = 1

# Configurations of parameters to simulate:
# [g_EE, g_IE, g_EI, g_II, tau_syn_AMPA_exc, tau_syn_AMPA_inh, tau_syn_GABA_exc, tau_syn_GABA_inh, ext_input_scale, v_0]

# Baseline configuration varying only v_0
# confs = [
#     [0.178, 0.233, -2.01, -2.70, 2.0, 1.0, 5.0, 5.0, 1.0, 1.5],
#     [0.178, 0.233, -2.01, -2.70, 2.0, 1.0, 5.0, 5.0, 1.0, 3.0],
# ]

# Configuration with increasing inhibitory conductances
confs = [
    [0.178, 0.233, -1.01, -1.35, 2.0, 1.0, 5.0, 5.0, 1.0, 3.0],
    [0.178, 0.233, -2.01, -2.70, 2.0, 1.0, 5.0, 5.0, 1.0, 3.0],
]


CONF_PARAM_LABELS = [
    r"$g_{EE}$",
    r"$g_{IE}$",
    r"$g_{EI}$",
    r"$g_{II}$",
    r"$\tau_{\mathrm{syn},AMPA}^{E}$",
    r"$\tau_{\mathrm{syn},AMPA}^{I}$",
    r"$\tau_{\mathrm{syn},GABA}^{E}$",
    r"$\tau_{\mathrm{syn},GABA}^{I}$",
    "ext_input_scale",
    r"$v_0$",
]

# Names of catch22 features
try:
    import pycatch22

    catch22_names = pycatch22.catch22_all([0])["names"]
except Exception:
    catch22_names = [
        "DN_HistogramMode_5",
        "DN_HistogramMode_10",
        "CO_f1ecac",
        "CO_FirstMin_ac",
        "CO_HistogramAMI_even_2_5",
        "CO_trev_1_num",
        "MD_hrv_classic_pnn40",
        "SB_BinaryStats_mean_longstretch1",
        "SB_TransitionMatrix_3ac_sumdiagcov",
        "PD_PeriodicityWang_th0_01",
        "CO_Embed2_Dist_tau_d_expfit_meandiff",
        "IN_AutoMutualInfoStats_40_gaussian_fmmi",
        "FC_LocalSimple_mean1_tauresrat",
        "DN_OutlierInclude_p_001_mdrmd",
        "DN_OutlierInclude_n_001_mdrmd",
        "SP_Summaries_welch_rect_area_5_1",
        "SB_BinaryStats_diff_longstretch0",
        "SB_MotifThree_quantile_hh",
        "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
        "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
        "SP_Summaries_welch_rect_centroid",
        "FC_LocalSimple_mean3_stderr",
    ]


def get_varying_conf_indices(configurations):
    """Return indices of configuration parameters that vary across conditions."""
    conf_array = np.asarray(configurations, dtype=float)
    return [
        idx for idx in range(conf_array.shape[1])
        if not np.allclose(conf_array[:, idx], conf_array[0, idx])
    ]


def format_conf_label(config, varying_indices, multiline=False):
    """Build a readable label from only the parameters that vary."""
    if not varying_indices:
        return "baseline"
    separator = "\n" if multiline else ", "
    return separator.join(
        f"{CONF_PARAM_LABELS[idx]} = {config[idx]:.2f}"
        for idx in varying_indices
    )


def get_conf_axis_label(varying_indices):
    """Return the x-axis label for the varied configuration parameters."""
    if len(varying_indices) == 1:
        return CONF_PARAM_LABELS[varying_indices[0]]
    return "Configuration"


def get_conf_ticklabels(configurations, varying_indices):
    """Return tick labels for the varied configuration parameters."""
    if len(varying_indices) == 1:
        idx = varying_indices[0]
        return [f"{config[idx]:g}" for config in configurations]
    return [format_conf_label(config, varying_indices, multiline=True) for config in configurations]


def get_spike_rate(times, transient, dt, tstop):
    """Bin spike times into counts per simulation time step."""
    bins = np.arange(transient, tstop + dt, dt)
    hist, _ = np.histogram(times, bins=bins)
    return bins, hist.astype(float)


def get_top_row_axes(fig, n_cols, y, h):
    """Create evenly spaced axes for one figure row."""
    left = 0.1
    gap = 0.05
    total_width = 0.85
    width = (total_width - gap * max(0, n_cols - 1)) / max(1, n_cols)
    return [fig.add_axes([left + col * (width + gap), y, width, h]) for col in range(n_cols)]


def get_data_path(conf_idx, trial):
    """Return the cached output path for one configuration and trial."""
    if signal_method == "proxy":
        return DATA_DIR / f"output_{conf_idx}_{trial}.pkl"
    return DATA_DIR / f"output_{signal_method}_{conf_idx}_{trial}.pkl"


def load_trial_output(path):
    """Load cached trial output, accepting both legacy tuple and dict payloads."""
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        if "proxy_data" not in payload and "cdm_data" in payload:
            payload["proxy_data"] = payload["cdm_data"]
        return payload

    times, gids, proxy_data, dt, tstop, transient, P_X, N_X = payload
    return {
        "times": times,
        "gids": gids,
        "proxy_data": proxy_data,
        "dt": dt,
        "tstop": tstop,
        "transient": transient,
        "P_X": P_X,
        "N_X": N_X,
    }


def is_cdm_components(signal_data):
    """Return True when a payload stores the four CDM pair components."""
    return isinstance(signal_data, dict) and all(
        key in signal_data for key in ("EE", "EI", "IE", "II")
    )


def wrap_cdm_components(cdm_components, transient, dt, P_X, decimation_factor=DECIMATION_FACTOR):
    """Wrap CDM components with summed signal data and a time axis."""
    component_order = [f"{X}{Y}" for X in P_X for Y in P_X]
    signal = np.sum(
        np.vstack([np.asarray(cdm_components[key], dtype=float) for key in component_order]),
        axis=0,
    )
    return {
        "signal": signal,
        "times": (
            float(transient)
            + np.arange(signal.size) * float(dt) * float(decimation_factor)
        ),
        "components": cdm_components,
        "decimation_factor": decimation_factor,
        "source": "kernel",
    }


def get_saved_signal_data(payload):
    """Return the signal payload matching the configured signal method."""
    if signal_method == "proxy":
        return payload["proxy_data"]

    if "signal_data" in payload:
        return payload["signal_data"]
    if "cdm_data" in payload:
        return wrap_cdm_components(
            payload["cdm_data"], payload["transient"], payload["dt"], payload["P_X"]
        )
    if "proxy_data" in payload and is_cdm_components(payload["proxy_data"]):
        return wrap_cdm_components(
            payload["proxy_data"], payload["transient"], payload["dt"], payload["P_X"]
        )
    raise KeyError(
        "Saved payload does not contain CDM data. Run again with compute_new_sim = True."
    )


def get_proxy_signal(proxy_data):
    """Return a 1D signal array from proxy or CDM payload variants."""
    if isinstance(proxy_data, dict) and "signal" in proxy_data:
        return np.asarray(proxy_data["signal"], dtype=float)
    if isinstance(proxy_data, dict) and "components" in proxy_data:
        return get_proxy_signal(proxy_data["components"])
    if is_cdm_components(proxy_data):
        return np.sum(
            np.vstack(
                [np.asarray(proxy_data[key], dtype=float) for key in ("EE", "EI", "IE", "II")]
            ),
            axis=0,
        )
    return np.asarray(proxy_data, dtype=float)


def get_feature_signal(signal_data):
    """Return the signal representation used for feature computation."""
    if signal_method == "kernel" and isinstance(signal_data, dict) and "components" in signal_data:
        return get_proxy_signal(signal_data["components"])
    return get_proxy_signal(signal_data)


def get_proxy_times(proxy_data):
    """Return the explicit time axis stored with a proxy or CDM signal payload."""
    if isinstance(proxy_data, dict) and "times" in proxy_data:
        return np.asarray(proxy_data["times"], dtype=float)
    raise KeyError("Signal payload does not contain an explicit time axis.")


def trim_signal_to_transient(signal_data, transient):
    """Discard signal samples before the configured transient time."""
    if not isinstance(signal_data, dict) or "signal" not in signal_data or "times" not in signal_data:
        return signal_data
    signal = get_proxy_signal(signal_data)
    times = get_proxy_times(signal_data)
    keep = times >= float(transient)
    trimmed = dict(signal_data)
    trimmed["signal"] = signal[keep]
    trimmed["times"] = times[keep]
    return trimmed


def _extract_exc_currents(exc_state_events):
    """Extract excitatory population currents from aggregated state events."""
    times = np.asarray(exc_state_events["times"], dtype=float)
    if {"AMPA", "GABA", "Vm"}.issubset(exc_state_events):
        return (
            times,
            np.asarray(exc_state_events["AMPA"], dtype=float),
            np.asarray(exc_state_events["GABA"], dtype=float),
            np.asarray(exc_state_events["Vm"], dtype=float),
        )
    raise KeyError("Expected aggregated excitatory-state payload with AMPA/GABA/Vm entries.")


def compute_proxy(exc_state_events, sim_step):
    """
    Reproduce ``tools.LFP(...)[5]`` followed by the normalization in
    pablomc88's save_results_1.py:
    ``LFP[5] = |AMPA_current| + |GABA_current|`` and
    normalize using samples from index ``int(500.0 / sim_step)`` onward.
    """
    times, AMPA_current, GABA_current, _ = _extract_exc_currents(exc_state_events)
    if times.size == 0:
        raise ValueError("No excitatory-state samples are available for proxy computation.")

    signal = np.abs(AMPA_current) + np.abs(GABA_current)
    start_time_pos = int(TRANSIENT_MS / float(sim_step))
    if start_time_pos >= signal.size:
        raise ValueError("Proxy normalization window starts after the end of the signal.")

    norm_window = signal[start_time_pos:]
    norm_std = np.std(norm_window)
    if np.isclose(norm_std, 0.0):
        raise ValueError("Proxy normalization failed because the post-500 ms signal is constant.")

    signal = (signal - np.mean(norm_window)) / norm_std
    return {
        "signal": signal,
        "times": times,
        "lfp_index": 5,
    }


def decimate_proxy(proxy_data, decimation_factor=DECIMATION_FACTOR):
    """Downsample a proxy payload while keeping its time axis aligned."""
    signal = get_proxy_signal(proxy_data)
    times = get_proxy_times(proxy_data)
    if signal.size == 0:
        raise ValueError("Cannot decimate an empty proxy signal.")
    if decimation_factor <= 1:
        return {
            "signal": signal,
            "times": times,
            "lfp_index": proxy_data.get("lfp_index", 5),
            "decimation_factor": decimation_factor,
        }

    decimated_signal = ss.decimate(signal, decimation_factor)
    decimated_times = times[::decimation_factor][: decimated_signal.size]
    return {
        "signal": decimated_signal,
        "times": decimated_times,
        "lfp_index": proxy_data.get("lfp_index", 5),
        "decimation_factor": decimation_factor,
    }


class KernelParamsAdapter:
    """Adapt MC kernel params whose fields depend on the postsynaptic population."""

    def __init__(self, kernel_params):
        """Store the original kernel-parameter object."""
        self._kernel_params = kernel_params

    def __getattr__(self, name):
        """Forward unknown attribute lookups to the wrapped parameters."""
        return getattr(self._kernel_params, name)

    @staticmethod
    def _current_post_index():
        """Find the postsynaptic loop index used inside ncpi kernel creation."""
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
        while frame is not None:
            if frame.f_code.co_name == "create_kernel" and "j" in frame.f_locals:
                return frame.f_locals["j"]
            frame = frame.f_back
        return None

    @property
    def extSynapseParameters(self):
        """Return external synapse parameters for the active postsynaptic population."""
        ext_params = self._kernel_params.extSynapseParameters
        post_idx = self._current_post_index()
        if post_idx is not None and isinstance(ext_params, (list, tuple)):
            return ext_params[post_idx]
        return ext_params

    @property
    def netstim_interval(self):
        """Return netstim interval data for the active postsynaptic population."""
        interval = self._kernel_params.netstim_interval
        post_idx = self._current_post_index()
        if post_idx is not None and isinstance(interval, (list, tuple, np.ndarray)):
            return float(np.asarray(interval, dtype=float)[post_idx])
        if isinstance(interval, (list, tuple)):
            return np.asarray(interval, dtype=float)
        return interval


def load_kernel_params():
    """Load multicompartment kernel parameters for the Cavallari model."""
    spec = util.spec_from_file_location("cavallari_mc_analysis_params", MC_ANALYSIS_PARAMS)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return KernelParamsAdapter(module.KernelParams)


def compute_kernel_cdm(times, dt, tstop, transient, P_X, kernel_params):
    """Compute decimated CDM components from spike times and MC kernels."""
    print("Computing the kernel...")
    potential = ncpi.FieldPotential()
    biophys = ["set_Ih_linearized_hay2011", "make_cell_uniform"]
    H_YX = potential.create_kernel(
        str(zenodo_dir),
        kernel_params,
        biophys,
        dt,
        tstop,
        output_sim_path=str(zenodo_dir / "output_Cavallari"),
        electrodeParameters=None,
        CDM=True,
    )

    probe = "KernelApproxCurrentDipoleMoment"
    cdm_signals = potential.compute_cdm_lfp_from_kernels(
        H_YX,
        spike_times=times,
        dt=dt,
        tstop=tstop,
        transient=transient,
        probe=probe,
        component=2,
        mode="same",
        scale=dt / 1000.0,
    )

    cdm_components = {}
    for X in P_X:
        for Y in P_X:
            key = f"{Y}:{X}"
            cdm_components[f"{X}{Y}"] = ss.decimate(
                cdm_signals[key],
                q=DECIMATION_FACTOR,
                zero_phase=True,
            )

    return wrap_cdm_components(cdm_components, transient, dt, P_X)


def build_cavallari_params(config):
    """Build a Cavallari LIF parameter payload for one configuration."""
    from network_params import Network_params as base_network_params
    from network_params import Neuron_params as base_neuron_params

    (
        g_EE,
        g_IE,
        g_EI,
        g_II,
        tau_syn_AMPA_exc,
        tau_syn_AMPA_inh,
        tau_syn_GABA_exc,
        tau_syn_GABA_inh,
        ext_input_scale,
        v_0,
    ) = config

    network_params = deepcopy(base_network_params)
    neuron_params = deepcopy(base_neuron_params)

    network_params["exc_exc_recurrent"] = g_EE
    network_params["exc_inh_recurrent"] = g_IE
    network_params["inh_exc_recurrent"] = g_EI
    network_params["inh_inh_recurrent"] = g_II
    network_params["th_exc_external"] *= ext_input_scale
    network_params["th_inh_external"] *= ext_input_scale
    network_params["v_0"] = v_0

    neuron_params[0]["tau_decay_AMPA"] = tau_syn_AMPA_exc
    neuron_params[1]["tau_decay_AMPA"] = tau_syn_AMPA_inh
    neuron_params[0]["tau_decay_GABA_A"] = tau_syn_GABA_exc
    neuron_params[1]["tau_decay_GABA_A"] = tau_syn_GABA_inh

    return {
        "X": ["E", "I"],
        "N_X": [network_params["N_exc"], network_params["N_inh"]],
        "model": "iaf_bw_2003",
        "neuron_params": {
            "E": neuron_params[0],
            "I": neuron_params[1],
        },
        "network_params": network_params,
    }


def main():
    """Run or load simulations, compute signals, plot results, and extract features."""
    # Validate the chosen signal pipeline.
    if signal_method not in {"proxy", "kernel"}:
        raise ValueError("signal_method must be either 'proxy' or 'kernel'.")

    # Identify which configuration parameters should be shown in labels.
    varying_conf_indices = get_varying_conf_indices(confs)

    # Prepare multicompartment kernel inputs only when requested.
    kernel_params = None
    if signal_method == "kernel":
        if zenodo_dw_mult:
            print("\n--- Downloading multicompartment data.")
            start_time = time.time()
            ncpi.tools.download_zenodo_record(zenodo_URL_mult, download_dir=str(zenodo_dir))
            end_time = time.time()
            print(f"All files downloaded in {(end_time - start_time) / 60:.2f} minutes.")
        kernel_params = load_kernel_params()

    # Match the deterministic seed style used by the Hagen example.
    np.random.seed(0)

    # Create the ncpi simulation runner for the local Cavallari folders.
    sim = ncpi.Simulation(
        param_folder=str(PARAMS_DIR),
        python_folder=str(PYTHON_DIR),
        output_folder=str(OUTPUT_DIR),
    )

    spikes = [[] for _ in range(trials)]
    signals = [[] for _ in range(trials)]

    # Run new simulations or load cached trial outputs, following the Hagen loop structure.
    for trial in range(trials):
        for k, params in enumerate(confs):
            if compute_new_sim:
                print(f"\nTrial {trial + 1}/{trials}, Configuration {k + 1}/{len(confs)}")

                # Build and save the parameter payload consumed by simulation.py.
                lif_params = build_cavallari_params(params)

                with open(OUTPUT_DIR / "network.pkl", "wb") as f:
                    pickle.dump(lif_params, f)

                # Run the NEST simulation through ncpi.
                start_time = time.time()
                sim.simulate("simulation.py", "simulation_params.py")
                end_time = time.time()
                print(f"Simulation finished in {end_time - start_time:.2f} s.")

                # Load simulation outputs written by the Cavallari simulation script.
                with open(OUTPUT_DIR / "times.pkl", "rb") as f:
                    times = pickle.load(f)
                with open(OUTPUT_DIR / "gids.pkl", "rb") as f:
                    gids = pickle.load(f)
                with open(OUTPUT_DIR / "tstop.pkl", "rb") as f:
                    tstop = pickle.load(f)
                with open(OUTPUT_DIR / "dt.pkl", "rb") as f:
                    dt = pickle.load(f)
                if signal_method == "proxy":
                    with open(OUTPUT_DIR / "exc_state_events.pkl", "rb") as f:
                        exc_state_events = pickle.load(f)

                # Remove the same initial transient period for proxy and kernel signals.
                transient = TRANSIENT_MS
                P_X = lif_params["X"]

                for X in P_X:
                    keep = times[X] >= transient
                    times[X] = times[X][keep]
                    gids[X] = gids[X][keep]

                # Compute the configured signal from either state variables or MC kernels.
                if signal_method == "proxy":
                    signal_data = trim_signal_to_transient(
                        decimate_proxy(compute_proxy(exc_state_events, dt)),
                        transient,
                    )
                else:
                    signal_data = compute_kernel_cdm(
                        times, dt, tstop, transient, P_X, kernel_params
                    )
                spikes[trial].append([times, gids])
                signals[trial].append(signal_data)

                # Cache outputs in a dict format while remaining compatible with legacy loaders.
                DATA_DIR.mkdir(exist_ok=True)
                payload = {
                    "times": times,
                    "gids": gids,
                    "dt": dt,
                    "tstop": tstop,
                    "transient": transient,
                    "P_X": P_X,
                    "N_X": lif_params["N_X"],
                }
                if signal_method == "proxy":
                    payload["proxy_data"] = signal_data
                else:
                    payload["signal_data"] = signal_data
                    payload["cdm_data"] = signal_data["components"]

                with open(get_data_path(k, trial), "wb") as f:
                    pickle.dump(
                        payload,
                        f,
                    )
            else:
                # Load cached outputs when simulations are not recomputed.
                try:
                    data_path = get_data_path(k, trial)
                    payload = load_trial_output(data_path)
                except FileNotFoundError:
                    print(f"File {data_path} not found. Please run the simulation first.")
                    sys.exit(1)

                times = payload["times"]
                gids = payload["gids"]
                signal_data = get_saved_signal_data(payload)
                dt = payload["dt"]
                tstop = payload["tstop"]
                transient = payload["transient"]
                P_X = payload["P_X"]
                if not np.isclose(float(transient), TRANSIENT_MS):
                    raise ValueError(
                        f"Cached payload at {data_path} uses transient={transient} ms. "
                        f"Recompute it so both proxy and kernel use {TRANSIENT_MS} ms."
                    )
                if (
                    signal_method == "proxy"
                    and (
                        not isinstance(signal_data, dict)
                        or signal_data.get("decimation_factor") != DECIMATION_FACTOR
                    )
                ):
                    signal_data = decimate_proxy(signal_data)
                if signal_method == "proxy":
                    signal_data = trim_signal_to_transient(signal_data, transient)
                spikes[trial].append([times, gids])
                signals[trial].append(signal_data)

    # Create a figure and set its properties.
    fig = plt.figure(figsize=(7.5, 6.0), dpi=300)
    plt.rcParams.update({"font.size": 10, "font.family": "Arial"})
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)

    # Time interval and signal sampling rate after decimation.
    T = PLOT_WINDOW_MS
    signal_fs = 1000.0 / (DECIMATION_FACTOR * float(dt))

    # Raster, firing-rate, and signal panels use the same dynamic column layout.
    colors = ["#1f77b4", "#ff7f0e"]
    top_axes = get_top_row_axes(fig, len(confs), 0.73, 0.22)
    middle_axes = get_top_row_axes(fig, len(confs), 0.6, 0.12)
    lower_axes = get_top_row_axes(fig, len(confs), 0.47, 0.12)

    # Raster plot of spike trains.
    for col, ax in enumerate(top_axes):
        for i, X in enumerate(P_X):
            t = spikes[0][col][0][X]
            gi = spikes[0][col][1][X]
            ii = (t >= T[0]) & (t <= T[1])
            ax.plot(t[ii], gi[ii], ".", color=colors[i], markersize=0.5)

        ax.set_title(format_conf_label(confs[col], varying_conf_indices, multiline=False))
        if col == 0:
            ax.set_ylabel("Neuron ID")
            ax.yaxis.set_label_coords(-0.22, 0.5)
            for j, Y in enumerate(P_X):
                ax.plot([], [], ".", color=colors[j], label=f"{Y}", markersize=4)
            ax.legend(loc=1, fontsize=8, labelspacing=0.2)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel("")
        ax.axis("tight")
        ax.set_xticklabels([])
        ax.set_xticks([])

    # Population firing-rate traces.
    for col, ax in enumerate(middle_axes):
        for i, X in enumerate(P_X):
            bins, spike_rate = get_spike_rate(spikes[0][col][0][X], transient, dt, tstop)
            bins = bins[:-1]
            ii = (bins >= T[0]) & (bins <= T[1])
            ax.plot(bins[ii], spike_rate[ii], color=f"C{i}", label=r"$\nu_\mathrm{%s}$" % X)

        if col == 0:
            ax.legend(loc=1)
            ax.set_ylabel(r"$\nu_X$ (spik./$\Delta t$)")
            ax.yaxis.set_label_coords(-0.22, 0.5)
        ax.axis("tight")
        ax.set_xticklabels([])
        ax.set_xticks([])

    # Proxy or CDM signal traces.
    for col, ax in enumerate(lower_axes):
        signal = get_proxy_signal(signals[0][col])
        bins = get_proxy_times(signals[0][col])
        ii = (bins >= T[0]) & (bins <= T[1])
        ax.plot(bins[ii], signal[ii], color="k")

        if col == 0:
            ax.set_ylabel("Proxy" if signal_method == "proxy" else r"CDM ($P_z$)")
            ax.yaxis.set_label_coords(-0.22, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("t (ms)")
        ax.axis("tight")

    # Power spectra.
    ax = fig.add_axes([0.1, 0.07, 0.27, 0.3])
    psd_colors = [f"C{i}" for i in range(len(confs))]
    for col in range(len(confs)):
        signal = np.asarray(
            [get_proxy_signal(signals[trial][col]) for trial in range(trials)],
            dtype=float,
        )
        f, Pxx = ss.welch(signal, fs=signal_fs)
        Pxx = np.mean(Pxx, axis=0)
        Pxx = Pxx / np.sum(Pxx)
        f1 = f[f >= PSD_FREQ_RANGE[0]]
        f2 = f1[f1 <= PSD_FREQ_RANGE[1]]
        ax.semilogy(
            f2,
            Pxx[(f >= PSD_FREQ_RANGE[0]) & (f <= PSD_FREQ_RANGE[1])],
            label=format_conf_label(confs[col], varying_conf_indices),
            color=psd_colors[col],
        )
    ax.legend(loc="lower left", fontsize=8, labelspacing=0.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")

    # Collect signals into the feature-computation format used by ncpi.
    all_signals = []
    IDs = []
    epochs = []
    for trial in range(trials):
        for conf_idx in range(len(confs)):
            all_signals.append(get_feature_signal(signals[trial][conf_idx]))
            IDs.append(conf_idx)
            epochs.append(trial)

    # Compute the same feature families used by the Hagen example.
    all_features = {}
    for method in FEATURE_METHODS:
        print(f"\n\n--- Method: {method}")
        df = pd.DataFrame(
            {
                "ID": IDs,
                "Group": IDs,
                "Epoch": epochs,
                "Sensor": np.zeros(len(IDs)),
                "Data": all_signals,
            }
        )
        df.Recording = "proxy" if signal_method == "proxy" else "LFP"
        df.fs = signal_fs

        if method == "catch22":
            features = ncpi.Features(method="catch22", params={"normalize": True})
            feats = features.compute_features(df["Data"].to_list())
            df = df.copy()
            df["Features"] = feats
        elif method == "power_spectrum_parameterization":
            fooof_setup_sim = {
                "peak_threshold": 1.0,
                "min_peak_height": 0.0,
                "max_n_peaks": 5,
                "peak_width_limits": (10.0, 50.0),
            }
            features = ncpi.Features(
                method="specparam",
                params={
                    "fs": df.fs,
                    "freq_range": PSD_FREQ_RANGE,
                    "specparam_model": dict(fooof_setup_sim),
                    "r_squared_th": 0.9,
                },
            )
            feats = features.compute_features(df["Data"].to_list())
            df = df.copy()
            df["Features"] = [float(np.asarray(d["aperiodic_params"])[1]) for d in feats]
        else:
            raise ValueError(f"Unknown method: {method}")

        all_features[method] = df

    # Plot selected features across configurations.
    feature_colors = ["lightcoral", "lightblue", "lightgreen", "lightgrey"]
    for row in range(2):
        for col in range(2):
            ax = fig.add_axes([0.5 + col * 0.27, 0.24 - row * 0.16, 0.18, 0.13])

            if row == 0 and col == 0:
                feats = np.array(
                    all_features["power_spectrum_parameterization"]["Features"].tolist()
                )
                ax.set_ylabel(r"$1/f$" + " " + r"$slope$")
            if row == 0 and col == 1:
                feats = np.array(all_features["catch22"]["Features"].tolist())
                idx = catch22_names.index("SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1")
                feats = feats[:, idx]
                ax.set_ylabel(r"$dfa$")
            if row == 1 and col == 0:
                feats = np.array(all_features["catch22"]["Features"].tolist())
                idx = catch22_names.index("SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1")
                feats = feats[:, idx]
                ax.set_ylabel(r"$rs\ range$")
            if row == 1 and col == 1:
                feats = np.array(all_features["catch22"]["Features"].tolist())
                idx = catch22_names.index("MD_hrv_classic_pnn40")
                feats = feats[:, idx]
                ax.set_ylabel(r"$high\ fluct.$")

            feats_plot = np.zeros((trials, len(confs)))
            for conf_idx in range(len(confs)):
                feats_plot[:, conf_idx] = feats[np.array(IDs) == conf_idx]

            ax.plot(
                np.arange(len(confs)),
                np.mean(feats_plot, axis=0),
                color=feature_colors[row * 2 + col],
            )
            ax.fill_between(
                np.arange(len(confs)),
                np.mean(feats_plot, axis=0) - np.std(feats_plot, axis=0),
                np.mean(feats_plot, axis=0) + np.std(feats_plot, axis=0),
                color=feature_colors[row * 2 + col],
                alpha=0.3,
            )

            if row == 1:
                ax.set_xlabel(get_conf_axis_label(varying_conf_indices))
                ax.set_xticks(np.arange(len(confs)))
                ax.set_xticklabels(get_conf_ticklabels(confs, varying_conf_indices))
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

    # Add figure panel letters and render the figure.
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.text(0.01, 0.97, "A", fontsize=12, fontweight="bold")
    ax.text(0.01, 0.37, "B", fontsize=12, fontweight="bold")
    ax.text(0.4, 0.37, "C", fontsize=12, fontweight="bold")

    plt.show()


if __name__ == "__main__":
    main()
