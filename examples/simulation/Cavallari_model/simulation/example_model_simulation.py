import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
import ncpi
import numpy as np
import pandas as pd
import scipy.signal as ss
from matplotlib import pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent
PARAMS_DIR = ROOT_DIR / "params"
PYTHON_DIR = ROOT_DIR / "python"
OUTPUT_DIR = ROOT_DIR / "output"
DATA_DIR = ROOT_DIR / "data"

sys.path.append(str(PARAMS_DIR))

# Set to True to run new simulations of the Cavallari LIF network model, or False to load
# precomputed results from pickle files located in the local 'data' folder.
compute_new_sim = True

# Number of repetitions of each simulation.
trials = 2

# Configurations of parameters to simulate:
# [g_EE, g_IE, g_EI, g_II, 
# tau_syn_AMPA_exc, tau_syn_AMPA_inh, tau_syn_GABA_exc, tau_syn_GABA_inh, 
# ext_input_scale, v_0]

# Baseline configuration varying only v_0
# confs = [
#     [0.178, 0.233, -2.01, -2.70, 2.0, 1.0, 5.0, 5.0, 1.0, 1.5],
#     [0.178, 0.233, -2.01, -2.70, 2.0, 1.0, 5.0, 5.0, 1.0, 3.0],
# ]

# Configuration with increased inhibitory conductances
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
    conf_array = np.asarray(configurations, dtype=float)
    return [
        idx for idx in range(conf_array.shape[1])
        if not np.allclose(conf_array[:, idx], conf_array[0, idx])
    ]


def format_conf_label(config, varying_indices, multiline=False):
    if not varying_indices:
        return "baseline"
    separator = "\n" if multiline else ", "
    return separator.join(
        f"{CONF_PARAM_LABELS[idx]} = {config[idx]:.2f}"
        for idx in varying_indices
    )


def get_conf_axis_label(varying_indices):
    if len(varying_indices) == 1:
        return CONF_PARAM_LABELS[varying_indices[0]]
    return "Configuration"


def get_conf_ticklabels(configurations, varying_indices):
    if len(varying_indices) == 1:
        idx = varying_indices[0]
        return [f"{config[idx]:g}" for config in configurations]
    return [format_conf_label(config, varying_indices, multiline=True) for config in configurations]


def get_spike_rate(times, transient, dt, tstop):
    bins = np.arange(transient, tstop + dt, dt)
    hist, _ = np.histogram(times, bins=bins)
    return bins, hist.astype(float)


def get_plot_window(transient, tstop, window_ms=500.0):
    end = float(tstop)
    start = max(float(transient), end - float(window_ms))
    if end <= start:
        start = max(0.0, end - float(window_ms))
    return [start, end]


def get_top_row_axes(fig, n_cols, y, h):
    left = 0.1
    gap = 0.05
    total_width = 0.85
    width = (total_width - gap * max(0, n_cols - 1)) / max(1, n_cols)
    return [fig.add_axes([left + col * (width + gap), y, width, h]) for col in range(n_cols)]


def load_trial_output(path):
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


def get_proxy_signal(proxy_data):
    if isinstance(proxy_data, dict) and "signal" in proxy_data:
        return np.asarray(proxy_data["signal"], dtype=float)
    return np.asarray(proxy_data, dtype=float)


def get_proxy_times(proxy_data):
    if isinstance(proxy_data, dict) and "times" in proxy_data:
        return np.asarray(proxy_data["times"], dtype=float)
    raise KeyError("Proxy payload does not contain an explicit time axis.")


def _extract_exc_currents(exc_state_events):
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
    start_time_pos = int(500.0 / float(sim_step))
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


def decimate_proxy(proxy_data, decimation_factor=10):
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


def build_cavallari_params(config):
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
    varying_conf_indices = get_varying_conf_indices(confs)
    transient_ms = 500.0

    np.random.seed(0)

    sim = ncpi.Simulation(
        param_folder=str(PARAMS_DIR),
        python_folder=str(PYTHON_DIR),
        output_folder=str(OUTPUT_DIR),
    )

    spikes = [[] for _ in range(trials)]
    proxies = [[] for _ in range(trials)]

    for trial in range(trials):
        for k, params in enumerate(confs):
            if compute_new_sim:
                print(f"\nTrial {trial + 1}/{trials}, Configuration {k + 1}/{len(confs)}")
                lif_params = build_cavallari_params(params)

                with open(OUTPUT_DIR / "network.pkl", "wb") as f:
                    pickle.dump(lif_params, f)

                start_time = time.time()
                sim.simulate("simulation.py", "simulation_params.py")
                end_time = time.time()
                print(f"Simulation finished in {end_time - start_time:.2f} s.")

                with open(OUTPUT_DIR / "times.pkl", "rb") as f:
                    times = pickle.load(f)
                with open(OUTPUT_DIR / "gids.pkl", "rb") as f:
                    gids = pickle.load(f)
                with open(OUTPUT_DIR / "tstop.pkl", "rb") as f:
                    tstop = pickle.load(f)
                with open(OUTPUT_DIR / "dt.pkl", "rb") as f:
                    dt = pickle.load(f)
                with open(OUTPUT_DIR / "exc_state_events.pkl", "rb") as f:
                    exc_state_events = pickle.load(f)

                transient = min(float(transient_ms), max(0.0, float(tstop) - float(dt)))
                P_X = lif_params["X"]
                N_X = lif_params["N_X"]

                for X in P_X:
                    keep = times[X] >= transient
                    times[X] = times[X][keep]
                    gids[X] = gids[X][keep]

                proxy_data = decimate_proxy(compute_proxy(exc_state_events, dt))
                spikes[trial].append([times, gids])
                proxies[trial].append(proxy_data)

                DATA_DIR.mkdir(exist_ok=True)
                with open(DATA_DIR / f"output_{k}_{trial}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "times": times,
                            "gids": gids,
                            "proxy_data": proxy_data,
                            "dt": dt,
                            "tstop": tstop,
                            "transient": transient,
                            "P_X": P_X,
                            "N_X": N_X,
                        },
                        f,
                    )
            else:
                try:
                    payload = load_trial_output(DATA_DIR / f"output_{k}_{trial}.pkl")
                except FileNotFoundError:
                    print(f"File {DATA_DIR / f'output_{k}_{trial}.pkl'} not found. Please run the simulation first.")
                    sys.exit(1)

                times = payload["times"]
                gids = payload["gids"]
                proxy_data = payload["proxy_data"]
                dt = payload["dt"]
                tstop = payload["tstop"]
                transient = payload["transient"]
                P_X = payload["P_X"]
                N_X = payload["N_X"]
                if not isinstance(proxy_data, dict) or proxy_data.get("decimation_factor") != 10:
                    proxy_data = decimate_proxy(proxy_data)
                spikes[trial].append([times, gids])
                proxies[trial].append(proxy_data)

    fig = plt.figure(figsize=(7.5, 6.0), dpi=300)
    plt.rcParams.update({"font.size": 10, "font.family": "Arial"})
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)

    T = get_plot_window(transient, tstop, window_ms=500.0)
    signal_fs = 1000.0 / (10.0 * float(dt))

    colors = ["#1f77b4", "#ff7f0e"]
    top_axes = get_top_row_axes(fig, len(confs), 0.73, 0.22)
    middle_axes = get_top_row_axes(fig, len(confs), 0.6, 0.12)
    lower_axes = get_top_row_axes(fig, len(confs), 0.47, 0.12)

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

    for col, ax in enumerate(lower_axes):
        proxy = get_proxy_signal(proxies[0][col])
        bins = get_proxy_times(proxies[0][col])
        ii = (bins >= T[0]) & (bins <= T[1])
        ax.plot(bins[ii], proxy[ii], color="k")

        if col == 0:
            ax.set_ylabel("Proxy")
            ax.yaxis.set_label_coords(-0.22, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("t (ms)")
        ax.axis("tight")

    ax = fig.add_axes([0.1, 0.07, 0.27, 0.3])
    psd_colors = [f"C{i}" for i in range(len(confs))]
    for col in range(len(confs)):
        proxy = np.asarray([get_proxy_signal(proxies[trial][col]) for trial in range(trials)], dtype=float)
        f, Pxx = ss.welch(proxy, fs=1000.0 / (10.0 * float(dt)))
        Pxx = np.mean(Pxx, axis=0)
        Pxx = Pxx / np.sum(Pxx)
        f1 = f[f >= 5]
        f2 = f1[f1 <= 200]
        ax.semilogy(
            f2,
            Pxx[(f >= 5) & (f <= 200)],
            label=format_conf_label(confs[col], varying_conf_indices),
            color=psd_colors[col],
        )
    ax.legend(loc="lower left", fontsize=8, labelspacing=0.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")

    all_proxies = []
    IDs = []
    epochs = []
    for trial in range(trials):
        for conf_idx in range(len(confs)):
            all_proxies.append(get_proxy_signal(proxies[trial][conf_idx]))
            IDs.append(conf_idx)
            epochs.append(trial)

    all_features = {}
    all_methods = ["catch22", "power_spectrum_parameterization"]
    for method in all_methods:
        print(f"\n\n--- Method: {method}")
        df = pd.DataFrame(
            {
                "ID": IDs,
                "Group": IDs,
                "Epoch": epochs,
                "Sensor": np.zeros(len(IDs)),
                "Data": all_proxies,
            }
        )
        df.Recording = "proxy"
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
                    "freq_range": (5.0, 200.0),
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

    feature_colors = ["lightcoral", "lightblue", "lightgreen", "lightgrey"]
    for row in range(2):
        for col in range(2):
            ax = fig.add_axes([0.5 + col * 0.27, 0.24 - row * 0.16, 0.18, 0.13])

            if row == 0 and col == 0:
                feats = np.array(all_features["power_spectrum_parameterization"]["Features"].tolist())
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

            ax.plot(np.arange(len(confs)), np.mean(feats_plot, axis=0), color=feature_colors[row * 2 + col])
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

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.text(0.01, 0.97, "A", fontsize=12, fontweight="bold")
    ax.text(0.01, 0.37, "B", fontsize=12, fontweight="bold")
    ax.text(0.4, 0.37, "C", fontsize=12, fontweight="bold")

    plt.show()


if __name__ == "__main__":
    main()
