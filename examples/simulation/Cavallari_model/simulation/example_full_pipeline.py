import os
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
from ncpi import tools


ROOT_DIR = Path(__file__).resolve().parent
PARAMS_DIR = ROOT_DIR / "params"
PYTHON_DIR = ROOT_DIR / "python"
OUTPUT_DIR = ROOT_DIR / "output"
DATA_DIR = ROOT_DIR / "data"

sys.path.append(str(PARAMS_DIR))

# Choose to either download files and precomputed outputs used in simulations of the reference multicompartment neuron
# network model (True) or load them from a local path (False)
zenodo_dw_mult = True

# Zenodo URL that contains the data (used if zenodo_dw_mult is True)
zenodo_URL_mult = "https://zenodo.org/api/records/15429373"

# Zenodo directory where the data is stored (must be an absolute path to correctly load morphologies in NEURON)
zenodo_dir = os.path.expandvars(os.path.expanduser(os.path.join("$HOME", "multicompartment_neuron_network")))

# Set to True to run new simulations of the Cavallari LIF network model, or False to load precomputed results
# from pickle files located in the local 'data' folder.
compute_new_sim = True

# Number of repetitions of each simulation
trials = 2

# Configurations of parameters to simulate:
# [g_EE, g_IE, g_EI, g_II, tau_syn_E, tau_syn_I, ext_input_scale]
confs = [
    [0.178, 0.233, -2.01, -2.70, 2.0, 5.0, 1.],
    [0.178, 0.233, -2.01, -2.70, 2.0, 5.0, 2.],
    [0.178, 0.233, -2.01, -2.70, 2.0, 5.0, 3.],
]

# Do not change these paths if the zenodo_dir has been correctly set:
# (1) Simulation output from the multicompartment neuron network model
output_path = os.path.join(zenodo_dir, "output", "adb947bfb931a5a8d09ad078a6d256b0")

# (2) Path to the data files of the multicompartment neuron models
multicompartment_neuron_network_path = zenodo_dir

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


def get_spike_rate(times, transient, dt, tstop):
    bins = np.arange(transient, tstop + dt, dt)
    hist, _ = np.histogram(times, bins=bins)
    return bins, hist.astype(float)


def _build_cavallari_params(config):
    from network_params import Network_params as base_network_params
    from network_params import Neuron_params as base_neuron_params

    g_EE = config[0]
    g_IE = config[1]
    g_EI = config[2]
    g_II = config[3]
    tau_syn_E = config[4]
    tau_syn_I = config[5]
    ext_input_scale = config[6]

    network_params = deepcopy(base_network_params)
    neuron_params = deepcopy(base_neuron_params)

    network_params["g_EE"] = g_EE
    network_params["g_IE"] = g_IE
    network_params["g_EI"] = g_EI
    network_params["g_II"] = g_II
    network_params["g_th_exc_external"] *= ext_input_scale
    network_params["g_th_inh_external"] *= ext_input_scale

    neuron_params[0]["tau_decay_AMPA"] = tau_syn_E
    neuron_params[1]["tau_decay_AMPA"] = tau_syn_E
    neuron_params[0]["tau_decay_GABA_A"] = tau_syn_I
    neuron_params[1]["tau_decay_GABA_A"] = tau_syn_I

    return {
        "X": ["E", "I"],
        "N_X": [network_params["N_exc"], network_params["N_inh"]],
        "neuron_params": {
            "E": neuron_params[0],
            "I": neuron_params[1],
        },
        "C_YX": [[network_params["P"], network_params["P"]], [network_params["P"], network_params["P"]]],
        "g_YX": [
            [network_params["g_EE"], network_params["g_EI"]],
            [network_params["g_IE"], network_params["g_II"]],
        ],
        "delay_YX": [[1.0, 1.0], [1.0, 1.0]],
        "extent": network_params["extent"],
        "model": "iaf_bw_2003",
        "network_params": network_params,
    }


def main():
    from analysis_params import KernelParams

    if zenodo_dw_mult:
        print("\n--- Downloading data.")
        start_time = time.time()
        tools.download_zenodo_record(zenodo_URL_mult, download_dir=zenodo_dir)
        end_time = time.time()
        print(f"All files downloaded in {(end_time - start_time) / 60:.2f} minutes.")

    np.random.seed(0)

    sim = ncpi.Simulation(
        param_folder=str(PARAMS_DIR),
        python_folder=str(PYTHON_DIR),
        output_folder=str(OUTPUT_DIR),
    )

    spikes = [[] for _ in range(trials)]
    CDMs = [[] for _ in range(trials)]

    for trial in range(trials):
        for k, params in enumerate(confs):
            if compute_new_sim:
                print(f"\nTrial {trial + 1}/{trials}, Configuration {k + 1}/{len(confs)}")

                lif_params = _build_cavallari_params(params)

                with open(OUTPUT_DIR / "network.pkl", "wb") as f:
                    pickle.dump(lif_params, f)

                sim.simulate("simulation.py", "simulation_params.py")

                with open(OUTPUT_DIR / "times.pkl", "rb") as f:
                    times = pickle.load(f)

                with open(OUTPUT_DIR / "gids.pkl", "rb") as f:
                    gids = pickle.load(f)

                with open(OUTPUT_DIR / "tstop.pkl", "rb") as f:
                    tstop = pickle.load(f)

                with open(OUTPUT_DIR / "dt.pkl", "rb") as f:
                    dt = pickle.load(f)

                with open(OUTPUT_DIR / "network.pkl", "rb") as f:
                    lif_params = pickle.load(f)
                    P_X = lif_params["X"]
                    N_X = lif_params["N_X"]

                transient = KernelParams.transient
                for X in P_X:
                    gids[X] = gids[X][times[X] >= transient]
                    times[X] = times[X][times[X] >= transient]

                print("Computing the kernel...")
                potential = ncpi.FieldPotential()
                biophys = ["set_Ih_linearized_hay2011", "make_cell_uniform"]

                H_YX = potential.create_kernel(
                    multicompartment_neuron_network_path,
                    KernelParams,
                    biophys,
                    dt,
                    tstop,
                    output_sim_path=output_path,
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

                CDM_data = dict(EE=[], EI=[], IE=[], II=[])
                for X in P_X:
                    for Y in P_X:
                        key = f"{Y}:{X}"
                        CDM_data[f"{X}{Y}"] = ss.decimate(cdm_signals[key], q=10, zero_phase=True)

                spikes[trial].append([times, gids])
                CDMs[trial].append(CDM_data)

                DATA_DIR.mkdir(exist_ok=True)
                with open(DATA_DIR / f"output_{k}_{trial}.pkl", "wb") as f:
                    pickle.dump([times, gids, CDM_data, dt, tstop, transient, P_X, N_X], f)

            else:
                try:
                    with open(DATA_DIR / f"output_{k}_{trial}.pkl", "rb") as f:
                        times, gids, CDM_data, dt, tstop, transient, P_X, N_X = pickle.load(f)
                    spikes[trial].append([times, gids])
                    CDMs[trial].append(CDM_data)
                except FileNotFoundError:
                    print(f"File {DATA_DIR / f'output_{k}_{trial}.pkl'} not found. Please run the simulation first.")
                    sys.exit(1)

    fig = plt.figure(figsize=(7.5, 6.0), dpi=300)
    plt.rcParams.update({"font.size": 10, "font.family": "Arial"})
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)

    T = [4000, 4100]

    colors = ["#1f77b4", "#ff7f0e"]
    for col in range(3):
        ax = fig.add_axes([0.1 + col * 0.3, 0.73, 0.25, 0.22])
        for i, X in enumerate(P_X):
            t = spikes[0][col][0][X]
            gi = spikes[0][col][1][X]
            ii = (t >= T[0]) & (t <= T[1])
            ax.plot(t[ii], gi[ii], ".", color=colors[i], markersize=0.5)

        ax.set_title(r"$s_\mathrm{ext}$ = %s" % confs[col][6])
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

    for col in range(3):
        ax = fig.add_axes([0.1 + col * 0.3, 0.6, 0.25, 0.12])
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

    for col in range(3):
        ax = fig.add_axes([0.1 + col * 0.3, 0.47, 0.25, 0.12])
        CDM = CDMs[0][col]["EE"] + CDMs[0][col]["EI"] + CDMs[0][col]["IE"] + CDMs[0][col]["II"]
        bins = np.arange(transient, tstop, dt)
        bins = bins[::10]
        ii = (bins >= T[0]) & (bins <= T[1])
        ax.plot(bins[ii], CDM[ii], color="k")

        if col == 0:
            ax.set_ylabel(r"CDM ($P_z$)")
            ax.yaxis.set_label_coords(-0.22, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("t (ms)")
        ax.axis("tight")

        y_max = np.max(CDM[ii])
        y_min = np.min(CDM[ii])
        scale = (y_max - y_min) / 5
        ax.plot(
            [T[0] if col < 2 else T[0] + 50, T[0] if col < 2 else T[0] + 50],
            [y_min + scale, y_min],
            "k",
        )
        ax.text(
            T[0] + 1 if col < 2 else T[0] + 51,
            y_min + scale / 4.0,
            r"$2^{%s}nAcm$" % np.round(np.log2(scale * 10 ** (-4))),
            fontsize=8,
        )

    ax = fig.add_axes([0.1, 0.07, 0.27, 0.3])
    colors = ["C0", "C1", "C2"]
    for col in range(3):
        CDM = [
            CDMs[trial][col]["EE"] + CDMs[trial][col]["EI"] + CDMs[trial][col]["IE"] + CDMs[trial][col]["II"]
            for trial in range(trials)
        ]
        f, Pxx = ss.welch(CDM, fs=1000.0 / (10.0 * dt))
        Pxx = np.mean(Pxx, axis=0)
        Pxx = Pxx / np.sum(Pxx)
        f1 = f[f >= 10]
        f2 = f1[f1 <= 200]
        ax.semilogy(
            f2,
            Pxx[(f >= 10) & (f <= 200)],
            label=r"$s_\mathrm{ext}$ = %s" % confs[col][6],
            color=colors[col],
        )
    ax.legend(loc="lower left", fontsize=8, labelspacing=0.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")

    all_CDMs = []
    IDs = []
    epochs = []
    for trial in range(trials):
        for k, params in enumerate(confs):
            all_CDMs.append(CDMs[trial][k]["EE"] + CDMs[trial][k]["EI"] + CDMs[trial][k]["IE"] + CDMs[trial][k]["II"])
            IDs.append(k)
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
                "Data": all_CDMs,
            }
        )
        df.Recording = "LFP"
        df.fs = 1000.0 / (10.0 * dt)

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

    colors = ["lightcoral", "lightblue", "lightgreen", "lightgrey"]
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
            for conf in range(len(confs)):
                feats_plot[:, conf] = feats[np.array(IDs) == conf]

            ax.plot(np.arange(len(confs)), np.mean(feats_plot, axis=0), color=colors[row * 2 + col])
            ax.fill_between(
                np.arange(len(confs)),
                np.mean(feats_plot, axis=0) - np.std(feats_plot, axis=0),
                np.mean(feats_plot, axis=0) + np.std(feats_plot, axis=0),
                color=colors[row * 2 + col],
                alpha=0.3,
            )

            if row == 1:
                ax.set_xlabel(r"$s_\mathrm{ext}$")
                ax.set_xticks(np.arange(3))
                ax.set_xticklabels([f"{confs[i][6]}" for i in range(3)])
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.text(0.01, 0.97, "A", fontsize=12, fontweight="bold")
    ax.text(0.01, 0.37, "B", fontsize=12, fontweight="bold")
    ax.text(0.4, 0.37, "C", fontsize=12, fontweight="bold")

    plt.show()
    # plt.savefig(ROOT_DIR / "example_full_pipeline.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
