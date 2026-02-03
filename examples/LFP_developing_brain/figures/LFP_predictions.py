import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import ncpi

# =============================================================================
# Configuration
# =============================================================================

# Folder with parameters of LIF model simulations
SIM_PARAM_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "simulation", "Hagen_model", "simulation", "params"
)
if SIM_PARAM_PATH not in sys.path:
    sys.path.append(SIM_PARAM_PATH)

# Path to the folder with prediction results
PRED_RESULTS = os.path.join("..", "data")

# Calculate new firing rates (True) or load them from file if they already exist (False).
# If firing rates do not exist, they will not be plotted.
COMPUTE_FIRING_RATE = False

# Path to saved firing rates
FR_PATH = os.path.join(".", "data")

# Number of samples to draw from the predictions for computing the firing rates
N_SAMPLES = 50

# Methods to plot
ALL_METHODS = ["catch22", "power_spectrum_parameterization"]

# Select the statistical analysis method ('cohen', 'lmer')
STATISTICAL_ANALYSIS = "lmer"

# Post-hoc config (match EEG_predictions style)
CONTROL_GROUP = "4"
GROUPS_TO_ANNOTATE = ["8", "9", "10", "11", "12"]

# Random seed for numpy
np.random.seed(0)


# =============================================================================
# Helpers (EEG_predictions-style)
# =============================================================================

def _fit_best_model_and_posthoc(
    df: pd.DataFrame,
    *,
    group_col: str,
    subj_col: str,
    control_group: str,
    specs: List[str],
    numeric: Optional[List[str]] = None,
    print_info: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Run BIC model selection + Holm post-hocs using Analysis.lmer_tests.

    Returns
    -------
    dict spec -> DataFrame
        For this script, we use specs=["group"], so the return is {"group": <emmeans contrasts>}.
    """
    analysis = ncpi.Analysis(df)

    # If there is only one subject level, mixed model can't be fit; use fixed-effects only.
    n_subj = df[subj_col].astype(str).nunique(dropna=True)
    full_fx = f"Predictions ~ {group_col}"
    if n_subj > 1:
        full_re = f"Predictions ~ {group_col} + (1|{subj_col})"
        models = [full_re, full_fx]
    else:
        models = [full_fx]

    out = analysis.lmer_tests(
        models=models,
        numeric=numeric,
        group_col=group_col,
        control_group=str(control_group),
        specs=specs,
        posthoc={"adjust": "holm"},
        print_info=print_info,
        lrt=False,
        return_model_info=False,
    )
    return out if isinstance(out, dict) else out.posthoc


def _pvalue_from_emm_contrasts(stat_df: pd.DataFrame, group: str, control: str) -> Optional[float]:
    """Extract p-value for contrasts, handling both '8 - 4' and 'group8 - group4' formats."""
    if stat_df is None or stat_df.empty or "contrast" not in stat_df.columns:
        return None

    candidates = [
        f"{group} - {control}",
        f"group{group} - group{control}",
    ]
    row = stat_df.loc[stat_df["contrast"].isin(candidates)]
    if row.empty:
        return None

    # Some emmeans tables may have NaN p-values (e.g., zero variance); treat as missing
    p = row["p.value"].iloc[0]
    try:
        p = float(p)
    except Exception:
        return None
    return None if np.isnan(p) else p



# =============================================================================
# Load predictions
# =============================================================================

sim_params: Dict[str, np.ndarray] = {}
IDs: Dict[str, np.ndarray] = {}
firing_rates: Dict[str, np.ndarray] = {}

all_IDs: Dict[str, np.ndarray] = {}
predictions_EI: Dict[str, np.ndarray] = {}
predictions_all: Dict[str, np.ndarray] = {}
ages: Dict[str, np.ndarray] = {}

for method in ALL_METHODS:
    # Load data
    try:
        data_EI = np.load(os.path.join(PRED_RESULTS, method, "emp_data_reduced.pkl"), allow_pickle=True)
        data_all = np.load(os.path.join(PRED_RESULTS, method, "emp_data_all.pkl"), allow_pickle=True)

        all_IDs[method] = np.array(data_all["subject_id"].tolist())
        predictions_EI[method] = np.array(data_EI["Predictions"].tolist())
        predictions_all[method] = np.array(data_all["Predictions"].tolist())
        ages[method] = np.array(data_EI["group"].tolist())

        # Pick only ages >= 4
        mask = ages[method] >= 4
        all_IDs[method] = all_IDs[method][mask]
        predictions_EI[method] = predictions_EI[method][mask, :]
        predictions_all[method] = predictions_all[method][mask, :]
        ages[method] = ages[method][mask]

    except Exception:
        all_IDs[method] = np.array([])
        predictions_EI[method] = np.array([])
        predictions_all[method] = np.array([])
        ages[method] = np.array([])

    firing_rates[method] = np.zeros((len(np.unique(ages[method])), N_SAMPLES))
    IDs[method] = np.zeros((len(np.unique(ages[method])), N_SAMPLES))

    # Parameter sampling for computing the firing rates
    if COMPUTE_FIRING_RATE:
        sim_params[method] = np.zeros((7, len(np.unique(ages[method])), N_SAMPLES))
        for param in range(4):
            for i, age in enumerate(np.unique(ages[method])):
                idx = np.where(ages[method] == age)[0]
                data_IDs = all_IDs[method][idx]
                data_EI = predictions_EI[method][idx, param]
                data_EI = data_EI[~np.isnan(data_EI)]

                # Randomly sample some predictions within the first and third quartile
                q1, q3 = np.percentile(data_EI, [25, 75])

                # Check if the quartiles are not NaN
                if not np.isnan(q1) and not np.isnan(q3):
                    within_quartiles = np.where((data_EI >= q1) & (data_EI <= q3))[0]

                    # Check within_quartiles is not empty
                    if len(within_quartiles) > 0:
                        # Randomly sample n_samples from within_quartiles
                        idx_samples = within_quartiles[np.random.randint(0, len(within_quartiles), N_SAMPLES)]
                        IDs[method][i, :] = data_IDs[idx_samples]
                        # E/I
                        if param == 0:
                            for j in range(4):
                                data_all_j = predictions_all[method][idx, j]
                                data_all_j = data_all_j[~np.isnan(data_all_j)]
                                sim_params[method][j, i, :] = data_all_j[idx_samples]
                        # tau_syn_exc, tau_syn_inh, J_syn_ext
                        else:
                            sim_params[method][param + 3, i, :] = data_EI[idx_samples]

        # Firing rates
        for i, age in enumerate(np.unique(ages[method])):
            for sample in range(N_SAMPLES):
                print(f"\nComputing firing rate for {method} at age {age} and sample {sample}")
                # Parameters of the model
                J_EE = sim_params[method][0, i, sample]
                J_IE = sim_params[method][1, i, sample]
                J_EI = sim_params[method][2, i, sample]
                J_II = sim_params[method][3, i, sample]
                tau_syn_E = sim_params[method][4, i, sample]
                tau_syn_I = sim_params[method][5, i, sample]
                J_ext = sim_params[method][6, i, sample]

                # Load LIF_params
                from network_params import LIF_params

                # Modify parameters
                LIF_params["J_YX"] = [[J_EE, J_IE], [J_EI, J_II]]
                LIF_params["tau_syn_YX"] = [[tau_syn_E, tau_syn_I], [tau_syn_E, tau_syn_I]]
                LIF_params["J_ext"] = J_ext

                # Create a Simulation object
                sim = ncpi.Simulation(
                    param_folder=os.path.join("../../simulation/Hagen_model/simulation/params"),
                    python_folder=os.path.join("../../simulation/Hagen_model/simulation/python"),
                    output_folder=os.path.join("../../simulation/Hagen_model/simulation/output"),
                )

                # Save parameters to a pickle file
                with open(
                    os.path.join("..", "..", "simulation", "Hagen_model", "simulation", "output", "network.pkl"), "wb"
                ) as f:
                    pickle.dump(LIF_params, f)

                # Run the simulation
                sim.simulate("simulation.py", "simulation_params.py")

                # Load spike times
                with open(
                    os.path.join("..", "..", "simulation", "Hagen_model", "simulation", "output", "times.pkl"), "rb"
                ) as f:
                    times = pickle.load(f)

                # Load tstop
                with open(
                    os.path.join("..", "..", "simulation", "Hagen_model", "simulation", "output", "tstop.pkl"), "rb"
                ) as f:
                    tstop = pickle.load(f)

                # Transient period
                from analysis_params import KernelParams
                transient = KernelParams.transient

                # Mean firing rate of excitatory cells
                times["E"] = times["E"][times["E"] >= transient]
                rate = ((times["E"].size / (tstop - transient)) * 1000) / LIF_params["N_X"][0]
                firing_rates[method][i, sample] = rate

        # Normalize firing rates to the maximum value
        if len(firing_rates[method]) > 0 and np.max(firing_rates[method]) > 0:
            firing_rates[method] /= np.max(firing_rates[method])

# Save firing rates to file
if COMPUTE_FIRING_RATE:
    if not os.path.exists("data"):
        os.makedirs("data")
    with open(os.path.join("data", "firing_rates_preds.pkl"), "wb") as f:
        pickle.dump(firing_rates, f)
    with open(os.path.join("data", "IDs.pkl"), "wb") as f:
        pickle.dump(IDs, f)
else:
    try:
        with open(os.path.join(FR_PATH, "firing_rates_preds.pkl"), "rb") as f:
            firing_rates = pickle.load(f)
        with open(os.path.join(FR_PATH, "IDs.pkl"), "rb") as f:
            IDs = pickle.load(f)
    except FileNotFoundError:
        print("Firing rates not found.")
        pass


# =============================================================================
# Plotting (kept the same as original)
# =============================================================================

fig = plt.figure(figsize=(7.5, 5), dpi=300)
plt.rcParams.update({"font.size": 10, "font.family": "Arial"})
plt.rc("xtick", labelsize=8)
plt.rc("ytick", labelsize=8)

titles = [r"$E/I$", r"$\tau_{syn}^{exc}$ (ms)", r"$\tau_{syn}^{inh}$ (ms)", r"$J_{syn}^{ext}$ (nA)", r"$Norm. fr$"]
y_labels = [r"$catch22$", r"$1/f$" + " " + r"$slope$"]

cmap = plt.colormaps["viridis"]

# Add rectangles to each row
for row in range(2):
    ax_bg = fig.add_axes([0.01, 0.53 - row * 0.52, 0.98, 0.45 if row == 0 else 0.52])
    ax_bg.add_patch(plt.Rectangle((0, 0), 1, 1, color="red" if row == 0 else "blue", alpha=0.1))
    ax_bg.set_xticks([])
    ax_bg.set_yticks([])

# Plots
for row in range(2):
    for col in range(5):
        ax = fig.add_axes([0.08 + col * 0.19, 0.55 - row * 0.45, 0.14, 0.32])
        try:
            method = ALL_METHODS[row]
        except Exception:
            method = ALL_METHODS[0]

        try:
            # Plot parameter predictions and firing rates as a function of age
            for i, age in enumerate(np.unique(ages[method])):
                idx = np.where(ages[method] == age)[0]
                if col < 4:
                    data_plot = predictions_EI[method][idx, col]
                else:
                    data_plot = firing_rates[method][i, :]

                data_plot = data_plot[~np.isnan(data_plot)]
                if data_plot.size == 0:
                    continue

                # Clip the data between the 5% and 95% quantiles
                q1, q3 = np.percentile(data_plot, [5, 95])
                clipped_data = data_plot[(data_plot >= q1) & (data_plot <= q3)]

                violin = ax.violinplot(clipped_data, positions=[age], widths=0.9, showextrema=False)
                for pc in violin["bodies"]:
                    pc.set_facecolor(cmap(i / len(np.unique(ages[method]))))
                    pc.set_edgecolor("black")
                    pc.set_alpha(0.8)
                    pc.set_linewidth(0.2)

                box = ax.boxplot(
                    data_plot,
                    positions=[age],
                    showfliers=False,
                    widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color="red", linewidth=0.8),
                    whiskerprops=dict(color="black", linewidth=0.5),
                    capprops=dict(color="black", linewidth=0.5),
                    boxprops=dict(linewidth=0.5, facecolor=(0, 0, 0, 0)),
                )
                for patch in box["boxes"]:
                    patch.set_linewidth(0.2)

            # Statistical analysis
            if STATISTICAL_ANALYSIS == "lmer":
                print("\n--- Linear mixed model analysis.")
            elif STATISTICAL_ANALYSIS == "cohen":
                print("\n--- Cohen's d analysis.")

            # Construct stats DataFrame similarly to the original script, but with EEG-style column names
            if col < 4:
                df_stat = pd.DataFrame(
                    {
                        "group": ages[method].astype(int).astype(str),
                        "subject_id": all_IDs[method].astype(str),
                        "epoch": np.arange(len(ages[method])),
                        "sensor": np.zeros(len(ages[method]), dtype=int),
                        "Predictions": predictions_EI[method][:, col],
                    }
                )
            else:
                df_stat = pd.DataFrame(
                    {
                        "group": np.repeat(np.unique(ages[method]).astype(int).astype(str), N_SAMPLES),
                        "subject_id": IDs[method].flatten().astype(str),
                        "epoch": np.arange(firing_rates[method].size),
                        "sensor": np.zeros(firing_rates[method].size, dtype=int),
                        "Predictions": firing_rates[method].flatten(),
                    }
                )

            df_stat = df_stat[~np.isnan(df_stat["Predictions"])].copy()
            df_stat["group"] = df_stat["group"].astype(str)

            if STATISTICAL_ANALYSIS == "lmer":
                posthoc = _fit_best_model_and_posthoc(
                    df_stat,
                    group_col="group",
                    subj_col="subject_id",
                    control_group=CONTROL_GROUP,
                    specs=["group"],
                    numeric=["Predictions"],
                    print_info=False,
                )
                stat_df = posthoc["group"]
            else:
                analysis = ncpi.Analysis(df_stat)
                stat_df = analysis.cohend(
                    control_group=CONTROL_GROUP,
                    data_col="Predictions",
                    data_index=-1,
                    group_col="group",
                    sensor_col="sensor",
                )

            # debug info
            print(f"\nStatistical analysis for {method}, column {col}:")
            print(stat_df)

            # Add p-values to the plot
            y_max = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            delta = (y_max - y_min) * 0.1

            groups = GROUPS_TO_ANNOTATE
            for i, group in enumerate(groups):
                if STATISTICAL_ANALYSIS == "lmer":
                    p_value = _pvalue_from_emm_contrasts(stat_df, group=group, control=CONTROL_GROUP)
                    if p_value is None:
                        continue

                    # Significance levels
                    if 0.05 > p_value >= 0.01:
                        pp = "*"
                    elif 0.01 > p_value >= 0.001:
                        pp = "**"
                    elif 0.001 > p_value >= 0.0001:
                        pp = "***"
                    elif p_value < 0.0001:
                        pp = "****"
                    else:
                        pp = "n.s."

                    offset = (-delta * 0.2) if pp != "n.s." else (delta * 0.05)

                else:
                    df_d = stat_df.get(f"{group}vs{CONTROL_GROUP}")
                    if df_d is None or df_d.empty:
                        continue
                    d_value = float(df_d["d"].iloc[0])
                    pp = f"{d_value:.2f}"
                    offset = 0.0

                ax.text(
                    (int(groups[i]) - int(CONTROL_GROUP)) / 2.0 + int(CONTROL_GROUP),
                    y_max + delta * i + delta * 0.1 + offset,
                    f"{pp}",
                    ha="center",
                    fontsize=8 if pp != "n.s." else 7,
                )
                ax.plot(
                    [int(CONTROL_GROUP), int(groups[0]) + i],
                    [y_max + delta * i, y_max + delta * i],
                    color="black",
                    linewidth=0.5,
                )

            # Change y-lim
            ax.set_ylim([y_min, y_max + delta * (len(groups))])

        except Exception:
            pass

        ax.set_title(titles[col])

        # X-axis labels
        try:
            if row == 1:
                ax.set_xticks(np.unique(ages[method])[::2])
                ax.set_xticklabels([f"{str(i)}" for i in np.unique(ages[method])[::2]])
                ax.set_xlabel("Postnatal days")
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
        except Exception:
            pass

# Plot letters
ax_letters = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax_letters.axis("off")
ax_letters.text(0.015, 0.945, "A", fontsize=12, fontweight="bold")
ax_letters.text(0.015, 0.495, "B", fontsize=12, fontweight="bold")

# Titles
ax_letters.text(0.5, 0.94, y_labels[0], color="red", alpha=0.5, fontsize=10, ha="center")
ax_letters.text(0.5, 0.49, y_labels[1], color="blue", alpha=0.5, fontsize=10, ha="center")

# Save the figure
plt.savefig(f"LFP_predictions_{STATISTICAL_ANALYSIS}.png", bbox_inches="tight")
# plt.show()