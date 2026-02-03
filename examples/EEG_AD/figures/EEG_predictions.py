from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import ncpi


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Path to the results folder
RESULTS_PATH = os.path.join("..", "data")

# Statistical analysis method ('cohend' or 'lmer')
STATISTICAL_ANALYSIS = "lmer"

# p-value threshold for displaying LMER stats on topomaps
P_VALUE_TH = 0.01


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _append_lmer_stat_for_sensor(
    lmer_df: pd.DataFrame,
    sensor_value: str,
    p_value_th: float,
    out: List[float],
) -> List[float]:
    """Append z/t statistic for a given sensor from an emmeans result frame.

    The expected `lmer_df` is already filtered down to a single contrast (e.g., 'ADMIL - HC')
    but potentially includes multiple sensors.
    """
    row = lmer_df[lmer_df["sensor"] == sensor_value]
    if row.empty:
        out.append(0.0)
        return out

    p_value = float(row["p.value"].iloc[0])
    if "z.ratio" in row.columns:
        stat = float(row["z.ratio"].iloc[0])
    elif "t.ratio" in row.columns:
        stat = float(row["t.ratio"].iloc[0])
    else:
        # Fallback: if neither column exists, append 0 (conservative)
        out.append(0.0)
        return out

    out.append(stat if p_value < p_value_th else 0.0)
    return out


def _fit_best_model_and_posthoc(
    df: pd.DataFrame,
    *,
    group_col: str,
    sensor_col: str,
    subj_col: str,
    control_group: str,
    specs: List[str],
    numeric: Optional[List[str]] = None,
    print_info: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Run BIC model selection + LRT (interaction) + Holm post-hocs.

    This is a lightweight wrapper around :meth:`ncpi.Analysis.lmer_tests` with
    ``lrt=True`` to keep the main script readable.

    Returns
    -------
    (posthoc_results, lrt_table)
        posthoc_results is the dict of post-hoc result tables keyed by spec.
        lrt_table is the nested-model comparison table (anova / LRT).
    """
    analysis = ncpi.Analysis(df)

    # Candidate models for BIC comparison
    full_re = f"Predictions ~ {group_col} * {sensor_col} + (1|{subj_col})"
    full_fx = f"Predictions ~ {group_col} * {sensor_col}"

    # Posthoc config: Holm adjustment (Holm-Bonferroni)
    posthoc_cfg = {"adjust": "holm"}

    out = analysis.lmer_tests(
        models=[full_re, full_fx],
        numeric=numeric,
        group_col=group_col,
        control_group=control_group,
        specs=specs,
        posthoc=posthoc_cfg,
        print_info=print_info,
        lrt=True,
        lrt_drop=[f"{group_col}:{sensor_col}"],
        return_model_info=False,
    )

    # Backward-compatible: lmer_tests returns a dict unless extra outputs were requested.
    if isinstance(out, dict):
        # Should not happen with lrt=True, but keep this safe.
        return out, pd.DataFrame()

    posthoc_results = out.posthoc
    lrt_df = out.lrt if out.lrt is not None else pd.DataFrame()
    return posthoc_results, lrt_df


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
def main() -> None:
    # Figure layout parameters
    ncols = 6
    nrows = 4

    left = 0.06
    right = 0.11
    width = (1.0 - left - right) / 6 - 0.03
    height = 1.0 / 5 - 0.025
    bottom = 1 - (1.0 / 5 + 0.07)

    new_spacing_x = 0.08
    new_spacing_y = 0.05
    spacing_x = 0.04
    spacing_y = 0.064

    fig1 = plt.figure(figsize=(7.5, 5.5), dpi=150)
    current_bottom = bottom

    # Background rectangles
    for row in range(2):
        ax = fig1.add_axes([0.01, 0.51 - row * 0.47, 0.98, 0.46 if row == 0 else 0.47])
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="red" if row == 0 else "blue", alpha=0.1))
        ax.set_xticks([])
        ax.set_yticks([])

    # Iterate rows/cols of the figure
    for row in range(nrows):
        method = "catch22" if row in (0, 1) else "power_spectrum_parameterization"

        try:
            data = pd.read_pickle(os.path.join(RESULTS_PATH, method, "emp_data_reduced.pkl"))
        except Exception as e:
            print(f"Error loading data for {method}: {e}")
            continue

        # Clean up types and remove MCI group so correction does not include it
        if "sensor" in data.columns:
            data["sensor"] = data["sensor"].apply(lambda x: str(x))
        if "group" not in data.columns:
            raise ValueError("Expected column 'group' in loaded data.")
        data = data[data["group"] != "MCI"]

        current_left = left
        for col in range(ncols):
            print(f"\n----- Processing row {row}, column {col} -----\n")

            # Group selection
            if col in (0, 3):
                group = "ADMIL"
                group_label = "ADMIL"
            elif col in (1, 4):
                group = "ADMOD"
                group_label = "ADMOD"
            else:
                group = "ADSEV"
                group_label = "ADSEV"

            ax1 = fig1.add_axes([current_left, current_bottom, width, height], frameon=False)

            # Update spacing
            current_left += width + (new_spacing_x if col == 2 else spacing_x)

            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(f"{group_label} vs HC", fontsize=10)

            # Pick prediction index (kept as in the original script)
            if col < 3:
                var = 0 if row in (0, 2) else 1
            else:
                var = 3 if row in (0, 2) else 2

            # Prepare per-panel predictions
            data_preds = data.copy()
            data_preds["Predictions"] = data_preds["Predictions"].apply(lambda x: x[var])

            # Statistical analysis
            if STATISTICAL_ANALYSIS == "lmer":
                # IMPORTANT: use new LMER workflow (BIC -> LRT -> Holm posthoc)
                posthoc_results, lrt_table = _fit_best_model_and_posthoc(
                    data_preds,
                    group_col="group",
                    sensor_col="sensor",
                    subj_col="subject_id",
                    control_group="HC",
                    specs=["group|sensor"],
                    numeric=["Predictions"],  # ensure Predictions is treated as numeric in R
                    print_info=False,
                )

                # For transparency (optional)
                print(posthoc_results)

                # Extract results for the current contrast and sensor
                res = posthoc_results["group|sensor"].query("contrast == @group + ' - HC'")
                stat_df = res

            elif STATISTICAL_ANALYSIS == "cohend":
                analysis = ncpi.Analysis(data_preds)
                stat_results = analysis.cohend(control_group="HC", data_col="Predictions", data_index=var)
                stat_df = stat_results[f"{group}vsHC"]  # columns: ['sensor', "d"]
            else:
                raise ValueError("STATISTICAL_ANALYSIS must be 'lmer' or 'cohend'.")

            # Build statistics vector aligned to sensor order
            data_stat: List[float] = []
            empirical_sensors = data['sensor'].unique()

            for elec in range(19):
                if elec >= len(empirical_sensors):
                    data_stat.append(0.0)
                    continue

                sensor_name = str(empirical_sensors[elec])

                if STATISTICAL_ANALYSIS == "lmer":
                    # lmer emmeans results contain a "Sensor" column
                    data_stat = _append_lmer_stat_for_sensor(stat_df, sensor_name, P_VALUE_TH, data_stat)
                else:
                    # Cohen's d: stat_df has columns ['sensor', "d"]
                    row = stat_df[stat_df['sensor'] == sensor_name]
                    data_stat.append(float(row["d"].iloc[0]) if not row.empty else 0.0)

            # Limits
            ylims_stat = [-6.0, 6.0] if STATISTICAL_ANALYSIS == "lmer" else [-1.0, 1.0]

            # Create EEG topography plot
            ch_names = [str(s) for s in empirical_sensors[:19]]

            analysis_plot = ncpi.Analysis(data_stat)
            analysis_plot.eeg_topomap(
                data_stat,
                ch_names=ch_names,
                axes=ax1,
                show=False,
                vmin=ylims_stat[0],
                vmax=ylims_stat[1],
                colorbar=False,
                sensors=True,
                montage="standard_1020",
                extrapolate="local",
            )

        # Update y spacing
        current_bottom -= height + (new_spacing_y if row == 1 else spacing_y)

    # Text and lines (unchanged)
    fontsize = 12
    fig1.text(0.46, 0.94, "catch22", color="red", alpha=0.5, fontsize=12, fontstyle="italic")
    fig1.text(0.46, 0.48, "1/f slope", color="blue", alpha=0.5, fontsize=12, fontstyle="italic")

    fig1.text(0.015, 0.94, "A", fontsize=12, fontweight="bold")
    fig1.text(0.015, 0.48, "B", fontsize=12, fontweight="bold")

    fig1.text(0.24, 0.94, r"$E/I$", ha="center", fontsize=fontsize)
    fig1.text(0.74, 0.94, r"$J_{syn}^{ext}$ (nA)", ha="center", fontsize=fontsize)

    fig1.text(0.24, 0.7, r"$\tau_{syn}^{exc}$ (ms)", ha="center", fontsize=fontsize)
    fig1.text(0.74, 0.7, r"$\tau_{syn}^{inh}$ (ms)", ha="center", fontsize=fontsize)

    fig1.text(0.24, 0.48, r"$E/I$", ha="center", fontsize=fontsize)
    fig1.text(0.74, 0.48, r"$J_{syn}^{ext}$ (nA)", ha="center", fontsize=fontsize)

    fig1.text(0.24, 0.245, r"$\tau_{syn}^{exc}$ (ms)", ha="center", fontsize=fontsize)
    fig1.text(0.74, 0.245, r"$\tau_{syn}^{inh}$ (ms)", ha="center", fontsize=fontsize)

    linepos1 = [0.925, 0.925]
    linepos2 = [0.686, 0.686]

    EI_line_c = mlines.Line2D([0.055, 0.46], linepos1, color="black", linewidth=0.5)
    tauexc_line_c = mlines.Line2D([0.055, 0.46], linepos2, color="black", linewidth=0.5)

    Jext_line_c = mlines.Line2D([0.54, 0.945], linepos1, color="black", linewidth=0.5)
    tauinh_line_c = mlines.Line2D([0.54, 0.945], linepos2, color="black", linewidth=0.5)

    linepos1 = [0.467, 0.467]
    linepos2 = [0.23, 0.23]

    EI_line_f = mlines.Line2D([0.055, 0.46], linepos1, color="black", linewidth=0.5)
    tauexc_line_f = mlines.Line2D([0.055, 0.46], linepos2, color="black", linewidth=0.5)

    Jext_line_f = mlines.Line2D([0.54, 0.945], linepos1, color="black", linewidth=0.5)
    tauinh_line_f = mlines.Line2D([0.54, 0.945], linepos2, color="black", linewidth=0.5)

    fig1.add_artist(EI_line_c)
    fig1.add_artist(Jext_line_c)
    fig1.add_artist(tauexc_line_c)
    fig1.add_artist(tauinh_line_c)

    fig1.add_artist(EI_line_f)
    fig1.add_artist(Jext_line_f)
    fig1.add_artist(tauexc_line_f)
    fig1.add_artist(tauinh_line_f)

    fig1.savefig(f"EEG_predictions_{STATISTICAL_ANALYSIS}.png")

if __name__ == "__main__":
    main()