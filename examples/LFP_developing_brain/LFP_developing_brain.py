import os
import sys
from ncpi import tools as ncpi_tools
import pandas as pd
from pathlib import Path
from ncpi.tools import timer, ensure_module
from ncpi.EphysDatasetParser import EphysDatasetParser, ParseConfig, CanonicalFields

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import tools as shared_tools


# ---------------------------
# User-config / constants
# ---------------------------

# Choose to either download data from Zenodo (True) or load it from a local path (False).
zenodo_dw_sim = True  # simulation data
zenodo_dw_emp = True  # empirical data

# Zenodo URL that contains the simulation data and ML models (used if zenodo_dw_sim is True)
zenodo_URL_sim = "https://zenodo.org/api/records/15351118"

# Zenodo URL that contains the empirical data (used if zenodo_dw_emp is True)
zenodo_URL_emp = "https://zenodo.org/api/records/15382047"

# Paths to zenodo files
zenodo_dir_sim = os.path.expandvars(os.path.expanduser(
    os.path.join("$HOME", "zenodo_sim_files")
))
zenodo_dir_emp = os.path.expandvars(os.path.expanduser(
    os.path.join("$HOME", "zenodo_emp_files")
))

# Methods used to compute the features
all_methods = ["catch22", "power_spectrum_parameterization"]

# ML model used to compute the predictions (MLPRegressor, Ridge)
ML_model = "MLPRegressor"

# ---------------------------
# Empirical LFP loading
# ---------------------------

@timer("Loading LFP data.")
def load_empirical_data(zenodo_dir_emp):
    """Load empirical LFP data from Zenodo and parse it into canonical schema."""
    base_dir = Path(zenodo_dir_emp) / "development_EI_decorrelation" / "baseline" / "LFP"
    files = sorted(base_dir.iterdir())

    rows = []

    for subject_id, file_path in enumerate(files):
        print(
            f"\r Progress: {subject_id + 1} of {len(files)} files loaded",
            end="",
            flush=True,
        )

        config = ParseConfig(
            fields=CanonicalFields(
                data=lambda d: d["LFP"].LFP,
                fs=lambda d: float(d["LFP"].fs),
                ch_names=lambda d: [f"ch{i}" for i in range(d["LFP"].LFP.shape[0])],
                metadata={
                    "subject_id": subject_id,
                    "group": lambda d: int(d["LFP"].age),  # age → group
                    "species": "mouse",
                    "recording_type": "LFP",
                },
            ),
            # 🔑 epoching (5 seconds, non-overlapping)
            epoch_length_s=5.0,
            epoch_step_s=5.0,

            # 🔑 collapse channels (sum LFP)
            aggregate_over=("sensor",),
            aggregate_method="sum",
        )

        parser = EphysDatasetParser(config)
        df = parser.parse(file_path)
        rows.append(df)

    print(f"\nFiles loaded: {len(rows)}")

    out = pd.concat(rows, ignore_index=True)
    return out


# --------------
# Main pipeline
# --------------

def main() -> None:
    # Download simulation data and ML models
    if zenodo_dw_sim:
        ncpi_tools.timer("Downloading simulation data and ML models from Zenodo.")(
            ncpi_tools.download_zenodo_record
        )(zenodo_URL_sim, download_dir=zenodo_dir_sim)

    # Download empirical data
    if zenodo_dw_emp:
        ncpi_tools.timer("Downloading empirical data from Zenodo.")(
            ncpi_tools.download_zenodo_record
        )(zenodo_URL_emp, download_dir=zenodo_dir_emp)

    # Empirical data (LFP) loaded once
    df_emp = load_empirical_data(zenodo_dir_emp)

    for method in all_methods:
        print(f"\n\n--- Method: {method}")

        # 1) Simulation data for this feature method
        X, theta = shared_tools.load_model_features(
            "catch22" if method == "catch22" else "power_spectrum_parameterization_1", # back compatibility
            zenodo_dir_sim,
        )

        # 2) Compute empirical features for this method
        emp_data = shared_tools.compute_features(method, df_emp)

        # 3) Predictions
        emp_data = shared_tools.compute_predictions(
            emp_data,
            data_kind="LFP",
            method="catch22" if method == "catch22" else "power_spectrum_parameterization_1", # back compatibility
            folder=method,
            ML_model=ML_model,
            X=X,
            theta=theta,
            zenodo_dir_sim=zenodo_dir_sim,
        )

        # 4) Save
        shared_tools.save_data(emp_data, method)


if __name__ == "__main__":
    main()
