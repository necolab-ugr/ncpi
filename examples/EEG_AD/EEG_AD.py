import os
import sys
import pandas as pd
from ncpi import tools as ncpi_tools
from ncpi.tools import timer, ensure_module
from ncpi.EphysDatasetParser import EphysDatasetParser, ParseConfig, CanonicalFields

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import tools as shared_tools

# sklearn models loaded in the EEG/LFP examples require scikit-learn ==1.3.2
ensure_module("sklearn", "1.3.2")

# ---------------------------
# User-config / constants
# ---------------------------

# Select either raw EEG data or source-reconstructed EEG data.
raw = True
if raw:
    data_path = os.path.join(os.sep, "DATOS", "pablomc", "empirical_datasets",
                             "POCTEP_data", "CLEAN", "SENSORS")
else:
    data_path = os.path.join(os.sep, "DATOS", "pablomc", "empirical_datasets",
                             "POCTEP_data", "CLEAN", "SOURCES", "dSPM", "DK")

# Choose to either download data from Zenodo (True) or load it from a local path (False).
zenodo_dw_sim = True  # simulation data
zenodo_URL_sim = "https://zenodo.org/api/records/15351118"
zenodo_dir_sim = os.path.join("/home/pablomc", "zenodo_sim_files")

# Methods used to compute the features
all_methods = ["catch22", "power_spectrum_parameterization"]

# ML model used to compute the predictions (MLPRegressor, Ridge)
ML_model = "MLPRegressor"

# sensor list
sensor_list = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1",
    "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]


# ---------------------------
# Empirical EEG loading
# ---------------------------

@timer("Loading POCTEP EEG data.")
def create_POCTEP_dataframe(data_path: str) -> pd.DataFrame:
    """
    Load the POCTEP dataset and return a DataFrame in the canonical schema
    defined by EphysDatasetParser.
    """
    if not isinstance(data_path, str) or not data_path:
        raise TypeError("`data_path` must be a non-empty string.")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"EEG data directory not found: {data_path}")

    ldir = sorted(os.listdir(data_path))
    out_frames = []

    for subject_id, fname in enumerate(ldir):
        # Skip non-.mat files (and common hidden files)
        if fname.startswith(".") or not fname.lower().endswith(".mat"):
            continue

        print(
            f"\r\033[KProcessing {fname} - {subject_id + 1}/{len(ldir)}",
            end="",
            flush=True,
        )

        file_path = os.path.join(data_path, fname)
        group = fname.split("_")[0]

        cfg = ParseConfig(
            fields=CanonicalFields(
                data=lambda d: d["data"].signal,
                fs=lambda d: float(d["data"].cfg.fs),
                ch_names=lambda d: list(d["data"].cfg.channels),
                metadata={
                    "subject_id": (lambda _d, sid=subject_id: sid),
                    "group": (lambda _d, g=group: g),
                    "species": (lambda _d: "human"),
                    "condition": (lambda _d: "resting-state"),
                    "recording_type": (lambda _d: "EEG"),
                },
            ),
            # epoching (5 seconds, non-overlapping)
            epoch_length_s=5.0,
            epoch_step_s=5.0,
            # z-score normalization (per epoch row)
            zscore=True,
        )

        parsed = EphysDatasetParser(cfg).parse(file_path)
        out_frames.append(parsed)

    if not out_frames:
        print("Error: not out_frames")

    return pd.concat(out_frames, ignore_index=True)


# --------------
# Main pipeline
# --------------

def main() -> None:
    if zenodo_dw_sim:
        ncpi_tools.timer("Downloading simulation data and ML models from Zenodo")(
            ncpi_tools.download_zenodo_record
        )(zenodo_URL_sim, download_dir=zenodo_dir_sim)

    # Load empirical EEG once
    df_emp = create_POCTEP_dataframe(data_path=data_path)

    for method in all_methods:
        print(f"\n\n--- Method: {method}")

        # 1) Simulation data for this feature method
        X, theta = shared_tools.load_model_features(
            "catch22" if method == "catch22" else "power_spectrum_parameterization_1", # back compatibility
            zenodo_dir_sim,
        )

        # 2) Features from empirical EEG
        emp_data = shared_tools.compute_features(method, df_emp)

        # 3) Predictions
        emp_data = shared_tools.compute_predictions(
            emp_data,
            data_kind="EEG",
            method="catch22" if method == "catch22" else "power_spectrum_parameterization_1", # back compatibility
            folder=method,
            ML_model=ML_model,
            X=X,
            theta=theta,
            zenodo_dir_sim=zenodo_dir_sim,
            sensor_list=sensor_list,
        )

        # 4) Save
        shared_tools.save_data(emp_data, method)

    # Restore scikit-learn to latest version
    ensure_module("sklearn")


if __name__ == "__main__":
    main()
