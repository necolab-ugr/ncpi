import os
import pickle
import pandas as pd
import numpy as np
import shutil
import ncpi
from ncpi import tools
from ncpi.tools import timer

from ncpi.EphysDatasetParser import EphysDatasetParser, ParseConfig, CanonicalFields, DEFAULT_COLUMNS


# Select either raw EEG data or source-reconstructed EEG data.
raw = True
if raw:
    data_path = os.path.join(os.sep, 'DATOS','pablomc', 'empirical_datasets',
                             'POCTEP_data', 'CLEAN', 'SENSORS')
else:
    data_path = os.path.join(os.sep, 'DATOS', 'pablomc', 'empirical_datasets',
                             'POCTEP_data', 'CLEAN', 'SOURCES', 'dSPM', 'DK')

# Choose to either download data from Zenodo (True) or load it from a local path (False).
zenodo_dw_sim = True  # simulation data
zenodo_URL_sim = "https://zenodo.org/api/records/15351118"
zenodo_dir_sim = os.path.join("/home/pablomc", "zenodo_sim_files")

# Methods used to compute the features
all_methods = ['catch22', 'power_spectrum_parameterization_1']

# ML model used to compute the predictions
ML_model = 'MLPRegressor'


def create_POCTEP_dataframe(data_path: str) -> pd.DataFrame:
    """
    Load the POCTEP dataset and return a DataFrame in the canonical schema
    defined by ``EphysDatasetParser.DEFAULT_COLUMNS``.
    """
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
        # Return an empty canonical DataFrame rather than erroring
        return pd.DataFrame(columns=EphysDatasetParser.DEFAULT_COLUMNS)

    # Canonical columns are already ensured by the parser;
    # concat preserves them (extras, if any, appear after canonical).
    return pd.concat(out_frames, ignore_index=True)


@timer("Loading simulation data.")
def load_model_features(method: str, zenodo_dir_sim: str):
    """Load model parameters (theta) and features (X) from simulation data."""
    with open(os.path.join(zenodo_dir_sim, 'data', method, 'sim_theta'), 'rb') as f:
        theta = pickle.load(f)
    with open(os.path.join(zenodo_dir_sim, 'data', method, 'sim_X'), 'rb') as f:
        X = pickle.load(f)

    # Print info (kept from old script)
    print('theta:')
    for key, value in theta.items():
        if isinstance(value, np.ndarray):
            print(f'--Shape of {key}: {value.shape}')
        else:
            print(f'--{key}: {value}')
    print(f'Shape of X: {X.shape}')
    return X, theta


@timer("Feature extraction")
def feature_extraction(method: str, df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from empirical data using specified method."""
    if "data" not in df.columns:
        raise ValueError("Expected input DataFrame to contain a 'data' column with 1D samples.")

    samples = df["data"].to_list()

    if method == "catch22":
        features = ncpi.Features(method="catch22", params={"normalize": True})
        feats = features.compute_features(samples)
        emp_data = df.copy()
        emp_data["Features"] = feats
        return emp_data

    if method == "power_spectrum_parameterization_1":
        fooof_setup_emp = {
            "peak_threshold": 1.0,
            "min_peak_height": 0.0,
            "max_n_peaks": 5,
            "peak_width_limits": (10.0, 50.0),
        }

        params = {
            "fs": float(df["fs"].iloc[0]),
            "freq_range": (5.0, 45.0),
            "specparam_model": dict(fooof_setup_emp),
            "metric_thresholds": {
                "gof_rsquared": 0.9
            },
            "metric_policy": "reject"
        }

        features = ncpi.Features(method="specparam", params=params)
        feats = features.compute_features(samples)

        emp_data = df.copy()
        emp_data["Features"] = [float(np.asarray(d["aperiodic_params"])[1]) for d in feats]
        return emp_data

    raise ValueError(f"Unknown method: {method}")


@timer('Computing predictions...')
def compute_predictions_neural_circuit(
    emp_data: pd.DataFrame,
    method: str,
    ML_model: str,
    X,
    theta,
    zenodo_dir_sim: str,
    sensor_list=None
) -> pd.DataFrame:
    emp_data = emp_data.copy()
    emp_data["Predictions"] = np.nan

    if sensor_list is None:
        sensor_list = [
            'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1',
            'O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'
        ]

    # Create folder to save results
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("data", method), exist_ok=True)

    # Create inference object
    inference = ncpi.Inference(model=ML_model)
    inference.add_simulation_data(X, theta['data'])

    for s, sensor in enumerate(sensor_list):
        print(f'--- Sensor: {sensor}')

        shutil.copy(
            os.path.join(zenodo_dir_sim, 'ML_models/EEG', sensor, method, 'model'),
            os.path.join('data', 'model.pkl')
        )
        shutil.copy(
            os.path.join(zenodo_dir_sim, 'ML_models/EEG', sensor, method, 'scaler'),
            os.path.join('data', 'scaler.pkl')
        )

        # Keep the old filtering behavior exactly: allow matching either by sensor name or numeric index
        sensor_df = emp_data[emp_data["sensor"].isin([sensor, s])].copy()
        if sensor_df.empty:
            continue

        predictions = inference.predict(np.array(sensor_df["Features"].to_list()))
        sensor_df["Predictions"] = [list(pred) for pred in predictions]

        # Correct write-back (old code intended to update the corresponding rows)
        emp_data.loc[sensor_df.index, "Predictions"] = sensor_df["Predictions"]

    return emp_data


def save_data(emp_data: pd.DataFrame, method: str) -> None:
    # Save the data including predictions of all parameters
    emp_data.to_pickle(os.path.join('data', method, 'emp_data_all.pkl'))

    # Replace parameters of recurrent synaptic conductances with the ratio (E/I)_net
    E_I_net = emp_data['Predictions'].apply(lambda x: (x[0]/x[2]) / (x[1]/x[3]))
    others = emp_data['Predictions'].apply(lambda x: x[4:])
    emp_data = emp_data.copy()
    emp_data['Predictions'] = (
        np.concatenate(
            (E_I_net.values.reshape(-1, 1), np.array(others.tolist())),
            axis=1
        )
    ).tolist()

    # Save the data including predictions of (E/I)_net
    emp_data.to_pickle(os.path.join('data', method, 'emp_data_reduced.pkl'))


if __name__ == "__main__":
    if zenodo_dw_sim:
        tools.timer("Downloading simulation data and ML models from Zenodo")(
            tools.download_zenodo_record
        )(zenodo_URL_sim, download_dir=zenodo_dir_sim)

    for method in all_methods:
        print(f'\n\n--- Method: {method}')

        X, theta = load_model_features(method, zenodo_dir_sim)

        df = create_POCTEP_dataframe(data_path=data_path)

        emp_data = feature_extraction(method, df)

        emp_data = compute_predictions_neural_circuit(emp_data, method, ML_model, X, theta, zenodo_dir_sim)

        save_data(emp_data, method)