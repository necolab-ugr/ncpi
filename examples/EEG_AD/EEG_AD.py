import os
import pickle
import pandas as pd
import numpy as np
import shutil
import ncpi
from ncpi import tools
from ncpi.tools import timer

from ncpi.EphysDatasetParser import EphysDatasetParser, ParseConfig


# Select either raw EEG data or source-reconstructed EEG data.
raw = True
if raw:
    data_path = os.path.join(os.sep, 'DATOS','pablomc', 'empirical_datasets', 'POCTEP_data', 'CLEAN', 'SENSORS')
else:
    data_path = os.path.join(os.sep, 'DATOS', 'pablomc', 'empirical_datasets', 'POCTEP_data', 'CLEAN', 'SOURCES', 'dSPM', 'DK')

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
    Load the POCTEP dataset and create a DataFrame that matches the old script's schema:
      ID, Group, Epoch, Sensor, Data, Recording, fs

    This function uses EphysDatasetParser for IO + epoching + z-scoring.
    """
    ldir = sorted(os.listdir(data_path))

    # Tune patterns to strongly prefer the POCTEP MATLAB struct layout:
    #   loadmat(... )['data']['signal'], ['data']['cfg']['fs'], ['data']['cfg']['channels']
    cfg = ParseConfig(
        epoch_length_s=5.0,
        epoch_step_s=5.0,
        zscore=True,
        # Keep default_fs=None so we DON'T silently invent an fs; POCTEP files should have it.
        default_fs=None,
        mat_signal_key_patterns=(r"data\.signal", r"\bsignal\b", r"\beeg\b", r"\bdata\b"),
        mat_fs_key_patterns=(r"data\.cfg\.fs", r"\bfs\b", r"\bsfreq\b", r"\bsrate\b"),
        mat_channel_key_patterns=(r"data\.cfg\.channels", r"\bchannels?\b", r"\bch_names\b", r"\belectrodes?\b"),
        mat_prefer_times_by_channels=True,
    )

    parser = EphysDatasetParser(config=cfg)

    out_frames = []
    for pt, file in enumerate(ldir):
        print(f'\rProcessing {file} - {pt + 1}/{len(ldir)}', end="", flush=True)
        file_path = os.path.join(data_path, file)

        parsed = parser.parse(
            file_path,
            subject_id=pt,                # old code: ID was file index
            species="human",
            group=file.split('_')[0],     # old code: group = prefix before underscore
            condition="resting-state",
            recording_type="EEG",
            data_kind="time_series",
        )
        if parsed is None or parsed.empty:
            continue

        # Map the parser schema -> old schema
        df_old = pd.DataFrame({
            "ID": parsed["subject_id"].astype(int, errors="ignore"),
            "Group": parsed["group"],
            "Epoch": parsed["epoch"].astype(int, errors="ignore"),
            "Sensor": parsed["sensor"],
            "Data": parsed["data"],
        })

        # Old script sets these for the whole DF (fs assumed constant); keep per-row fs too.
        df_old["Recording"] = "EEG"
        df_old["fs"] = parsed["fs"].astype(float, errors="ignore")

        out_frames.append(df_old)

    if not out_frames:
        return pd.DataFrame(columns=["ID", "Group", "Epoch", "Sensor", "Data", "Recording", "fs"])

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
    if "Data" not in df.columns:
        raise ValueError("Expected input DataFrame to contain a 'Data' column with 1D samples.")

    samples = df["Data"].to_list()

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
        sensor_df = emp_data[emp_data["Sensor"].isin([sensor, s])].copy()
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