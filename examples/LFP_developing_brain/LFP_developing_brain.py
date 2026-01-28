import os
import shutil
import pickle
import numpy as np
import pandas as pd
import scipy.io
import ncpi
from ncpi import tools
from ncpi.tools import timer

from ncpi.EphysDatasetParser import EphysDatasetParser, ParseConfig

# Choose to either download data from Zenodo (True) or load it from a local path (False).
zenodo_dw_sim = True  # simulation data
zenodo_dw_emp = True  # empirical data

# Zenodo URL that contains the simulation data and ML models (used if zenodo_dw_sim is True)
zenodo_URL_sim = "https://zenodo.org/api/records/15351118"

# Zenodo URL that contains the empirical data (used if zenodo_dw_emp is True)
zenodo_URL_emp = "https://zenodo.org/api/records/15382047"

# Paths to zenodo files
zenodo_dir_sim = os.path.join("/home/pablomc", "zenodo_sim_files")
zenodo_dir_emp = os.path.join("/home/pablomc", "zenodo_emp_files")

# Methods used to compute the features
all_methods = ["catch22", "power_spectrum_parameterization_1"]

# ML model used to compute the predictions (MLPRegressor, Ridge or NPE)
ML_model = "MLPRegressor"


@timer("Loading simulation data.")
def load_model_features(method, zenodo_dir_sim):
    """Load model parameters (theta) and features (X) from simulation data."""
    with open(os.path.join(zenodo_dir_sim, "data", method, "sim_theta"), "rb") as file:
        theta = pickle.load(file)
    with open(os.path.join(zenodo_dir_sim, "data", method, "sim_X"), "rb") as file:
        X = pickle.load(file)

    # Print info
    print("theta:")
    for key, value in theta.items():
        if isinstance(value, np.ndarray):
            print(f"--Shape of {key}: {value.shape}")
        else:
            print(f"--{key}: {value}")
    print(f"Shape of X: {X.shape}")

    return X, theta


@timer("Loading the inverse model.")
def load_inference_data(method, X, theta, zenodo_dir_sim, ML_model):
    """Load the Inference object and copy scaler/model locally."""
    inference = ncpi.Inference(model=ML_model)
    inference.add_simulation_data(X, theta["data"])

    # Create folder to save results
    os.makedirs(os.path.join("data", method), exist_ok=True)

    # Transfer model and scaler to the data folder
    if ML_model == "MLPRegressor":
        folder = "MLP"
    elif ML_model == "Ridge":
        folder = "Ridge"
    elif ML_model == "NPE":
        folder = "SBI"
    else:
        raise ValueError(f"Unknown ML_model: {ML_model}")

    shutil.copy(
        os.path.join(zenodo_dir_sim, "ML_models", "4_param", folder, method, "scaler"),
        os.path.join("data", "scaler.pkl"),
    )
    shutil.copy(
        os.path.join(zenodo_dir_sim, "ML_models", "4_param", folder, method, "model"),
        os.path.join("data", "model.pkl"),
    )

    if ML_model == "NPE":
        shutil.copy(
            os.path.join(
                zenodo_dir_sim, "ML_models", "4_param", folder, method, "density_estimator"
            ),
            os.path.join("data", "density_estimator.pkl"),
        )

    return inference


@timer("Loading LFP data (parser-based, old-schema output).")
def load_empirical_data(zenodo_dir_emp):
    """
    Matches the old script output schema using your EphysDatasetParser for:
      - loading LFP signal
      - summing across channels
      - 5-second epoching

    Returns a DataFrame with columns:
      ID, Group (age), Epoch, Sensor(=0), Data, Recording, fs
    """
    lfp_dir = os.path.join(zenodo_dir_emp, "development_EI_decorrelation", "baseline", "LFP")
    file_list = sorted([f for f in os.listdir(lfp_dir) if f.lower().endswith(".mat")])

    # Configure parser to mimic old pipeline:
    # - epoch into 5s windows (non-overlapping)
    # - aggregate sensors by sum (equivalent to np.sum(LFP, axis=0) before epoching)
    cfg = ParseConfig(
        epoch_length_s=5.0,
        epoch_step_s=5.0,
        aggregate_sensors=True,
        aggregate_method="sum",
        aggregate_sensor_label="aggregate",
        # (optional) make it more likely to pick the intended structure
        mat_signal_key_patterns=(r"\bLFP\b", r"\blfp\b", r"\bdata\b"),
        mat_fs_key_patterns=(r"\bfs\b", r"\bsfreq\b", r"\bsrate\b"),
        mat_channel_key_patterns=(r"\bchannels?\b", r"\bsensors?\b", r"\blabels?\b"),
    )
    parser = EphysDatasetParser(config=cfg)

    out_rows = []
    fs0 = None

    for i, file_name in enumerate(file_list):
        print(f"\r Progress: {i+1} of {len(file_list)} files loaded", end="", flush=True)
        fp = os.path.join(lfp_dir, file_name)

        # --- Read age exactly like the old script ---
        structure = scipy.io.loadmat(fp)
        age = float(structure["LFP"]["age"][0, 0][0, 0])

        # --- Use parser to load + epoch + aggregate ---
        dfp = parser.parse(
            fp,
            subject_id=i,
            group=age,              # keep age in "group" like old Group column
            recording_type="LFP",
            data_kind="raw",
        )

        if dfp.empty:
            continue

        # fs used in old script is fs of the first file
        if fs0 is None:
            fs0 = float(dfp["fs"].iloc[0])

        # dfp has one row per epoch (because we aggregated sensors)
        # Convert to old schema
        for _, r in dfp.iterrows():
            out_rows.append(
                {
                    "ID": int(i),
                    "Group": age,
                    "Epoch": int(r["epoch"]),
                    "Sensor": 0.0,  # old code uses zeros
                    "Data": np.asarray(r["data"], dtype=float),
                }
            )

    print(f"\nFiles loaded: {len(file_list)}")
    if fs0 is None:
        raise RuntimeError("No LFP data parsed. Check paths / file structure.")

    df = pd.DataFrame(out_rows)
    df["Recording"] = "LFP"
    df["fs"] = float(fs0)
    return df


@timer("Computing features from empirical data.")
def compute_features_empirical_data(method, df):
    """
    Takes the old-schema df: ID, Group, Epoch, Sensor, Data, Recording, fs
    Adds Features column.
    """
    if method == "catch22":
        features_method = "catch22"
        params = {"normalize": True}

    elif method == "power_spectrum_parameterization_1":
        features_method = "specparam"

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

    else:
        raise ValueError(f"Unknown method: {method}")

    features = ncpi.Features(method=features_method, params=params)
    feats = features.compute_features(df["Data"].to_list())

    out = df.copy()
    if method == "power_spectrum_parameterization_1":
        out["Features"] = [float(np.asarray(d["aperiodic_params"])[1]) for d in feats]
    else:
        out["Features"] = feats

    return out


@timer("Computing predictions from empirical data.")
def compute_predictions_empirical_data(emp_data, inference):
    predictions = inference.predict(np.array(emp_data["Features"].tolist()))
    emp_data = emp_data.copy()
    emp_data["Predictions"] = [list(p) for p in predictions]
    return emp_data


def save_data(emp_data, method):
    os.makedirs(os.path.join("data", method), exist_ok=True)

    # Save the data including predictions of all parameters
    emp_data.to_pickle(os.path.join("data", method, "emp_data_all.pkl"))

    # Replace parameters of recurrent synaptic conductances with the ratio (E/I)_net
    E_I_net = emp_data["Predictions"].apply(lambda x: (x[0] / x[2]) / (x[1] / x[3]))
    others = emp_data["Predictions"].apply(lambda x: x[4:])
    emp_data = emp_data.copy()
    emp_data["Predictions"] = (
        np.concatenate((E_I_net.values.reshape(-1, 1), np.array(others.tolist())), axis=1)
    ).tolist()

    # Save the data including predictions of (E/I)_net
    emp_data.to_pickle(os.path.join("data", method, "emp_data_reduced.pkl"))


if __name__ == "__main__":
    # Download simulation data and ML models
    if zenodo_dw_sim:
        tools.timer("Downloading simulation data and ML models from Zenodo.")(
            tools.download_zenodo_record
        )(zenodo_URL_sim, download_dir=zenodo_dir_sim)

    # Download empirical data
    if zenodo_dw_emp:
        tools.timer("Downloading empirical data from Zenodo.")(
            tools.download_zenodo_record
        )(zenodo_URL_emp, download_dir=zenodo_dir_emp)

    for method in all_methods:
        print(f"\n\n--- Method: {method}")

        X, theta = load_model_features(method, zenodo_dir_sim)
        inference = load_inference_data(method, X, theta, zenodo_dir_sim, ML_model)

        # Empirical data as the *old* schema, but parsed via your parser
        df_emp = load_empirical_data(zenodo_dir_emp)

        # Features
        emp_data = compute_features_empirical_data(method, df_emp)

        # Predictions
        emp_data = compute_predictions_empirical_data(emp_data, inference)

        # Save
        save_data(emp_data, method)
