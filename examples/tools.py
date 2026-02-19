"""
Shared utilities used by the example empirical-data pipelines.
"""

from __future__ import annotations

import os
import pickle
import shutil
from typing import Any, Dict, List, Tuple, Iterable, Optional, Literal
import numpy as np
import pandas as pd
import ncpi
from ncpi.tools import timer

# ---------
# Helpers
# ---------

def _features_to_2d_numpy(feat_list):
    """
    Convert a list-like of features into a 2D numpy array:
      - list of scalars -> (n, 1)
      - list of vectors -> (n, d) via np.stack
      - already 2D -> unchanged
    """
    feats = list(feat_list)

    if len(feats) == 0:
        return np.empty((0, 0), dtype=float)

    # If entries are arrays/lists (vector features), stack them
    first = feats[0]
    if isinstance(first, (list, tuple, np.ndarray)) and not np.isscalar(first):
        try:
            X = np.stack([np.asarray(f) for f in feats], axis=0)
        except Exception:
            # fallback: let numpy make an object array (Inference may reject later)
            X = np.asarray(feats, dtype=object)
    else:
        # scalar features
        X = np.asarray(feats)

    # Critical fix: if 1D, interpret as many 1-feature samples
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X


# ---------------------------
# Simulation data loading
# ---------------------------

@timer("Loading simulation data.")
def load_model_features(method: str, zenodo_dir_sim: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model parameters (theta) and features (X) from simulation data.

    Parameters
    ----------
    method:
        Feature method folder name (e.g., "catch22", "power_spectrum_parameterization_1")
    zenodo_dir_sim:
        Local path to the Zenodo simulation record directory.

    Returns
    -------
    X:
        Simulation features array/object.
    theta:
        Dict-like object containing simulation parameters. Must include key "data".
    """

    # --- Input validation ---
    if not isinstance(method, str) or not method:
        raise TypeError("`method` must be a non-empty string.")

    if not isinstance(zenodo_dir_sim, str) or not zenodo_dir_sim:
        raise TypeError("`zenodo_dir_sim` must be a non-empty string.")

    if not os.path.isdir(zenodo_dir_sim):
        raise FileNotFoundError(f"Zenodo directory not found: {zenodo_dir_sim}")

    base_dir = os.path.join(zenodo_dir_sim, "data", method)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Method directory not found: {base_dir}")

    theta_path = os.path.join(base_dir, "sim_theta")
    x_path = os.path.join(base_dir, "sim_X")

    if not os.path.isfile(theta_path):
        raise FileNotFoundError(f"Missing theta file: {theta_path}")

    if not os.path.isfile(x_path):
        raise FileNotFoundError(f"Missing X file: {x_path}")

    # --- Load files ---
    try:
        with open(theta_path, "rb") as f:
            theta = pickle.load(f)
    except Exception as exc:
        raise IOError(f"Failed to load theta from {theta_path}") from exc

    try:
        with open(x_path, "rb") as f:
            X = pickle.load(f)
    except Exception as exc:
        raise IOError(f"Failed to load X from {x_path}") from exc

    # --- Minimal output validation ---
    if not isinstance(theta, dict):
        raise TypeError("Loaded `theta` must be a dict-like object.")

    # --- Verbose printing behavior ---
    print("theta:")
    for key, value in theta.items():
        shape = getattr(value, "shape", None)
        if isinstance(value, np.ndarray) and shape is not None:
            print(f"--Shape of {key}: {shape}")
        else:
            print(f"--{key}: {value}")

    x_shape = getattr(X, "shape", None)
    print(f"Shape of X: {x_shape if x_shape is not None else '<unknown>'}")

    return X, theta


# ---------------------------
# Feature computation
# ---------------------------

def _specparam_params(fs: float) -> Dict[str, Any]:
    """Build the specparam parameter dict."""
    
    # --- Input validation ---
    try:
        fs = float(fs)
    except (TypeError, ValueError) as exc:
        raise TypeError("`fs` must be convertible to float.") from exc

    if fs <= 0:
        raise ValueError(f"`fs` must be positive, got {fs}.")

    specparam_setup_emp = {
        "peak_threshold": 1.0,
        "min_peak_height": 0.0,
        "max_n_peaks": 5,
        "peak_width_limits": (10.0, 50.0),
    }

    return {
        "fs": fs,
        "freq_range": (5.0, 45.0),
        "specparam_model": dict(specparam_setup_emp),
        "metric_thresholds": {"gof_rsquared": 0.9},
        "metric_policy": "reject",
    }


@timer("Feature extraction")
def compute_features(method: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features and append a ``Features`` column to a canonical-schema dataframe.

    - method == "catch22" -> stores the full catch22 feature vector per epoch.
    - method == "power_spectrum_parameterization" -> stores the aperiodic exponent
      (the second value of aperiodic_params) per epoch.
    """

    # --- Minimal input validation ---
    if not isinstance(method, str) or not method:
        raise TypeError("`method` must be a non-empty string.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if "data" not in df.columns:
        raise ValueError("Expected `df` to contain a 'data' column with 1D samples.")

    # Normalize method name
    method_norm = method.strip()

    samples = df["data"].to_list()
    out = df.copy()

    if method_norm == "catch22":
        features = ncpi.Features(method="catch22", params={"normalize": True})
        out["Features"] = features.compute_features(samples)
        return out

    if method_norm == "power_spectrum_parameterization":
        # --- Extract sampling rate from DataFrame ---
        if "fs" not in df.columns:
            raise ValueError("Expected `df` to contain an 'fs' column for specparam features.")

        try:
            fs = float(df["fs"].iloc[0])
        except (TypeError, ValueError) as exc:
            raise ValueError("`df['fs']` must be convertible to float.") from exc

        if fs <= 0:
            raise ValueError(f"Sampling rate `fs` must be positive, got {fs}.")

        params = _specparam_params(fs)
        features = ncpi.Features(method="specparam", params=params)

        feats = features.compute_features(samples)

        # Store exponent only.
        def _extract_exponent(d: Any) -> float:
            if not isinstance(d, dict) or "aperiodic_params" not in d:
                raise ValueError(
                    "Specparam feature output must be a dict containing 'aperiodic_params'."
                )
            ap = np.asarray(d["aperiodic_params"])
            if ap.ndim != 1 or ap.size < 2:
                raise ValueError(
                    f"'aperiodic_params' must be a 1D array with >=2 values, got shape {ap.shape}."
                )
            return float(ap[1])

        out["Features"] = [_extract_exponent(d) for d in feats]
        return out

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Inference model + predictions
# -----------------------------

_EEG_DEFAULT_SENSORS: List[str] = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1",
    "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]

def _ensure_local_data_dir(method: str) -> str:
    """Create local output dirs and return path to the method subdir."""
    os.makedirs("data", exist_ok=True)
    method_dir = os.path.join("data", method)
    os.makedirs(method_dir, exist_ok=True)
    return method_dir


def _LFP_ml_model_folder(ML_model: str) -> str:
    """Map ML_model name to Zenodo folder name for LFP assets."""
    mapping = {
        "MLPRegressor": "MLP",
        "Ridge": "Ridge",
        "NPE": "SBI",
    }
    try:
        return mapping[ML_model]
    except KeyError as exc:
        raise ValueError(f"Unknown ML_model: {ML_model}") from exc


def _copy_assets_LFP(*, method: str, ML_model: str, zenodo_dir_sim: str) -> None:
    """Copy LFP model/scaler (and density_estimator for NPE) to local data/."""
    folder = _LFP_ml_model_folder(ML_model)
    base = os.path.join(zenodo_dir_sim, "ML_models", "4_param", folder, method)

    scaler_src = os.path.join(base, "scaler")
    model_src = os.path.join(base, "model")

    if not os.path.isfile(scaler_src):
        raise FileNotFoundError(f"Missing scaler file: {scaler_src}")
    if not os.path.isfile(model_src):
        raise FileNotFoundError(f"Missing model file: {model_src}")

    shutil.copy(scaler_src, os.path.join("data", "scaler.pkl"))
    shutil.copy(model_src, os.path.join("data", "model.pkl"))

    if ML_model == "NPE":
        de_src = os.path.join(base, "density_estimator")
        if not os.path.isfile(de_src):
            raise FileNotFoundError(f"Missing density_estimator file: {de_src}")
        shutil.copy(de_src, os.path.join("data", "density_estimator.pkl"))


def _copy_assets_EEG(*, method: str, sensor: str, zenodo_dir_sim: str) -> None:
    """Copy EEG model/scaler for a specific sensor to local data/."""
    base = os.path.join(zenodo_dir_sim, "ML_models", "EEG", sensor, method)

    scaler_src = os.path.join(base, "scaler")
    model_src = os.path.join(base, "model")

    if not os.path.isfile(scaler_src):
        raise FileNotFoundError(f"Missing scaler file: {scaler_src}")
    if not os.path.isfile(model_src):
        raise FileNotFoundError(f"Missing model file: {model_src}")

    shutil.copy(scaler_src, os.path.join("data", "scaler.pkl"))
    shutil.copy(model_src, os.path.join("data", "model.pkl"))


def compute_predictions(
    emp_data: pd.DataFrame,
    *,
    data_kind: Literal["EEG", "LFP"],
    method: str,
    folder: str,
    ML_model: str,
    X: Any,
    theta: Dict[str, Any],
    zenodo_dir_sim: str,
    sensor_list: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Unified prediction entry point for EEG and LFP.

    Parameters
    ----------
    emp_data:
        Must contain a 'Features' column. For EEG must also contain 'sensor'.
    data_kind:
        "EEG" or "LFP".
    method:
        Feature method folder name used to locate trained assets under zenodo_dir_sim.
    folder:
        Folder to save results.
    ML_model:
        ncpi inference model name (e.g., "Ridge", "MLPRegressor", "NPE").
        For EEG it configures ncpi.Inference(model=...), but asset paths are sensor/method-based.
        For LFP it also selects the correct asset subfolder.
    X, theta:
        Simulation feature matrix and parameter dict. theta must contain key "data".
    zenodo_dir_sim:
        Local path to the Zenodo simulation record directory.
    sensor_list:
        EEG only. If None, uses a default 19-channel list.

    Returns
    -------
    DataFrame copy of emp_data with a 'Predictions' column appended/overwritten.
    """
    # --- Minimal input validation ---
    if not isinstance(emp_data, pd.DataFrame):
        raise TypeError("`emp_data` must be a pandas DataFrame.")
    if not isinstance(method, str) or not method:
        raise TypeError("`method` must be a non-empty string.")
    if not isinstance(ML_model, str) or not ML_model:
        raise TypeError("`ML_model` must be a non-empty string.")
    if not isinstance(zenodo_dir_sim, str) or not zenodo_dir_sim:
        raise TypeError("`zenodo_dir_sim` must be a non-empty string.")
    if "Features" not in emp_data.columns:
        raise ValueError("Expected `emp_data` to contain a 'Features' column.")

    if not isinstance(theta, dict) or "data" not in theta:
        raise ValueError("`theta` must be a dict containing key 'data'.")

    if data_kind not in ("EEG", "LFP"):
        raise ValueError(f"`data_kind` must be 'EEG' or 'LFP', got {data_kind!r}.")

    # Local output dirs
    _ensure_local_data_dir(folder)

    # Create inference object and attach simulation data
    inference = ncpi.Inference(model=ML_model)
    inference.add_simulation_data(X, theta["data"])

    out = emp_data.copy()
    out["Predictions"] = np.nan  # initialization behavior

    # --- LFP branch ---
    if data_kind == "LFP":
        _copy_assets_LFP(method=method, ML_model=ML_model, zenodo_dir_sim=zenodo_dir_sim)

        X_emp = _features_to_2d_numpy(out["Features"])
        predictions = inference.predict(X_emp, scaler=True)
        out["Predictions"] = [list(p) if isinstance(p, (list, np.ndarray)) else [p] for p in predictions]
        return out

    # --- EEG branch ---
    if "sensor" not in out.columns:
        raise ValueError("EEG mode requires `emp_data` to contain a 'sensor' column.")

    sensors = list(sensor_list) if sensor_list is not None else list(_EEG_DEFAULT_SENSORS)

    for s, sensor in enumerate(sensors):
        print(f"--- Sensor: {sensor}")

        # Copy sensor-specific assets locally (overwrites each loop, as before)
        _copy_assets_EEG(method=method, sensor=sensor, zenodo_dir_sim=zenodo_dir_sim)

        # Filtering semantics: match by sensor name or numeric index
        sensor_df = out[out["sensor"].isin([sensor, s])].copy()
        if sensor_df.empty:
            continue

        X_emp = _features_to_2d_numpy(sensor_df["Features"])
        predictions = inference.predict(X_emp, scaler=True)
        sensor_df["Predictions"] = [list(p) if isinstance(p, (list, np.ndarray)) else [p] for p in predictions]

        # Write back into out
        out.loc[sensor_df.index, "Predictions"] = sensor_df["Predictions"]

    return out

# ---------------------------
# Saving results
# ---------------------------

def ensure_method_dir(method: str, base_dir: str = "data") -> str:
    """Create and return the directory where results for `method` are stored."""
    out_dir = os.path.join(base_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_data(emp_data: pd.DataFrame, method: str, base_dir: str = "data") -> None:
    """
    Save predictions in two forms:

    - emp_data_all.pkl: raw per-epoch predicted parameters.
    - emp_data_reduced.pkl: first element replaced by (E/I)_net and remaining parameters kept.

    This matches the original behavior in both scripts.
    """
    out_dir = ensure_method_dir(method, base_dir=base_dir)

    emp_data.to_pickle(os.path.join(out_dir, "emp_data_all.pkl"))

    E_I_net = emp_data["Predictions"].apply(lambda x: (x[0] / x[2]) / (x[1] / x[3]))
    others = emp_data["Predictions"].apply(lambda x: x[4:])

    reduced = emp_data.copy()
    reduced["Predictions"] = (
        np.concatenate((E_I_net.values.reshape(-1, 1), np.array(others.tolist())), axis=1)
    ).tolist()

    reduced.to_pickle(os.path.join(out_dir, "emp_data_reduced.pkl"))
