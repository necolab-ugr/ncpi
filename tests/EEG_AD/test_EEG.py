import os
import sys
import numpy as np
import pytest
import importlib.util
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))

tests_dir = os.path.join(project_root, "tests")
examples_dir = os.path.join(project_root, "examples")
eeg_example_dir = os.path.join(examples_dir, "EEG_AD")

# Ensure EEG example can be imported
if eeg_example_dir not in sys.path:
    sys.path.insert(0, eeg_example_dir)

# ----------------------------
# Import test helpers explicitly (tests/tools.py) as test_tools
# ----------------------------
_test_tools_path = os.path.join(tests_dir, "tools.py")
_spec_test = importlib.util.spec_from_file_location("test_tools", _test_tools_path)
test_tools = importlib.util.module_from_spec(_spec_test)
_spec_test.loader.exec_module(test_tools)

# These are Paths in your repo (PosixPath)
ZENODO_SIM_DIR = test_tools.ZENODO_SIM_DIR
ZENODO_EMP_DIR = test_tools.ZENODO_EMP_DIR

# ----------------------------
# Import shared tools explicitly (examples/tools.py) as shared_tools
# (avoids "tools" name collisions in sys.modules)
# ----------------------------
_examples_tools_path = os.path.join(examples_dir, "tools.py")
_spec_shared = importlib.util.spec_from_file_location("shared_tools", _examples_tools_path)
shared_tools = importlib.util.module_from_spec(_spec_shared)
_spec_shared.loader.exec_module(shared_tools)

if not hasattr(shared_tools, "load_model_features"):
    raise ImportError(f"'load_model_features' not found in {_examples_tools_path}")

# ----------------------------
# Import the example pipeline
# ----------------------------
try:
    import EEG_AD as eeg
except ImportError as e:
    raise ImportError(f"Failed to import EEG_AD from {eeg_example_dir}: {e}")

# ----------------------------
# Config
# ----------------------------
zenodo_URL_test = "https://zenodo.org/api/records/17483670"
ML_model = "MLPRegressor"

# Keep the test small/fast
sensor_list = ["Fp1", "O1"]

# Keep Path objects for test_tools (it expects Path-like API)
zenodo_dir_sim_path = Path(ZENODO_SIM_DIR)
zenodo_dir_emp_path = Path(ZENODO_EMP_DIR)

# But pass strings into shared_tools / example loader (shared_tools validates `str`)
zenodo_dir_sim = str(zenodo_dir_sim_path)

# Select either raw EEG data or source-reconstructed EEG data.
raw = True
if raw:
    data_path = os.path.join(os.sep, "DATOS", "pablomc", "empirical_datasets",
                             "POCTEP_data", "CLEAN", "SENSORS")
else:
    data_path = os.path.join(os.sep, "DATOS", "pablomc", "empirical_datasets",
                             "POCTEP_data", "CLEAN", "SOURCES", "dSPM", "DK")

# Download simulation data + models used by the test (if needed) using PATHS
test_tools.download_data_if_needed(zenodo_URL_test, zenodo_dir_sim_path)


def _method_to_sim_id(method: str) -> str:
    if method == "catch22":
        return "catch22"
    if method == "power_spectrum_parameterization":
        return "power_spectrum_parameterization_1"
    raise ValueError(f"Unknown method: {method}")


def EEG_mean(method: str) -> float:
    """
    Run the new EEG_AD pipeline for one feature method and return a mean value for testing.
    """
    print(f"\n\n--- Method: {method}")
    sim_method_id = _method_to_sim_id(method)

    # 1) Simulation data for this feature method
    X, theta = shared_tools.load_model_features(sim_method_id, zenodo_dir_sim)

    # 2) Empirical EEG -> canonical DataFrame (new loader)
    df_emp = eeg.create_POCTEP_dataframe(data_path=data_path)

    # 3) Feature extraction (shared tools)
    emp_data = shared_tools.compute_features(method, df_emp)

    # 4) Predictions (shared tools)
    emp_data = shared_tools.compute_predictions(
        emp_data,
        data_kind="EEG",
        method=sim_method_id,
        folder=method,          # output folder uses new method name
        ML_model=ML_model,
        X=X,
        theta=theta,
        zenodo_dir_sim=zenodo_dir_sim,
        sensor_list=sensor_list,
    )

    predictions = emp_data["Predictions"].tolist()
    average = np.nanmean(tuple(np.nanmean(x) for x in predictions))
    return float(average)


def test_EEG():
    assert EEG_mean("catch22") == pytest.approx(-0.906795302405384, abs=1e-1)
    assert EEG_mean("power_spectrum_parameterization") == pytest.approx(
        -1.3593237111519152, abs=1e-1
    )
