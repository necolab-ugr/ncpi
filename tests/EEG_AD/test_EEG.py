import os
import numpy as np
import pytest
import sys

# Import EEG_AD.py file of this repository to use its functions
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, "..", "..")) # root directory of this repository
EEG_path = os.path.join(project_root, 'examples', 'EEG_AD')

if EEG_path not in sys.path:
    sys.path.append(EEG_path)

# Import test tools
tools_path = os.path.join(project_root, 'tests')

if tools_path not in sys.path:
    sys.path.append(tools_path)

from tools import ZENODO_SIM_DIR, ZENODO_EMP_DIR

# Now try to import the file EEG_developing_brain.py and the test tools
try:
    import EEG_AD as eeg
    print(f"Successfully imported EEG AD from: {eeg.__file__}")
    import tools
    print(f"Successfully imported tools from: {tools.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")

# Select either raw EEG data or source-reconstructed EEG data. This study used the raw EEG data for all analyses.
raw = True
if raw:
    data_path = os.path.join(os.sep, 'DATOS','pablomc', 'empirical_datasets', 'POCTEP_data', 'CLEAN', 'SENSORS') # Specify you data path here
else:
    data_path = os.path.join(os.sep, 'DATOS', 'pablomc', 'empirical_datasets', 'POCTEP_data', 'CLEAN', 'SOURCES', 'dSPM', 'DK') # Specify you data path here


# Methods used to compute the features
# all_methods = ['catch22','power_spectrum_parameterization_1']

# Zenodo URL that contains only the simulation data and ML models needed to test the EEG method
zenodo_URL_test = "https://zenodo.org/api/records/17483670"

# Get the directory where this test file is located
test_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to zenodo simulation files
zenodo_dir_sim = ZENODO_SIM_DIR

# ML model used to compute the predictions
ML_model = 'MLPRegressor'

# Download the data if it's not already downloaded
tools.download_data_if_needed(zenodo_URL_test, zenodo_dir_sim)


def EEG_mean(method):
    """
    Compute only certain data for testing and calculate the average
    """
    print(f'\n\n--- Method: {method}')

    # Load parameters of the model (theta) and features (X) from simulation data
    X, theta = eeg.load_model_features(method, zenodo_dir_sim)

    # Load empirical data and create DataFrame
    df = eeg.create_POCTEP_dataframe(data_path=data_path)

    # Feature extraction
    emp_data = eeg.feature_extraction(method, df)

    # Predictions of parameters of the neural circuit
    emp_data = eeg.compute_predictions_neural_circuit(emp_data, method, ML_model, X, theta, zenodo_dir_sim,
                                                      sensor_list=['Fp1', 'O1'])
    predictions = emp_data['Predictions'].tolist()

    # Calculate the average for testing
    average = np.nanmean(tuple(np.nanmean(x) for x in predictions))

    return average

def test_EEG():
    """Test average of the data"""
    # Check if the values are within ±0.1 of the expected values.
    assert(EEG_mean('catch22') == pytest.approx(-0.906795302405384, abs=1e-1))
    assert(EEG_mean('power_spectrum_parameterization_1') == pytest.approx(-1.3593237111519152, abs=1e-1))
