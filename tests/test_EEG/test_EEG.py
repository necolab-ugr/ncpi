import os
import pickle
import time
import pandas as pd
import scipy
import numpy as np
import shutil
import ncpi
from ncpi import tools

# Import the test module
import pytest

# Import EEG_AD.py file of this repository to use its functions
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, "..", "..")) # root directory of this repository
EEG_path = os.path.join(project_root, 'examples', 'EEG_AD')

if EEG_path not in sys.path:
    sys.path.append(EEG_path)

# Import test tools
test_tools_path = os.path.join(project_root, 'tests')  

if test_tools_path not in sys.path:
    sys.path.append(test_tools_path) 

# Now try to import the file EEG_developing_brain.py and the test tools
try:
    import EEG_AD as eeg
    print(f"Successfully imported EEG AD from: {eeg.__file__}")
    import test_tools
    print(f"Successfully imported test_tools from: {test_tools.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")

# Select either raw EEG data or source-reconstructed EEG data. This study used the raw EEG data for all analyses.
raw = True
if raw:
    data_path = os.path.join('DATA', 'empirical_datasets', 'POCTEP_data', 'CLEAN', 'SENSORS') 
else:
    data_path = os.path.join('DATA', 'empirical_datasets', 'POCTEP_data', 'CLEAN', 'SOURCES', 'dSPM', 'DK')


# Methods used to compute the features
# all_methods = ['catch22','power_spectrum_parameterization_1']

# Zenodo URL that contains only the simulation data and ML models needed to test the EEG method
zenodo_URL_test = "https://zenodo.org/api/records/17483670"

# Get the directory where this test file is located
test_dir = os.path.dirname(os.path.abspath(__file__))

# Path to save zenodo test files locally
zenodo_test_files = os.path.join(test_dir, "..", "data") # You have to write here your data path if you already downloaded it

# Paths to zenodo simulation files
zenodo_dir_sim = os.path.join(zenodo_test_files, "zenodo_sim_files") 

# ML model used to compute the predictions
ML_model = 'MLPRegressor'

# Download the data if it's not already downloaded
test_tools.download_data_if_needed(zenodo_URL_test, zenodo_test_files)


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
    emp_data = eeg.compute_predictions_neural_circuit(emp_data, method, ML_model, X, theta, zenodo_dir_sim)
    predictions = emp_data['Predictions'].tolist()

    # Calculate the average for testing
    average = np.nanmean(tuple(np.nanmean(x) for x in predictions))

    return average

def test_EEG():
    """Test average of the data"""
    # Check if the values are within ±0.0001 of the expected values.
    assert(EEG_mean('catch22') == pytest.approx(-1.147342026387889, abs=1e-4))
    assert(EEG_mean('power_spectrum_parameterization_1') == pytest.approx(-1.3593237111519152, abs=1e-4))


# When testing using python test_EEG.py:
# mean_catch = EEG_mean('catch22')
# mean_power = EEG_mean('power_spectrum_parameterization_1')
# print(f'Is \n{mean_catch} equal or similar to \n-1.147342026387889?')
# print(f'Is \n{mean_power} equal or similar to \n-1.3593237111519152?')

# print('If no error messages were showed, the tests completed successfully.')

