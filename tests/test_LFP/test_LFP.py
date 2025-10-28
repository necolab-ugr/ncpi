import os
import shutil
import pickle
import pandas as pd
import scipy
import numpy as np

# Import the python package 'ncpi'
import ncpi

# Import the test module
import pytest

# Import LFP_developing_brain.py file of this repository to use its functions
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, "..", "..")) # root directory of this repository
LFP_path = os.path.join(project_root, 'examples', 'LFP_developing_brain')

if LFP_path not in sys.path:
    sys.path.append(LFP_path)

# Now try to import the file LFP_developing_brain.py
try:
    import LFP_developing_brain as lfp
    print(f"Successfully imported LFP_developing_brain from: {lfp.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")


# Methods used to compute the features
# all_methods = ['catch22','power_spectrum_parameterization_1']

# Get the directory where this test file is located
test_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to zenodo simulation files
zenodo_dir_sim = os.path.join(test_dir, "..", "data", "zenodo_test_files_LFP", "zenodo_sim_files") # Dir of GitHub testing downloaded files (set in tests.yml)

# Paths to zenodo empirical files
zenodo_dir_emp= os.path.join(test_dir, "..", "data", "zenodo_test_files_LFP", "zenodo_emp_files") # Dir of GitHub testing downloaded files (set in tests.yml)

# ML model used to compute the predictions (MLPRegressor, Ridge or NPE)
ML_model = 'MLPRegressor'

def LFP_mean(method):
    """
    Compute only certain data for testing and calculate the average
    """
    print(f'\n\n--- Method: {method}')
    # Load parameters of the model (theta) and features (X) from simulation data
    X, theta = lfp.load_model_features(method, zenodo_dir_sim)

    # Load the Inference objects and add the simulation data
    inference = lfp.load_inference_data(method, X, theta, zenodo_dir_sim, ML_model)

    # Load empirical data
    emp_data = lfp.load_empirical_data(zenodo_dir_emp)
    
    # Compute features from empirical data
    emp_data = lfp.compute_features_empirical_data(method, emp_data)

    # Compute predictions from the empirical data
    emp_data = lfp.compute_predictions_empirical_data(emp_data, inference)
    predictions = emp_data['Predictions'].tolist()

    # Calculate the average for testing
    average = np.nanmean(tuple(np.nanmean(x) for x in predictions))

    return average


def test_LFP():
    """Test average of the data"""
    # Check if the values are within ±0.0001 of the expected values.
    assert(LFP_mean('catch22') == pytest.approx(-0.8690581451462226, abs=1e-4))
    assert(LFP_mean('power_spectrum_parameterization_1') == pytest.approx(-1.2395402659827317, abs=1e-4))


# When testing using python test_LFP.py:
# mean_catch = LFP_mean('catch22')
# mean_power = LFP_mean('power_spectrum_parameterization_1')
# print(f'Is \n{mean_catch} equal or similar to \n-0.8690581451462226?')
# print(f'Is \n{mean_power} equal or similar to \n-1.2395402659827317?')
