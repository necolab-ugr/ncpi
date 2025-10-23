import os
import shutil
import pickle
import pandas as pd
import scipy
import numpy as np


# Test python package (comment the import of repository files)
# import ncpi

# Test files of this repository (comment # import ncpi)
import sys

# Debug import
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))

# Add project root to Python path (for ncpi package files)
if project_root not in sys.path:
    sys.path.insert(0, project_root)




# Import ncpi/tools.py as file, not as the Python package. It gets the current directory of test_LFP.py
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "ncpi"))
# sys.path.insert(0, os.path.join(project_root, 'ncpi'))
# import ncpi

# Import functions from LFP_developing_brain.py
LFP_path = os.path.join(project_root, 'examples', 'LFP_developing_brain')
if LFP_path not in sys.path:
    sys.path.insert(0, LFP_path)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")
print(f"LFP dir: {LFP_path}")


# Now try to import
try:
    import ncpi
    print("Successfully imported ncpi from: {ncpi.__file__}")

    import LFP_developing_brain
    print(f"Successfully imported LFP_developing_brain from: {LFP_developing_brain.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")
    # List contents to debug
    ncpi_dir = os.path.join(project_root, "ncpi")
    if os.path.exists(ncpi_dir):
        print(f"Contents of ncpi directory {ncpi_dir}: {os.listdir(ncpi_dir)}")
    else:
        print(f"ncpi directory not found at: {ncpi_dir}")


import pytest

# Methods used to compute the features
# all_methods = ['catch22','power_spectrum_parameterization_1']

# Get the directory where this test file is located
test_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to zenodo simulation files
zenodo_dir_sim = os.path.join(test_dir, "..", "data", "zenodo_sim_files") 

# Paths to zenodo empirical files
zenodo_dir_emp= os.path.join(test_dir, "..", "data", "zenodo_emp_files")

# ML model used to compute the predictions
ML_model = 'MLPRegressor'


def LFP_mean(method):
    """
    Compute only certain data for testing and calculate the average
    """

    print(f'\n\n--- Method: {method}')
    # Load parameters of the model (theta) and features (X) from simulation data
    X, theta = LFP_developing_brain.load_model_features(method, zenodo_dir_sim)

    # Reshape X to 2D if it's 1D
    # if X.ndim == 1:
    #    X = X.reshape(1, -1)

    # Load the Inference objects and add the simulation data
    inference = LFP_developing_brain.load_inference_data(method, X, theta, zenodo_dir_sim)

    # Load empirical data
    emp_data = LFP_developing_brain.load_empirical_data(zenodo_dir_emp)
    
    # Compute features from empirical data
    emp_data = LFP_developing_brain.compute_features_empirical_data(method, emp_data)

    # Compute predictions from the empirical data
    emp_data = LFP_developing_brain.compute_predictions_empirical_data(emp_data, inference)
    predictions = emp_data['Predictions'].tolist()

    # Calculate the average for testing
    average = np.nanmean(tuple(np.nanmean(x) for x in predictions))

    return average


def test_LFP():
    """Test average of the data"""
    # Check if the values are within ±0.0001 of the expected values.
    assert(LFP_mean('catch22') == pytest.approx(-0.8690581451462226, abs=1e-4))
    assert(LFP_mean('power_spectrum_parameterization_1') == pytest.approx(-1.2395402659827317, abs=1e-4))