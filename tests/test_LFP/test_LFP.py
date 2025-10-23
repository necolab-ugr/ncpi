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
# Import ncpi/tools.py as file, not as the Python package. It gets the current directory of test_LFP.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "ncpi"))
import ncpi

# Import functions from LFP_developing_brain.py
LFP_path = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'LFP_developing_brain')
sys.path.insert(0, LFP_path)
import LFP_developing_brain

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
    X, theta = LFP_developing_brain.load_model_features(method)

    # Reshape X to 2D if it's 1D
    # if X.ndim == 1:
    #    X = X.reshape(1, -1)

    # Load the Inference objects and add the simulation data
    inference = LFP_developing_brain.load_inference_data(method, X, theta)

    # Load empirical data
    emp_data = LFP_developing_brain.load_empirical_data()
    
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
    assert(LFP_mean('catch22') == -0.8690581451462226)
    assert(LFP_mean('power_spectrum_parameterization_1') == -1.2395402659827317)