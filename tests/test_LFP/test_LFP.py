import os
import shutil
import pickle
import pandas as pd
import scipy
import numpy as np
import time

# Test python library
# import ncpi
# from ncpi import tools

# Test library files of these folders (comment # import ncpi)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ncpi"))
from ncpi import tools

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
    print('\n--- Loading simulation data.')
    start_time = time.time()

    try:
        with open(os.path.join(zenodo_dir_sim, 'data', method, 'sim_theta'), 'rb') as file:
            theta = pickle.load(file)
        with open(os.path.join(zenodo_dir_sim, 'data', method, 'sim_X'), 'rb') as file:
            X = pickle.load(file)
    except Exception as e:
        print(f"Error loading simulation data: {e}")

    # Print info
    print('theta:')
    for key, value in theta.items():
        if isinstance(value, np.ndarray):
            print(f'--Shape of {key}: {value.shape}')
        else:
            print(f'--{key}: {value}')
    print(f'Shape of X: {X.shape}')

    end_time = time.time()
    print(f"Done in {(end_time - start_time):.2f} sec.")

    # Load the Inference objects and add the simulation data
    print('\n--- Loading the inverse model.')
    start_time = time.time()

    # Create inference object
    inference = ncpi.Inference(model=ML_model)
    # Not sure if this is really needed
    inference.add_simulation_data(X, theta['data'])

    # Create folder to save results
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(os.path.join('data', method)):
        os.makedirs(os.path.join('data', method))

    # Transfer model and scaler to the data folder
    if ML_model == 'MLPRegressor':
        folder = 'MLP'
    elif ML_model == 'Ridge':
        folder = 'Ridge'
    elif ML_model == 'NPE':
        folder = 'SBI'
        
    shutil.copy(
        os.path.join(zenodo_dir_sim, 'ML_models/4_param', folder, method, 'scaler'),
        os.path.join('data', 'scaler.pkl')
    )

    shutil.copy(
        os.path.join(zenodo_dir_sim, 'ML_models/4_param', folder, method, 'model'),
        os.path.join('data', 'model.pkl')
    )

    if ML_model == 'NPE':
        shutil.copy(
            os.path.join(zenodo_dir_sim, 'ML_models/4_param', folder, method,
                            'density_estimator'),
            os.path.join('data', 'density_estimator.pkl')
        )

    end_time = time.time()
    print(f'Done in {(end_time - start_time):.2f} sec')

    # Load empirical data
    print('\n--- Loading LFP data.')
    start_time = time.time()

    file_list = os.listdir(os.path.join(zenodo_dir_emp, 'development_EI_decorrelation/baseline/LFP'))
    emp_data = {'LFP': [], 'fs': [], 'age': []}

    for i,file_name in enumerate(file_list):
        print(f'\r Progress: {i+1} of {len(file_list)} files loaded', end='', flush=True)
        structure = scipy.io.loadmat(os.path.join(os.path.join(zenodo_dir_emp,
                                                                'development_EI_decorrelation/baseline/LFP'),
                                                    file_name))
        LFP = structure['LFP']['LFP'][0,0]
        sum_LFP = np.sum(LFP, axis=0)  # sum LFP across channels
        fs = structure['LFP']['fs'][0, 0][0, 0]
        age = structure['LFP']['age'][0,0][0,0]

        emp_data['LFP'].append(sum_LFP)
        emp_data['fs'].append(fs)
        emp_data['age'].append(age)

    end_time = time.time()
    print(f'\nFiles loaded: {len(emp_data["LFP"])}')
    print(f'Done in {(end_time - start_time):.2f} sec')

    # Compute features from empirical data
    print('\n--- Computing features from empirical data.')
    start_time = time.time()

    # Epoch length in seconds
    chunk_size = 5

    # Parameters of the feature extraction method
    if method == 'catch22':
        params = None
    elif method == 'power_spectrum_parameterization_1':
        fooof_setup_emp = {'peak_threshold': 1.,
                            'min_peak_height': 0.,
                            'max_n_peaks': 5,
                            'peak_width_limits': (10., 50.)}
        params={'fs': emp_data['fs'][0],
                'fmin': 5.,
                'fmax': 45.,
                'fooof_setup': fooof_setup_emp,
                'r_squared_th':0.9}

    # Split the data into chunks (epochs)
    chunk_size = int(chunk_size * emp_data['fs'][0])
    chunked_data = []
    ID = []
    epoch = []
    group = []
    for i in range(len(emp_data['LFP'])):
        for e,j in enumerate(range(0, len(emp_data['LFP'][i]), chunk_size)):
            if len(emp_data['LFP'][i][j:j+chunk_size]) == chunk_size:
                chunked_data.append(emp_data['LFP'][i][j:j+chunk_size])
                ID.append(i)
                epoch.append(e)
                group.append(emp_data['age'][i])

    # Create the Pandas DataFrame
    df = pd.DataFrame({'ID': ID,
                        'Group': group,
                        'Epoch': epoch,
                        'Sensor': np.zeros(len(ID)), # dummy sensor
                        'Data': chunked_data})
    df.Recording = 'LFP'
    df.fs = emp_data['fs'][0]

    # Compute features
    features = ncpi.Features(method=method if method == 'catch22' else 'power_spectrum_parameterization',
                                params=params)
    emp_data = features.compute_features(df)

    # Keep only the aperiodic exponent (1/f slope)
    if method == 'power_spectrum_parameterization_1':
        emp_data['Features'] = emp_data['Features'].apply(lambda x: x[1])

    end_time = time.time()
    print(f'Done in {(end_time - start_time):.2f} sec')

    # Compute predictions from the empirical data
    print('\n--- Computing predictions from empirical data.')
    start_time = time.time()
    # Predict the parameters from the features of the empirical data
    predictions = inference.predict(np.array(emp_data['Features'].tolist()))

    # Calculate the average for testing
    average = np.nanmean(tuple(np.nanmean(x) for x in predictions))

    return average


def test_LFP():
    """Test average of the data"""
    assert(LFP_mean('catch22') == -0.8690581451462226)
    assert(LFP_mean('power_spectrum_parameterization_1') == -1.2395402659827317)
