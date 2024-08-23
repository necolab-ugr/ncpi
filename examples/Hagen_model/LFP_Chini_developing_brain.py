"""
To do.
"""
import importlib
import os
import sys
import pickle
import json

import pandas as pd
import scipy
import numpy as np
import time
import matplotlib.pyplot as plt

# ccpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ccpi

def load_simulation_data(file_path):
    """
    Load simulation data from a file.

    Parameters
    ----------
    file_path : str
        Path to the file containing the simulation data.

    Returns
    -------
    data : ndarray
        Simulation data loaded from the file.
    """

    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f'Loaded file: {file_path}')
    except Exception as e:
        print(f'Error loading file: {file_path}')
        print(e)

    # Check if the data is a dictionary
    if isinstance(data, dict):
        print(f'The file contains a dictionary. {data.keys()}')
        # Print info about each key in the dictionary
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                print(f'Shape of {key}: {data[key].shape}')
            else:
                print(f'{key}: {data[key]}')
    # Check if the data is a ndarray and print its shape
    elif isinstance(data, np.ndarray):
        print(f'Shape of data: {data.shape}')
    print('')

    return data

def load_empirical_data(folder_path):
    '''
    Load empirical data from a folder containing LFP data files.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the LFP data files.

    Returns
    -------
    emp_data : dict
        Dictionary containing the loaded data. The keys are 'LFP', 'fs', and 'age'.
    '''
    file_list = os.listdir(folder_path)
    emp_data = {'LFP': [], 'fs': [], 'age': []}

    for i,file_name in enumerate(file_list):
        print(f'\r Progress: {i+1} of {len(file_list)} files loaded', end='', flush=True)
        structure = scipy.io.loadmat(os.path.join(folder_path, file_name))
        LFP = structure['LFP']['LFP'][0,0]
        sum_LFP = np.sum(LFP, axis=0)  # sum LFP across channels
        fs = structure['LFP']['fs'][0, 0][0, 0]
        age = structure['LFP']['age'][0,0][0,0]

        emp_data['LFP'].append(sum_LFP)
        emp_data['fs'].append(fs)
        emp_data['age'].append(age)

    return emp_data


def compute_features(data, chunk_size=5., method='catch22'):
    '''
    Compute features from the LFP data.

    Parameters
    ----------
    data : dict
        Dictionary containing the LFP data.
    chunk_size : float
        Size of the chunks in seconds.
    method : str
        Method used to compute the features.

    Returns
    -------
    df : DataFrame
        Pandas DataFrame containing the computed features.
    '''

    # Split the data into chunks
    chunk_size = int(chunk_size * data['fs'][0])
    chunked_data = []
    ID = []
    epoch = []
    group = []
    for i in range(len(data['LFP'])):
        for e,j in enumerate(range(0, len(data['LFP'][i]), chunk_size)):
            chunked_data.append(data['LFP'][i][j:j+chunk_size])
            ID.append(i)
            epoch.append(e)
            group.append(data['age'][i])

    # Create the Pandas DataFrame
    df = pd.DataFrame({'ID': ID,
                       'Group': group,
                       'Epoch': epoch,
                       'Sensor': np.zeros(len(ID)), # dummy sensor
                       'Data': chunked_data})
    df.Recording = 'LFP'
    df.fs = data['fs'][0]

    # Compute features
    features = ccpi.Features(method=method)
    df = features.compute_features(df)

    return df


if __name__ == "__main__":
    # Load the configuration file that stores all file paths used in the script
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_features_path']
    emp_data_path = config['LFP_development_data_path']

    # Load parameters of the model (theta) and features from simulation data (X)
    print('\n--- Loading simulation data.')
    start_time = time.time()
    theta = load_simulation_data(os.path.join(sim_file_path, 'sim_theta'))
    X = load_simulation_data(os.path.join(sim_file_path, 'sim_X'))
    end_time = time.time()
    print(f'Samples loaded: {len(theta["data"])}')
    print(f'Done in {end_time - start_time} s')

    # Load empirical data
    print('\n--- Loading empirical data.')
    start_time = time.time()
    emp_data = load_empirical_data(emp_data_path)
    print(f'\nFiles loaded: {len(emp_data["LFP"])}')
    end_time = time.time()
    print(f'Done in {end_time - start_time} s')

    # Compute features from empirical data
    print('\n--- Computing features from empirical data.')
    start_time = time.time()
    emp_data = compute_features(emp_data, chunk_size=5., method='catch22')
    end_time = time.time()
    print(f'Done in {end_time - start_time} s')
