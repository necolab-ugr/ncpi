import os
import pickle
import time
import pandas as pd
import scipy
import numpy as np
import shutil

# Test python library: ncpi
#import ncpi
#from ncpi import tools

# Test library files of these folders (comment # import ncpi)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ncpi"))
from ncpi import tools


data_path = '../data/SENSORS' # private files
# Paths to zenodo simulation files
zenodo_dir_sim = "../data/zenodo_sim_files"

# Methods used to compute the features
all_methods = ['catch22','power_spectrum_parameterization_1']

# ML model used to compute the predictions
ML_model = 'MLPRegressor'

def create_POCTEP_dataframe(data_path):
    '''
    Load the POCTEP dataset and create a DataFrame with the data.

    Parameters
    ----------
    data_path: str
        Path to the directory containing the POCTEP data files.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the POCTEP data.
    '''


    # List files in the directory
    ldir = os.listdir(data_path)

    ID = []
    group = []
    epoch = []
    sensor = []
    EEG = []

    for pt,file in enumerate(ldir):
        print(f'\rProcessing {file} - {pt + 1 }/{len(ldir)}', end="", flush=True)

        # load data
        data = scipy.io.loadmat(data_path + '/' + file)['data']
        signal = data['signal'][0, 0]

        # get sampling frequency
        fs = data['cfg'][0, 0]['fs'][0, 0][0, 0]

        # Electrodes (raw data)/ regions (if source data)
        regions = np.arange(signal.shape[1])

        # get channels
        ch_names = data['cfg'][0, 0]['channels'][0, 0][0]
        ch_names = [ch_names[ll][0] for ll in range(len(ch_names))]

        # 5-second epochs
        epochs = np.arange(0, signal.shape[0], int(fs * 5))

        for i in range(len(epochs) - 1):
            ep = signal[epochs[i]:epochs[i + 1], :]
            # z-score normalization
            ep = (ep - np.mean(ep, axis=0)) / np.std(ep, axis=0)

            # Append data
            for rg in regions:
                ID.append(pt)
                group.append(file.split('_')[0])
                epoch.append(i)
                sensor.append(ch_names[rg])
                EEG.append(ep[:, rg])

    # Create the Pandas DataFrame
    df = pd.DataFrame({'ID': ID,
                       'Group': group,
                       'Epoch': epoch,
                       'Sensor': sensor,
                       'Data': EEG})
    df['Recording'] = 'EEG'
    df['fs'] = fs

    return df

def EEG_mean(method):
    """
    Compute data for testing and calculate the average
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

    # Load empirical data and create DataFrame
    df = create_POCTEP_dataframe(data_path=data_path)

    ##########################
    #   FEATURE EXTRACTION   #
    ##########################

    start_time = time.time()

    # Parameters of the feature extraction method
    if method == 'catch22':
        params = None
    elif method == 'power_spectrum_parameterization_1':
        fooof_setup_emp = {'peak_threshold': 1.,
                            'min_peak_height': 0.,
                            'max_n_peaks': 5,
                            'peak_width_limits': (10., 50.)}
        params={'fs': df['fs'][0],
                'fmin': 5.,
                'fmax': 45.,
                'fooof_setup': fooof_setup_emp,
                'r_squared_th':0.9}
        
    features = ncpi.Features(method=method if method == 'catch22' else 'power_spectrum_parameterization',
                                params=params)
    emp_data = features.compute_features(df)

    # Keep only the aperiodic exponent (1/f slope)
    if method == 'power_spectrum_parameterization_1':
        emp_data['Features'] = emp_data['Features'].apply(lambda x: x[1])

    end_time = time.time()
    print(f'Done in {(end_time - start_time):.2f} sec')

    #######################################################
    #   PREDICTIONS OF PARAMETERS OF THE NEURAL CIRCUIT   #
    #######################################################

    print('\nComputing predictions...')
    start_time = time.time()

    # Add "Predictions" column to later store the parameters infered
    emp_data['Predictions'] = np.nan

    # List of sensors
    sensor_list = [
        'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1',
        'O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'
    ]

    # Create folder to save results
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(os.path.join('data', method)):
        os.makedirs(os.path.join('data', method))

    # Create inference object
    inference = ncpi.Inference(model=ML_model)
    # Not sure if this is really needed
    inference.add_simulation_data(X, theta['data'])

    for s, sensor in enumerate(sensor_list):
        print(f'--- Sensor: {sensor}')

        shutil.copy(
            os.path.join(zenodo_dir_sim, 'ML_models/EEG', sensor, method, 'model'),
            os.path.join('data', 'model.pkl')
        )

        shutil.copy(
            os.path.join(zenodo_dir_sim, 'ML_models/EEG', sensor, method, 'scaler'),
            os.path.join('data', 'scaler.pkl')
        )

        sensor_df = emp_data[emp_data['Sensor'].isin([sensor, s])]

        predictions = inference.predict(np.array(sensor_df['Features'].to_list()))

        sensor_df['Predictions'] = [list(pred) for pred in predictions]
        emp_data.update(sensor_df['Predictions'])
    print('---------------------------------------------EMP DATA:\n', emp_data)
    end_time = time.time()
    print(f'Done in {(end_time - start_time):.2f} sec')

    # Calculate the average for testing
    average = emp_data['Predictions'].apply(np.mean).mean()

    return average


def test_EEG():
    """Test average of the data"""
    assert(EEG_mean('catch22') == -1.043846650253303)
    assert(EEG_mean_mean('power_spectrum_parameterization_1') == -1.3940573275196155)

