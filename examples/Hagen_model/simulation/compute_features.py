import os
import sys
import json
import pickle
import shutil
import numpy as np
import pandas as pd

# ccpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ccpi

if __name__ == '__main__':
    # Path to the folder containing the processed data
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_processed_data_path']

    # Path to the folder where the features will be saved
    features_path = config['simulation_features_path']

    for method in ['catch22', 'power_spectrum_parameterization', 'fEI']:
        # Check if the features have already been computed
        if os.path.isfile(os.path.join(features_path, method, 'sim_X')):
            print(f'Features have already been computed for the method {method}.')
        else:
            print(f'Computing features for the method {method}.')
            # Create folders
            if not os.path.isdir(features_path):
                os.mkdir(features_path)
            if not os.path.isdir(os.path.join(features_path, method)):
                os.mkdir(os.path.join(features_path, method))
            if not os.path.isdir(os.path.join(features_path, method, 'tmp')):
                os.mkdir(os.path.join(features_path, method, 'tmp'))

            # Process files in the folder
            ldir = os.listdir(sim_file_path)
            for file in ldir:
                print(file)

                # CDM data
                if file[:3] == 'CDM':
                    CDM = pickle.load(open(os.path.join(sim_file_path,file),'rb'))

                    # Create a fake Pandas DataFrame (only Data and fs are relevant)
                    df = pd.DataFrame({'ID': np.zeros(len(CDM)),
                                       'Group': np.arange(len(CDM)),
                                       'Epoch': np.zeros(len(CDM)),
                                       'Sensor': np.zeros(len(CDM)),
                                       'Data': list(CDM)})
                    df.Recording = 'CDM'
                    df.fs = 1000. / 0.625 # samples/s

                    # Compute features
                    if method == 'catch22':
                        features = ccpi.Features(method='catch22')
                    elif method == 'power_spectrum_parameterization':
                        # Parameters of the fooof algorithm
                        fooof_setup_sim = {'peak_threshold': 1.,
                                           'min_peak_height': 0.,
                                           'max_n_peaks': 2,
                                           'peak_width_limits': (10., 50.)}
                        features = ccpi.Features(method='power_spectrum_parameterization',
                                                 params={'fs': df.fs,
                                                         'fmin': 5.,
                                                         'fmax': 200.,
                                                         'fooof_setup': fooof_setup_sim,
                                                         'r_squared_th':0.9})
                    elif method == 'fEI':
                        features = ccpi.Features(method='fEI',
                                                 params={'fs': df.fs,
                                                         'fmin': 8.,
                                                         'fmax': 30.,
                                                         'fEI_folder': '../../../ccpi/Matlab'})

                    df = features.compute_features(df)

                    # Keep only the aperiodic exponent
                    if method == 'power_spectrum_parameterization':
                        df['Features'] = df['Features'].apply(lambda x: x[1])

                    # Save the features to a file
                    pickle.dump(np.array(df['Features'].tolist()),
                                open(os.path.join(features_path, method, 'tmp',
                                                  'sim_X_'+file.split('_')[-1]), 'wb'))

                    # clear memory
                    del CDM, df, features

                # Theta data
                elif file[:5] == 'theta':
                    theta = pickle.load(open(os.path.join(sim_file_path, file), 'rb'))

                    # Save parameters to a file
                    pickle.dump(theta, open(os.path.join(features_path, method, 'tmp',
                                                         'sim_theta_'+file.split('_')[-1]), 'wb'))

            # Merge the features and parameters into single files
            print('\nMerging features and parameters into single files.')
            X = []
            theta = []
            parameters = []

            ldir = os.listdir(os.path.join(features_path, method, 'tmp'))
            for file in ldir:
                data = pickle.load(open(os.path.join(features_path, method, 'tmp', file), 'rb'))
                if file[:5] == 'sim_X':
                    X.append(data)
                elif file[:9] == 'sim_theta':
                    theta.append(data['data'])
                    parameters.append(data['parameters'])

            # Save the features and parameters to files
            pickle.dump(np.concatenate(X), open(os.path.join(features_path, method, 'sim_X'), 'wb'))
            th = {'data': np.concatenate(theta), 'parameters': parameters[0]}
            pickle.dump(th, open(os.path.join(features_path, method, 'sim_theta'), 'wb'))
            print(f'\nFeatures computed for {len(np.concatenate(X))} samples.')

            # Remove the 'tmp' folder
            shutil.rmtree(os.path.join(features_path, method, 'tmp'))
