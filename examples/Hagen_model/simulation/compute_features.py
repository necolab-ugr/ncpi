import os
import sys
import json
import pickle
import shutil
import numpy as np
import pandas as pd

# ncpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ncpi

# Set to True if features should be computed for the EEG data instead of the CDM data
compute_EEG = False

if __name__ == '__main__':
    # Path to the folder containing the processed data
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_processed_data_path']

    # Path to the folder where the features will be saved
    features_path = config['simulation_features_path']

    # Create the FieldPotential object
    if compute_EEG:
        potential = ncpi.FieldPotential(nyhead = True, kernel = False)

    for method in ['catch22', 'power_spectrum_parameterization_1','power_spectrum_parameterization_2']:
        # Check if the features have already been computed
        folder = 'EEG' if compute_EEG else ''
        if os.path.isfile(os.path.join(features_path, method, folder, 'sim_X')):
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
                    CDM_data = pickle.load(open(os.path.join(sim_file_path,file),'rb'))

                    # Split CDM data into 10 chunks when computing EEGs to avoid memory issues
                    if compute_EEG:
                        all_CDM_data = np.array_split(CDM_data,10)
                    else:
                        all_CDM_data = [CDM_data]

                    all_features = []
                    for j, CDM_data in enumerate(all_CDM_data):
                        if compute_EEG:
                            all_data = np.zeros((20,len(CDM_data),len(CDM_data[0])))
                            for sample,CDM in enumerate(CDM_data):
                                # print progress
                                print(f'Computing EEGs: {sample}/{len(CDM_data)}', end='\r')
                                EEGs = potential.compute_EEG(CDM)
                                for i, EEG in enumerate(EEGs):
                                    all_data[i,sample,:] = EEG
                                del EEGs
                            all_data = all_data.tolist()
                        else:
                            all_data = [CDM_data]

                        # Get the features for each chunk
                        for i, data_chunk in enumerate(all_data):
                            print(f'Chunk {i+1}/{len(all_data)} of CDM_data {j+1}/{len(all_CDM_data)}')
                            # Create a fake Pandas DataFrame (only Data and fs are relevant)
                            df = pd.DataFrame({'ID': np.zeros(len(data_chunk)),
                                               'Group': np.arange(len(data_chunk)),
                                               'Epoch': np.zeros(len(data_chunk)),
                                               'Sensor': np.zeros(len(data_chunk)),
                                               'Data': list(data_chunk)})
                            df.Recording = 'EEG' if compute_EEG else 'CDM'
                            df.fs = 1000. / 0.625 # samples/s

                            # Compute features
                            if method == 'catch22':
                                features = ncpi.Features(method='catch22')
                            elif method == 'power_spectrum_parameterization_1' or method == 'power_spectrum_parameterization_2':
                                # Parameters of the fooof algorithm
                                fooof_setup_sim = {'peak_threshold': 1.,
                                                   'min_peak_height': 0.,
                                                   'max_n_peaks': 5,
                                                   'peak_width_limits': (10., 50.)}
                                features = ncpi.Features(method='power_spectrum_parameterization',
                                                         params={'fs': df.fs,
                                                                 'fmin': 5.,
                                                                 'fmax': 200.,
                                                                 'fooof_setup': fooof_setup_sim,
                                                                 'r_squared_th':0.9})
                            elif method == 'fEI':
                                features = ncpi.Features(method='fEI',
                                                         params={'fs': df.fs,
                                                                 'fmin': 5.,
                                                                 'fmax': 150.,
                                                                 'fEI_folder': '../../../ncpi/Matlab'})

                            df = features.compute_features(df)

                            # Keep only the aperiodic exponent
                            if method == 'power_spectrum_parameterization_1':
                                df['Features'] = df['Features'].apply(lambda x: x[1])
                            # Keep aperiodic exponent, peak frequency, peak power, knee frequency, and mean power
                            if method == 'power_spectrum_parameterization_2':
                                df['Features'] = df['Features'].apply(lambda x: x[[1, 2, 3, 6, 11]])

                            # Append the feature dataframes to a list
                            all_features.append(df)

                    if compute_EEG:
                        for i in range(20):
                            df = pd.concat([all_features[j*20+i] for j in range(len(all_CDM_data))])

                            # Save the features to a file
                            pickle.dump(np.array(df['Features'].tolist()),
                                        open(os.path.join(features_path, method, 'tmp',
                                                          'sim_X_'+file.split('_')[-1]+'_'+str(i)), 'wb'))

                    else:
                        df = pd.concat(all_features)

                        # Save the features to a file
                        pickle.dump(np.array(df['Features'].tolist()),
                                    open(os.path.join(features_path, method, 'tmp',
                                                      'sim_X_'+file.split('_')[-1]), 'wb'))

                    # clear memory
                    del all_data, CDM_data, df, all_features

                # Theta data
                elif file[:5] == 'theta':
                    theta = pickle.load(open(os.path.join(sim_file_path, file), 'rb'))

                    # Save parameters to a file
                    pickle.dump(theta, open(os.path.join(features_path, method, 'tmp',
                                                         'sim_theta_'+file.split('_')[-1]), 'wb'))

            # Merge the features and parameters into single files
            print('\nMerging features and parameters into single files.')

            ldir = os.listdir(os.path.join(features_path, method, 'tmp'))
            theta_files = [file for file in ldir if file[:9] == 'sim_theta']
            num_files = len(theta_files)

            if compute_EEG:
                # Create EEG folder
                if not os.path.isdir(os.path.join(features_path, method, 'EEG')):
                    os.mkdir(os.path.join(features_path, method, 'EEG'))

                X = [[] for _ in range(20)]
                theta = []
                parameters = []

                for ii in range(int(num_files)):
                    data_theta = pickle.load(open(os.path.join(features_path, method, 'tmp','sim_theta_'+str(ii)), 'rb'))
                    theta.append(data_theta['data'])
                    parameters.append(data_theta['parameters'])

                    for jj in range(20):
                        data_X = pickle.load(open(os.path.join(features_path, method,
                                                               'tmp','sim_X_'+str(ii)+'_'+str(jj)), 'rb'))
                        X[jj].append(data_X)

                # Save features to files
                for jj in range(20):
                    pickle.dump(np.concatenate(X[jj]),
                                open(os.path.join(features_path, method, 'EEG', 'sim_X_'+str(jj)), 'wb'))

            else:
                X = []
                theta = []
                parameters = []

                for ii in range(int(num_files)):
                    data_theta = pickle.load(open(os.path.join(features_path, method, 'tmp','sim_theta_'+str(ii)), 'rb'))
                    theta.append(data_theta['data'])
                    parameters.append(data_theta['parameters'])
                    data_X = pickle.load(open(os.path.join(features_path, method, 'tmp', 'sim_X_' + str(ii)), 'rb'))
                    X.append(data_X)

                # Save features to files
                pickle.dump(np.concatenate(X), open(os.path.join(features_path, method, 'sim_X'), 'wb'))

            # Save the parameters to a file
            if os.path.isfile(os.path.join(features_path, method, 'sim_theta')) == False:
                th = {'data': np.concatenate(theta), 'parameters': parameters[0]}
                pickle.dump(th, open(os.path.join(features_path, method, 'sim_theta'), 'wb'))
                print(f"\nFeatures computed for {len(th['data'])} samples.")

            # Remove the 'tmp' folder
            shutil.rmtree(os.path.join(features_path, method, 'tmp'))