import os
import pickle
import shutil
import numpy as np
import pandas as pd
import ncpi

# Path to the folder containing the processed data
sim_file_path = '/DATOS/pablomc/data/Hagen_model_v1'

# Path to the folder where the features will be saved
features_path = '/DATOS/pablomc/data'

# Set to True if features should be computed for the EEG data instead of the CDM data
compute_EEG = False

if __name__ == '__main__':
    # Create the FieldPotential object
    if compute_EEG:
        potential = ncpi.FieldPotential(kernel=False, nyhead=True, MEEG ='EEG')

    for method in ['catch22', 'power_spectrum_parameterization_1']:
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
                        # Check if the features have already been computed for this file
                        if os.path.isfile(os.path.join(features_path, method, 'tmp',
                                                       'sim_X_'+file.split('_')[-1]+'_0')) == False:
                            if len(CDM_data) > 10:
                                all_CDM_data = np.array_split(CDM_data, 10)
                            else:
                                all_CDM_data = [CDM_data]
                        else:
                            print(f'Features have already been computed for CDM data {file.split("_")[-1]}.')
                            continue
                    else:
                        all_CDM_data = [CDM_data]

                    all_features = []
                    for ii, data_chunk_1 in enumerate(all_CDM_data):
                        print(f'Computing features for CDM data chunk {ii+1}/{len(all_CDM_data)}')
                        # Computation of EEGs
                        if compute_EEG:
                            # Check if the features have already been computed for this chunk
                            if os.path.isfile(os.path.join(features_path, method, 'tmp',
                                                           'all_features_' + file.split('_')[-1] + '_' + str(ii))) == False:
                                print(f'Computing EEGs for CDM data chunk {ii+1}/{len(all_CDM_data)}')
                                all_data = np.zeros((len(data_chunk_1), 20, len(data_chunk_1[0])))
                                for k,CDM_data in enumerate(data_chunk_1):
                                    # print(f'EEG {k+1}/{len(data_chunk_1)}', end='\r')
                                    EEGs = potential.compute_MEEG(CDM_data)
                                    all_data[k,:,:] = EEGs
                            else:
                                print(f'Features have already been computed for CDM data chunk {ii+1}/{len(all_CDM_data)}')
                                continue
                        else:
                            all_data = [data_chunk_1]

                        # Get the features for each chunk
                        for jj, data_chunk_2 in enumerate(all_data):
                            # print(f'Chunk {jj+1}/{len(all_data)} of CDM_data {ii+1}/{len(all_CDM_data)}')
                            # Create a fake Pandas DataFrame (only Data and fs are relevant)
                            df = pd.DataFrame({'ID': np.zeros(len(data_chunk_2)),
                                               'Group': np.arange(len(data_chunk_2)),
                                               'Epoch': np.zeros(len(data_chunk_2)),
                                               'Sensor': np.zeros(len(data_chunk_2)),
                                               'Data': list(data_chunk_2)})
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
                            all_features.append(df['Features'].tolist())

                        # Save the features to a tmp file
                        if compute_EEG:
                            pickle.dump(all_features, open(os.path.join(features_path, method, 'tmp',
                                                           'all_features_' + file.split('_')[-1] + '_' + str(ii)), 'wb'))
                            # Kill the process to clear memory and start again
                            if ii < len(all_CDM_data)-1:
                                os._exit(0)

                    if compute_EEG:
                        # Merge the features into a single list
                        all_features = []
                        for ii in range(len(all_CDM_data)):
                            feats = pickle.load(open(os.path.join(features_path, method, 'tmp',
                                                           'all_features_' + file.split('_')[-1] + '_' + str(ii)), 'rb'))
                            all_features.extend(feats)

                        # Remove feature files
                        for ii in range(len(all_CDM_data)):
                            os.remove(os.path.join(features_path, method, 'tmp',
                                                           'all_features_' + file.split('_')[-1] + '_' + str(ii)))

                        # Save the features to a file
                        print('\nSaving EEG features to files.')
                        for i in range(20):
                            elec_data = []
                            for j in range(len(all_features)):
                                elec_data.append(all_features[j][i])

                            pickle.dump(np.array(elec_data),open(os.path.join(features_path, method, 'tmp',
                                                          'sim_X_'+file.split('_')[-1]+'_'+str(i)), 'wb'))
                        # Kill the process
                        os._exit(0)

                    else:
                        df = all_features[0]

                        # Save the features to a file
                        pickle.dump(np.array(df),
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