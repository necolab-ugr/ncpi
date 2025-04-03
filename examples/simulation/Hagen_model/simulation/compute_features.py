import os
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import mne
from mne.filter import next_fast_len
from scipy.signal import hilbert
from joblib import Parallel, delayed
import ncpi

# Set to True if features should be computed for the EEG data instead of the CDM data
compute_EEG = False

def get_frequency_bins(frequency_range):
    """ Get frequency bins for the frequency range of interest. The code has been adapted from 
    https://github.com/arthur-ervin/crosci/tree/main. This code is licensed under creative commons license CC-BY-NC 
    https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.

    Parameters
    ----------
    frequency_range : array, shape (1,2)
        The frequency range over which to create frequency bins.
        The lower edge should be equal or more than 1 Hz, and the upper edge should be equal or less than 150 Hz.

    Returns
    -------
    frequency_bins : list, shape (n_bins,2)
        The lower and upper range in Hz per frequency bin.
    """

    assert frequency_range[0] >= 1.0 and frequency_range[1] <= 150.0, \
        'The frequency range should cannot be less than 1 Hz or more than 150 Hz'

    frequency_bin_delta = [1.0, 4.0]
    frequency_range_full = [frequency_bin_delta[1], 150]
    n_bins_full = 16

    # Create logarithmically-spaced bins over the full frequency range
    frequencies_full = np.logspace(np.log10(frequency_range_full[0]), np.log10(frequency_range_full[-1]), n_bins_full)
    frequencies = np.append(frequency_bin_delta[0],frequencies_full)
    # Get frequencies that fall within the frequency range of interest
    myfrequencies = frequencies[np.where((np.round(frequencies, 4) >= frequency_range[0]) & (
                np.round(frequencies, 4) <= frequency_range[1]))[0]]

    # Get all frequency bin ranges
    frequency_bins = [[myfrequencies[i], myfrequencies[i + 1]] for i in range(len(myfrequencies) - 1)]

    n_bins = len(frequency_bins)

    return frequency_bins


def get_DFA_fitting_interval(frequency_interval):
    """ Get a fitting interval for DFA computation. The code has been adapted from 
    https://github.com/arthur-ervin/crosci/tree/main. This code is licensed under creative commons license CC-BY-NC 
    https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.

    Parameters
    ----------
    frequency_interval : array, shape (1,2)
        The lower and upper bound of the frequency bin in Hz for which the fitting interval will be inferred.
        The fitting interval is where the regression line is fit for log-log coordinates of the fluctuation function vs.
        time windows.

    Returns
    -------
    fit_interval : array, shape (1,2)
        The lower and upper bound of the fitting range in seconds for a frequency bin.
    """

    # Upper fitting margin in seconds
    upper_fit = 30
    # Default lower fitting margins in seconds per frequency bin
    default_lower_fits = [5., 5., 5., 3.981, 3.162, 2.238, 1.412, 1.122, 0.794,
                          0.562, 0.398, 0.281, 0.141, 0.1, 0.1, 0.1]

    frequency_bins = get_frequency_bins([1, 150])
    # Find the fitting interval. In case when frequency range is not exactly one from the defined frequency bins,
    # it finds the fitting interval of the bin for which the lowest of the provided frequencies falls into.
    idx_freq = np.where((np.array(frequency_bins)[:, 0] <= frequency_interval[0]))[0][-1]

    fit_interval = [default_lower_fits[idx_freq],upper_fit]

    return fit_interval

if __name__ == '__main__':
    # Path to the folder containing the processed data
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_processed_data_path']

    # Path to the folder where the features will be saved
    features_path = config['simulation_features_path']

    # Create the FieldPotential object
    if compute_EEG:
        potential = ncpi.FieldPotential(kernel=False, nyhead=True, MEEG ='EEG')

    for method in ['catch22', 'power_spectrum_parameterization_1','power_spectrum_parameterization_2','fEI']:
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
                                # Parameters
                                fEI_window_seconds = 5
                                fEI_overlap = 0.8

                                # Frequency range for the band-pass filter
                                frequency_bins = [[8.,12.]]

                                # Get fit interval
                                DFA_compute_interval = get_DFA_fitting_interval(frequency_bins[0])

                                # Band-pass filtering
                                print('Band-pass filtering and computing amplitude envelope...')
                                all_envelopes = []
                                for ss in df['Data']:
                                    signal_matrix = np.array(ss).reshape(1,-1).astype(np.float64)
                                    # Filter signal in the given frequency bin
                                    filtered_signal = mne.filter.filter_data(data=signal_matrix,
                                                                             sfreq=df.fs,
                                                                             l_freq=frequency_bins[0][0],
                                                                             h_freq=frequency_bins[0][1],
                                                                             filter_length='auto',
                                                                             l_trans_bandwidth='auto',
                                                                             h_trans_bandwidth='auto',
                                                                             fir_window='hamming', phase='zero',
                                                                             fir_design="firwin", pad='reflect_limited',
                                                                             verbose=0)

                                    filtered_signal = filtered_signal[:, 1 * int(df.fs):filtered_signal.shape[1] - 1 *
                                                                                        int(df.fs)]
                                    # Get the amplitude envelope
                                    n_fft = next_fast_len(signal_matrix.shape[1])
                                    amplitude_envelope = Parallel(n_jobs=1, backend='threading', verbose=0)(
                                        delayed(hilbert)
                                        (filtered_signal[idx_channel, :], n_fft)
                                        for idx_channel in range(1))
                                    amplitude_envelope = np.abs(np.array(amplitude_envelope))
                                    all_envelopes.append(amplitude_envelope)

                                # Compute first DFA features
                                print('Computing DFA features...')
                                params = {'fs': df.fs,
                                          'fit_interval': DFA_compute_interval,
                                          'compute_interval': DFA_compute_interval,
                                          'overlap': True,
                                          'bad_idxes': []}
                                df_DFA = pd.DataFrame({'Data': all_envelopes})
                                features = ncpi.Features(method='DFA', params=params)
                                df_DFA = features.compute_features(df_DFA)
                                all_dfa_array = [df_DFA['Features'][k][0] for k in range(len(df_DFA['Features']))]

                                # Compute fE/I for each sample
                                print('Computing fE/I features...')
                                all_fEI_feats = []
                                for xx,ss in enumerate(all_envelopes):
                                    params = {'fs': df.fs,
                                              'window_size_sec': fEI_window_seconds,
                                              'window_overlap': fEI_overlap,
                                              'DFA_array': all_dfa_array[xx],
                                              'bad_idxes': []}
                                    df_fEI = pd.DataFrame({'Data': [ss]})
                                    features = ncpi.Features(method='fEI', params=params)
                                    df_fEI = features.compute_features(df_fEI)
                                    (fEI_outliers_removed, fEI_val, num_outliers, wAmp, wDNF) = df_fEI['Features'][0]
                                    all_fEI_feats.append(np.squeeze(fEI_outliers_removed))

                                # Pass the features to the dataframe
                                df['Features'] = all_fEI_feats

                            if method != 'fEI':
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