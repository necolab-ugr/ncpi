import json
import os
import pickle
import sys
import time

import mne
import pandas as pd
import scipy
import numpy as np
from matplotlib import pyplot as plt

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

    except Exception as e:
        print(f'Error loading file: {file_path}')
        print(e)

    return data


def load_empirical_data(dataset):
    # Check if features have been already computed
    if os.path.exists(os.path.join('data', method, f'emp_data_{dataset}.pkl')):
        emp_data = pd.read_pickle(os.path.join('data', method, f'emp_data_{dataset}.pkl'))
        print(f'emp_data_{dataset}.pkl loaded.')

    # Compute features from empirical data
    else:
        # Create folder to save features
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists(os.path.join('data', method)):
            os.makedirs(os.path.join('data', method))

        print(f'\n--- Computing features for {dataset} data.')
        start_time = time.time()
        if method == 'catch22':
            if dataset == 'POCTEP':
                emp_data = compute_features_POCTEP(method='catch22', params=None)
            elif dataset == 'OpenNEURO':
                emp_data = compute_features_OpenNEURO(method='catch22', params=None)
        else:
            print(f'Error: method {method} not implemented.')
        end_time = time.time()
        print(f'Done in {(end_time - start_time) / 60.} min')

        # Save the features
        emp_data.to_pickle(os.path.join('data', method, f'emp_data_{dataset}.pkl'))

    return emp_data

def compute_features_POCTEP(method='catch22', params=None):
    data_path = '/DATOS/pablomc/empirical_datasets/POCTEP_data/CLEAN/SOURCES/dSPM/DK'

    # List files in the directory
    ldir = os.listdir(data_path)

    ID = []
    group = []
    epoch = []
    sensor = []
    EEG = []

    for pt,file in enumerate(ldir):
        print(file)

        # load data
        data = scipy.io.loadmat(data_path + '/' + file)['data']
        signal = data['signal'][0, 0]

        # get sampling frequency
        fs = data['cfg'][0, 0]['fs'][0, 0][0, 0]

        # Brain regions
        regions = np.arange(signal.shape[1])

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
                sensor.append(rg)
                EEG.append(ep[:, rg])

    # Create the Pandas DataFrame
    df = pd.DataFrame({'ID': ID,
                       'Group': group,
                       'Epoch': epoch,
                       'Sensor': sensor,
                       'Data': EEG})
    df.Recording = 'EEG'
    df.fs = fs

    # Compute features
    features = ccpi.Features(method=method, params=params)
    df = features.compute_features(df)

    return df

def compute_features_OpenNEURO(method='catch22', params=None):
    data_path = '/DATOS/pablomc/empirical_datasets/OpenNEURO_data'

    # load participants file
    participants = pd.read_csv(os.path.join(data_path,'participants.tsv'), sep='\t')

    ID = []
    group = []
    epoch = []
    sensor = []
    EEG = []

    for gp in ['A', 'C', 'F']:
        pt = participants.loc[participants['Group'] == gp]
        folders = np.array(pt['participant_id'])

        for folder in folders:
            if folder[:3] == 'sub':
                print(folder)
                dir = os.path.join(data_path,'derivatives', folder, 'eeg')

                # find the .set file
                for file in os.listdir(dir):
                    if file[-3:] == 'set':
                        EEG_file = file
                # load raw data
                raw = mne.io.read_raw_eeglab(os.path.join(data_path, 'derivatives', folder, 'eeg', EEG_file))
                # get data
                data, times = raw[:]
                ch_names = raw.ch_names
                fs = 1. / (times[1] - times[0])

                # 5-second epochs
                epochs = np.arange(0, data.shape[1], int(fs * 5))

                for i in range(len(epochs) - 1):
                    ep = data[:, epochs[i]:epochs[i + 1]]
                    ep = ep.T
                    # z-score normalization
                    ep = (ep - np.mean(ep, axis=0)) / np.std(ep, axis=0)

                    # Append data
                    for elec in range(len(ch_names)):
                        ID.append(folder)
                        group.append(gp)
                        epoch.append(i)
                        sensor.append(ch_names[elec])
                        EEG.append(ep[:, elec])

    # Create the Pandas DataFrame
    df = pd.DataFrame({'ID': ID,
                       'Group': group,
                       'Epoch': epoch,
                       'Sensor': sensor,
                       'Data': EEG})
    df.Recording = 'EEG'
    df.fs = fs

    # Compute features
    features = ccpi.Features(method=method, params=params)
    df = features.compute_features(df)

    return df

if __name__ == "__main__":
    # Load the configuration file that stores all file paths used in the script
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_features_path']

    # Iterate over the methods used to compute the features
    all_methods = ['catch22']
    for method in all_methods:
        print(f'\n\n--- Method: {method}')
        # # Load parameters of the model (theta) and features from simulation data (X)
        # print('\n--- Loading simulation data.')
        # start_time = time.time()
        # theta = load_simulation_data(os.path.join(sim_file_path, method, 'sim_theta'))
        # X = load_simulation_data(os.path.join(sim_file_path, method, 'sim_X'))
        # end_time = time.time()
        # print(f'Samples loaded: {len(theta["data"])}')
        # print(f'Done in {(end_time - start_time)/60.} min')


        # Compute features from empirical data or load them if they have been already computed
        # POCTEP dataset
        emp_data_POCTEP = load_empirical_data('POCTEP')
        # OpenNEURO dataset
        emp_data_OpenNeuro = load_empirical_data('OpenNEURO')

        # Plots
        # Create a figure and set its properties
        fig = plt.figure(figsize=(7.5, 3.5), dpi=300)
        plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

        # Define 4 colors
        colors_1 = ['lightgrey', 'lightcoral', 'lightblue', 'lightgreen']

        # Define 3 colors
        colors_2 = ['lightgrey', 'peachpuff', 'cornflowerblue']

        for row in range(2):
            for col in range(4):
                ax = fig.add_axes([0.08 + col * 0.24, 0.59 - row * 0.43, 0.15, 0.33])

                if row == 0 and col < 2:
                    feat = 18
                if row == 0 and col >= 2:
                    feat = 8
                if row == 1 and col < 2:
                    feat = 4
                if row == 1 and col >= 2:
                    feat = 19

                # DB1
                if col % 2 == 0:
                    groups = ['HC','ADMIL', 'ADMOD', 'ADSEV']
                    dataset = emp_data_POCTEP
                    colors = colors_1
                # DB2
                else:
                    groups = ['C','F','A']
                    dataset = emp_data_OpenNeuro
                    colors = colors_2

                # Boxplots
                for i,group in enumerate(groups):
                    emp_data_group = dataset[dataset.Group == group]
                    feature = emp_data_group['Features'].apply(lambda x: x[feat])
                    box = ax.boxplot(feature, positions=[i], showfliers=False,
                                 widths=0.7, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                                 whiskerprops=dict(color='black', linewidth=0.5),
                                 capprops=dict(color='black', linewidth=0.5),
                                 boxprops=dict(linewidth=0.5))
                    for patch in box['boxes']:
                        patch.set_facecolor(colors[i])

                # x-labels
                if row == 1:
                    if col % 2 == 0:
                        ax.set_xticks([0, 1, 2, 3])
                        ax.set_xticklabels(['HC', 'ADMIL', 'ADMOD', 'ADSEV'], rotation=45, fontsize = 8)
                    else:
                        ax.set_xticks([0, 1, 2])
                        ax.set_xticklabels(['HC', 'FTD', 'AD'], rotation=45, fontsize = 8)
                else:
                    ax.set_xticks([])

                # y-labels
                if row == 0 and col < 2:
                    ax.set_ylabel(r'$rs\_range$')
                if row == 0 and col >= 2:
                    ax.set_ylabel(r'$TransVar$')
                if row == 1 and col < 2:
                    ax.set_ylabel(r'$ami2$')
                if row == 1 and col >= 2:
                    ax.set_ylabel(r'$dfa$')

                # Titles
                ax.set_title(f'DB{1 if col%2 == 0 else 2}')


        plt.show()