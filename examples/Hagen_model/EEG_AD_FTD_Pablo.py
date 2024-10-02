import json
import os
import pickle
import sys
import time
import mne
import pandas as pd
import scipy
import numpy as np
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro

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


def load_empirical_data(dataset, raw = False):
    # Check if features have been already computed
    compute_f = False
    if dataset == 'POCTEP':
        if os.path.exists(os.path.join('data', method, f'emp_data_{dataset}_{raw}.pkl')):
            emp_data = pd.read_pickle(os.path.join('data', method, f'emp_data_{dataset}_{raw}.pkl'))
            print(f'Loaded file: emp_data_{dataset}_{raw}.pkl')
        else:
            compute_f = True
    if dataset == 'OpenNEURO':
        if os.path.exists(os.path.join('data', method, f'emp_data_{dataset}.pkl')):
            emp_data = pd.read_pickle(os.path.join('data', method, f'emp_data_{dataset}.pkl'))
            print(f'Loaded file: emp_data_{dataset}.pkl')
        else:
            compute_f = True

    # Compute features from empirical data
    if compute_f:
        # Create folder to save features
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists(os.path.join('data', method)):
            os.makedirs(os.path.join('data', method))

        print(f'\n--- Computing features for {dataset} data.')
        start_time = time.time()
        if method == 'catch22':
            if dataset == 'POCTEP':
                emp_data = compute_features_POCTEP(method='catch22', params=None, raw=raw)
            elif dataset == 'OpenNEURO':
                emp_data = compute_features_OpenNEURO(method='catch22', params=None)
        else:
            print(f'Error: method {method} not implemented.')
        end_time = time.time()
        print(f'Done in {(end_time - start_time) / 60.} min')

        # Save the features
        if dataset == 'POCTEP':
            emp_data.to_pickle(os.path.join('data', method, f'emp_data_{dataset}_{raw}.pkl'))
        else:
            emp_data.to_pickle(os.path.join('data', method, f'emp_data_{dataset}.pkl'))

    return emp_data

def compute_features_POCTEP(method='catch22', params=None, raw = False):
    if raw:
        data_path = '/DATOS/pablomc/empirical_datasets/POCTEP_data/RAW'
    else:
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

        # Electrodes (raw data)/regions (if source data)
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

def lmer(df, feat, elec = False):

    # Activate pandas2ri
    pandas2ri.activate()

    # Load R libraries directly in R
    r('''
    library(dplyr)
    library(lme4)
    library(emmeans)
    library(ggplot2)
    library(repr)
    library(mgcv)
    ''')

    # Copy the dataframe
    df = df.copy()

    # Change the name of the C group to HC
    if 'HC' not in df['Group'].unique():
        df['Group'] = df['Group'].apply(lambda x: 'HC' if x == 'C' else x)

    # Remove the 'Data' column
    df = df.drop(columns=['Data'])

    # Select the desired feature to analyze
    feature = df['Features'].apply(lambda x: x[feat])
    df['Features'] = feature

    # Filter out 'HC' and 'MCI' from the list of unique groups
    groups = df['Group'].unique()
    groups = [group for group in groups if group != 'HC' and group != 'MCI']

    # Create a list with the different group comparisons
    groups_comp = [f'{group}vsHC' for group in groups]

    # Remove rows where the variable is zero
    df = df[df['Features'] != 0]

    results = {}
    for label, label_comp in zip(groups, groups_comp):
        print(f'\n\n--- Group: {label}')
        # Filter DataFrame to obtain the desired groups
        df_pair = df[(df['Group'] == 'HC') | (df['Group'] == label)]
        ro.globalenv['df_pair'] = pandas2ri.py2rpy(df_pair)
        ro.globalenv['label'] = label
        # print(df_pair)

        # Convert columns to factors
        r(''' 
        df_pair$ID = as.factor(df_pair$ID)
        df_pair$Group = factor(df_pair$Group, levels = c(label, 'HC'))
        df_pair$Epoch = as.factor(df_pair$Epoch)
        df_pair$Sensor = as.factor(df_pair$Sensor)
        print(table(df_pair$Group))
        ''')

        # if table in R is empty for any group, skip the analysis
        if r('table(df_pair$Group)')[0] == 0 or r('table(df_pair$Group)')[1] == 0:
            results[label_comp] = pd.DataFrame({'p.value': [1], 'z.ratio': [0]})
        else:
            # Fit the linear mixed-effects model
            if elec == False:
                r('''
                mod00 = Features ~ Group  + (1 | ID)
                m00 <- lmer(mod00, data=df_pair)
                print(summary(m00))
                ''')
            else:
                r('''
                mod00 = Features ~ Group * Sensor + (1 | ID)
                m00 <- lmer(mod00, data=df_pair)
                print(summary(m00))
                ''')

            # Compute the pairwise comparisons between groups
            r('''
            emm <- suppressMessages(emmeans(m00, specs=~Group))
            res <- pairs(emm, adjust='holm')
            df_res <- as.data.frame(res)
            print(df_res)
            filtered_df <- df_res[df_res$p.value < 0.05, ]
            ''')

            df_res_r = ro.r['df_res']

            # Convert the R DataFrame to a pandas DataFrame
            with (pandas2ri.converter + pandas2ri.converter).context():
                df_res_pd = pandas2ri.conversion.get_conversion().rpy2py(df_res_r)

            results[label_comp] = df_res_pd

    return results


if __name__ == "__main__":
    # Load the configuration file that stores all file paths used in the script
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_features_path']

    # Iterate over the methods used to compute the features
    all_methods = ['catch22']
    for method in all_methods:
        print(f'\n\n--- Method: {method}')

        # Load parameters of the model (theta) and features from simulation data (X)
        print('\n--- Loading simulation data.')
        start_time = time.time()
        theta = load_simulation_data(os.path.join(sim_file_path, method, 'sim_theta'))
        X = load_simulation_data(os.path.join(sim_file_path, method, 'sim_X'))
        end_time = time.time()
        print(f'Samples loaded: {len(theta["data"])}')
        print(f'Done in {(end_time - start_time)/60.} min')

        # Compute features from empirical data or load them if they have been already computed
        # POCTEP dataset
        print('\n--- Loading empirical data.')
        start_time = time.time()
        emp_data_POCTEP_source = load_empirical_data('POCTEP', raw = False)
        emp_data_POCTEP_raw = load_empirical_data('POCTEP', raw=True)
        # OpenNEURO dataset
        emp_data_OpenNeuro = load_empirical_data('OpenNEURO')
        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')

        # LMER analysis
        print('\n--- LMER analysis.')
        start_time = time.time()
        lmer_feat = [{'DB1_raw': {}, 'DB1_source': {}, 'DB2': {}} for k in range(2)]
        for ii,elec in enumerate([False, True]):
            for DB in range(2):
                for feat in [18, 8, 4, 19]:
                    print(f'\n--- Feature: {feat}, DB: {DB}, Elec: {elec}')
                    # Compute the linear mixed-effects model
                    if DB == 0:
                        lmer_feat_raw = lmer(emp_data_POCTEP_raw, feat, elec)
                        lmer_feat_source = lmer(emp_data_POCTEP_source, feat, elec)
                        lmer_feat[ii]['DB1_raw'][f'{feat}'] = lmer_feat_raw
                        lmer_feat[ii]['DB1_source'][f'{feat}'] = lmer_feat_source
                    else:
                        lmer_feat_raw = lmer(emp_data_OpenNeuro, feat, elec)
                        lmer_feat[ii]['DB2'][f'{feat}'] = lmer_feat_raw

        # Save the results
        with open(os.path.join('data', method, 'lmer_feat.pkl'), 'wb') as file:
            pickle.dump(lmer_feat, file)
            print(f'lmer_feat.pkl saved.')

        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')

        # Predictions
        # To do

