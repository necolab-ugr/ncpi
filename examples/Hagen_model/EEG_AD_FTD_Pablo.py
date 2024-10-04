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

EEG_AD_FTD_path = '/DATOS/pablomc/EEG_AD_FTD_results'

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
        if os.path.exists(os.path.join(EEG_AD_FTD_path, method, f'emp_data_{dataset}_{raw}.pkl')):
            emp_data = pd.read_pickle(os.path.join(EEG_AD_FTD_path, method, f'emp_data_{dataset}_{raw}.pkl'))
            print(f'Loaded file: emp_data_{dataset}_{raw}.pkl')
        else:
            compute_f = True
    if dataset == 'OpenNEURO':
        if os.path.exists(os.path.join(EEG_AD_FTD_path, method, f'emp_data_{dataset}.pkl')):
            emp_data = pd.read_pickle(os.path.join(EEG_AD_FTD_path, method, f'emp_data_{dataset}.pkl'))
            print(f'Loaded file: emp_data_{dataset}.pkl')
        else:
            compute_f = True

    # Compute features from empirical data
    if compute_f:
        # Create folder to save features
        if not os.path.exists(EEG_AD_FTD_path):
            os.makedirs(EEG_AD_FTD_path)
        if not os.path.exists(os.path.join(EEG_AD_FTD_path, method)):
            os.makedirs(os.path.join(EEG_AD_FTD_path, method))

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
            emp_data.to_pickle(os.path.join(EEG_AD_FTD_path, method, f'emp_data_{dataset}_{raw}.pkl'))
        else:
            emp_data.to_pickle(os.path.join(EEG_AD_FTD_path, method, f'emp_data_{dataset}.pkl'))

    return emp_data

def compute_features_POCTEP(method='catch22', params=None, raw = False):
    if raw:
        data_path = '/DATOS/pablomc/empirical_datasets/POCTEP_data/CLEAN/SENSORS'
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
                        sensor.append(elec)
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
    if not np.isnan(feat):
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
            # Fit the linear mixed-effects models
            if elec == False:
                r('''
                mod00 = Features ~ Group  + (1 | ID)
                mod01 = Features ~ Group
                m00 <- lmer(mod00, data=df_pair)
                m01 <- lmer(mod01, data=df_pair)
                print(summary(m00))
                print(summary(m01))
                ''')
            else:
                r('''
                mod00 = Features ~ Group * Sensor + (1 | ID)
                m00 <- lmer(mod00, data=df_pair)
                print(summary(m00))
                ''')

            # BIC
            if elec == False:
                r('''
                all_models <- c('m00', 'm01')
                bics <- c(BIC(m00), BIC(m01))
                print(bics)
                index <- which.min(bics)
                mod_sel <- all_models[index]
                ''')

            # Compute the pairwise comparisons between groups
            if elec == False:
                r('''
                emm <- suppressMessages(emmeans(mod_sel, specs=~Group))
                ''')
            else:
                r('''
                emm <- suppressMessages(emmeans(m00, specs=~Group | Sensor))
                ''')

            r('''
            res <- pairs(emm, adjust='holm')
            df_res <- as.data.frame(res)
            print(df_res)
            ''')

            df_res_r = ro.r['df_res']

            # Convert the R DataFrame to a pandas DataFrame
            with (pandas2ri.converter + pandas2ri.converter).context():
                df_res_pd = pandas2ri.conversion.get_conversion().rpy2py(df_res_r)

            results[label_comp] = df_res_pd

    return results

def compute_predictions(inference, data):
    """
    Compute predictions from the empirical data.

    Parameters
    ----------
    inference : Inference
        Inference object containing the trained model.
    data : DataFrame
        DataFrame containing the features of the empirical data.

    Returns
    -------
    data : DataFrame
        DataFrame containing the features of the empirical data and the predictions.
    """

    # Predict the parameters from the features of the empirical data
    predictions = inference.predict(np.array(data['Features'].tolist()))

    return predictions

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
        print(f'All features computed/loaded in {(end_time - start_time)/60.} min')

        # LMER analysis
        print('\n--- LMER analysis.')
        # Check if the results have been already computed
        if os.path.exists(os.path.join(EEG_AD_FTD_path, method, 'lmer_feat.pkl')):
            print(f'lmer_feat.pkl already computed.')
        else:
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
            with open(os.path.join(EEG_AD_FTD_path, method, 'lmer_feat.pkl'), 'wb') as file:
                pickle.dump(lmer_feat, file)
                print(f'lmer_feat.pkl saved.')

            end_time = time.time()
            print(f'Done in {(end_time - start_time)/60.} min')

        # Predictions
        tr_model = 'MLPRegressor'
        all_confs = ['SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
                     'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
                     'CO_HistogramAMI_even_2_5',
                     'SB_TransitionMatrix_3ac_sumdiagcov',
                     'catch22']
        lmer_preds = [{'DB1': {}, 'DB2': {}} for k in range(2)]

        for n_var in [1,2]:
            MLP_path = f'/DATOS/pablomc/MLP_models/{n_var}_var'

            for conf in all_confs:
                print(f'\n--- Configuration: {conf}, n_var: {n_var}')

                # Check if predictions have been already computed
                if os.path.exists(os.path.join(EEG_AD_FTD_path, method,f'preds_data_POCTEP_raw_{conf}_{n_var}.pkl')) and \
                        os.path.exists(os.path.join(EEG_AD_FTD_path, method,f'preds_data_OpenNeuro_{conf}_{n_var}.pkl')):
                    print(f'Predictions already computed.')
                else:
                    # Load the best model and the StandardScaler
                    model = pickle.load(open(os.path.join(MLP_path, conf, 'model'), 'rb'))
                    scaler = pickle.load(open(os.path.join(MLP_path, conf, 'scaler'), 'rb'))

                    # Transfer the model and scaler to the data folder
                    if not os.path.exists('data'):
                        os.makedirs('data')
                    pickle.dump(model, open('data/model.pkl', 'wb'))
                    pickle.dump(scaler, open('data/scaler.pkl', 'wb'))

                    # Adapt features to the regression model
                    new_data_POCTEP = emp_data_POCTEP_raw.copy()
                    new_data_OpenNeuro = emp_data_OpenNeuro.copy()
                    if conf != 'catch22':
                        if conf == 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1':
                            new_data_POCTEP['Features'] = new_data_POCTEP['Features'].apply(lambda x: x[18])
                            new_data_OpenNeuro['Features'] = new_data_OpenNeuro['Features'].apply(lambda x: x[18])
                        elif conf == 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1':
                            new_data_POCTEP['Features'] = new_data_POCTEP['Features'].apply(lambda x: x[19])
                            new_data_OpenNeuro['Features'] = new_data_OpenNeuro['Features'].apply(lambda x: x[19])
                        elif conf == 'CO_HistogramAMI_even_2_5':
                            new_data_POCTEP['Features'] = new_data_POCTEP['Features'].apply(lambda x: x[4])
                            new_data_OpenNeuro['Features'] = new_data_OpenNeuro['Features'].apply(lambda x: x[4])
                        elif conf == 'SB_TransitionMatrix_3ac_sumdiagcov':
                            new_data_POCTEP['Features'] = new_data_POCTEP['Features'].apply(lambda x: x[8])
                            new_data_OpenNeuro['Features'] = new_data_OpenNeuro['Features'].apply(lambda x: x[8])

                    # Compute predictions from the empirical data
                    print('\n--- Computing predictions from empirical data.')
                    start_time = time.time()
                    inference = ccpi.Inference(model=tr_model)
                    # Add fake simulation data. Not sure if this is necessary
                    inference.add_simulation_data(np.zeros((len(X),22 if conf == 'catch22' else 1)),
                                                  np.zeros((len(X),3+n_var)))
                    predictions_POCTEP = compute_predictions(inference, new_data_POCTEP)
                    predictions_OpenNeuro = compute_predictions(inference, new_data_OpenNeuro)
                    end_time = time.time()
                    print(f'Done in {(end_time - start_time) / 60.} min')

                    # Save the predictions
                    pickle.dump(predictions_POCTEP, open(os.path.join(EEG_AD_FTD_path, method,
                                                                  f'preds_data_POCTEP_raw_{conf}_{n_var}.pkl'), 'wb'))
                    pickle.dump(predictions_OpenNeuro, open(os.path.join(EEG_AD_FTD_path, method,
                                                                    f'preds_data_OpenNeuro_{conf}_{n_var}.pkl'), 'wb'))

                    # LMER analysis
                    print('\n--- LMER analysis.')
                    lmer_preds_POCTEP = []
                    lmer_preds_OpenNeuro = []
                    for param in range(n_var):
                        # E/I
                        if param == 0:
                            preds_POCTEP = (predictions_POCTEP[:,0]/predictions_POCTEP[:,2]) /\
                                         (predictions_POCTEP[:, 1] / predictions_POCTEP[:, 3])
                            preds_OpenNeuro = (predictions_OpenNeuro[:,0]/predictions_OpenNeuro[:,2]) /\
                                            (predictions_OpenNeuro[:, 1] / predictions_OpenNeuro[:, 3])
                        # J_ext
                        else:
                            preds_POCTEP = predictions_POCTEP[:,4]
                            preds_OpenNeuro = predictions_OpenNeuro[:,4]

                        # Replace the features with the predictions for the lmer analysis (this should be improved
                        # in the future)
                        new_data_POCTEP['Features'] = preds_POCTEP
                        new_data_OpenNeuro['Features'] = preds_OpenNeuro
                        # Compute the linear mixed-effects model
                        lmer_preds_POCTEP.append(lmer(new_data_POCTEP, np.nan, True))
                        lmer_preds_OpenNeuro.append(lmer(new_data_OpenNeuro, np.nan, True))
                    lmer_preds[n_var-1]['DB1'][conf] = lmer_preds_POCTEP
                    lmer_preds[n_var-1]['DB2'][conf] = lmer_preds_OpenNeuro

        # Check if the LMER results have been already computed
        if os.path.exists(os.path.join(EEG_AD_FTD_path, method, 'lmer_preds.pkl')):
            print(f'lmer_preds.pkl already computed.')
        else:
            # Save the results
            with open(os.path.join(EEG_AD_FTD_path, method, 'lmer_preds.pkl'), 'wb') as file:
                pickle.dump(lmer_preds, file)
                print(f'lmer_preds.pkl saved.')
