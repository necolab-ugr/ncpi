import json
import os
import pickle
import sys
import time
import mne
import pandas as pd
import scipy
import numpy as np
import shutil
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro

# ncpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ncpi

EEG_AD_FTD_path = '/DATOS/pablomc/EEG_AD_FTD_results'

databases = [
    'POCTEP', 
    'OpenNEURO'
    ]

all_methods = [
    'catch22',
    'power_spectrum_parameterization_1', 
    # 'CO_HistogramAMI_even_2_5',
    # 'SB_TransitionMatrix_3ac_sumdiagcov',
    # 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
    # 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
]

print('Default parameters:')
print('---------------------------------------------------------------------------')
print('Inference method: CDM')
print('Number of variables (n_var): 4')
print('Aperiodic component interval (if aperiodic component is desired): 5., 45.')
print('---------------------------------------------------------------------------')


default = input('\n--Use default parameters? (y/n): ')

if default == 'y':
    inference_method = 'CDM'
    n_var = 4
    models_path = f'/DATOS/pablomc/ML_models/{n_var}_var/MLP'
    fmin, fmax = 5., 45.

if default == 'n':

    inference_method = input(f'\nUse CDM o EEG models? (cdm/eeg): ')

    if inference_method == 'EEG' or inference_method == 'eeg':
        n_var = int(input('Number of variables (2 or 4): '))
        models_path = f'/DATOS/pablomc/ML_models/EEG/{n_var}_var'

    if inference_method == 'CDM' or inference_method == 'cdm':
        n_var = int(input('Number of variables (2 or 4): '))
        models_path = f'/DATOS/pablomc/ML_models/{n_var}_var/MLP'


    if 'power_spectrum_parameterization_1' in all_methods:
        print(f'\nIntroduce the interval to compute the aperiodic component (fmin, fmax): ')
        fmin = int(input('fmin: '))
        fmax = int(input('fmax: '))




catch22_names = [
    'DN_HistogramMode_5',
    'DN_HistogramMode_10',
    'CO_f1ecac',
    'CO_FirstMin_ac',
    'CO_HistogramAMI_even_2_5',
    'CO_trev_1_num',
    'MD_hrv_classic_pnn40',
    'SB_BinaryStats_mean_longstretch1',
    'SB_TransitionMatrix_3ac_sumdiagcov',
    'PD_PeriodicityWang_th0_01',
    'CO_Embed2_Dist_tau_d_expfit_meandiff',
    'IN_AutoMutualInfoStats_40_gaussian_fmmi',
    'FC_LocalSimple_mean1_tauresrat',
    'DN_OutlierInclude_p_001_mdrmd',
    'DN_OutlierInclude_n_001_mdrmd',
    'SP_Summaries_welch_rect_area_5_1',
    'SB_BinaryStats_diff_longstretch0',
    'SB_MotifThree_quantile_hh',
    'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
    'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
    'SP_Summaries_welch_rect_centroid',
    'FC_LocalSimple_mean3_stderr'
]

def load_empirical_data(dataset, method, n_var, inference_method, raw=False):

    print(f'Loading {dataset} data...')

    if dataset=='POCTEP':
        file_name = f'{dataset}_{raw}-{method}-{n_var}_var-{inference_method}'
        if os.path.exists(os.path.join('results', file_name+'.pkl')):
            emp_data = pd.read_pickle(os.path.join('results', file_name+'.pkl'))
            print(f'Loaded file: {file_name}.pkl')

        else:
            print(f'{dataset}_{raw}-{method}-{n_var}_var-{inference_method} not found. Creating DataFrame...')
            emp_data = create_POCTEP_dataframe(raw=raw)

    if dataset == 'OpenNEURO':
        file_name = f'{dataset}-{method}-{n_var}_var-{inference_method}'
        if os.path.exists(os.path.join('results', file_name+'.pkl')):
            emp_data = pd.read_pickle(os.path.join('results', file_name+'.pkl'))
            print(f'Loaded file: {file_name}.pkl')
        else:
            emp_data = create_OpenNEURO_dataframe()

    pd.to_pickle(emp_data, os.path.join('results', file_name+'.pkl'))

    return emp_data, file_name

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


def create_POCTEP_dataframe(raw=False):
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
        print(f'\rProcessiing {file} - {pt + 1 }/{len(ldir)}', end="", flush=True)

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

    return df

def create_OpenNEURO_dataframe():
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
    df = df.drop(columns=['Predictions'])

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
                m01 <- lm(mod01, data=df_pair)
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
                
                if (mod_sel == 'm00') {
                    m_sel <- lmer(mod00, data=df_pair)
                }
                if (mod_sel == 'm01') {
                    m_sel <- lm(mod01, data=df_pair)
                }                
                ''')

            # Compute the pairwise comparisons between groups
            if elec == False:
                r('''
                emm <- suppressMessages(emmeans(m_sel, specs=~Group))
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

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

sim_file_path = config['simulation_features_path']

# Check if the 'results' folder to store results already exists
if not os.path.exists(os.path.join('results')):
    os.makedirs(os.path.join('results'))

for db in databases:
 
    database_init_time = time.time()
    print(f'\n\n### Database: {db}')
    for method in all_methods:
        print(f'\n=== Method: {method}')
        

        # Load empirical data. It will create the DataFrame if it does not exist
        data, file_name = load_empirical_data(db, method, n_var, inference_method, True)
          
        # Check if 'Features' and 'Predictions' columns are in the DataFrame to skip the unnecessary computations

        features_computed = True
        predictions_computed = True
        if 'Features' not in data.columns:

            # If there is no Features column, there are no predictions either
            print(f'No features computed for {method}.')
            print(f'No predictions computed for {method}.')

            features_computed = False
            predictions_computed = False

        else:
            print(f'Features already computed for {method}.')

            if 'Predictions' not in data.columns:
                print(f'No predictions computed for {method}.')
                predictions_computed = False

        #######################
        #   FEATURE SECTION   #
        #######################
        if not features_computed:
            # Compute features if the DataFrame does not hat 'Features' column
            if method == 'power_spectrum_parameterization_1':
                fooof_setup_emp = {
                    'peak_threshold': 1.,
                    'min_peak_height': 0.,
                    'max_n_peaks': 5,
                    'peak_width_limits': (10., 50.)
                }
                
                params = {
                    'fs': 500,
                    'fmin': fmin,
                    'fmax': fmax,
                    'fooof_setup': fooof_setup_emp,
                    'r_squared_th':0.9
                }
            else:
                params = None

            feat_init_time = time.time()
            print(f'Computing {method} features from {db}')
            
            if method in catch22_names or method == 'catch22':
                features = ncpi.Features(method='catch22', params=params)
                data = features.compute_features(data)
            
            if method == 'power_spectrum_parameterization_1':
                features = ncpi.Features(method='power_spectrum_parameterization', params=params)
                data = features.compute_features(data)
                data['Features'] = data['Features'].apply(lambda x: x[1])
            
            data.to_pickle(os.path.join('results', file_name+'.pkl'))

            feat_end_time = time.time()
            print(f'{method} computed in {feat_end_time - feat_init_time} seconds')


        ### LMER ###
        print(f'Computing lmer for {method}...')

        lmer_init_time = time.time()
        for ii, elec in enumerate([False, True]):
            is_elec = 'elec' if elec else 'noelec'
            
            lmer_file_name = file_name + f'-{is_elec}-feat_lmer.pkl'
            
            # Check if the lmer results have already been computed
            if os.path.exists(os.path.join('results', lmer_file_name)):
                print(f'{lmer_file_name} already computed.')
            
            else:
                
                # Need to read feature file to do lmer 
                data = pd.read_pickle(os.path.join('results', file_name+'.pkl'))
                
                if method in catch22_names:
                    method_index = catch22_names.index(method)
                    lmer_result = lmer(data, method_index, elec)

                    with open(os.path.join('results', lmer_file_name), 'wb') as results_file:
                        pickle.dump(lmer_result, results_file)

                if method == 'power_spectrum_parameterization_1':
                    lmer_result = lmer(data, np.nan, elec)


                    with open(os.path.join('results', lmer_file_name), 'wb') as results_file:
                        pickle.dump(lmer_result, results_file)
                
                if method == 'catch22':
                    print('Can not compute lmer to the whole dataset of catch22.')

        lmer_end_time = time.time()
        print(f'Lmer computed in {lmer_end_time - lmer_init_time} seconds')


        #########################
        #   INFERENCE SECTION   #
        #########################
        

        if not predictions_computed:

            # Read feauture file if computed before
            data = pd.read_pickle(os.path.join('results', file_name+'.pkl')) if features_computed else data

            # Load simulation data
            theta = load_simulation_data(os.path.join(sim_file_path, method, 'sim_theta'))
            X = load_simulation_data(os.path.join(sim_file_path, method, 'sim_X'))

            print(f'Samples loaded: {len(theta["data"])}')

            inference = ncpi.Inference(model='MLPRegressor')
            inference.add_simulation_data(
                np.zeros((len(X), 22 if method == 'catch22' else 1)),
                np.zeros((len(X), 3+n_var))    
            )

            data['Predictions'] = np.nan

            # Transfer model and scaler to the data folder is necessary everytime
            if not os.path.exists('data'):
                os.makedirs('data')

            predictions_init_time = time.time()
            if inference_method == 'cdm' or inference_method == 'CDM':
                shutil.copy(
                    os.path.join(models_path, method, 'scaler'),
                    os.path.join('data', 'scaler.pkl')
                    )
                
                shutil.copy(
                    os.path.join(models_path, method, 'model'),
                    os.path.join('data', 'model.pkl')
                    )

                if method in catch22_names:
                    predictions = inference.predict(
                        np.array(data['Features'].apply(lambda x: x[catch22_names.index(method)]).to_list())
                    )
                elif method == 'catch22' or method == 'power_spectrum_parameterization_1':
                    predictions = inference.predict(
                        np.array(data['Features'].to_list())
                    )
                
                # Store predictions in the DataFrame
                data['Predictions'] = [list(pred) for pred in predictions]

            

            if inference_method == 'eeg' or inference_method == 'EEG':
                if n_var == 4:
                    path = os.path.join(models_path)
                if n_var == 2:
                    path = os.path.join(models_path, method)

                # List of sensors following the order of DB1
                sensor_list = [
                    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1',
                    'O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'
                ]

                for s, sensor in enumerate(sensor_list):
                    print(f'--- Sensor: {sensor}')

                    shutil.copy(
                        os.path.join(path, sensor, method, 'model'),
                        os.path.join('data', 'model.pkl')
                    )

                    shutil.copy(
                        os.path.join(path, sensor, method, 'scaler'),
                        os.path.join('data', 'scaler.pkl')
                    )
                    
                    sensor_df = data[data['Sensor'].isin([sensor, s])]
                    print(sensor_df)
                    if method in catch22_names:
                        predictions = inference.predict(
                            np.array(sensor_df['Features'].apply(lambda x: x[catch22_names.index(method)]).to_list())
                        )
                    elif method == 'catch22' or method == 'power_spectrum_parameterization_1':
                        predictions = inference.predict(
                            np.array(sensor_df['Features'].to_list())
                        )

                    sensor_df['Predictions'] = [list(pred) for pred in predictions]
                    data.update(sensor_df['Predictions'])  
                print(data['Predictions'])
                predictions = np.array(data['Predictions'].to_list())   
            
            predictions_end_time = time.time()
            print(f'--Predictions computed in {predictions_end_time - predictions_init_time} seconds')

            # Save the DataFrame with the predictions
            data.to_pickle(os.path.join('results', file_name+'.pkl'))


        ### LMER ###
        
        # Check if the lmer results have already been computed
        if os.path.exists(os.path.join('results', file_name+'-elec-pred_lmer.pkl')):
            print(f'{file_name}-elec-pred_lmer.pkl already computed.')

        else:
            # Read DataFrame with predictions if computed before
            data = pd.read_pickle(os.path.join('results', file_name+'.pkl')) if predictions_computed else data
            predictions = np.array(data['Predictions'].to_list())

            lmer_init_time = time.time()
            lmer_dict = {}
            for i in range(n_var):
                if i == 0:  # E/I
                    param = (predictions[:, 0] / predictions[:, 2]) /\
                            (predictions[:, 1] / predictions[:, 3])
                    param_name = 'E/I'

                if i == 1: # Jext if n_var == 2 or tau_exc if n_var == 4
                    param = predictions[:, 4]
                    param_name = 'Jext' if n_var == 2 else 'tau_exc'

                if i == 2: # tau_inh            
                    param = predictions[:, 5]
                    param_name = 'tau_inh'

                if i == 3: # Jext if n_var == 4
                    param = predictions[:, 6]                
                    param_name = 'Jext'    

                # lmer function uses 'Features' column 
                data['Features'] = param

                lmer_dict[param_name] = lmer(data, np.nan, elec=True)  
            

            with open(os.path.join('results', file_name+'-elec-pred_lmer.pkl'), 'wb') as results_file:
                pickle.dump(lmer_dict, results_file)      
            
            lmer_end_time = time.time()
            print(f'--Lmer computed in {lmer_end_time - lmer_init_time} seconds')

    database_end_time = time.time()

    print(f'\n\n=== Database {db} completed in {database_end_time - database_init_time} seconds')
    