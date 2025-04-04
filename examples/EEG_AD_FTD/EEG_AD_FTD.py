import json
import os
import pickle
import time
import mne
import pandas as pd
import scipy
import numpy as np
import shutil
import ncpi

# Select the statistical analysis method ('cohen', 'lmer')
statistical_analysis = 'lmer'

databases = [
    'POCTEP', 
    # 'OpenNEURO'
    ]

all_methods = [
    'catch22',
    'power_spectrum_parameterization_1'
    # 'CO_HistogramAMI_even_2_5',
    # 'SB_TransitionMatrix_3ac_sumdiagcov',
    # 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
    # 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
]

print('\nDefault parameters:')
print('---------------------------------------------------------------------------')
print('Inference method trained on: CDM')
print('Number of variables to predict: 4 (E/I, tau_exc, tau_inh, J_ext)')
print('Frequency range used to fit the aperiodic component: (5, 45) Hz')
print('---------------------------------------------------------------------------\n')


default = input('\n--Use default parameters? (y/n): ')

if default == 'y':
    inference_method = 'CDM'
    n_var = 4
    models_path = f'/DATOS/pablomc/ML_models/{n_var}_var/MLP'
    fmin, fmax = 5., 45.

if default == 'n':

    inference_method = input(f'\nUse models trained on CDM o EEG data? (cdm/eeg): ')

    if inference_method == 'EEG' or inference_method == 'eeg':
        n_var = int(input('Number of variables (2 or 4): '))
        models_path = f'/DATOS/pablomc/ML_models/EEG/{n_var}_var'

    if inference_method == 'CDM' or inference_method == 'cdm':
        n_var = int(input('Number of variables (2 or 4): '))
        models_path = f'/DATOS/pablomc/ML_models/{n_var}_var/MLP'

    if 'power_spectrum_parameterization_1' in all_methods:
        print(f'\nIntroduce the frequency range to compute the aperiodic component (fmin, fmax): ')
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


def load_simulation_data(file_path):
    """
    Load simulation data from a file.

    Parameters
    ----------
    file_path : str
        Path to the file containing the simulation data.

    Returns
    -------
    data : dict, ndarray, or None
        Simulation data loaded from the file. Returns None if an error occurs.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pickle.UnpicklingError
        If the file cannot be unpickled.
    TypeError
        If the loaded data is not a dictionary or ndarray.
    """

    data = None  # Initialize to avoid returning an undefined variable

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        # Load the file using pickle
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f'Loaded file: {file_path}')

        # Check if the data is a dictionary
        if isinstance(data, dict):
            print(f'The file contains a dictionary. {list(data.keys())}')
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f'Shape of {key}: {value.shape}')
                else:
                    print(f'{key}: {value}')
        # Check if the data is an ndarray
        elif isinstance(data, np.ndarray):
            print(f'Shape of data: {data.shape}')
        else:
            raise TypeError("Loaded data is neither a dictionary nor an ndarray.")

    except (pickle.UnpicklingError, TypeError) as e:
        print(f"Error: Unable to load the file '{file_path}'. Invalid data format.")
        print(e)
        data = None  # Explicitly set data to None on error
    except Exception as e:
        print(f"An unexpected error occurred while loading the file '{file_path}'.")
        print(e)
        data = None

    return data


def load_empirical_data(dataset, method, n_var, inference_method, raw=False):
    """
    Load empirical data from a file and create a DataFrame with the appropriate format.

    Parameters
    ----------
    dataset : str
        Name of the empirical dataset to load.
    method : str
        Name of the method used to compute the features.
    n_var : int
        Number of variables to predict.
    inference_method : str
        Name of the inference method used to compute the predictions.
    raw : bool, optional
        If True, load the raw data. If False, load the source data.

    Returns
    -------
    emp_data : pandas.DataFrame
        DataFrame containing the empirical data.
    """

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


def create_POCTEP_dataframe(raw=False):
    '''
    Load the POCTEP dataset and create a DataFrame with the data.

    Parameters
    ----------
    raw : bool, optional
        If True, load the raw data. If False, load the source data.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the POCTEP data.
    '''

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
        print(f'\rProcessing {file} - {pt + 1 }/{len(ldir)}', end="", flush=True)

        # load data
        data = scipy.io.loadmat(data_path + '/' + file)['data']
        signal = data['signal'][0, 0]

        # get sampling frequency
        fs = data['cfg'][0, 0]['fs'][0, 0][0, 0]

        # Electrodes (raw data)/regions (if source data)
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

    # Save ch_names
    pd.to_pickle(ch_names, os.path.join('results', 'ch_names_POCTEP.pkl'))

    return df

def create_OpenNEURO_dataframe():
    '''
    Load the OpenNEURO dataset and create a DataFrame with the data.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the OpenNEURO data.
    '''

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
                        group.append(gp if gp != 'C' else 'HC')
                        epoch.append(i)
                        sensor.append(ch_names[elec])
                        EEG.append(ep[:, elec])

    # Create the Pandas DataFrame
    df = pd.DataFrame({'ID': ID,
                       'Group': group,
                       'Epoch': epoch,
                       'Sensor': sensor,
                       'Data': EEG})
    df['Recording'] = 'EEG'
    df['fs'] = fs

    # Save ch_names
    pd.to_pickle(ch_names, os.path.join('results', 'ch_names_OpenNEURO.pkl'))

    return df

if __name__ == "__main__":

    # Paths to the simulation data
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

            # Check if 'Features' and 'Predictions' columns are in the DataFrame
            features_computed = 'Features' in data.columns
            predictions_computed = 'Predictions' in data.columns

            if not features_computed:
                print(f'\nNo features computed for {method}.')
                print(f'No predictions computed for {method}.')
                predictions_computed = False
            else:
                print(f'\nFeatures already computed for {method}.')
                if not predictions_computed:
                    print(f'No predictions computed for {method}.')

            ##########################
            #   FEATURE EXTRACTION   #
            ##########################

            if not features_computed:
                if method == 'power_spectrum_parameterization_1':
                    fooof_setup_emp = {
                        'peak_threshold': 1.,
                        'min_peak_height': 0.,
                        'max_n_peaks': 5,
                        'peak_width_limits': (10., 50.)
                    }

                    params = {
                        'fs': data['fs'][0],
                        'fmin': fmin,
                        'fmax': fmax,
                        'fooof_setup': fooof_setup_emp,
                        'r_squared_th':0.9
                    }
                else:
                    params = None

                feat_init_time = time.time()
                print(f'\nComputing {method} features from {db}')

                if method in catch22_names or method == 'catch22':
                    features = ncpi.Features(method='catch22', params=params)
                    data = features.compute_features(data)

                if method == 'power_spectrum_parameterization_1':
                    features = ncpi.Features(method='power_spectrum_parameterization', params=params)
                    data = features.compute_features(data)
                    # 1/f slope
                    data['Features'] = data['Features'].apply(lambda x: x[1])

                data.to_pickle(os.path.join('results', file_name+'.pkl'))

                feat_end_time = time.time()
                print(f'{method} computed in {(feat_end_time - feat_init_time)/60.} min')


            ######################################
            #   Statistical analysis: FEATURES   #
            ######################################

            if statistical_analysis == 'lmer':
                print(f'\nLinear mixed model analysis of features for {method}...')
            elif statistical_analysis == 'cohen':
                print(f'\nCohen\'s d analysis of features for {method}...')

            stat_init_time = time.time()
            for ii, elec in enumerate([False, True]):
                is_elec = 'elec' if elec else 'noelec'

                stat_file_name = file_name + f'-{is_elec}-feat_{statistical_analysis}.pkl'

                # Check if statistical results have already been computed
                if os.path.exists(os.path.join('results', stat_file_name)):
                    print(f'{stat_file_name} already computed.')

                else:
                    # data with features
                    data = pd.read_pickle(os.path.join('results', file_name+'.pkl'))
                    Analysis = ncpi.Analysis(data)

                    if method in catch22_names:
                        method_index = catch22_names.index(method)
                        if statistical_analysis == 'lmer':
                            stat_result = Analysis.lmer(control_group = 'HC', data_col = 'Features',
                                                        data_index = method_index,
                                                        models={
                                                            'mod00': 'Y ~ Group * Sensor + (1 | ID)',
                                                            'mod01': 'Y ~ Group * Sensor',
                                                            'mod02': 'Y ~ Group + Sensor + (1 | ID)',
                                                            'mod03': 'Y ~ Group + Sensor'
                                                        } if elec else
                                                            {'mod00': 'Y ~ Group + (1 | ID)',
                                                             'mod01': 'Y ~ Group'},
                                                        bic_models=["mod00", "mod01"],
                                                        anova_tests={
                                                            "test1": ["mod00", "mod02"],
                                                            "test2": ["mod01", "mod03"]
                                                        } if elec else None,
                                                        specs = '~Group | Sensor' if elec else '~Group')
                        elif statistical_analysis == 'cohen':
                            stat_result = Analysis.cohend(control_group = 'HC', data_col = 'Features',
                                                        data_index = method_index)

                        with open(os.path.join('results', stat_file_name), 'wb') as results_file:
                            pickle.dump(stat_result, results_file)

                    if method == 'power_spectrum_parameterization_1':
                        if statistical_analysis == 'lmer':
                            stat_result = Analysis.lmer(control_group='HC', data_col='Features',
                                                        data_index=-1,
                                                        models={
                                                            'mod00': 'Y ~ Group * Sensor + (1 | ID)',
                                                            'mod01': 'Y ~ Group * Sensor',
                                                            'mod02': 'Y ~ Group + Sensor + (1 | ID)',
                                                            'mod03': 'Y ~ Group + Sensor'
                                                        } if elec else
                                                        {'mod00': 'Y ~ Group + (1 | ID)',
                                                         'mod01': 'Y ~ Group'},
                                                        bic_models=["mod00", "mod01"],
                                                        anova_tests={
                                                            "test1": ["mod00", "mod02"],
                                                            "test2": ["mod01", "mod03"]
                                                        } if elec else None,
                                                        specs='~Group | Sensor' if elec else '~Group')
                        elif statistical_analysis == 'cohen':
                            stat_result = Analysis.cohend(control_group='HC', data_col='Features',
                                                          data_index=-1)


                        with open(os.path.join('results', stat_file_name), 'wb') as results_file:
                            pickle.dump(stat_result, results_file)

                    if method == 'catch22':
                        print(f'{statistical_analysis} is not computed for the whole catch22 set. '
                              f'Use a specific catch22 feature instead.')

            stat_end_time = time.time()
            print(f'Statistical analysis computed in {(stat_end_time - stat_init_time)/60.} min')

            #######################################################
            #   PREDICTIONS OF PARAMETERS OF THE NEURAL CIRCUIT   #
            #######################################################

            if not predictions_computed:
                print('\nComputing predictions...')
                # Load data with features
                data = pd.read_pickle(os.path.join('results', file_name+'.pkl')) if features_computed else data

                # Load simulation data
                theta = load_simulation_data(os.path.join(sim_file_path, method, 'sim_theta'))
                X = load_simulation_data(os.path.join(sim_file_path, method, 'sim_X'))

                print(f'Samples loaded: {len(theta["data"])}')

                # Create the Inference object
                inference = ncpi.Inference(model='MLPRegressor')

                # Add simulation data to the Inference object (not sure if necessary)
                inference.add_simulation_data(
                    np.zeros((len(X), 22 if method == 'catch22' else 1)),
                    np.zeros((len(X), 3+n_var))
                )

                data['Predictions'] = np.nan

                if not os.path.exists('data'):
                    os.makedirs('data')

                predictions_init_time = time.time()
                if inference_method == 'cdm' or inference_method == 'CDM':
                    # Transfer model and scaler to the data folder
                    shutil.copy(
                        os.path.join(models_path, method, 'scaler'),
                        os.path.join('data', 'scaler.pkl')
                        )

                    shutil.copy(
                        os.path.join(models_path, method, 'model'),
                        os.path.join('data', 'model.pkl')
                        )

                    # Predictions
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

                    # List of sensors for DB1
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
                print(f'--Predictions computed in {(predictions_end_time - predictions_init_time)/60.} min')

                # Save the DataFrame with the predictions
                data.to_pickle(os.path.join('results', file_name+'.pkl'))

            ########################################
            #   Statistical analysis: PARAMETERS   #
            ########################################

            if statistical_analysis == 'lmer':
                print(f'\nLinear mixed model analysis of inferred parameters for {method}...')
            elif statistical_analysis == 'cohen':
                print(f'\nCohen\'s d analysis of inferred parameters for {method}...')

            # Check if the statistical results have already been computed
            if os.path.exists(os.path.join('results', file_name+f'-elec-pred_{statistical_analysis}.pkl')):
                print(f'{file_name}-elec-pred_{statistical_analysis}.pkl already computed.')

            else:
                # Load data with predictions
                data = pd.read_pickle(os.path.join('results', file_name+'.pkl')) if predictions_computed else data
                predictions = np.array(data['Predictions'].to_list())

                stat_init_time = time.time()
                stat_dict = {}
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

                    data['Predictions'] = param
                    Analysis = ncpi.Analysis(data)
                    print(f'\n--- Parameter: {param_name}\n')
                    if statistical_analysis == 'lmer':
                        stat_dict[param_name] = Analysis.lmer(control_group='HC', data_col='Predictions',
                                                              data_index=-1,
                                                              models={
                                                                  'mod00': 'Y ~ Group * Sensor + (1 | ID)',
                                                                  'mod01': 'Y ~ Group * Sensor',
                                                                  'mod02': 'Y ~ Group + Sensor + (1 | ID)',
                                                                  'mod03': 'Y ~ Group + Sensor'
                                                              },
                                                              bic_models = ["mod00", "mod01"],
                                                              anova_tests={
                                                                  "test1": ["mod00", "mod02"],
                                                                  "test2": ["mod01", "mod03"]
                                                              },
                                                              specs='~Group | Sensor')
                    elif statistical_analysis == 'cohen':
                        stat_dict[param_name] = Analysis.cohend(control_group='HC', data_col='Predictions',
                                                                data_index=-1)

                with open(os.path.join('results', file_name+f'-elec-pred_{statistical_analysis}.pkl'), 'wb') as results_file:
                    pickle.dump(stat_dict, results_file)

                stat_end_time = time.time()
                print(f'--Statistical Analysis computed in {stat_end_time - stat_init_time} seconds')

        database_end_time = time.time()

        print(f'\n\n=== Database {db} completed in {(database_end_time - database_init_time)/60.} min')
