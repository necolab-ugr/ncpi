import os
import shutil
import pickle
import json
import pandas as pd
import scipy
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import ncpi

# Path to ML models
models_path = f'/DATOS/pablomc/ML_models/4_var/MLP'
ML_model = 'MLPRegressor'

# Names of catch22 features
try:
    import pycatch22
    catch22_names = pycatch22.catch22_all([0])['names']
except:
    catch22_names = ['DN_HistogramMode_5',
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
                     'FC_LocalSimple_mean3_stderr']


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

def load_empirical_data(folder_path):
    '''
    Collect the LFP data from all LFP data files and merge them into a single dictionary.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the LFP data files.

    Returns
    -------
    emp_data : dict
        Dictionary containing the loaded data. The keys are 'LFP', 'fs', and 'age'.
    '''

    print(f'Loading files from {folder_path}')
    file_list = os.listdir(folder_path)
    emp_data = {'LFP': [], 'fs': [], 'age': []}

    for i,file_name in enumerate(file_list):
        print(f'\r Progress: {i+1} of {len(file_list)} files loaded', end='', flush=True)
        structure = scipy.io.loadmat(os.path.join(folder_path, file_name))
        LFP = structure['LFP']['LFP'][0,0]
        sum_LFP = np.sum(LFP, axis=0)  # sum LFP across channels
        fs = structure['LFP']['fs'][0, 0][0, 0]
        age = structure['LFP']['age'][0,0][0,0]

        emp_data['LFP'].append(sum_LFP)
        emp_data['fs'].append(fs)
        emp_data['age'].append(age)

    return emp_data


def compute_features(data, chunk_size=5., method='catch22', params=None):
    '''
    Creates a Pandas DataFrame containing the computed features from the LFP data.

    Parameters
    ----------
    data : dict
        Dictionary containing the LFP data.
    chunk_size : float
        Size of the chunks (epochs) in seconds.
    method : str
        Method used to compute the features.
    params : dict
        Dictionary containing the parameters of the method used to compute the features.

    Returns
    -------
    df : DataFrame
        Pandas DataFrame containing the computed features.
    '''

    # Split the data into chunks (epochs)
    chunk_size = int(chunk_size * data['fs'][0])
    chunked_data = []
    ID = []
    epoch = []
    group = []
    for i in range(len(data['LFP'])):
        for e,j in enumerate(range(0, len(data['LFP'][i]), chunk_size)):
            if len(data['LFP'][i][j:j+chunk_size]) == chunk_size:
                chunked_data.append(data['LFP'][i][j:j+chunk_size])
                ID.append(i)
                epoch.append(e)
                group.append(data['age'][i])

    # Create the Pandas DataFrame
    df = pd.DataFrame({'ID': ID,
                       'Group': group,
                       'Epoch': epoch,
                       'Sensor': np.zeros(len(ID)), # dummy sensor
                       'Data': chunked_data})
    df.Recording = 'LFP'
    df.fs = data['fs'][0]

    # Compute features
    features = ncpi.Features(method=method, params=params)
    df = features.compute_features(df)

    return df


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

    # Append the predictions to the DataFrame
    pd_preds = pd.DataFrame({'Predictions': predictions})
    data = pd.concat([data, pd_preds], axis=1)

    return data

if __name__ == "__main__":
    # Load the configuration file that stores all file paths used in the script
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_features_path']
    emp_data_path = config['LFP_development_data_path']

    # # Define a catch22 feature subset
    # catch22_subset = ['SP_Summaries_welch_rect_centroid',
    #                   'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1']
    # catch22_subset_idx = [catch22_names.index(f) for f in catch22_subset]

    # Iterate over the methods used to compute the features
    all_methods = ['catch22','power_spectrum_parameterization_1']
    for method in all_methods:
        print(f'\n\n--- Method: {method}')
        # Load parameters of the model (theta) and features from simulation data (X)
        print('\n--- Loading simulation data.')
        start_time = time.time()
        if method == 'catch22' or method == 'catch22_subset' or method in catch22_names:
            folder = 'catch22'
        else:
            folder = method
        theta = load_simulation_data(os.path.join(sim_file_path, folder, 'sim_theta'))
        X = load_simulation_data(os.path.join(sim_file_path, folder, 'sim_X'))
        end_time = time.time()
        print(f'Samples loaded: {len(theta["data"])}')
        print(f'Done in {(end_time - start_time)/60.} min')

        # # Select features from the catch22 feature set
        # if method == 'catch22_subset':
        #     X = X[:, catch22_subset_idx]
        # elif method in catch22_names:
        #     X = X[:, catch22_names.index(method)]

        # Load empirical data
        print('\n--- Loading empirical data.')
        start_time = time.time()
        emp_data = load_empirical_data(emp_data_path)
        print(f'\nFiles loaded: {len(emp_data["LFP"])}')
        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')

        # Compute features from empirical data
        print('\n--- Computing features from empirical data.')
        start_time = time.time()
        chunk_size = 5
        if method == 'catch22' or method == 'catch22_subset' or method in catch22_names:
            emp_data = compute_features(emp_data, chunk_size=chunk_size, method='catch22')

            # # Subsets of catch22 features
            # if method == 'catch22_subset':
            #     new_features = []
            #     for jj in range(len(emp_data)):
            #         # print(f'\r Arranging the catch22 subset. Progress: {jj+1} of {len(emp_data)}', end='', flush=True)
            #         new_features.append([emp_data['Features'][jj][i] for i in catch22_subset_idx])
            #     emp_data['Features'] = new_features
            #
            # if method in catch22_names:
            #     emp_data['Features'] = emp_data['Features'].apply(lambda x: x[catch22_names.index(method)])

        elif method == 'power_spectrum_parameterization_1':
            # Parameters of the fooof algorithm
            fooof_setup_emp = {'peak_threshold': 1.,
                               'min_peak_height': 0.,
                               'max_n_peaks': 5,
                               'peak_width_limits': (10., 50.)}
            emp_data = compute_features(emp_data, chunk_size=chunk_size,
                                        method='power_spectrum_parameterization',
                                        params={'fs': emp_data['fs'][0],
                                                'fmin': 5.,
                                                'fmax': 45.,
                                                'fooof_setup': fooof_setup_emp,
                                                'r_squared_th':0.9})

            # Keep only the aperiodic exponent (1/f slope)
            emp_data['Features'] = emp_data['Features'].apply(lambda x: x[1])

        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')

        # Load the Inference objects and add the simulation data
        print('\n--- Loading the inverse model.')
        start_time = time.time()

        # Create inference object
        inference = ncpi.Inference(model=ML_model)
        # Not sure if this is needed
        inference.add_simulation_data(X, theta['data'])

        # Create folder to save results
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists(os.path.join('data', method)):
            os.makedirs(os.path.join('data', method))

        # Transfer model and scaler to the data folder
        shutil.copy(
            os.path.join(models_path, method, 'scaler'),
            os.path.join('data', 'scaler.pkl')
        )

        shutil.copy(
            os.path.join(models_path, method, 'model'),
            os.path.join('data', 'model.pkl')
        )

        if ML_model == 'SNPE':
            shutil.copy(
                os.path.join(models_path, method, 'density_estimator'),
                os.path.join('data', 'density_estimator.pkl')
            )

        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')

        # Compute predictions from the empirical data
        print('\n--- Computing predictions from empirical data.')
        start_time = time.time()
        emp_data = compute_predictions(inference, emp_data)
        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')

        # Save the data including predictions of all parameters
        emp_data.to_pickle(os.path.join('data', method, 'emp_data_all.pkl'))

        # Replace parameters of recurrent synaptic conductances with the ratio (E/I)_net
        E_I_net = emp_data['Predictions'].apply(lambda x: (x[0]/x[2]) / (x[1]/x[3]))
        others = emp_data['Predictions'].apply(lambda x: x[4:])
        emp_data['Predictions'] = (np.concatenate((E_I_net.values.reshape(-1,1),
                                                   np.array(others.tolist())), axis=1)).tolist()

        # Save the data including predictions of (E/I)_net
        emp_data.to_pickle(os.path.join('data', method, 'emp_data_reduced.pkl'))

        # Plot predictions as a function of age
        plt.figure(dpi = 300)
        plt.rc('font', size=8)
        plt.rc('font', family='Arial')
        titles = [r'$(E/I)_{net}$', r'$\tau_{exc}^{syn}$', r'$\tau_{inh}^{syn}$', r'$J_{ext}^{syn}$']

        for param in range(4):
            plt.subplot(1,4,param+1)
            param_pd = pd.DataFrame({'Group': emp_data['Group'],
                                     'Predictions': emp_data['Predictions'].apply(lambda x: x[param])})
            ax = sns.boxplot(x='Group', y='Predictions', data=param_pd, showfliers=False,
                             palette='Set2', legend=False, hue='Group')
            ax.set_title(titles[param])
            if param == 0:
                ax.set_ylabel('Predictions')
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Postnatal days')
            ax.set_xticks(np.arange(0, len(np.unique(emp_data['Group'])), 2))
            ax.set_xticklabels([f'P{str(i)}' for i in np.unique(emp_data['Group'])[::2]])
            plt.tight_layout()

        # Save the plot
        if not os.path.exists('figures'):
            os.makedirs('figures')
        plt.savefig(f'figures/predictions_{method}.png')