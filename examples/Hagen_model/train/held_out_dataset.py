import json
import time
import os
import sys
import pickle
import numpy as np

# ncpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ncpi

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
        # Check if the data is a ndarray
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

if __name__ == "__main__":
    # Load the configuration file that stores file paths used in the script
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_features_path']

    # Iterate over the methods used to compute the features
    all_methods = ['catch22','power_spectrum_parameterization_1', 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
                   'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', 'MD_hrv_classic_pnn40', 'catch22_psp_1']
    for method in all_methods:
        print(f'\n\n--- Method: {method}')
        # Load parameters of the model (theta) and features from simulation data (X)
        print('\n--- Loading simulation data.')
        start_time = time.time()

        # Choose either the catch22 features or the power spectrum parameterization features
        if method != 'catch22_psp_1':
            if method == 'catch22' or method in catch22_names:
                folder = 'catch22'
            else:
                folder = method
            theta = load_simulation_data(os.path.join(sim_file_path, folder, 'sim_theta'))
            X = load_simulation_data(os.path.join(sim_file_path, folder, 'sim_X'))
            if method in catch22_names:
                X = X[:, catch22_names.index(method)]
                print(f'X shape: {X.shape}, column selected: {catch22_names.index(method)}')

        # Concatenate both catch22 and power spectrum parameterization features
        else:
            theta = load_simulation_data(os.path.join(sim_file_path, 'catch22', 'sim_theta'))
            X_1 = load_simulation_data(os.path.join(sim_file_path, 'catch22', 'sim_X'))
            X_2 = load_simulation_data(os.path.join(sim_file_path, 'power_spectrum_parameterization_1', 'sim_X'))
            X = np.concatenate((X_1, X_2.reshape(-1,1)), axis=1)
            print(f'X shape: {X.shape}')

        end_time = time.time()
        print(f'Samples loaded: {len(theta["data"])}')
        print(f'Done in {(end_time - start_time)/60.} min')

        # Create a held-out dataset (90% training, 10% testing)
        print('\n--- Creating a held-out dataset.')
        np.random.seed(0)
        start_time = time.time()
        indices = np.arange(len(theta['data']))
        np.random.shuffle(indices)
        split = int(0.9 * len(indices))
        train_indices = indices[:split]
        test_indices = indices[split:]

        X_train = X[train_indices]
        X_test = X[test_indices]
        theta_train = theta['data'][train_indices]
        theta_test = theta['data'][test_indices]
        end_time = time.time()

        # Create the Inference object, add the simulation data and train the model
        print('\n--- Training the regression model.')
        start_time = time.time()

        model = 'MLPRegressor'
        if method == 'catch22' or method == 'catch22_psp_1':
            hyperparams = [{'hidden_layer_sizes': (25,25), 'max_iter': 100, 'tol': 1e-1, 'n_iter_no_change': 5},
                           {'hidden_layer_sizes': (50,50), 'max_iter': 100, 'tol': 1e-1, 'n_iter_no_change': 5}]
        else:
            hyperparams = [{'hidden_layer_sizes': (2,2), 'max_iter': 100, 'tol': 1e-1, 'n_iter_no_change': 5},
                           {'hidden_layer_sizes': (4,4), 'max_iter': 100, 'tol': 1e-1, 'n_iter_no_change': 5}]

        inference = ncpi.Inference(model=model)
        inference.add_simulation_data(X_train, theta_train)

        # Create folder to save results
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists(os.path.join('data', method)):
            os.makedirs(os.path.join('data', method))

        # Train the model
        inference.train(param_grid=hyperparams,n_splits=10, n_repeats=20)
        # Save the best model and the StandardScaler
        pickle.dump(pickle.load(open('data/model.pkl', 'rb')),
                    open(os.path.join('data', method, 'model'), 'wb'))
        pickle.dump(pickle.load(open('data/scaler.pkl', 'rb')),
                    open(os.path.join('data', method, 'scaler'), 'wb'))

        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')

        # Evaluate the model using the test data
        print('\n--- Evaluating the model.')
        start_time = time.time()

        # Predict the parameters from the test data
        predictions = inference.predict(X_test)

        # Save predictions
        with open(os.path.join('data', method, 'predictions'), 'wb') as file:
            pickle.dump(predictions, file)
        with open(os.path.join('data', method, 'parameters'), 'wb') as file:
            pickle.dump(theta_test, file)

        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')