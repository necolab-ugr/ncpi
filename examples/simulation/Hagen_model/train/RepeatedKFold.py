import json
import time
import os
import pickle
import numpy as np
import ncpi

# Choose whether to use a held-out dataset or the full dataset
held_out_dataset = False

# List of parameters to be included in the training
# Full list of parameters:
# params =  ['J_EE', 'J_IE', 'J_EI', 'J_II', 'tau_syn_E', 'tau_syn_I', 'J_ext']

# Special case that includes the E/I parameter and the synaptic time constants
# params = ['E_I', 'tau_syn_E', 'tau_syn_I']

# Special case that only includes the E/I parameter
params = ['E_I']

# ML model to train
model = 'MLPRegressor'

# List of feature sets used to train the model
# all_methods = ['catch22','power_spectrum_parameterization_1', 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
#                'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', 'MD_hrv_classic_pnn40', 'catch22_psp_1']
all_methods = ['catch22', 'power_spectrum_parameterization_1']

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

        # Subset the data to include only the parameters of interest
        if params is not None:
            print(f'\n--- Subsetting the data to include only the parameters of interest.')
            # First compute the E/I parameter
            if 'E_I' in params:
                E_I = (theta['data'][:,0]/theta['data'][:,2]) / (theta['data'][:,1]/theta['data'][:,3])
                E_I = np.reshape(E_I, (-1, 1))
                theta['data'] = np.concatenate((E_I, theta['data']), axis=1)
                theta['parameters'] = ['E_I'] + theta['parameters']

            # Next generate the subset of parameters
            theta['data'] = theta['data'][:, [theta['parameters'].index(param) for param in params]]
            theta['parameters'] = params
            print(f'Subset of parameters: {params}')

            # If theta['data'] is a 2D array with one column, reshape it to a 1D array
            if theta['data'].shape[1] == 1:
                theta['data'] = np.reshape(theta['data'], (-1,))

        print(f'Shape of theta data: {theta["data"].shape}')
        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')

        # Create a directory to save results
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists(os.path.join('data', method)):
            os.makedirs(os.path.join('data', method))

        # Create a held-out dataset (90% training, 10% testing)
        if held_out_dataset:
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

            # Save the held-out dataset
            with open(os.path.join('data', method, 'held_out_dataset'), 'wb') as file:
                pickle.dump((X_test, theta_test), file)
            print(f'\n--- The held-out dataset has been saved.')
            print(f'Done in {(end_time - start_time)/60.} min')

        else:
            print('\n--- Using the full dataset.')
            X_train = X
            X_test = None
            theta_train = theta['data']
            theta_test = None

        # Create the Inference object, add the simulation data and train the model
        print('\n--- Training the regression model.')
        start_time = time.time()

        if model == 'MLPRegressor':
            print('--- Using MLPRegressor')
            if method == 'catch22' or method == 'catch22_psp_1':
                hyperparams = [{'hidden_layer_sizes': (25,25), 'max_iter': 100, 'tol': 1e-1, 'n_iter_no_change': 5},
                               {'hidden_layer_sizes': (50,50), 'max_iter': 100, 'tol': 1e-1, 'n_iter_no_change': 5}]
            else:
                hyperparams = [{'hidden_layer_sizes': (2,2), 'max_iter': 100, 'tol': 1e-1, 'n_iter_no_change': 5},
                               {'hidden_layer_sizes': (4,4), 'max_iter': 100, 'tol': 1e-1, 'n_iter_no_change': 5}]

        if model == 'SNPE':
            print('--- Using SNPE')
            if method == 'catch22' or method == 'catch22_psp_1':
                hyperparams = [{'prior': None, 'density_estimator': {'model':"maf", 'hidden_features':10,
                                                                     'num_transforms':2}}]
            else:
                hyperparams = [{'prior': None, 'density_estimator': {'model':"maf", 'hidden_features':2,
                                                                     'num_transforms':2}}]

        if model == 'Ridge':
            print('--- Using Ridge')
            hyperparams = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1.}, {'alpha': 10.}, {'alpha': 100.}]


        inference = ncpi.Inference(model=model)
        inference.add_simulation_data(X_train, theta_train)

        # Train the model
        if model == 'SNPE':
            # inference.train(param_grid=None, train_params={
            #     'learning_rate': 1e-1,
            #     'stop_after_epochs': 5,
            #     'max_num_epochs': 100})
            # inference.train(param_grid=None)
            inference.train(param_grid=hyperparams, n_splits=10, n_repeats=1)
        else:
            inference.train(param_grid=hyperparams,n_splits=10, n_repeats=20)

        # Save the best model and the StandardScaler
        pickle.dump(pickle.load(open('data/model.pkl', 'rb')),
                    open(os.path.join('data', method, 'model'), 'wb'))
        pickle.dump(pickle.load(open('data/scaler.pkl', 'rb')),
                    open(os.path.join('data', method, 'scaler'), 'wb'))

        # Save density estimator
        if model == 'SNPE':
            pickle.dump(pickle.load(open('data/density_estimator.pkl', 'rb')),
                        open(os.path.join('data', method, 'density_estimator'), 'wb'))

        end_time = time.time()
        print(f'Done in {(end_time - start_time)/60.} min')

        # Evaluate the model using the test data
        if held_out_dataset:
            print('\n--- Evaluating the model.')
            start_time = time.time()

            # Predict the parameters from the test data
            predictions = inference.predict(X_test)

            # Save predictions
            with open(os.path.join('data', method, 'predictions'), 'wb') as file:
                pickle.dump(predictions, file)

            end_time = time.time()
            print(f'Done in {(end_time - start_time)/60.} min')