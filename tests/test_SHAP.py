import json
import os
import pickle
import gc
import numpy as np
import shap
from matplotlib import pyplot as plt
from mpi4py import MPI
from joblib import Parallel, delayed

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to calculate SHAP values for a single row
def compute_shap(explainer, row):
    return explainer(np.array([row]))

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
    except Exception as e:
        print(f'Error loading file: {file_path}')
        print(e)

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

    return data

# Labels for the features
features_ID = ['mode_5',
               'mode_10',
               'acf_timescale',
               'acf_first_min',
               'ami2',
               'trev',
               'high_fluctuation',
               'stretch_high',
               'transition_variance',
               'periodicity',
               'embedding_dist',
               'ami_timescale',
               'whiten_timescale',
               'outlier_timing_pos',
               'outlier_timing_neg',
               'low_freq_power',
               'stretch_decreasing',
               'entropy_pairs',
               'rs_range',
               'dfa',
               'centroid_freq',
               'forecast_error']

if __name__ == '__main__':
    # Load the configuration file that stores the file path
    with open('../examples/Hagen_model/config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_features_path']

    # Load the simulation data
    print('Loading simulation data...')
    X = load_simulation_data(os.path.join(sim_file_path, 'catch22', 'sim_X'))

    # Randomly subsample the simulation data
    np.random.seed(42)
    idx = np.random.choice(len(X), 200000, replace=False)
    X = X[idx]

    # Load the machine learning model and scaler
    with open('data/model', 'rb') as file:
        model = pickle.load(file)
    with open('data/scaler', 'rb') as file:
        scaler = pickle.load(file)
    if type(model) is list:
        reg = model[0].__class__.__name__
    else:
        reg = model.__class__.__name__
    print(f'Loaded {reg} model and scaler')

    # # Randomly subsample the machine learning model
    # if type(model) is list:
    #     idx = np.random.choice(len(model), 2, replace=False)
    #     model = [model[i] for i in idx]
    # else:
    #     print('Error: The model is not a list.')
    #     exit()

    # Split the data among MPI ranks
    n_samples_per_rank = len(model)//size
    start = rank * n_samples_per_rank
    end = start + n_samples_per_rank
    model = model[start:end]

    # Scale the features
    print('Scaling the features...')
    feats = scaler.transform(X)

    # Compute SHAP values
    print('Computing SHAP values...')
    if type(model) is list:
        all_SHAP_values = {'JEE': [], 'JIE': [], 'JEI': [], 'JII': [], 'tau_exc': [], 'tau_inh': [], 'J_ext': []}
        for i,m in enumerate(model):
            print(f'\n Rank {rank} - Model {i+1}')

            # Explain the model's predictions using SHAP
            if reg == 'Ridge':
                explainer = shap.Explainer(m, feats)
            elif reg == 'MLPRegressor':
                explainer = shap.PermutationExplainer(m.predict, feats)
            print('Explaining the model...')
            # shap_values = explainer(feats)

            # Parallel SHAP value computation
            shap_values = Parallel(n_jobs=-1)(delayed(compute_shap)(explainer, row) for row in feats)
            for j in range(len(shap_values)):
                shap_values[0].values = np.concatenate((shap_values[0].values, shap_values[j].values),
                                                       axis=0)
                shap_values[0].base_values = np.concatenate((shap_values[0].base_values, shap_values[j].base_values),
                                                            axis=0)
                shap_values[0].data = np.concatenate((shap_values[0].data, shap_values[j].data),
                                                     axis=0)
            shap_values = shap_values[0]

            # Transform to absolute values
            # shap_values.values = np.abs(shap_values.values)

            # Store the SHAP values
            print(f'Storing SHAP values for model {i+1}...')
            if i == 0:
                all_SHAP_values['JEE'] = shap_values[:, :, 0]
                all_SHAP_values['JIE'] = shap_values[:, :, 1]
                all_SHAP_values['JEI'] = shap_values[:, :, 2]
                all_SHAP_values['JII'] = shap_values[:, :, 3]
                all_SHAP_values['tau_exc'] = shap_values[:, :, 4]
                all_SHAP_values['tau_inh'] = shap_values[:, :, 5]
                all_SHAP_values['J_ext'] = shap_values[:, :, 6]
            else:
                all_SHAP_values['JEE'] += shap_values[:, :, 0]
                all_SHAP_values['JIE'] += shap_values[:, :, 1]
                all_SHAP_values['JEI'] += shap_values[:, :, 2]
                all_SHAP_values['JII'] += shap_values[:, :, 3]
                all_SHAP_values['tau_exc'] += shap_values[:, :, 4]
                all_SHAP_values['tau_inh'] += shap_values[:, :, 5]
                all_SHAP_values['J_ext'] += shap_values[:, :, 6]

            # Release memory
            del shap_values, explainer
            gc.collect()

    else:
        print('Error: The model is not a list.')
        exit()

    # Compute the average SHAP values of the rank
    for key in all_SHAP_values.keys():
        all_SHAP_values[key] /= len(model)

    # Gather the SHAP values from all ranks
    gathered_SHAP_values = all_SHAP_values.copy()
    for key in gathered_SHAP_values.keys():
        gathered_SHAP_values[key].values = comm.gather(all_SHAP_values[key].values, root=0)
        gathered_SHAP_values[key].base_values = comm.gather(all_SHAP_values[key].base_values, root=0)
        gathered_SHAP_values[key].data = comm.gather(all_SHAP_values[key].data, root=0)
    # gathered_SHAP_values = comm.gather(all_SHAP_values, root=0)

    if rank == 0:
        # Sum the SHAP values from all ranks
        for i in range(1, size):
            for key in gathered_SHAP_values.keys():
                gathered_SHAP_values[key].values[0] += gathered_SHAP_values[key].values[i]
                gathered_SHAP_values[key].base_values[0] += gathered_SHAP_values[key].base_values[i]
                gathered_SHAP_values[key].data[0] += gathered_SHAP_values[key].data[i]

        # Compute the average SHAP values
        for key in gathered_SHAP_values.keys():
            all_SHAP_values[key].values = gathered_SHAP_values[key].values[0] / size
            all_SHAP_values[key].base_values = gathered_SHAP_values[key].base_values[0] / size
            all_SHAP_values[key].data = gathered_SHAP_values[key].data[0] / size

            # Feature names
            all_SHAP_values[key].feature_names = features_ID

        # Plot the SHAP values
        print('Plotting SHAP values...')
        fig = plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.size': 8, 'font.family': 'Arial'})
        for row in range(4):
            for col in range(2):
                ax = fig.add_axes([0.17 + col * 0.5, 0.77 - row * 0.24, 0.28, 0.19])
                if row == 0 and col == 0:
                    shap.plots.bar(all_SHAP_values['JEE'], max_display=15, show=False)
                    ax.set_title(r'$J_{EE}$')
                elif row == 0 and col == 1:
                    shap.plots.bar(all_SHAP_values['JIE'], max_display=15, show=False)
                    ax.set_title(r'$J_{IE}$')
                elif row == 1 and col == 0:
                    shap.plots.bar(all_SHAP_values['JEI'], max_display=15, show=False)
                    ax.set_title(r'$J_{EI}$')
                elif row == 1 and col == 1:
                    shap.plots.bar(all_SHAP_values['JII'], max_display=15, show=False)
                    ax.set_title(r'$J_{II}$')
                elif row == 2 and col == 0:
                    shap.plots.bar(all_SHAP_values['tau_exc'], max_display=15, show=False)
                    ax.set_title(r'$\tau_{syn}^{exc}$ (ms)')
                elif row == 2 and col == 1:
                    shap.plots.bar(all_SHAP_values['tau_inh'], max_display=15, show=False)
                    ax.set_title(r'$\tau_{syn}^{inh}$ (ms)')
                elif row == 3 and col == 0:
                    shap.plots.bar(all_SHAP_values['J_ext'], max_display=15, show=False)
                    ax.set_title(r'$J_{syn}^{ext}$ (nA)')
                else:
                    # Plot the sum of the SHAP values
                    shap.plots.bar(all_SHAP_values['JEE']/np.max(all_SHAP_values['JEE'].values)+
                                   all_SHAP_values['JEI']/np.max(all_SHAP_values['JEI'].values)+
                                   all_SHAP_values['JIE']/np.max(all_SHAP_values['JIE'].values)+
                                   all_SHAP_values['JII']/np.max(all_SHAP_values['JII'].values)+
                                   all_SHAP_values['tau_exc']/np.max(all_SHAP_values['tau_exc'].values)+
                                   all_SHAP_values['tau_inh']/np.max(all_SHAP_values['tau_inh'].values)+
                                   all_SHAP_values['J_ext']/np.max(all_SHAP_values['J_ext'].values),
                                   max_display=15, show=False)
                    ax.set_title('Sum of SHAP values')

                # Change the font size of the axis labels
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=6)
                if row == 3:
                    ax.set_xlabel('SHAP values', fontsize=8)
                else:
                    ax.set_xlabel('')
                ax.set_xticks([])

                # Change font size of text
                for text_obj in ax.texts:
                    text_obj.set_fontsize(6)

        plt.savefig('SHAP_values.png')
        # plt.show()