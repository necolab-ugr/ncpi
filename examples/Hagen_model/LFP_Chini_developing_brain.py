import os
import sys
import pickle
import json

import pandas as pd
import scipy
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_empirical_data(folder_path):
    '''
    Load empirical data from a folder containing LFP data files.

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


def compute_features(data, chunk_size=5., method='catch22'):
    '''
    Compute features from the LFP data.

    Parameters
    ----------
    data : dict
        Dictionary containing the LFP data.
    chunk_size : float
        Size of the chunks (epochs) in seconds.
    method : str
        Method used to compute the features.

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
    features = ccpi.Features(method=method)
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

    # Load parameters of the model (theta) and features from simulation data (X)
    print('\n--- Loading simulation data.')
    start_time = time.time()
    theta = load_simulation_data(os.path.join(sim_file_path, 'sim_theta'))
    X = load_simulation_data(os.path.join(sim_file_path, 'sim_X'))
    end_time = time.time()
    print(f'Samples loaded: {len(theta["data"])}')
    print(f'Done in {end_time - start_time} s')

    # # Randomly subsample the simulation data
    # idx = np.random.choice(len(theta['data']), 100000, replace=False)
    # X = X[idx]
    # theta['data'] = theta['data'][idx]

    # # Plot some statistics of the simulation data
    # plt.figure(dpi = 300)
    # plt.rc('font', size=8)
    # plt.rc('font', family='Arial')
    #
    # for param in range(7):
    #     print(f'Parameter {theta["parameters"][param]}')
    #     plt.subplot(2,4,param+1)
    #     ax = sns.histplot(theta['data'][:,param], kde=True, bins=50, color='blue')
    #     ax.set_title(f'Parameter {theta["parameters"][param]}')
    #     ax.set_xlabel('')
    #     ax.set_ylabel('')
    #     plt.tight_layout()
    #
    # plt.figure(figsize=(15, 15))
    # plt.rc('font', size=8)
    # plt.rc('font', family='Arial')
    #
    # # Iterate over pairs of columns in theta['data']
    # for i in range(7):
    #     for j in range(i + 1, 7):
    #         print(f'Parameter {theta["parameters"][i]} vs Parameter {theta["parameters"][j]}')
    #         plt.subplot(7, 7, i * 7 + j + 1)
    #         hist, xedges, yedges = np.histogram2d(theta['data'][:, i], theta['data'][:, j], bins=50)
    #         plt.imshow(hist.T, origin='lower', interpolation='bilinear', cmap='viridis', aspect='auto',
    #                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    #         plt.colorbar()
    #         plt.xlabel(f'{theta["parameters"][i]}')
    #         plt.ylabel(f'{theta["parameters"][j]}')
    #         plt.title(f'{theta["parameters"][i]} vs {theta["parameters"][j]}')
    #         plt.tight_layout()
    #
    # plt.show()

    # Load empirical data
    print('\n--- Loading empirical data.')
    start_time = time.time()
    emp_data = load_empirical_data(emp_data_path)
    print(f'\nFiles loaded: {len(emp_data["LFP"])}')
    end_time = time.time()
    print(f'Done in {end_time - start_time} s')

    # Compute features from empirical data
    print('\n--- Computing features from empirical data.')
    start_time = time.time()
    emp_data = compute_features(emp_data, chunk_size=5., method='catch22')
    end_time = time.time()
    print(f'Done in {end_time - start_time} s')

    # Create the Inference object, add the simulation data and train the model
    print('\n--- Training the regression model.')
    start_time = time.time()
    model = 'MLPRegressor'
    hyperparams = {'hidden_layer_sizes': (50,50), 'max_iter': 1000, 'tol': 1e-4, 'n_iter_no_change': 20,
                   'verbose': True}
    inference = ccpi.Inference(model=model, hyperparams=hyperparams)
    inference.add_simulation_data(X, theta['data'])
    inference.train(param_grid=None)
    end_time = time.time()
    print(f'Done in {end_time - start_time} s')

    # Compute predictions from the empirical data
    print('\n--- Computing predictions from empirical data.')
    start_time = time.time()
    emp_data = compute_predictions(inference, emp_data)
    end_time = time.time()
    print(f'Done in {end_time - start_time} s')

    # Save the data including predictions of all parameters
    if not os.path.exists('data'):
        os.makedirs('data')
    emp_data.to_pickle('data/emp_data_all.pkl')

    # Replace parameters of recurrent synaptic conductances with the ratio (E/I)_net
    E_I_net = emp_data['Predictions'].apply(lambda x: (x[0]/x[2]) / (x[1]/x[3]))
    others = emp_data['Predictions'].apply(lambda x: x[4:])
    emp_data['Predictions'] = (np.concatenate((E_I_net.values.reshape(-1,1),
                                               np.array(others.tolist())), axis=1)).tolist()

    # Save the data including predictions of (E/I)_net
    emp_data.to_pickle('data/emp_data_reduced.pkl')

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
    plt.savefig('figures/predictions.png')