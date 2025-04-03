import os
import pickle
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

if __name__ == "__main__":
    # Load the configuration file that stores all file paths used in the script
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_features_path']

    # Analyze parameters of the simulation data
    for method in ['catch22']:
        print(f'\n\n--- Method: {method}')
        # Load parameters of the model (theta)
        print('\n--- Loading simulation data.')
        start_time = time.time()
        theta = load_simulation_data(os.path.join(sim_file_path, method, 'sim_theta'))
        end_time = time.time()
        print(f'Samples loaded: {len(theta["data"])}')
        print(f'Done in {(end_time - start_time)/60.} min')

        # Plot some statistics of the simulation data
        plt.figure(dpi = 300)
        plt.rc('font', size=8)
        plt.rc('font', family='Arial')

        # 1D histograms
        for param in range(7):
            print(f'Parameter {theta["parameters"][param]}')
            plt.subplot(2,4,param+1)
            ax = sns.histplot(theta['data'][:,param], kde=True, bins=50, color='blue')
            ax.set_title(f'Parameter {theta["parameters"][param]}')
            ax.set_xlabel('')
            ax.set_ylabel('')
            plt.tight_layout()

        plt.figure(figsize=(15, 15))
        plt.rc('font', size=8)
        plt.rc('font', family='Arial')

        # 2D histograms
        for i in range(7):
            for j in range(i + 1, 7):
                print(f'Parameter {theta["parameters"][i]} vs Parameter {theta["parameters"][j]}')
                plt.subplot(7, 7, i * 7 + j + 1)
                hist, xedges, yedges = np.histogram2d(theta['data'][:, i], theta['data'][:, j], bins=50)
                plt.imshow(hist.T, origin='lower', interpolation='bilinear', cmap='viridis', aspect='auto',
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                plt.colorbar()
                plt.xlabel(f'{theta["parameters"][i]}')
                plt.ylabel(f'{theta["parameters"][j]}')
                plt.title(f'{theta["parameters"][i]} vs {theta["parameters"][j]}')
                plt.tight_layout()

        plt.show()