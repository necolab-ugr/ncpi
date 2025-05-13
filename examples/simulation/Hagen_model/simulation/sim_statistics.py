import os
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from ncpi import tools

# Choose to either download data from Zenodo (True) or load it from a local path (False).
# Important: the zenodo downloads will take a while, so if you have already downloaded the data, set this to False and
# configure the zenodo_dir variable to point to the local path where the data is stored.
zenodo_dw_sim = True # simulation data

# Zenodo URL that contains the simulation data and ML models (used if zenodo_dw_sim is True)
zenodo_URL_sim = "https://zenodo.org/api/records/15351118"

# Paths to zenodo files
zenodo_dir_sim = "zenodo_sim_files"

# Download simulation data and ML models
if zenodo_dw_sim:
    print('\n--- Downloading simulation data and ML models from Zenodo.')
    start_time = time.time()
    tools.download_zenodo_record(zenodo_URL_sim, download_dir=zenodo_dir_sim)
    end_time = time.time()
    print(f"All files downloaded in {(end_time - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    # Analyze parameters of the simulation data
    for method in ['catch22']:
        print(f'\n\n--- Method: {method}')

        # Load parameters of the model (theta) and features (X) from simulation data
        print('\n--- Loading simulation data.')
        start_time = time.time()

        try:
            with open(os.path.join(zenodo_dir_sim, 'data', method, 'sim_theta'), 'rb') as file:
                theta = pickle.load(file)
            with open(os.path.join(zenodo_dir_sim, 'data', method, 'sim_X'), 'rb') as file:
                X = pickle.load(file)
        except Exception as e:
            print(f"Error loading simulation data: {e}")

        # Print info
        print('theta:')
        for key, value in theta.items():
            if isinstance(value, np.ndarray):
                print(f'--Shape of {key}: {value.shape}')
            else:
                print(f'--{key}: {value}')
        print(f'Shape of X: {X.shape}')

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