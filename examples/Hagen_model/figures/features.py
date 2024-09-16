import os
import pickle
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

# Load the configuration file that stores all file paths used in the script
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)
sim_file_path = config['simulation_features_path']
emp_data_path = config['LFP_development_data_path']

# Dictionaries to store the features and parameters
features = {}
parameters = {'catch22':{},
              'power_spectrum_parameterization':{},
              'fEI':{}}
emp = {}
ages = {}

# Iterate over the methods used to compute the features
for method in ['catch22', 'power_spectrum_parameterization', 'fEI']:
    print(f'\n\n--- Method: {method}')
    try:
        # Load simulation data
        print('\n--- Loading simulation data.')
        theta = load_simulation_data(os.path.join(sim_file_path, method, 'sim_theta'))
        X = load_simulation_data(os.path.join(sim_file_path, method, 'sim_X'))
        print(f'Samples loaded: {len(theta["data"])}')
        # Load empirical data
        data_EI = np.load(os.path.join('../data', method, 'emp_data_reduced.pkl'), allow_pickle=True)
        ages[method] = np.array(data_EI['Group'].tolist())
        # Pick only ages >= 4
        data_EI = data_EI[data_EI['Group'] >= 4]
        ages[method] = ages[method][ages[method] >= 4]
    except:
        print(f'Error loading data for {method}.')
        # Fake data
        theta = {'data': np.ones((10,7))}
        X = np.zeros((10,22)) if method == 'catch22' else np.zeros((10, 1))
        data_EI = pd.DataFrame({'Features': [X[i] for i in range(10)]})
        ages = np.arange(10)

    # Remove nan features from simulation data
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ii = np.where(~np.isnan(X).any(axis=1))[0]
    X = X[ii]
    theta['data'] = theta['data'][ii]
    print(f'Number of samples after removing nan features: {len(X)}')

    # Remove nan features from empirical data
    if np.array(data_EI['Features'].tolist()).ndim == 1:
        ii = np.where(~np.isnan(np.array(data_EI['Features'].tolist())))[0]
    else:
        ii = np.where(~np.isnan(np.array(data_EI['Features'].tolist())).any(axis=1))[0]
    data_EI = data_EI.iloc[ii]
    ages[method] = ages[method][ii]

    # Collect parameters
    parameters[method]['E_I'] = ((theta['data'][:, 0] / theta['data'][:, 2]) /
                                 (theta['data'][:, 1] / theta['data'][:, 3]))
    parameters[method]['tau_syn_exc'] = theta['data'][:, 4]
    parameters[method]['tau_syn_inh'] = theta['data'][:, 5]
    parameters[method]['J_syn_ext'] = theta['data'][:, 6]

    # Collect features
    if method == 'catch22':
        features['dfa'] = X[:, 19]
        features['rs_range'] = X[:, 18]
        features['high_fluct'] = X[:, 6]
        emp['dfa'] = np.array(data_EI['Features'].apply(lambda x: x[19]).tolist())
        emp['rs_range'] = np.array(data_EI['Features'].apply(lambda x: x[18]).tolist())
        emp['high_fluct'] = np.array(data_EI['Features'].apply(lambda x: x[6]).tolist())
    elif method == 'power_spectrum_parameterization':
        features['slope'] = X
        emp['slope'] = np.array(data_EI['Features'].tolist())
    elif method == 'fEI':
        features['fEI'] = X
        emp['fEI'] = np.array(data_EI['Features'].tolist())

# Create a figure and set its properties
fig = plt.figure(figsize=(7.5, 4.5), dpi=300)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Labels for the parameters
param_labels = [r'$E/I$', r'$\tau_{syn}^{exc}$ (ms)', r'$\tau_{syn}^{inh}$ (ms)',
          r'$J_{syn}^{ext}$ (nA)', 'LFP data']

# Define 5 colormaps for simulation data
cmaps = ['Blues', 'Greens', 'Reds', 'Purples', 'YlOrBr']

# Define a colormap for empirical data
cmap = plt.colormaps['viridis']

# Plots
for row in range(5):
    for col in range(5):
        ax = fig.add_axes([0.1 + col * 0.18, 0.83 - row * 0.18, 0.13, 0.13])

        # Get the keys for the parameters and features
        if row == 0:
            feat = 'dfa'
            method = 'catch22'
        elif row == 1:
            feat = 'rs_range'
            method = 'catch22'
        elif row == 2:
            feat = 'high_fluct'
            method = 'catch22'
        elif row == 3:
            feat = 'slope'
            method = 'power_spectrum_parameterization'
        else:
            feat = 'fEI'
            method = 'fEI'

        if col == 0:
            param = 'E_I'
        elif col == 1:
            param = 'tau_syn_exc'
        elif col == 2:
            param = 'tau_syn_inh'
        elif col == 3:
            param = 'J_syn_ext'

        # Simulation data
        if col < 4:
            try:
                # Bins for the boxplots
                # n_bins = np.unique(parameters[method][param]).shape[0]
                # if n_bins > 10:
                #     n_bins = 10
                # bins = np.linspace(np.min(parameters[method][param]), 1.05*np.max(parameters[method][param]), n_bins)

                # Constraints for the bins
                if col == 0:
                    minp = 0.1
                    maxp = 15.0
                elif col == 1:
                    minp = 0.1
                    maxp = 2.0
                elif col == 2:
                    minp = 0.1
                    maxp = 8.0
                elif col == 3:
                    minp = 10.
                    maxp = 40.
                ii = np.where((parameters[method][param] >= minp) & (parameters[method][param] <= maxp))[0]
                n_bins = np.unique(parameters[method][param][ii]).shape[0]
                if n_bins > 15:
                    n_bins = 15
                bins = np.linspace(minp, maxp, n_bins)

                for jj in range(len(bins) - 1):
                    pos = np.where((parameters[method][param] >= bins[jj]) & (parameters[method][param] < bins[jj + 1]))[0]
                    box = ax.boxplot(features[feat][pos], positions=[jj], showfliers=False,
                                     widths=0.9, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                                     whiskerprops=dict(color='black', linewidth=0.5),
                                     capprops=dict(color='black', linewidth=0.5),
                                     boxprops=dict(linewidth=0.5))

                    for patch in box['boxes']:
                        patch.set_facecolor(plt.get_cmap(cmaps[row])(jj / (len(bins) - 1)))

            except:
                pass

        # Empirical data
        if col == 4:
            try:
                for i, age in enumerate(np.unique(ages[method])):
                    idx = np.where(ages[method] == age)[0]
                    data_plot = emp[feat][idx]

                    # Boxplot
                    box = ax.boxplot(data_plot, positions=[age], showfliers=False,
                                     widths=0.9, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                                     whiskerprops=dict(color='black', linewidth=0.5),
                                     capprops=dict(color='black', linewidth=0.5),
                                     boxprops=dict(linewidth=0.5))
                    for patch in box['boxes']:
                        patch.set_facecolor(cmap(i / len(np.unique(ages[method]))))
            except:
                pass

        # Labels
        if col < 4:
            ax.set_xticks(np.arange(0,len(bins) - 1,3))
            ax.set_xticklabels(['%.2f' % ((bins[jj] + bins[jj + 1]) / 2) for jj in np.arange(0,len(bins) - 1,3)],
                               fontsize = 6)

        if col < 4 and row == 4:
            ax.set_xlabel(param_labels[col])

        if col == 4:
            # X-axis labels
            try:
                ax.set_xticks(np.unique(ages[method])[::2])
                ax.set_xticklabels([f'{str(i)}' for i in np.unique(ages[method])[::2]])
            except:
                pass

        if col == 4 and row == 4:
            ax.set_xlabel('Postnatal days')

        if col == 0:
            ax.yaxis.set_label_coords(-0.5, 0.5)
            if row == 0:
                ax.set_ylabel(r'dfa')
            elif row == 1:
                ax.set_ylabel(r'$rs\_range$')
            elif row == 2:
                ax.set_ylabel(r'$high\_fluct$')
            elif row == 3:
                ax.set_ylabel(r'$1/f$' + ' ' + r'$slope$')
            else:
                ax.set_ylabel(r'$fE/I$')

# Save the figure
plt.savefig('features.png', bbox_inches='tight')
#plt.show()
