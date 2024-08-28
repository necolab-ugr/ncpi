import pickle
import sys
import os
import numpy as np
from matplotlib import pyplot as plt

# LIF network model
sys.path.append(os.path.join(os.path.dirname(__file__), '../simulation'))
import LIF_simulation

# Load data
data_EI = np.load('../data/emp_data_reduced.pkl', allow_pickle=True)
data_all = np.load('../data/emp_data_all.pkl', allow_pickle=True)
predictions_EI = np.array(data_EI['Predictions'].tolist())
predictions_all = np.array(data_all['Predictions'].tolist())
ages = np.array(data_EI['Group'].tolist())

# Create a figure and set its properties
fig = plt.figure(figsize=(7.5, 4.5), dpi=300)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Titles for the subplots
titles = [r'$E/I$', r'$\tau_{syn}^{exc}$ (ms)', r'$\tau_{syn}^{inh}$ (ms)',
          r'$J_{syn}^{ext}$ (nA)', r'$fr$ (spikes/s)']

# Define a colormap
cmap = plt.colormaps['viridis']

# Sample parameters for computing the firing rate
n_samples = 10 # Number of samples to draw from the predictions
sim_params = np.zeros((7, len(np.unique(ages)), n_samples))
for param in range(4):
    for i, age in enumerate(np.unique(ages)):
        idx = np.where(ages == age)[0]

        # Randomly sample some predictions within the first and third quartile
        q1, q3 = np.percentile(predictions_EI[idx, param], [25, 75])

        # Check if the quartiles are not NaN
        if not np.isnan(q1) and not np.isnan(q3):
            within_quartiles = np.where((predictions_EI[idx, param] >= q1) & (predictions_EI[idx, param] <= q3))[0]

            # Check within_quartiles is not empty
            if len(within_quartiles) > 0:
                # Randomly sample n_samples from within_quartiles
                idx_samples = within_quartiles[np.random.randint(0, len(within_quartiles), n_samples)]
                # E/I
                if param == 0:
                    for j in range(4):
                        sim_params[j, i, :] = predictions_all[idx, j][idx_samples]
                # tau_syn_exc, tau_syn_inh, J_syn_ext
                else:
                    sim_params[param+3, i, :] = predictions_EI[idx, param][idx_samples]

# Firing rates
fr = np.zeros((len(np.unique(ages)), n_samples))

# Change to folder simulation
os.chdir('../simulation')

for i in range(len(np.unique(ages))):
    for j in range(n_samples):
        print(f'\nSimulating age {np.unique(ages)[i]}, sample {j}')

        # Create the network and run the simulation
        sim = LIF_simulation.LIF_simulation(params=sim_params[:, i, j])
        try:
            sim.simulate(n_threads=64)
        except:
            print('Simulation failed')
            continue

        # Find the folder with simulation data
        folder = [f for f in os.listdir('LIF_simulations') if f != 'H_YX'][0]

        # Load firing rates
        lif_mean_nu_X = pickle.load(open(f'LIF_simulations/{folder}/lif_mean_nu_X', 'rb'))

        # Load parameters of the network model
        LIF_params = pickle.load(open(f'LIF_simulations/{folder}/LIF_params', 'rb'))

        # Compute firing rate of the excitatory neuron population
        fr[i,j] = lif_mean_nu_X['E'] / LIF_params['N_X'][0]

        # Remove the folder with simulation data
        os.system(f'rm -r LIF_simulations/{folder}')

# Change to folder figures
os.chdir('../figures')

# Plots
for row in range(3):
    for col in range(5):
        ax = fig.add_axes([0.08 + col * 0.19, 0.68 - row * 0.29, 0.14, 0.24])

        # Plot parameter predictions and firing rates as a function of age
        if row == 0:
            for i, age in enumerate(np.unique(ages)):
                idx = np.where(ages == age)[0]
                if col < 4:
                    data_plot = predictions_EI[idx, col]
                else:
                    data_plot = fr[i, :]

                box = ax.boxplot(data_plot, positions=[age], showfliers=False,
                                 widths=0.9, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                                 whiskerprops=dict(color='black', linewidth=0.5),
                                 capprops=dict(color='black', linewidth=0.5),
                                 boxprops=dict(linewidth=0.5))
                for patch in box['boxes']:
                    patch.set_facecolor(cmap(i / len(np.unique(ages))))

                # Plot samples selected to calculate the firing rate
                if 0 < col < 4 and age!=2:
                    ax.scatter([age]*n_samples, sim_params[col+3, i, :], color='black', s=2, zorder = 3)
                elif col == 0 and age!=2:
                    ax.scatter([age]*n_samples, (sim_params[0, i, :]/sim_params[2, i, :]) /
                               (sim_params[1, i, :]/sim_params[3, i, :]),
                               color='black', s=2, zorder = 3)

        # Titles
        if row == 0:
            ax.set_title(titles[col])

        # X-axis labels
        ax.set_xticks(np.unique(ages)[::2])
        ax.set_xticklabels([f'{str(i)}' for i in np.unique(ages)[::2]])
        if row == 2:
            ax.set_xlabel('Postnatal days')

        # Y-axis labels
        if col == 0:
            if row == 0:
                ax.set_ylabel(r'$catch22$')
            elif row == 1:
                ax.set_ylabel(r'$1/f$'+' '+r'$slope$')
            else:
                ax.set_ylabel(r'$fE/I$')

# Save the figure
plt.savefig('LFP_predictions.png', bbox_inches='tight')