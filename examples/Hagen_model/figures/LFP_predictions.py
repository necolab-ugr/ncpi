import os
import pickle
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ccpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ccpi

# Parameters of LIF model simulations
sys.path.append(os.path.join(os.path.dirname(__file__), '../simulation/params'))

def cohen_d(x, y):
    """ Compute Cohen's d effect size.

    Compute Cohen's d effect size for two independent samples.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.

    Returns
    -------
    d : float
        Cohen's d effect size.
    """

    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.var(x) + (ny-1)*np.var(y)) / dof)

# Debug
compute_firing_rate = True

# Number of samples to draw from the predictions for computing the firing rates
n_samples = 10
sim_params = {}
firing_rates = {}

# Load data
predictions_EI = {}
predictions_all = {}
ages = {}
for method in ['catch22','power_spectrum_parameterization', 'fEI']:
    try:
        data_EI = np.load(os.path.join('../data',method,'emp_data_reduced.pkl'), allow_pickle=True)
        data_all = np.load(os.path.join('../data',method,'emp_data_all.pkl'), allow_pickle=True)
        predictions_EI[method] = np.array(data_EI['Predictions'].tolist())
        predictions_all[method] = np.array(data_all['Predictions'].tolist())
        ages[method] = np.array(data_EI['Group'].tolist())
    except:
        predictions_EI[method] = []
        predictions_all[method] = []
        ages[method] = []

    # Sample parameters for computing the firing rate
    sim_params[method] = np.zeros((7, len(np.unique(ages[method])), n_samples))
    for param in range(4):
        for i, age in enumerate(np.unique(ages[method])):
            idx = np.where(ages[method] == age)[0]
            data_EI = predictions_EI[method][idx, param]
            data_EI = data_EI[~np.isnan(data_EI)]

            # Randomly sample some predictions within the first and third quartile
            q1, q3 = np.percentile(data_EI, [25, 75])

            # Check if the quartiles are not NaN
            if not np.isnan(q1) and not np.isnan(q3):
                within_quartiles = np.where((data_EI >= q1) & (data_EI <= q3))[0]

                # Check within_quartiles is not empty
                if len(within_quartiles) > 0:
                    # Randomly sample n_samples from within_quartiles
                    idx_samples = within_quartiles[np.random.randint(0, len(within_quartiles), n_samples)]
                    # E/I
                    if param == 0:
                        for j in range(4):
                            data_all = predictions_all[method][idx, j]
                            data_all = data_all[~np.isnan(data_all)]
                            sim_params[method][j, i, :] = data_all[idx_samples]
                    # tau_syn_exc, tau_syn_inh, J_syn_ext
                    else:
                        sim_params[method][param+3, i, :] = data_EI[idx_samples]

    # Firing rates
    firing_rates[method] = np.zeros((len(np.unique(ages[method])), n_samples))

    for i, age in enumerate(np.unique(ages[method])):
        for sample in range(n_samples):
            if compute_firing_rate:
                print(f'Computing firing rate for {method} at age {age} and sample {sample}')
                # Parameters of the model
                J_EE = sim_params[method][0, i, sample]
                J_IE = sim_params[method][1, i, sample]
                J_EI = sim_params[method][2, i, sample]
                J_II = sim_params[method][3, i, sample]
                tau_syn_E = sim_params[method][4, i, sample]
                tau_syn_I = sim_params[method][5, i, sample]
                J_ext = sim_params[method][6, i, sample]

                # Load LIF_params
                from network_params import LIF_params

                # Modify parameters
                LIF_params['J_YX'] = [[J_EE, J_IE], [J_EI, J_II]]
                LIF_params['tau_syn_YX'] = [[tau_syn_E, tau_syn_I],
                                            [tau_syn_E, tau_syn_I]]
                LIF_params['J_ext'] = J_ext

                # Create a Simulation object
                sim = ccpi.Simulation(param_folder='../simulation/params',
                                      python_folder='../simulation/python',
                                      output_folder='../simulation/output')

                # Save parameters to a pickle file
                with open(os.path.join('../simulation/output', 'network.pkl'), 'wb') as f:
                    pickle.dump(LIF_params, f)

                # Run the simulation
                sim.simulate('simulation.py', 'simulation_params.py')

                # Load spike times
                with open(os.path.join('../simulation/output', 'times.pkl'), 'rb') as f:
                    times = pickle.load(f)

                # Load tstop
                with open(os.path.join('../simulation/output', 'tstop.pkl'), 'rb') as f:
                    tstop = pickle.load(f)

                # Transient period
                from analysis_params import KernelParams
                transient = KernelParams.transient

                # Mean firing rate of excitatory cells
                rate = ((times['E'].size / (tstop - transient)) * 1000) / LIF_params['N_X'][0]
                firing_rates[method][i, sample] = rate


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

# Plots
for row in range(3):
    for col in range(5):
        ax = fig.add_axes([0.08 + col * 0.19, 0.68 - row * 0.29, 0.14, 0.24])
        if row == 0:
            method = 'catch22'
        elif row == 1:
            method = 'power_spectrum_parameterization'
        else:
            method = 'fEI'

        # Plot parameter predictions and firing rates as a function of age
        try:
            for i, age in enumerate(np.unique(ages[method])):
                idx = np.where(ages[method] == age)[0]
                if col < 4:
                    data_plot = predictions_EI[method][idx, col]
                else:
                    data_plot = firing_rates[method][i, :]

                # Remove NaNs
                data_plot = data_plot[~np.isnan(data_plot)]

                # Boxplot
                box = ax.boxplot(data_plot, positions=[age], showfliers=False,
                                 widths=0.9, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                                 whiskerprops=dict(color='black', linewidth=0.5),
                                 capprops=dict(color='black', linewidth=0.5),
                                 boxprops=dict(linewidth=0.5))
                for patch in box['boxes']:
                    patch.set_facecolor(cmap(i / len(np.unique(ages[method]))))

                # Debug: plot samples selected for the firing rate over the parameter predictions
                if 0 < col < 4:
                    ax.scatter([age]*n_samples, sim_params[method][col+3, i, :], color='black', s=2, zorder = 3)
                elif col == 0:
                    ax.scatter([age]*n_samples, (sim_params[method][0, i, :]/sim_params[method][2, i, :]) /
                               (sim_params[method][1, i, :]/sim_params[method][3, i, :]),
                               color='black', s=2, zorder = 3)

            # Statistical analysis
            if col < 4:
                data_analyse = predictions_EI[method][:, col]
                data_analyse = data_analyse[~np.isnan(data_analyse)]
                ages_analyse = ages[method][~np.isnan(predictions_EI[method][:, col])]
            else:
                data_analyse = firing_rates[method].flatten()
                ages_analyse = np.repeat(np.unique(ages[method]), n_samples)

            data = {'value': data_analyse, 'age': ages_analyse}
            df = pd.DataFrame(data)

            # Post-hoc test
            tukey = pairwise_tukeyhsd(endog=df['value'], groups=df['age'], alpha=0.05)
            # Plot the Tukey HSD results manually for age 4
            for comparison in tukey.summary().data[1:]:
                group1, group2, meandiff, p_adj, lower, upper, reject = comparison
                if group1 == 4 and group2 > 4 and reject:
                    y_max = ax.get_ylim()[1]
                    y_min = ax.get_ylim()[0]
                    ax.plot([group1, group2], [y_max, y_max], color='black', linewidth = 0.5)

                    # Significance levels
                    if p_adj < 0.05 and p_adj >= 0.01:
                        p_value = '*'
                    elif p_adj < 0.01 and p_adj >= 0.001:
                        p_value = '**'
                    elif p_adj < 0.001 and p_adj >= 0.0001:
                        p_value = '***'
                    elif p_adj < 0.0001:
                        p_value = '****'
                    else:
                        p_value = 'n.s.'

                    # Compute Cohen's d effect size
                    data_group1 = df[df['age'] == group1]['value']
                    data_group2 = df[df['age'] == group2]['value']
                    d = cohen_d(data_group1, data_group2)

                    # Add the significance level and Cohen's d effect size to the plot
                    ax.text((group1 + group2) / 2, y_max - (y_max-y_min) * 0.005, p_value, ha='center', va='center',
                            color='black', fontsize=6)
                    ax.text((group1 + group2) / 2, y_max + (y_max-y_min) * 0.015,
                            f'                                     d = {d:.2f}', ha='center',
                            va='center', color='black', fontsize=3)

                    # Plot confidence interval
                    ax.text((group1 + group2) / 3, y_max + (y_max-y_min) * 0.015,
                            f'CI = {upper-lower:.5f}',
                            ha='center', va='center', color='black', fontsize=3)

        except:
            pass

        # Titles
        if row == 0:
            ax.set_title(titles[col])

        # X-axis labels
        try:
            ax.set_xticks(np.unique(ages[method])[::2])
            ax.set_xticklabels([f'{str(i)}' for i in np.unique(ages[method])[::2]])
            if row == 2:
                ax.set_xlabel('Postnatal days')
        except:
            pass

        # Y-axis labels
        if col == 0:
            if row == 0:
                ax.set_ylabel(r'$catch22$')
            elif row == 1:
                ax.set_ylabel(r'$1/f$'+' '+r'$slope$')
            else:
                ax.set_ylabel(r'$fE/I$')

# Save the figure
# plt.savefig('LFP_predictions.png', bbox_inches='tight')
plt.show()