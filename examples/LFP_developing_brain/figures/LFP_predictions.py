import os
import pickle
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ncpi

# Folder with parameters of LIF model simulations
sys.path.append(os.path.join(os.path.dirname(__file__), '../../simulation/Hagen_model/simulation/params'))

# Path to the folder with prediction results
pred_results = '../data'

# Calculate new firing rates (True) or load them from file if they already exist (False). If firing rates do not
# exist, they will not be plotted.
compute_firing_rate = False

# Path to saved firing rates
fr_path = '/DATA/ML_models/4_var/MLP'

# Number of samples to draw from the predictions for computing the firing rates
n_samples = 50
sim_params = {}
IDs = {}
firing_rates = {}

# Methods to plot
all_methods = ['catch22','power_spectrum_parameterization_1']

# Select the statistical analysis method ('cohen', 'lmer')
statistical_analysis = 'cohen'

# Random seed for numpy
np.random.seed(0)

all_IDs = {}
predictions_EI = {}
predictions_all = {}
ages = {}
for method in all_methods:
    # Load data
    try:
        data_EI = np.load(os.path.join(pred_results,method,'emp_data_reduced.pkl'), allow_pickle=True)
        data_all = np.load(os.path.join(pred_results,method,'emp_data_all.pkl'), allow_pickle=True)
        all_IDs[method] = np.array(data_all['ID'].tolist())
        predictions_EI[method] = np.array(data_EI['Predictions'].tolist())
        predictions_all[method] = np.array(data_all['Predictions'].tolist())
        ages[method] = np.array(data_EI['Group'].tolist())

        # Pick only ages >= 4
        all_IDs[method] = all_IDs[method][ages[method] >= 4]
        predictions_EI[method] = predictions_EI[method][ages[method] >= 4, :]
        predictions_all[method] = predictions_all[method][ages[method] >= 4, :]
        ages[method] = ages[method][ages[method] >= 4]

    except:
        all_IDs[method] = []
        predictions_EI[method] = []
        predictions_all[method] = []
        ages[method] = []

    firing_rates[method] = np.zeros((len(np.unique(ages[method])), n_samples))
    IDs[method] = np.zeros((len(np.unique(ages[method])), n_samples))

    # Parameter sampling for computing the firing rates
    if compute_firing_rate:
        sim_params[method] = np.zeros((7, len(np.unique(ages[method])), n_samples))
        for param in range(4):
            for i, age in enumerate(np.unique(ages[method])):
                idx = np.where(ages[method] == age)[0]
                data_IDs = all_IDs[method][idx]
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
                        IDs[method][i, :] = data_IDs[idx_samples]
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
        for i, age in enumerate(np.unique(ages[method])):
            for sample in range(n_samples):
                print(f'\nComputing firing rate for {method} at age {age} and sample {sample}')
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
                sim = ncpi.Simulation(param_folder='../../Hagen_model/simulation/params',
                                      python_folder='../../Hagen_model/simulation/python',
                                      output_folder='../../Hagen_model/simulation/output')

                # Save parameters to a pickle file
                with open(os.path.join('../../simulation/Hagen_model/simulation/output', 'network.pkl'), 'wb') as f:
                    pickle.dump(LIF_params, f)

                # Run the simulation
                sim.simulate('simulation.py', 'simulation_params.py')

                # Load spike times
                with open(os.path.join('../../simulation/Hagen_model/simulation/output', 'times.pkl'), 'rb') as f:
                    times = pickle.load(f)

                # Load tstop
                with open(os.path.join('../../simulation/Hagen_model/simulation/output', 'tstop.pkl'), 'rb') as f:
                    tstop = pickle.load(f)

                # Transient period
                from analysis_params import KernelParams
                transient = KernelParams.transient

                # Mean firing rate of excitatory cells
                times['E'] = times['E'][times['E'] >= transient]
                rate = ((times['E'].size / (tstop - transient)) * 1000) / LIF_params['N_X'][0]
                firing_rates[method][i, sample] = rate

        # Normalize firing rates to the maximum value
        if len(firing_rates[method]) > 0:
            if np.max(firing_rates[method]) > 0:
                firing_rates[method] /= np.max(firing_rates[method])

# Save firing rates to file
if compute_firing_rate:
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/firing_rates_preds.pkl', 'wb') as f:
        pickle.dump(firing_rates, f)
    with open('data/IDs.pkl', 'wb') as f:
        pickle.dump(IDs, f)
else:
    try:
        with open(os.path.join(fr_path,'firing_rates_preds.pkl'), 'rb') as f:
            firing_rates = pickle.load(f)
        with open(os.path.join(fr_path,'IDs.pkl'), 'rb') as f:
            IDs = pickle.load(f)
    except FileNotFoundError:
        print('Firing rates not found.')
        pass

# Create a figure and set its properties
fig = plt.figure(figsize=(7.5, 5), dpi=300)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Titles for the subplots
titles = [r'$E/I$', r'$\tau_{syn}^{exc}$ (ms)', r'$\tau_{syn}^{inh}$ (ms)',
          r'$J_{syn}^{ext}$ (nA)', r'$Norm. fr$']
# y-axis labels
y_labels = [r'$catch22$', r'$1/f$'+' '+r'$slope$']

# Define a colormap
cmap = plt.colormaps['viridis']

# Add rectangles to each row
for row in range(2):
    ax = fig.add_axes([0.01, 0.53 - row * 0.52, 0.98, 0.45 if row == 0 else 0.52])
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='red' if row == 0 else 'blue', alpha=0.1))
    ax.set_xticks([])
    ax.set_yticks([])

# Plots
for row in range(2):
    for col in range(5):
        ax = fig.add_axes([0.08 + col * 0.19, 0.55 - row * 0.45, 0.14, 0.32])
        try:
            method = all_methods[row]
        except:
            method = all_methods[0]

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

                # # Boxplot
                # box = ax.boxplot(data_plot, positions=[age], showfliers=False,
                #                  widths=0.9, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                #                  whiskerprops=dict(color='black', linewidth=0.5),
                #                  capprops=dict(color='black', linewidth=0.5),
                #                  boxprops=dict(linewidth=0.5))
                # for patch in box['boxes']:
                #     patch.set_facecolor(cmap(i / len(np.unique(ages[method]))))

                # Clip the data between the 5 % and 95 % quantiles
                q1, q3 = np.percentile(data_plot, [5, 95])
                clipped_data = data_plot[(data_plot >= q1) & (data_plot <= q3)]

                # Violin plot
                violin = ax.violinplot(clipped_data, positions=[age], widths=0.9, showextrema=False)

                for pc in violin['bodies']:
                    pc.set_facecolor(cmap(i / len(np.unique(ages[method]))))
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.8)
                    pc.set_linewidth(0.2)

                # violin['cmedians'].set_linewidth(0.6)
                # violin['cmedians'].set_color('red')

                # Boxplot
                box = ax.boxplot(data_plot, positions=[age], showfliers=False,
                                 widths=0.5, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                                 whiskerprops=dict(color='black', linewidth=0.5),
                                 capprops=dict(color='black', linewidth=0.5),
                                 boxprops=dict(linewidth=0.5, facecolor=(0, 0, 0, 0)))

                for patch in box['boxes']:
                    patch.set_linewidth(0.2)

                # # Debug: plot samples selected for the firing rate over the parameter predictions
                # if compute_firing_rate:
                #     if 0 < col < 4:
                #         ax.scatter([age]*n_samples, sim_params[method][col+3, i, :], color='black', s=0.5, zorder = 3)
                #     elif col == 0:
                #         ax.scatter([age]*n_samples, (sim_params[method][0, i, :]/sim_params[method][2, i, :]) /
                #                    (sim_params[method][1, i, :]/sim_params[method][3, i, :]),
                #                    color='black', s=2, zorder = 3)

            # stat. analysis
            if statistical_analysis == 'lmer':
                print('\n--- Linear mixed model analysis.')
            elif statistical_analysis == 'cohen':
                print('\n--- Cohen\'s d analysis.')

            data_EI = np.load(os.path.join(pred_results, method, 'emp_data_reduced.pkl'), allow_pickle=True)
            # Pick only ages >= 4
            data_EI = data_EI[data_EI['Group'] >= 4]

            # Create a DataFrame with the predictions
            if col < 4:
                data_EI['Y'] = predictions_EI[method][:, col]
            else:
                # Create a DataFrame with the firing rates
                data_fr = {'Group': np.repeat(np.unique(ages[method]), n_samples),
                           'ID': IDs[method].flatten(),
                           'Y': firing_rates[method].flatten(),
                           'Epoch': np.arange(firing_rates[method].size),
                           'Sensor': np.zeros(firing_rates[method].size)}
                data_EI = pd.DataFrame(data_fr)

            # Transform the 'Group' column to string type
            data_EI['Group'] = data_EI['Group'].astype(str)

            # Remove nan values from Y column
            data_EI = data_EI[~np.isnan(data_EI['Y'])]

            # Compute the statistical analysis
            Analysis = ncpi.Analysis(data_EI)
            if statistical_analysis == 'lmer':
                stat_result = Analysis.lmer(control_group='4', 
                                         data_col='Y',
                                         other_col=['ID', 'Group', 'Epoch', 'Sensor'],
                                         data_index=-1,
                                         models = {'mod00': 'Y ~ Group + (1 | ID)',
                                                             'mod01': 'Y ~ Group'},
                                         bic_models=["mod00", "mod01"],
                                         anova_tests = None,
                                         specs= '~Group')
            elif statistical_analysis == 'cohen':
                stat_result = Analysis.cohend(control_group='4', data_col='Y',data_index=-1)

            # Add p-values to the plot
            y_max = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            delta = (y_max - y_min) * 0.1

            # groups = ['5', '6', '7', '8', '9', '10', '11', '12']
            groups = ['8', '9', '10', '11', '12']
            for i, group in enumerate(groups):
                if statistical_analysis == 'lmer':
                    p_value = stat_result[f'{group}vs4']['p.value']
                    if p_value.empty:
                        continue
                elif statistical_analysis == 'cohen':
                    d_value = stat_result[f'{group}vs4']['d']
                    if d_value.empty:
                        continue

                # Significance levels
                if statistical_analysis == 'lmer':
                    if p_value[0] < 0.05 and p_value[0] >= 0.01:
                        pp = '*'
                    elif p_value[0] < 0.01 and p_value[0] >= 0.001:
                        pp = '**'
                    elif p_value[0] < 0.001 and p_value[0] >= 0.0001:
                        pp = '***'
                    elif p_value[0] < 0.0001:
                        pp = '****'
                    else:
                        pp = 'n.s.'

                    if pp != 'n.s.':
                        offset = -delta*0.2
                    else:
                        offset = delta*0.05

                elif statistical_analysis == 'cohen':
                    pp = f'{d_value[0]:.2f}'
                    offset = 0.

                ax.text((int(groups[i]) - 4)/2. + 4, y_max + delta*i + delta*0.1 + offset,
                        f'{pp}', ha='center', fontsize=8 if pp != 'n.s.' else 7)
                ax.plot([4, int(groups[0])+i], [y_max + delta*i, y_max + delta*i], color='black',
                        linewidth=0.5)

            # Change y-lim
            ax.set_ylim([y_min, y_max + delta*(len(groups))])

        except:
            pass

        # Titles
        ax.set_title(titles[col])

        # X-axis labels
        try:
            if row == 1:
                ax.set_xticks(np.unique(ages[method])[::2])
                ax.set_xticklabels([f'{str(i)}' for i in np.unique(ages[method])[::2]])
                ax.set_xlabel('Postnatal days')
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
        except:
            pass

        # # Y-axis labels
        # if col == 0:
        #     ax.set_ylabel(y_labels[row], color = 'red' if row == 0 else 'blue', alpha = 0.5)
        #     if row == 0:
        #         ax.yaxis.set_label_coords(-0.3, 0.5)
        #     else:
        #         ax.yaxis.set_label_coords(-0.3, 0.5)

# Plot letters
ax = fig.add_axes([0., 0., 1., 1.])
ax.axis('off')
ax.text(0.015, 0.945, 'A', fontsize=12, fontweight='bold')
ax.text(0.015, 0.495, 'B', fontsize=12, fontweight='bold')

# Titles
ax.text(0.5, 0.94, y_labels[0], color = 'red', alpha = 0.5, fontsize = 10, ha='center')
ax.text(0.5, 0.49, y_labels[1], color = 'blue', alpha = 0.5, fontsize = 10, ha='center')

# Save the figure
plt.savefig('LFP_predictions.png', bbox_inches='tight')
# plt.show()
