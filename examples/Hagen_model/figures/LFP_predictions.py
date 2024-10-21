import os
import pickle
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro

# ncpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ncpi

# Parameters of LIF model simulations
sys.path.append(os.path.join(os.path.dirname(__file__), '../simulation/params'))

def lmer(df):
    # Activate pandas2ri
    pandas2ri.activate()

    # Load R libraries directly in R
    r('''
    library(dplyr)
    library(lme4)
    library(emmeans)
    library(ggplot2)
    library(repr)
    library(mgcv)
    ''')

    # Copy the dataframe
    df = df.copy()

    # The age=4 group is considered as the control group
    df['Group'] = df['Group'].apply(lambda x: 'HC' if x == 4 else str(x))

    # Remove the 'Data' and 'Features' columns
    df = df.drop(columns=['Data', 'Features'])

    # Filter out 'HC' from the list of unique groups
    groups = df['Group'].unique()
    groups = [group for group in groups if group != 'HC']

    # Create a list with the different group comparisons
    groups_comp = [f'{group}vsHC' for group in groups]

    # Remove rows where the variable is zero
    df = df[df['Predictions'] != 0]

    results = {}
    for label, label_comp in zip(groups, groups_comp):
        print(f'\n\n--- Group: {label}')
        # Filter DataFrame to obtain the desired groups
        df_pair = df[(df['Group'] == 'HC') | (df['Group'] == label)]
        ro.globalenv['df_pair'] = pandas2ri.py2rpy(df_pair)
        ro.globalenv['label'] = label
        # print(df_pair)

        # Convert columns to factors
        r(''' 
        df_pair$ID = as.factor(df_pair$ID)
        df_pair$Group = factor(df_pair$Group, levels = c(label, 'HC'))
        df_pair$Epoch = as.factor(df_pair$Epoch)
        df_pair$Sensor = as.factor(df_pair$Sensor)
        print(table(df_pair$Group))
        ''')

        # if table in R is empty for any group, skip the analysis
        if r('table(df_pair$Group)')[0] == 0 or r('table(df_pair$Group)')[1] == 0:
            results[label_comp] = pd.DataFrame({'p.value': [1], 'z.ratio': [0]})
        else:
            # Fit the linear models
            r('''
            mod00 = Predictions ~ Group  + (1 | ID)
            mod01 = Predictions ~ Group
            m00 <- lmer(mod00, data=df_pair)
            m01 <- lm(mod01, data=df_pair)
            print(summary(m00))
            print(summary(m01))
            ''')

            # BIC
            r('''
            all_models <- c('m00', 'm01')
            bics <- c(BIC(m00), BIC(m01))
            print(bics)
            index <- which.min(bics)
            mod_sel <- all_models[index]

            if (mod_sel == 'm00') {
                m_sel <- lmer(mod00, data=df_pair)
            }
            if (mod_sel == 'm01') {
                m_sel <- lm(mod01, data=df_pair)
            }                
            ''')

            # Compute the pairwise comparisons between groups
            r('''
            emm <- suppressMessages(emmeans(m_sel, specs=~Group))
            ''')

            r('''
            res <- pairs(emm, adjust='holm')
            df_res <- as.data.frame(res)
            print(df_res)
            ''')

            df_res_r = ro.r['df_res']

            # Convert the R DataFrame to a pandas DataFrame
            with (pandas2ri.converter + pandas2ri.converter).context():
                df_res_pd = pandas2ri.conversion.get_conversion().rpy2py(df_res_r)

            results[label_comp] = df_res_pd

    return results

# Debug
compute_firing_rate = True

# Random seed for numpy
np.random.seed(0)

# Number of samples to draw from the predictions for computing the firing rates
n_samples = 50
sim_params = {}
IDs = {}
firing_rates = {}

# Methods to plot
all_methods = ['catch22','power_spectrum_parameterization']

# Load data
all_IDs = {}
predictions_EI = {}
predictions_all = {}
ages = {}
for method in all_methods:
    try:
        data_EI = np.load(os.path.join('../data',method,'emp_data_reduced.pkl'), allow_pickle=True)
        data_all = np.load(os.path.join('../data',method,'emp_data_all.pkl'), allow_pickle=True)
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

    # Sample parameters for computing the firing rate
    if compute_firing_rate:
        sim_params[method] = np.zeros((7, len(np.unique(ages[method])), n_samples))
        IDs[method] = np.zeros((len(np.unique(ages[method])), n_samples))
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
        firing_rates[method] = np.zeros((len(np.unique(ages[method])), n_samples))
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
                sim = ncpi.Simulation(param_folder='../simulation/params',
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
                times['E'] = times['E'][times['E'] >= transient]
                rate = ((times['E'].size / (tstop - transient)) * 1000) / LIF_params['N_X'][0]
                firing_rates[method][i, sample] = rate

        # Normalize firing rates to the maximum value
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
    with open('data/firing_rates_preds.pkl', 'rb') as f:
        firing_rates = pickle.load(f)
    with open('data/IDs.pkl', 'rb') as f:
        IDs = pickle.load(f)

# Create a figure and set its properties
fig = plt.figure(figsize=(7.5, 4.), dpi=300)
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

# Plots
for row in range(2):
    for col in range(5):
        ax = fig.add_axes([0.09 + col * 0.19, 0.52 - row * 0.42, 0.14, 0.4])
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

                # Boxplot
                box = ax.boxplot(data_plot, positions=[age], showfliers=False,
                                 widths=0.9, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                                 whiskerprops=dict(color='black', linewidth=0.5),
                                 capprops=dict(color='black', linewidth=0.5),
                                 boxprops=dict(linewidth=0.5))
                for patch in box['boxes']:
                    patch.set_facecolor(cmap(i / len(np.unique(ages[method]))))

                # # Debug: plot samples selected for the firing rate over the parameter predictions
                # if compute_firing_rate:
                #     if 0 < col < 4:
                #         ax.scatter([age]*n_samples, sim_params[method][col+3, i, :], color='black', s=0.5, zorder = 3)
                #     elif col == 0:
                #         ax.scatter([age]*n_samples, (sim_params[method][0, i, :]/sim_params[method][2, i, :]) /
                #                    (sim_params[method][1, i, :]/sim_params[method][3, i, :]),
                #                    color='black', s=2, zorder = 3)

            # LMER analysis
            print('\n--- LMER analysis.')
            data_EI = np.load(os.path.join('../data', method, 'emp_data_reduced.pkl'), allow_pickle=True)
            # Pick only ages >= 4
            data_EI = data_EI[data_EI['Group'] >= 4]

            # Insert correct predictions for the lmer analysis (this should be improved in the future)
            if col < 4:
                data_EI['Predictions'] = predictions_EI[method][:, col]
            else:
                # Create a DataFrame with the firing rates
                data_fr = {'Group': np.repeat(np.unique(ages[method]), n_samples),
                           'ID': IDs[method].flatten(),
                           'Predictions': firing_rates[method].flatten(),
                           # unused columns
                           'Data': np.zeros(firing_rates[method].size),
                           'Epoch': np.arange(firing_rates[method].size),
                           'Sensor': np.zeros(firing_rates[method].size),
                           'Features': np.zeros(firing_rates[method].size)}
                data_EI = pd.DataFrame(data_fr)
            lmer_res = lmer(data_EI)

            # Add p-values to the plot
            y_max = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            delta = (y_max - y_min) * 0.1

            groups = ['5', '6', '7', '8', '9', '10', '11', '12']
            for i, group in enumerate(groups):
                p_value = lmer_res[f'{group}vsHC']['p.value']
                if p_value.empty:
                    continue

                # Significance levels
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
                    offset = 0

                ax.text(0.5*i+4.5, y_max + delta*i + delta*0.1 + offset, f'{pp}', ha='center',
                        fontsize=8 if pp != 'n.s.' else 7)
                ax.plot([4, 5+i], [y_max + delta*i, y_max + delta*i], color='black', linewidth=0.5)

            # Change y-lim
            ax.set_ylim([y_min, y_max + delta*(len(groups))])

        except:
            pass

        # Titles
        if row == 0:
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

        # Y-axis labels
        if col == 0:
            ax.set_ylabel(y_labels[row])
            if row == 0:
                ax.yaxis.set_label_coords(-0.3, 0.5)
            else:
                ax.yaxis.set_label_coords(-0.3, 0.5)

# Save the figure
plt.savefig('LFP_predictions.png', bbox_inches='tight')
# plt.show()