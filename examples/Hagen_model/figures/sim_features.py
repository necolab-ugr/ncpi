import os
import pickle
import json
import numpy as np
import pandas as pd
# from rpy2.robjects import pandas2ri, r
# import rpy2.robjects as ro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt

# def lm(df):
#     # Activate pandas2ri
#     pandas2ri.activate()
#
#     # Load R libraries directly in R
#     r('''
#     library(dplyr)
#     library(lme4)
#     library(emmeans)
#     library(ggplot2)
#     library(repr)
#     library(mgcv)
#     ''')
#
#     # Copy the dataframe
#     df = df.copy()
#
#     # The bin=0 group is considered as the control group
#     df['Group'] = df['Group'].apply(lambda x: 'HC' if x == 0 else str(x))
#
#     # Filter out 'HC' from the list of unique groups
#     groups = df['Group'].unique()
#     groups = [group for group in groups if group != 'HC']
#
#     # Create a list with the different group comparisons
#     groups_comp = [f'{group}vsHC' for group in groups]
#
#     # Remove rows where the variable is zero
#     df = df[df['Feature'] != 0]
#
#     results = {}
#     for label, label_comp in zip(groups, groups_comp):
#         print(f'\n\n--- Group: {label}')
#         # Filter DataFrame to obtain the desired groups
#         df_pair = df[(df['Group'] == 'HC') | (df['Group'] == label)]
#         ro.globalenv['df_pair'] = pandas2ri.py2rpy(df_pair)
#         ro.globalenv['label'] = label
#         # print(df_pair)
#
#         # Convert columns to factors
#         r('''
#         df_pair$Group = factor(df_pair$Group, levels = c(label, 'HC'))
#         print(table(df_pair$Group))
#         ''')
#
#         # if table in R is empty for any group, skip the analysis
#         if r('table(df_pair$Group)')[0] == 0 or r('table(df_pair$Group)')[1] == 0:
#             results[label_comp] = pd.DataFrame({'p.value': [1], 'z.ratio': [0]})
#         else:
#             # Fit the linear model
#             r('''
#             mod = Feature ~ Group
#             m <- lm(mod, data=df_pair)
#             print(summary(m))
#             ''')
#
#             # Compute the pairwise comparisons between groups
#             r('''
#             emm <- suppressMessages(emmeans(m, specs=~Group))
#             ''')
#
#             r('''
#             res <- pairs(emm, adjust='holm')
#             df_res <- as.data.frame(res)
#             print(df_res)
#             ''')
#
#             df_res_r = ro.r['df_res']
#
#             # Convert the R DataFrame to a pandas DataFrame
#             with (pandas2ri.converter + pandas2ri.converter).context():
#                 df_res_pd = pandas2ri.conversion.get_conversion().rpy2py(df_res_r)
#
#             results[label_comp] = df_res_pd
#
#     return results

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
    data : ndarray
        Simulation data loaded from the file.
    """

    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f'Loaded file: {file_path}')

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

    except Exception as e:
        print(f'Error loading file: {file_path}')
        print(e)

    return data

# Load the configuration file that stores all file paths used in the script
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)
sim_file_path = config['simulation_features_path']
# emp_data_path = config['LFP_development_data_path']

# Dictionaries to store the features and parameters
features = {}
parameters = {'catch22':{},
              'power_spectrum_parameterization_1':{}}
# emp = {}
# ages = {}

# Iterate over the methods used to compute the features
for method in ['catch22', 'power_spectrum_parameterization_1']:
    print(f'\n\n--- Method: {method}')
    try:
        # Load simulation data
        print('\n--- Loading simulation data.')
        theta = load_simulation_data(os.path.join(sim_file_path, method, 'sim_theta'))
        X = load_simulation_data(os.path.join(sim_file_path, method, 'sim_X'))
        print(f'Samples loaded: {len(theta["data"])}')
    except:
        print(f'Error loading data for {method}.')
        # Fake data
        theta = {'data': np.ones((10,7))}
        X = np.zeros((10,22)) if method == 'catch22' else np.zeros((10, 1))

    # try:
    #     # Load empirical data
    #     data_EI = np.load(os.path.join('../data', method, 'emp_data_reduced.pkl'), allow_pickle=True)
    #     ages[method] = np.array(data_EI['Group'].tolist())
    #     # Pick only ages >= 4
    #     data_EI = data_EI[data_EI['Group'] >= 4]
    #     ages[method] = ages[method][ages[method] >= 4]
    # except:
    #     print(f'Error loading empirical data for {method}.')
    #     # Fake data
    #     data_EI = pd.DataFrame({'Features': [X[i] for i in range(10)]})
    #     ages[method] = np.arange(20)

    # Remove nan features from simulation data
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ii = np.where(~np.isnan(X).any(axis=1))[0]
    X = X[ii]
    theta['data'] = theta['data'][ii]
    print(f'Number of samples after removing nan features: {len(X)}')

    # # Remove nan features from empirical data
    # if np.array(data_EI['Features'].tolist()).ndim == 1:
    #     ii = np.where(~np.isnan(np.array(data_EI['Features'].tolist())))[0]
    # else:
    #     ii = np.where(~np.isnan(np.array(data_EI['Features'].tolist())).any(axis=1))[0]
    # data_EI = data_EI.iloc[ii]
    # ages[method] = ages[method][ii]

    # Collect parameters
    parameters[method]['E_I'] = ((theta['data'][:, 0] / theta['data'][:, 2]) /
                                 (theta['data'][:, 1] / theta['data'][:, 3]))
    parameters[method]['tau_syn_exc'] = theta['data'][:, 4]
    parameters[method]['tau_syn_inh'] = theta['data'][:, 5]
    parameters[method]['J_syn_ext'] = theta['data'][:, 6]

    # Collect features
    if method == 'catch22':
        features['dfa'] = X[:, catch22_names.index('SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1')]
        features['rs_range'] = X[:, catch22_names.index('SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1')]
        features['high_fluct'] = X[:, catch22_names.index('MD_hrv_classic_pnn40')]
        # emp['dfa'] = np.array(data_EI['Features'].apply(
        #     lambda x: x[catch22_names.index('SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1')]).tolist())
        # emp['rs_range'] = np.array(data_EI['Features'].apply(
        #     lambda x: x[catch22_names.index('SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1')]).tolist())
        # emp['high_fluct'] = np.array(data_EI['Features'].apply(
        #     lambda x: x[catch22_names.index('MD_hrv_classic_pnn40')]).tolist())
    elif method == 'power_spectrum_parameterization_1':
        features['slope'] = X[:,0]
        # emp['slope'] = np.array(data_EI['Features'].tolist())

# Create a figure and set its properties
fig = plt.figure(figsize=(7.5, 4.5), dpi=300)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Labels for the parameters
# param_labels = [r'$E/I$', r'$\tau_{syn}^{exc}$ (ms)', r'$\tau_{syn}^{inh}$ (ms)',
#           r'$J_{syn}^{ext}$ (nA)', 'LFP data']
param_labels = [r'$E/I$', r'$\tau_{syn}^{exc}$ (ms)', r'$\tau_{syn}^{inh}$ (ms)',
          r'$J_{syn}^{ext}$ (nA)']

# Define 4 colormaps for simulation data
cmaps = ['Blues', 'Greens', 'Reds', 'Purples']

# Define a colormap for empirical data
# cmap = plt.colormaps['viridis']

# Plots
for row in range(4):
    for col in range(4):
        ax = fig.add_axes([0.09 + col * 0.23, 0.76 - row * 0.21, 0.18, 0.19])

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
        else:
            feat = 'slope'
            method = 'power_spectrum_parameterization_1'

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
                    maxp = 12.0
                elif col == 1:
                    minp = 0.1
                    maxp = 2.0
                elif col == 2:
                    minp = 1.
                    maxp = 8.0
                elif col == 3:
                    minp = 10.
                    maxp = 40.
                ii = np.where((parameters[method][param] >= minp) & (parameters[method][param] <= maxp))[0]
                n_bins = np.unique(parameters[method][param][ii]).shape[0]
                if n_bins > 15:
                    n_bins = 15
                bins = np.linspace(minp, maxp, n_bins)

                bin_features = []
                group =  []
                for jj in range(len(bins) - 1):
                    pos = np.where((parameters[method][param] >= bins[jj]) & (parameters[method][param] < bins[jj + 1]))[0]
                    bin_features.extend([features[feat][pos]])
                    group.extend([jj for _ in range(len(pos))])

                    # Clip the data between the 5 % and 95 % quantiles
                    q1, q3 = np.percentile(features[feat][pos], [5, 95])
                    clipped_data = features[feat][pos][(features[feat][pos] >= q1) & (features[feat][pos] <= q3)]

                    # Violin plot
                    violin = ax.violinplot(clipped_data, positions=[jj], widths=0.9, showextrema=False)

                    for pc in violin['bodies']:
                        pc.set_facecolor(plt.get_cmap(cmaps[row])(jj / (len(bins) - 1)))
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.8)
                        pc.set_linewidth(0.2)

                    # violin['cmedians'].set_linewidth(0.6)
                    # violin['cmedians'].set_color('red')

                    # Boxplot
                    box = ax.boxplot(features[feat][pos], positions=[jj], showfliers=False,
                                     widths=0.5, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                                     whiskerprops=dict(color='black', linewidth=0.5),
                                     capprops=dict(color='black', linewidth=0.5),
                                     boxprops=dict(linewidth=0.5, facecolor=(0, 0, 0, 0)))

                    for patch in box['boxes']:
                        patch.set_linewidth(0.2)

                # Eta squared
                print(f'\n--- Eta squared for {feat} and {param}.')
                # Create the dataframe
                df = pd.DataFrame({'Feature': np.concatenate(bin_features),
                                   'Group': np.array(group)})

                # Perform ANOVA
                model = ols('Feature ~ C(Group)', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)

                # Compute eta squared
                ss_between = anova_table['sum_sq']['C(Group)']  # Sum of squares for the groups
                ss_total = anova_table['sum_sq'].sum()  # Total sum of squares
                eta_squared = ss_between / ss_total
                print(f"Eta squared: {eta_squared}")

                # Plot eta squared
                y_max = ax.get_ylim()[1]
                y_min = ax.get_ylim()[0]
                delta = (y_max - y_min) * 0.1
                ax.text((len(np.unique(group))-1)/2., y_max + delta/4.,
                        f'$\eta^2$ = {eta_squared:.3f}' if eta_squared > 0.001 else f'$\eta^2$ < 0.001',
                        ha='center', fontsize=8, color = 'black' if eta_squared >= .05 else 'red')

                # Change y-lim
                ax.set_ylim([y_min, y_max + 2 * delta])

                # # LM analysis
                # print('\n--- LM analysis.')
                # # Create the dataframe
                # df = pd.DataFrame({'Feature': np.concatenate(bin_features),
                #                    'Group': np.array(group)})
                # lmer_res = lm(df)
                #
                # # Add p-values to the plot
                # y_max = ax.get_ylim()[1]
                # y_min = ax.get_ylim()[0]
                # delta = (y_max - y_min) * 0.1
                #
                # if col == 0:
                #     groups = ['3', '6', '9','12']
                # elif col == 1 or col == 2:
                #     groups = ['2', '4', '6']
                # else:
                #     groups = ['2', '4']
                #
                # for i, group in enumerate(groups):
                #     p_value = lmer_res[f'{group}vsHC']['p.value']
                #     if p_value.empty:
                #         continue
                #
                #     # Significance levels
                #     if p_value[0] < 0.05 and p_value[0] >= 0.01:
                #         pp = '*'
                #     elif p_value[0] < 0.01 and p_value[0] >= 0.001:
                #         pp = '**'
                #     elif p_value[0] < 0.001 and p_value[0] >= 0.0001:
                #         pp = '***'
                #     elif p_value[0] < 0.0001:
                #         pp = '****'
                #     else:
                #         pp = 'n.s.'
                #
                #     if pp != 'n.s.':
                #         offset = -delta*0.2
                #     else:
                #         offset = delta*0.05
                #
                #     ax.text(int(group)/2., y_max + delta*i + delta*0.05 + offset,
                #             f'{pp}', ha='center', fontsize=8 if pp != 'n.s.' else 7)
                #     ax.plot([0, int(group)], [y_max + delta*i, y_max + delta*i], color='black',
                #             linewidth=0.5)
                #
                # # Change y-lim
                # ax.set_ylim([y_min, y_max + delta*(len(groups))])

            except:
                pass

        # # Empirical data
        # if col == 4:
        #     try:
        #         for i, age in enumerate(np.unique(ages[method])):
        #             idx = np.where(ages[method] == age)[0]
        #             data_plot = emp[feat][idx]
        #
        #             # Boxplot
        #             box = ax.boxplot(data_plot, positions=[age], showfliers=False,
        #                              widths=0.9, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
        #                              whiskerprops=dict(color='black', linewidth=0.5),
        #                              capprops=dict(color='black', linewidth=0.5),
        #                              boxprops=dict(linewidth=0.5))
        #             for patch in box['boxes']:
        #                 patch.set_facecolor(cmap(i / len(np.unique(ages[method]))))
        #     except:
        #         pass

        # Labels
        if col < 4 and row == 3:
            if col == 0:
                step = 4
            elif col == 1 or col == 2:
                step = 3
            else:
                step = 2

            ax.set_xticks(np.arange(0,len(bins) - 1,step))
            ax.set_xticklabels(['%.2f' % ((bins[jj] + bins[jj + 1]) / 2) for jj in np.arange(0,len(bins) - 1,step)],
                               fontsize = 8)
            ax.set_xlabel(param_labels[col])

        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        # if col == 4 and row == 3:
        #     # X-axis labels
        #     try:
        #         ax.set_xticks(np.unique(ages[method])[::2])
        #         ax.set_xticklabels([f'{str(i)}' for i in np.unique(ages[method])[::2]],fontsize = 8)
        #     except:
        #         pass
        #
        #     ax.set_xlabel('Postnatal days')

        if col == 0:
            ax.yaxis.set_label_coords(-0.35, 0.5)
            if row == 0:
                ax.set_ylabel(r'$dfa$')
            elif row == 1:
                ax.set_ylabel(r'$rs\ range$')
            elif row == 2:
                ax.set_ylabel(r'$high\ fluct.$')
            else:
                ax.set_ylabel(r'$1/f$' + ' ' + r'$slope$')

# Save the figure
plt.savefig('sim_features.png', bbox_inches='tight')
# plt.show()
