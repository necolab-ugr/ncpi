import os
import numpy as np
import matplotlib.pyplot as plt

# Path to the folder with prediction results
pred_results = '../data'

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

# Dictionaries to store the features and ages
emp = {}
ages = {}

# Iterate over the methods used to compute the features
for method in ['catch22', 'power_spectrum_parameterization_1']:
    print(f'\n\n--- Method: {method}')

    # Load empirical data
    try:
        data_EI = np.load(os.path.join(pred_results, method, 'emp_data_reduced.pkl'), allow_pickle=True)
        ages[method] = np.array(data_EI['Group'].tolist())
        # Pick only ages >= 4
        data_EI = data_EI[data_EI['Group'] >= 4]
        ages[method] = ages[method][ages[method] >= 4]
    except:
        raise RuntimeError(f'Error loading empirical data for {method}. Execution stopped.')

    # Remove nan features from empirical data
    if np.array(data_EI['Features'].tolist()).ndim == 1:
        ii = np.where(~np.isnan(np.array(data_EI['Features'].tolist())))[0]
    else:
        ii = np.where(~np.isnan(np.array(data_EI['Features'].tolist())).any(axis=1))[0]
    data_EI = data_EI.iloc[ii]
    ages[method] = ages[method][ii]

    # Collect features
    if method == 'catch22':
        emp['dfa'] = np.array(data_EI['Features'].apply(
            lambda x: x[catch22_names.index('SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1')]).tolist())
        emp['rs_range'] = np.array(data_EI['Features'].apply(
            lambda x: x[catch22_names.index('SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1')]).tolist())
        emp['high_fluct'] = np.array(data_EI['Features'].apply(
            lambda x: x[catch22_names.index('MD_hrv_classic_pnn40')]).tolist())
    elif method == 'power_spectrum_parameterization_1':
        emp['slope'] = np.array(data_EI['Features'].tolist())

# Create a figure and set its properties
fig = plt.figure(figsize=(4, 3), dpi=300)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Define a colormap for empirical data
cmap = plt.colormaps['viridis']

# Plots
for row in range(2):
    for col in range(2):
        ax = fig.add_axes([0.15 + col * 0.5, 0.67 - row * 0.5, 0.29, 0.29])

        # Get the keys for the parameters and features
        if row == 0 and col == 0:
            feat = 'dfa'
            method = 'catch22'
        elif row == 0 and col == 1:
            feat = 'rs_range'
            method = 'catch22'
        elif row == 1 and col == 0:
            feat = 'high_fluct'
            method = 'catch22'
        else:
            feat = 'slope'
            method = 'power_spectrum_parameterization_1'

        # Show empirical data
        try:
            for i, age in enumerate(np.unique(ages[method])):
                idx = np.where(ages[method] == age)[0]
                data_plot = emp[feat][idx]

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
        except:
            pass

        # Set labels
        # X-axis labels
        try:
            ax.set_xticks(np.unique(ages[method])[::2])
            ax.set_xticklabels([f'{str(i)}' for i in np.unique(ages[method])[::2]],fontsize = 8)
        except:
            pass

        ax.set_xlabel('Postnatal days')

        # Y-axis labels
        ax.yaxis.set_label_coords(-0.35, 0.5)
        if row == 0 and col == 0:
            ax.set_ylabel(r'$dfa$')
        elif row == 0 and col == 1:
            ax.set_ylabel(r'$rs\ range$')
        elif row == 1 and col == 0:
            ax.set_ylabel(r'$high\ fluct.$')
        else:
            ax.set_ylabel(r'$1/f$' + ' ' + r'$slope$')

# Save the figure
plt.savefig('emp_features.png', bbox_inches='tight')
# plt.show()