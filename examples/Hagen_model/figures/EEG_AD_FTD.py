import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import scipy
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from matplotlib import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines

if '__file__' not in globals():
    __file__ = os.path.abspath('')

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ncpi

script_dir = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(script_dir, 'config.json')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

EEG_AD_FTD_path = config['empirical_features_path']

n_var = 2

all_confs = ['catch22',
            'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
            'SB_TransitionMatrix_3ac_sumdiagcov',
            'CO_HistogramAMI_even_2_5',
            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1']




if __name__ == "__main__":
    # Load features
    # POCTEP dataset
    # emp_data_POCTEP_source = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_POCTEP_False.pkl'))
    emp_data_POCTEP_raw = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_POCTEP_True.pkl'))
    # OpenNEURO dataset
    emp_data_OpenNeuro = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_OpenNEURO.pkl'))

    # Load LMER results
    lmer_feat = pickle.load(open(os.path.join(EEG_AD_FTD_path, 'catch22', 'lmer_feat.pkl'), 'rb'))
    lmer_preds = pickle.load(open(os.path.join(EEG_AD_FTD_path, 'catch22', 'lmer_preds.pkl'), 'rb'))


    # Fig. 2
    fig2 = plt.figure(figsize=(7.5, 3.5), dpi=300)
    plt.rcParams.update({'font.family': 'Arial'})

    # Define 4 colors for DB1
    colors_1 = ['lightgrey', 'lightcoral', 'lightblue', 'lightgreen']

    # Define 3 colors for DB2
    colors_2 = ['lightgrey', 'peachpuff', 'cornflowerblue']

    for row in range(2):
        for col in range(4):
            ax = fig2.add_axes([0.08 + col * 0.24, 0.59 - row * 0.43, 0.15, 0.33])

            if row == 0 and col < 2:
                feat = 18
            if row == 0 and col >= 2:
                feat = 8
            if row == 1 and col < 2:
                feat = 4
            if row == 1 and col >= 2:
                feat = 19

            # DB1
            if col % 2 == 0:
                groups = ['HC','ADMIL', 'ADMOD', 'ADSEV']
                dataset = emp_data_POCTEP_raw
                colors = colors_1
            # DB2
            else:
                groups = ['C','F','A']
                dataset = emp_data_OpenNeuro
                colors = colors_2

            # Boxplots
            for i,group in enumerate(groups):
                emp_data_group = dataset[dataset.Group == group]
                feature = emp_data_group['Features'].apply(lambda x: x[feat])
                # Remove nan values
                feature = feature[~np.isnan(feature)]
                box = ax.boxplot(feature, positions=[i], showfliers=False,
                             widths=0.7, patch_artist=True, medianprops=dict(color='red', linewidth=0.8),
                             whiskerprops=dict(color='black', linewidth=0.5),
                             capprops=dict(color='black', linewidth=0.5),
                             boxprops=dict(linewidth=0.5))
                for patch in box['boxes']:
                    patch.set_facecolor(colors[i])

            # Compute the linear mixed-effects model
            if col % 2 == 0:
                lmer_results = lmer_feat[0]['DB1_raw'][f'{feat}']
            else:
                lmer_results = lmer_feat[0]['DB2'][f'{feat}']

            # Add p-values to the plot
            y_max = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            delta = (y_max - y_min) * 0.2

            for i, group in enumerate(groups[1:]):
                p_value = lmer_results[f'{group}vsHC']['p.value']
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
                    offset = -delta*0.1
                else:
                    offset = 0

                ax.text(0.5*i+0.5, y_max + delta*i + delta*0.1 + offset, f'{pp}', ha='center', fontsize=8)
                ax.plot([0, i+1], [y_max + delta*i, y_max + delta*i], color='black', linewidth=0.5)

            # Change y-lim
            ax.set_ylim([y_min, y_max + delta*(len(groups)-1)])

            # x-labels
            if row == 1:
                if col % 2 == 0:
                    ax.set_xticks([0, 1, 2, 3])
                    ax.set_xticklabels(['HC', 'ADMIL', 'ADMOD', 'ADSEV'], rotation=45, fontsize = 8)
                else:
                    ax.set_xticks([0, 1, 2])
                    ax.set_xticklabels(['HC', 'FTD', 'AD'], rotation=45, fontsize = 8)
            else:
                ax.set_xticks([])

            # y-labels
            if row == 0 and col < 2:
                ax.set_ylabel(r'$rs\_range$')
            if row == 0 and col >= 2:
                ax.set_ylabel(r'$TransVar$')
            if row == 1 and col < 2:
                ax.set_ylabel(r'$ami2$')
            if row == 1 and col >= 2:
                ax.set_ylabel(r'$dfa$')

            # Titles
            ax.set_title(f'DB{1 if col%2 == 0 else 2}')
    sizex = 3.5
    sizey = 4
    left = 0.03
    right = 0.17
    width = (1.0 - left - right) / (2 * 5) - 0.037  
    height = 1.0 / 4 + 0.06
    bottom = 1 - height 
    spacing_x = 0.14
    new_spacing_x = 0.16
    spacing_y = -0.095

    # # Fig. 3
    fig3 = plt.figure(figsize=(4.5, 5), dpi=300)
    current_bottom = bottom
    for row in range(4):
        current_left = left
        for col in range(5):

            if row == 0:
                feat = 18
            if row == 1:
                feat = 4
            if row == 2:
                feat = 8
            if row == 3:
                feat = 19

            if col == 0:
                group = 'ADMIL'
                group_label = 'ADMIL'

            if col == 1:
                group = 'ADMOD'
                group_label = 'ADMOD'
            if col == 2:
                group = 'ADSEV'
                group_label = 'ADSEV'
            if col == 3:
                group = 'F'
                group_label = 'FTD'
            if col == 4:
                group = 'A'
                group_label = 'AD'

            ax = fig3.add_axes([current_left, current_bottom, 0.15, 0.15], frameon=False)
            if col == 2:
                    current_left += width + new_spacing_x
            else:
                current_left += width + spacing_x
            # Get lmer results
            if col < 3:
                lmer_results = lmer_feat[1]['DB1_raw'][f'{feat}']
            else:
                lmer_results = lmer_feat[1]['DB2'][f'{feat}']

            data = []
            for elec in range(19):
                p_value = lmer_results[f'{group}vsHC']['p.value'][elec]
                z_score = lmer_results[f'{group}vsHC']['z.ratio'][elec]
                if p_value < 0.01 and np.abs(z_score) > 2.5:
                    data.append(z_score)
                else:
                    data.append(0)


            if col < 3:
                if row == 0:
                    ylims = [-11.,11.]
                if row == 1 or row == 2:
                    ylims = [-6.,6.]
                if row == 3:
                    ylims = [-3.5, 3.5]
            else:
                if row == 0 or row == 1 or row == 3:
                    ylims = [-6.,6.]
                if row == 2:
                    ylims = [-3.5, 3.5]

            
            # ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Titles
            if row == 0:
                ax.set_title(f'{group_label} vs HC', fontsize=8)

            # Labels
            if col == 0:
                if row == 0:
                    ax.set_ylabel(r'$rs\_range$', fontsize=5)
                if row == 1:
                    ax.set_ylabel(r'$ami2$',fontsize=5)
                if row == 2:
                    ax.set_ylabel(r'$TransVar$',fontsize=5)
                if row == 3:
                    ax.set_ylabel(r'$dfa$', fontsize=5)
            # Create a topographic plot

            analysis = ncpi.Analysis(data)
            analysis.EEG_topographic_plot(
                group = f'{group}vsHC',
                system = 19,
                p_value = 0.01,
                electrode_size = 0.6,
                ax = ax,
                fig=fig3,
                vmin = ylims[0],
                vmax = ylims[1]
            )
        current_bottom -= height + spacing_y
    
    fig3.text(0.29, 0.95, 'DB1', ha='center', fontsize=8)
    fig3.text(0.8, 0.95, 'DB2', ha='center', fontsize=8)
    linepos = [0.94, 0.94]
    line1 = mlines.Line2D([0.013, 0.57], linepos, color='black', linewidth=0.7)
    line2 = mlines.Line2D([0.62, 0.95], linepos, color='black', linewidth=0.7)
    fig3.add_artist(line1)
    fig3.add_artist(line2)
    
    height = 1.0 / 5
    sizex = 3.5
    sizey = 4
    left = 0.03
    right = 0.17
    width = (1.0 - left - right) / (2 * 5) - 0.037  
    height = 1.0 / 4 
    bottom = 1 - height - 0.03
    spacing_x = 0.15
    new_spacing_x = 0.16
    spacing_y = -0.075

    max = 0
    yname = [r'$[E/I]_{net}$', r'$J_{ext}$']
    order = [1,3,2,4,0]
    # Figs. 5 & 6
    for param in range(n_var):
        fig = plt.figure(figsize=(4.5, 5), dpi=300)
        current_bottom = bottom
        for row in order:
            current_left = left
            for col in range(5):
                
                if col == 0:
                    group = 'ADMIL'
                    group_label = 'ADMIL'
                if col == 1:
                    group = 'ADMOD'
                    group_label = 'ADMOD'
                if col == 2:
                    group = 'ADSEV'
                    group_label = 'ADSEV'
                if col == 3:
                    group = 'F'
                    group_label = 'FTD'
                if col == 4:
                    group = 'A'
                    group_label = 'AD'
                
                ax = fig.add_axes([current_left, current_bottom, 0.15, 0.15], frameon=False)
                if col == 2:
                    current_left += width + new_spacing_x
                else:
                    current_left += width + spacing_x


                # Get lmer results
                if col < 3:
                    lmer_results = lmer_preds[n_var-1]['DB1'][all_confs[row]][param]
                else:
                    lmer_results = lmer_preds[n_var-1]['DB2'][all_confs[row]][param]

                data = []
                for elec in range(19):
                    p_value = lmer_results[f'{group}vsHC']['p.value'][elec]
                    z_score = lmer_results[f'{group}vsHC']['z.ratio'][elec]
                    if p_value < 0.01:
                        data.append(z_score)
                    else:
                        data.append(0)
                range_max = np.max(np.abs(data))
                ylims = [-6., 6.]
                if row == 1 and col == 0:
                    vmin = -4.720872348422581
                    vmax = 4.720872348422581

                # ticks
                ax.set_xticks([])
                ax.set_yticks([])

                # Titles
                if row == 1:
                    ax.set_title(f'{group_label} vs HC', fontsize=8)

                # Labels
                if col == 0:
                    if row == 0:
                        ax.set_ylabel(f'{yname[param]}'r'($catch22$)', fontsize=5)
                    if row == 1:
                        ax.set_ylabel(f'{yname[param]}'r'($rs\_range$)', fontsize=5)
                    if row == 2:
                        ax.set_ylabel(f'{yname[param]}'r'($TransVar$)', fontsize=5)
                    if row == 3:
                        ax.set_ylabel(f'{yname[param]}'r'($ami2$)', fontsize=5)
                    if row == 4:
                        ax.set_ylabel(f'{yname[param]}'r'($dfa$)', fontsize=5)
                analysis = ncpi.Analysis(data)
                analysis.EEG_topographic_plot(
                    group = f'{group}vsHC',
                    system = 19,
                    p_value = 0.01,
                    electrode_size = 0.6,
                    ax = ax,
                    fig=fig3,
                    vmin = ylims[0],
                    vmax = ylims[1]
                )
            current_bottom -= height + spacing_y

        fig.text(0.29, 0.95, 'DB1', ha='center', fontsize=8)
        fig.text(0.79, 0.95, 'DB2', ha='center', fontsize=8)
        linepos = [0.94, 0.94]
        line1 = mlines.Line2D([0.03, 0.56], linepos, color='black', linewidth=0.7)
        line2 = mlines.Line2D([0.63, 0.93], linepos, color='black', linewidth=0.7)
        fig.add_artist(line1)
        fig.add_artist(line2)

    plt.show()