import os

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt

EEG_AD_FTD_path = '/DATOS/pablomc/EEG_AD_FTD_results'

if __name__ == "__main__":
    # Load features
    # POCTEP dataset
    emp_data_POCTEP_source = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_POCTEP_False.pkl'))
    emp_data_POCTEP_raw = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_POCTEP_True.pkl'))
    # OpenNEURO dataset
    emp_data_OpenNeuro = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_OpenNEURO.pkl'))

    # Load LMER results
    lmer_feat = pickle.load(open(os.path.join(EEG_AD_FTD_path, 'catch22', 'lmer_feat.pkl'), 'rb'))

    # Fig. 2
    fig2 = plt.figure(figsize=(7.5, 3.5), dpi=300)
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

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
                dataset = emp_data_POCTEP_source
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
                lmer_results = lmer_feat[0]['DB1_source'][f'{feat}']
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


    plt.show()