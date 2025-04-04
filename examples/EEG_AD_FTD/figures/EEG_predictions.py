import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import ncpi

# Set the path to the results folder
results_path = '../results'

# Select the statistical analysis method ('cohen', 'lmer')
statistical_analysis = 'lmer'

databases = [
    'POCTEP',
    # 'OpenNEURO'
    ]

# Load channel names
if 'POCTEP' in databases:
    ch_names_POCTEP = pd.read_pickle(os.path.join(results_path, 'ch_names_POCTEP.pkl'))
if 'OpenNEURO' in databases:
    ch_names_OpenNEURO = pd.read_pickle(os.path.join(results_path, 'ch_names_OpenNEURO.pkl'))

def append_lmer_results(lmer_results, group, elec, p_value_th, data_lmer):
    '''
    Create a list with the z-scores of the linear mixed model analysis for a given group and electrode.

    Parameters
    ----------
    lmer_results : dict
        Dictionary with the results of the linear mixed model analysis.
    group : str
        Group name.
    elec : int
        Electrode index.
    p_value_th : float
        P-value threshold.
    data_lmer : list
        List with the z-scores of the linear mixed model analysis.

    Returns
    -------
    data_lmer : list
        List with the z-scores of the linear mixed model analysis.
    '''

    p_value = lmer_results[f'{group}vsHC']['p.value'][elec]
    z_score = lmer_results[f'{group}vsHC']['z.ratio'][elec]

    if p_value < p_value_th:
        data_lmer.append(z_score)
    else:
        data_lmer.append(0)

    return data_lmer


if __name__ == "__main__":

    # Some parameters
    p_value_th = 0.01
    inference_method = 'CDM'

    print('\n---------------------------------------------------------------------------------------------------------')
    print(f'Which figures do you want to generate?')
    print(f'a) Figures from the paper "ncpi: A Python toolbox for neural circuit parameter inference."')
    print(f'b) Figures from the paper "A Hybrid Machine Learning and Mechanistic Modelling Approach for Probing '
          f'Potential Biomarkers of Excitation/Inhibition Imbalance in Cortical Circuits in Dementia."')
    print('---------------------------------------------------------------------------------------------------------\n')

    option = input(f'a or b: ')

    ncols = 5 if option == 'b' else 6
    nrows = 5

    left = 0.06
    right = 0.11

    width = (1.0 - left - right) / (6) - 0.03 if option == 'a' else (1.0 - left - right) / (6) - 0.01
    height = 1.0 / 5 - 0.025
    bottom = 1 - (1. / 5 + 0.07)

    new_spacing_x = 0.08 if option == 'a' else 0.14
    new_spacing_y = 0.05

    spacing_x = 0.04
    spacing_y = 0.064 if option == 'a' else 0.004

    fig1 = plt.figure(figsize=(7.5, 5.5), dpi=300)
    if option == 'b':
        fig2 = plt.figure(figsize=(7.5, 5.5), dpi=300) 
        fig3 = plt.figure(figsize=(7.5, 5.5), dpi=300)

    max = 0

    current_bottom = bottom

    if option == 'a':
        for row in range(2):
            ax = fig1.add_axes([0.01, 0.51 - row * 0.47, 0.98, 0.46 if row == 0 else 0.47])
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='red' if row == 0 else 'blue', alpha=0.1))
            ax.set_xticks([])
            ax.set_yticks([])

    for row in range(nrows):
        if option == 'a':
            if row == 0 or row == 1:
                method = 'catch22'
            if row == 2 or row == 3:
                method = 'power_spectrum_parameterization_1'
            try:
                data = pd.read_pickle(f'{results_path}/POCTEP_True-{method}-4_var-{inference_method}-'
                                      f'elec-pred_{statistical_analysis}.pkl')
            except Exception as e:
                print(f'Error loading data for {method}: {e}')
                continue

        current_left = left
        for col in range(ncols):
            if option == 'b':
                if row == 0:
                    method = 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1'
                if row == 1:
                    method = 'CO_HistogramAMI_even_2_5'
                if row == 2:
                    method = 'SB_TransitionMatrix_3ac_sumdiagcov'
                if row == 3:
                    method = 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1'
                if row == 4:
                    method = 'catch22'

                database = 'POCTEP_True' if col < 3 else 'OpenNEURO'
                try:
                    data = pd.read_pickle(f'{results_path}/{database}-{method}-2_var-{inference_method}-'
                                          f'elec-pred_{statistical_analysis}.pkl')
                    feature_data = pd.read_pickle(f'{results_path}/{database}-{method}-2_var-{inference_method}-'
                                                  f'elec-feat_{statistical_analysis}.pkl') if row != 4 else None
                except Exception as e:
                    print(f'Error loading data for {method}: {e}')
                    continue
            
            if option == 'a':
                if col == 0 or col == 3:
                    group = 'ADMIL'
                    group_label = 'ADMIL'

                if col == 1 or col == 4:
                    group = 'ADMOD'
                    group_label = 'ADMOD'

                if col == 2 or col == 5:
                    group = 'ADSEV'
                    group_label = 'ADSEV'

            if option == 'b':
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
                    group = 'A'
                    group_label = 'AD'
                if col == 4:
                    group = 'F'
                    group_label = 'FTD'

            # Add ax --> [left, bottom, width, height]
            ax1 = fig1.add_axes([current_left, current_bottom, width, height], frameon=False)
            ax2 = fig2.add_axes([current_left, current_bottom, width, height], frameon=False) if option == 'b' else None
            ax3 = fig3.add_axes([current_left, current_bottom, width, height], frameon=False) if (option == 'b'
                                                                                                  and row != 4) else None

            # Compute new left position (x spacing)
            if (col == 2 and option == 'b') or (col == 2 and option == 'a'):
                # More spacing for separate plots 
                current_left += width + new_spacing_x
            else:
                current_left += width + spacing_x

            # Disable ticks
            ax1.set_xticks([])
            ax1.set_yticks([])

            if option == 'b':
                ax2.set_xticks([])
                ax2.set_yticks([])
                
                if ax3 != None:
                    ax3.set_xticks([])
                    ax3.set_yticks([])

            # Titles 
            if option == 'a' or (option == 'b' and row == 0):
                ax1.set_title(f'{group_label} vs HC', fontsize=10)
                if option == 'b':
                    ax2.set_title(f'{group_label} vs HC', fontsize=10)
                    ax3.set_title(f'{group_label} vs HC', fontsize=10) 

            # Labels
            if option == 'a':
                if col < 3:
                    var = 'E/I' if row == 0 or row == 2 else 'tau_exc'

                if col >= 3:
                    var = r'Jext' if row == 0 or row == 2 else 'tau_inh'
       
                stat_results = data[var]
            
            if option == 'b':
                if col == 0:
                    if row == 0:
                        ax1.set_ylabel(r'$[E/I]_{net}$(rs_range)', fontsize=8)
                        ax2.set_ylabel(r'$J_{ext}$(rs_range)', fontsize=8)
                        ax3.set_ylabel(r'rs_range', fontsize=8) 

                    if row == 1:
                        ax1.set_ylabel(r'$[E/I]_{net}$(ami2)', fontsize=8)
                        ax2.set_ylabel(r'$J_{ext}$(ami2)', fontsize=8)
                        ax3.set_ylabel(r'ami2', fontsize=8) 

                    if row == 2:
                        ax1.set_ylabel(r'$[E/I]_{net}$(TransVar)', fontsize=8)
                        ax2.set_ylabel(r'$J_{ext}$(TransVar)', fontsize=8)    
                        ax3.set_ylabel(r'TransVar', fontsize=8) 

                    if row == 3:
                        ax1.set_ylabel(r'$[E/I]_{net}$(dfa)', fontsize=8)
                        ax2.set_ylabel(r'$J_{ext}$(dfa)', fontsize=8)
                        ax3.set_ylabel(r'dfa', fontsize=8) 

                    if row == 4:
                        ax1.set_ylabel(r'$[E/I]_{net}$(catch22)', fontsize=8)
                        ax2.set_ylabel(r'$J_{ext}$(catch22)', fontsize=8)

                stat_results = data['E/I']
                stat_results_jext = data['Jext']

            data_stat = []
            data_stat_jext = []
            data_stat_feat = []
            for elec in range(19):
                # Find position of the electrode in the stat results
                if option == 'a' or database == 'POCTEP_True':
                    pos_results = np.where(stat_results[f'{group}vsHC']['Sensor'] == ch_names_POCTEP[elec])[0]
                else:
                    pos_results = np.where(stat_results[f'{group}vsHC']['Sensor'] == ch_names_OpenNEURO[elec])[0]
                    pos_results_jext = np.where(stat_results_jext[f'{group}vsHC']['Sensor'] == ch_names_OpenNEURO[elec])[0]
                    pos_feature_data = np.where(feature_data[f'{group}vsHC']['Sensor'] == ch_names_OpenNEURO[elec])[0]

                if len(pos_results) > 0:
                    if statistical_analysis == 'lmer':
                        data_stat = append_lmer_results(stat_results, group, pos_results[0], p_value_th, data_stat)
                    elif statistical_analysis == 'cohen':
                        data_stat.append(stat_results[f'{group}vsHC']['d'][pos_results[0]])
                else:
                    data_stat.append(0)

                if option == 'b':
                    if len(pos_results_jext) > 0:
                        if statistical_analysis == 'lmer':
                            data_stat_jext = append_lmer_results(stat_results_jext, group, pos_results_jext[0],
                                                                 p_value_th, data_stat_jext)
                        elif statistical_analysis == 'cohen':
                            data_stat_jext.append(stat_results_jext[f'{group}vsHC']['d'][pos_results_jext[0]])
                    else:
                        data_stat_jext.append(0)
                    if row != 4:
                        if len(pos_feature_data) > 0:
                            if statistical_analysis == 'lmer':
                                data_stat_feat = append_lmer_results(feature_data, group, pos_feature_data[0],
                                                                     p_value_th, data_stat_feat)
                            elif statistical_analysis == 'cohen':
                                data_stat_feat.append(feature_data[f'{group}vsHC']['d'][pos_feature_data[0]])
                        else:
                            data_stat_feat.append(0)

            if statistical_analysis == 'lmer':
                ylims_stat = [-6., 6.]
            else:
                ylims_stat = [-1., 1.]

            # Create brainplot
            analysis = ncpi.Analysis(data_stat)
            analysis.EEG_topographic_plot(
                        electrode_size = 0.6,
                        ax = ax1,
                        fig=fig1,
                        vmin = ylims_stat[0],
                        vmax = ylims_stat[1],
                        label=False
            )

            if option == 'b':
                analysis = ncpi.Analysis(data_stat_jext)
                analysis.EEG_topographic_plot(
                            electrode_size = 0.6,
                            ax = ax2,
                            fig=fig2,
                            vmin = ylims_stat[0],
                            vmax = ylims_stat[1],
                            label=False
                )
                if row != 4:
                    analysis = ncpi.Analysis(data_stat_feat)
                    analysis.EEG_topographic_plot(
                                electrode_size = 0.6,
                                ax = ax3,
                                fig=fig3,
                                vmin = ylims_stat[0],
                                vmax = ylims_stat[1],
                                label=False
                    )

        # Update "y" spacing
        if row == 1 and option == 'a':
            # More spacing for separate plots 
            current_bottom -= height + new_spacing_y
        else:
            current_bottom -= height + spacing_y


    fontsize = 12
    if option == 'a':
        fig1.text(0.46, 0.94, 'catch22', color='red', alpha=0.5, fontsize=12, fontstyle='italic')
        fig1.text(0.46, 0.48, '1/f slope', color='blue', alpha=0.5, fontsize=12, fontstyle='italic')

        fig1.text(0.015, 0.94, 'A', fontsize=12, fontweight='bold')
        fig1.text(0.015, 0.48, 'B', fontsize=12, fontweight='bold')

        fig1.text(0.24, 0.94, r'$E/I$', ha='center', fontsize=fontsize)
        fig1.text(0.74, 0.94, r'$J_{syn}^{ext}$ (nA)', ha='center', fontsize=fontsize)

        fig1.text(0.24, 0.7, r'$\tau_{syn}^{exc}$ (ms)', ha='center', fontsize=fontsize)
        fig1.text(0.74, 0.7, r'$\tau_{syn}^{inh}$ (ms)', ha='center', fontsize=fontsize)

        # Parameters for 1/f slope
        fig1.text(0.24, 0.48, r'$E/I$', ha='center', fontsize=fontsize)
        fig1.text(0.74, 0.48, r'$J_{syn}^{ext}$ (nA)', ha='center', fontsize=fontsize)

        fig1.text(0.24, 0.245, r'$\tau_{syn}^{exc}$ (ms)', ha='center', fontsize=fontsize)
        fig1.text(0.74, 0.245, r'$\tau_{syn}^{inh}$ (ms)', ha='center', fontsize=fontsize)

        linepos1 = [0.925, 0.925]
        linepos2 = [0.686, 0.686]

        EI_line_c = mlines.Line2D([0.055, 0.46], linepos1, color='black', linewidth=0.5)
        tauexc_line_c = mlines.Line2D([0.055, 0.46], linepos2, color='black', linewidth=0.5)

        Jext_line_c = mlines.Line2D([0.54, 0.945], linepos1, color='black', linewidth=0.5)
        tauinh_line_c = mlines.Line2D([0.54, 0.945], linepos2, color='black', linewidth=0.5)

        # 1/f slope lines
        linepos1 = [0.467, 0.467]
        linepos2 = [0.23, 0.23]

        EI_line_f = mlines.Line2D([0.055, 0.46], linepos1, color='black', linewidth=0.5)
        tauexc_line_f = mlines.Line2D([0.055, 0.46], linepos2, color='black', linewidth=0.5)

        Jext_line_f = mlines.Line2D([0.54, 0.945], linepos1, color='black', linewidth=0.5)
        tauinh_line_f = mlines.Line2D([0.54, 0.945], linepos2, color='black', linewidth=0.5)

        # Add catch22 lines
        fig1.add_artist(EI_line_c)
        fig1.add_artist(Jext_line_c)
        fig1.add_artist(tauexc_line_c)
        fig1.add_artist(tauinh_line_c)

        # Add 1/f slope lines
        fig1.add_artist(EI_line_f)
        fig1.add_artist(Jext_line_f)
        fig1.add_artist(tauexc_line_f)
        fig1.add_artist(tauinh_line_f)

        fig1.savefig('EEG-predictions-ncpi.png')

    else:    
        fig1.text(0.28, 0.94, r'DB1', ha='center', fontsize=fontsize)
        fig1.text(0.81, 0.94, r'DB2', ha='center', fontsize=fontsize)

        fig2.text(0.28, 0.94, r'DB1', ha='center', fontsize=fontsize)
        fig2.text(0.81, 0.94, r'DB2', ha='center', fontsize=fontsize)

        fig3.text(0.28, 0.94, r'DB1', ha='center', fontsize=fontsize)
        fig3.text(0.81, 0.94, r'DB2', ha='center', fontsize=fontsize)

        linepos1 = [0.935, 0.935]

        DB1 = mlines.Line2D([0.056, 0.51], linepos1, color='black', linewidth=0.6)
        DB2 = mlines.Line2D([0.68, 0.945], linepos1, color='black', linewidth=0.6)

        fig1.add_artist(DB1)
        fig1.add_artist(DB2)
        
        DB1 = mlines.Line2D([0.056, 0.51], linepos1, color='black', linewidth=0.6)
        DB2 = mlines.Line2D([0.68, 0.945], linepos1, color='black', linewidth=0.6)
        
        fig2.add_artist(DB1)
        fig2.add_artist(DB2)
        
        DB1 = mlines.Line2D([0.056, 0.51], linepos1, color='black', linewidth=0.6)
        DB2 = mlines.Line2D([0.68, 0.945], linepos1, color='black', linewidth=0.6)
        
        fig3.add_artist(DB1)
        fig3.add_artist(DB2)
        fig3.savefig('features-Hybrid.png')

        fig1.savefig('EI-predictions-Hybrid.png')
        fig2.savefig('Jext-predictions-Hybrid.png')

# plt.show()
