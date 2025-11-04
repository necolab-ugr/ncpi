import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import ncpi

# Set the path to the results folder
results_path = os.path.join('..', 'data')

# Select the statistical analysis method ('cohen', 'lmer')
statistical_analysis = 'lmer'

# Set the p-value threshold
p_value_th = 0.01

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

    p_value = lmer_results[f'{group}vsHC']['p.value'].iloc[elec]
    z_score = lmer_results[f'{group}vsHC']['z.ratio'].iloc[elec]

    if p_value < p_value_th:
        data_lmer.append(z_score)
    else:
        data_lmer.append(0)

    return data_lmer


if __name__ == "__main__":
    # Some parameters for the figure
    ncols = 6 
    nrows = 5

    left = 0.06
    right = 0.11

    width = (1.0 - left - right) / (6) - 0.03 
    height = 1.0 / 5 - 0.025
    bottom = 1 - (1. / 5 + 0.07)

    new_spacing_x = 0.08 
    new_spacing_y = 0.05

    spacing_x = 0.04
    spacing_y = 0.064 

    # Create figure
    fig1 = plt.figure(figsize=(7.5, 5.5), dpi=300)
    current_bottom = bottom

    for row in range(2):
        ax = fig1.add_axes([0.01, 0.51 - row * 0.47, 0.98, 0.46 if row == 0 else 0.47])
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='red' if row == 0 else 'blue', alpha=0.1))
        ax.set_xticks([])
        ax.set_yticks([])

    for row in range(nrows):
        if row == 0 or row == 1:
            method = 'catch22'
        if row == 2 or row == 3:
            method = 'power_spectrum_parameterization_1'
        try:
            data = pd.read_pickle(os.path.join(results_path, method, 'emp_data_reduced.pkl'))
            
        except Exception as e:
            print(f'Error loading data for {method}: {e}')
            continue

        current_left = left
        for col in range(ncols):
            if col == 0 or col == 3:
                group = 'ADMIL'
                group_label = 'ADMIL'

            if col == 1 or col == 4:
                group = 'ADMOD'
                group_label = 'ADMOD'

            if col == 2 or col == 5:
                group = 'ADSEV'
                group_label = 'ADSEV'

            # Add ax --> [left, bottom, width, height]
            ax1 = fig1.add_axes([current_left, current_bottom, width, height], frameon=False)
            
            # Compute new left position (x spacing)
            if col == 2:
                # More spacing for separate plots 
                current_left += width + new_spacing_x
            else:
                current_left += width + spacing_x

            # Disable ticks
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Titles 
            ax1.set_title(f'{group_label} vs HC', fontsize=10)

            # Labels
            if col < 3:
                var = 0 if row == 0 or row == 2 else 1

            if col >= 3:
                var = 3 if row == 0 or row == 2 else 2
    
            # Statistical analysis
            analysis = ncpi.Analysis(data)
            if statistical_analysis == 'lmer':
                stat_results = analysis.lmer(control_group='HC', data_col='Predictions', data_index=var,
                                        other_col=['ID', 'Group', 'Epoch', 'Sensor'],
                                        models={
                                            'mod00': 'Y ~ Group * Sensor + (1 | ID)',
                                            'mod01': 'Y ~ Group * Sensor',
                                            'mod02': 'Y ~ Group + Sensor + (1 | ID)',
                                            'mod03': 'Y ~ Group + Sensor',
                                        },
                                        bic_models=["mod00", "mod01"],
                                        anova_tests={
                                            'test1': ["mod00", "mod01"],
                                            'test2': ["mod02", "mod03"],
                                        },
                                        specs='~Group | Sensor')
            elif statistical_analysis == 'cohen':
                stat_results = analysis.cohend(control_group='HC', data_col='Predictions', data_index=var)
            
            data_stat = []
            # Extract sensor names
            empirical_sensors = data['Sensor'].unique()
            for elec in range(19):
                # Find position of the electrode in the stat results
                pos_results = np.where(stat_results[f'{group}vsHC']['Sensor'] == empirical_sensors[elec])[0]
                
                if len(pos_results) > 0:
                    if statistical_analysis == 'lmer':
                        data_stat = append_lmer_results(stat_results, group, pos_results[0], p_value_th, data_stat)
                    elif statistical_analysis == 'cohen':
                        data_stat.append(stat_results[f'{group}vsHC']['d'][pos_results[0]])
                else:
                    data_stat.append(0)

            # Limits
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


        # Update "y" spacing
        if row == 1:
            # More spacing for separate plots 
            current_bottom -= height + new_spacing_y
        else:
            current_bottom -= height + spacing_y

    # Text and lines
    fontsize = 12
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

    fig1.savefig(f'EEG_predictions_{statistical_analysis}.png')

    
