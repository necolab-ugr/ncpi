import os
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
# EEG_AD_FTD_path = '/DATOS/pablomc/EEG_AD_FTD_results'
EEG_AD_FTD_path = '/home/alejandro/CCPI/DATA/features'

n_var = 2

all_confs = ['catch22',
            'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
            'SB_TransitionMatrix_3ac_sumdiagcov',
            'CO_HistogramAMI_even_2_5',
            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1']

def plot_simple_head_model(ax, radius, pos):
    '''
    Plot a simple head model for adding results of the EEG data analysis later.

    Parameters
    ----------
    ax: matplotlib Axes object
    radius: float,
        radius of the head circumference.
    pos: float
        Position of the head on the x-axis.
    '''

    ax.set_aspect('equal')


    # Cabeza
    head_circle = mpatches.Circle((pos, 0), radius+0.02, edgecolor='k', facecolor='none', linewidth=0.5)
    ax.add_patch(head_circle)

    # Orejas
    right_ear = mpatches.FancyBboxPatch([pos + radius + radius / 20, -radius / 10],
                                        radius / 50, radius / 5,
                                        boxstyle=mpatches.BoxStyle("Round", pad=radius / 20),
                                        linewidth=0.5)
    ax.add_patch(right_ear)

    left_ear = mpatches.FancyBboxPatch([pos - radius - radius / 20 - radius / 50, -radius / 10],
                                       radius / 50, radius / 5,
                                       boxstyle=mpatches.BoxStyle("Round", pad=radius / 20),
                                       linewidth=0.5)
    ax.add_patch(left_ear)

    # Nariz
    ax.plot([pos - radius / 10, pos, pos + radius / 10], 
            [radius + 0.02, radius + radius / 10 + 0.02,0.02 + radius], 
            'k', linewidth=0.5)

def plot_EEG(fig, Vax, data, radius, pos, vmin, vmax, label=True):
    '''
    Plot slopes or E/I predictions on EEG electrodes (20 EEG montage) as a
    contour plot.

    Parameters
    ----------
    fig: matplotlib figure
    Vax: matplotlib Axes object
    data: list
        Data containing slopes or E/I predictions.
    radius: float,
        radius of the head circumference.
    pos: float
        Position of the head on the x-axis.
    vmin, vmax: float
        Min and max values used for plotting.
    '''
    # Some parameters
    N = 100             # number of points for interpolation
    xy_center = [pos,0]   # center of the plot

    # Coordinates of the EEG electrodes in the 20 montage
    koord = [[pos-0.25*radius,0.8*radius], # "Fp1"
            [pos+0.25*radius,0.8*radius], # "Fp2"
            [pos-0.3*radius,0.35*radius], # "F3"
            [pos+0.3*radius,0.35*radius], # "F4"
            [pos-0.35*radius,0.0], # "C3"
            [pos+0.35*radius,0.], # "C4"
            [pos-0.3*radius,-0.4*radius], # "P3"
            [pos+0.3*radius,-0.4*radius], # "P4"
            [pos-0.35*radius,-0.8*radius], # "O1"
            [pos+0.35*radius,-0.8*radius], # "O2"
            [pos-0.6*radius,0.45*radius], # "F7"
            [pos+0.6*radius,0.45*radius], # "F8"
            [pos-0.8*radius,0.0], # "T3"
            [pos+0.8*radius,0.0], # "T4"
            [pos-0.6*radius,-0.2], # "T5"
            [pos+0.6*radius,-0.2], # "T6"
            [pos,0.35*radius], # "Fz"
            [pos,0.], # "Cz"
            [pos,-0.4*radius]] # "Pz"


    # External fake electrodes for completing interpolation
    for xx in np.linspace(pos-radius,pos+radius,50):
        koord.append([xx,np.sqrt(radius**2 - (xx)**2)])
        koord.append([xx,-np.sqrt(radius**2 - (xx)**2)])
        data.append(0)
        data.append(0)

    # Interpolate data points
    x,y = [],[]
    for i in koord:
        x.append(i[0])
        y.append(i[1])
    z = data

    xi = np.linspace(-radius, radius, N)
    yi = np.linspace(-radius, radius, N)
    zi = scipy.interpolate.griddata((np.array(x), np.array(y)), z,
                                    (xi[None,:], yi[:,None]), method='cubic')

    # # set points > radius to not-a-number. They will not be plotted.
    # # the dr/2 makes the edges a bit smoother
    # dr = xi[1] - xi[0]
    # for i in range(N):
    #     for j in range(N):
    #         r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
    #         if (r - dr/2) > radius:
    #             zi[j,i] = "nan"

    # Use different number of levels for the fill and the lines
    CS = Vax.contourf(xi, yi, zi, 30, cmap = plt.cm.bwr, zorder = 1,
                      vmin = vmin,vmax = vmax)
    Vax.contour(xi, yi, zi, 5, colors = "grey", zorder = 2,linewidths = 0.4,
                vmin = vmin,vmax = vmax)

    # Make a color bar
    # cbar = fig.colorbar(CS, ax=Vax)
    divider = make_axes_locatable(Vax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if np.sum(np.abs(data)) > 2: 
        colorbar = fig.colorbar(ScalarMappable(norm=CS.norm, cmap=CS.cmap), cax=cax)
        colorbar.ax.tick_params(labelsize=5)
        if label == True:
            colorbar.ax.xaxis.set_label_position('top')
            bbox = colorbar.ax.get_position()
            # print(bbox)
            colorbar.set_label('Ratio', size=5, labelpad=0, rotation=270, va='center')
            
    else:
        # Hide the colorbar if the data is not significant
        cax.axis('off')

    # Add the EEG electrode positions
    Vax.scatter(x[:20], y[:20], marker = 'o', c = 'k', s = 0.5, zorder = 3)


if __name__ == "__main__":
    # Load features
    # POCTEP dataset
    # emp_data_POCTEP_source = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_POCTEP_False.pkl'))
    # emp_data_POCTEP_raw = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_POCTEP_True.pkl'))
    # OpenNEURO dataset
    # emp_data_OpenNeuro = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_OpenNEURO.pkl'))

    # Load LMER results
    lmer_feat = pickle.load(open(os.path.join(EEG_AD_FTD_path, 'databases', 'lmer_feat.pkl'), 'rb'))
    lmer_preds = pickle.load(open(os.path.join(EEG_AD_FTD_path, 'databases', 'lmer_preds.pkl'), 'rb'))

    # Fig. 2
    # fig2 = plt.figure(figsize=(7.5, 3.5), dpi=300)
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
            if col == 1:
                group = 'ADMOD'
            if col == 2:
                group = 'ADSEV'
            if col == 3:
                group = 'F'
            if col == 4:
                group = 'A'

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

            # Plot EEG
            plot_simple_head_model(ax, 0.6, 0)
            # plot_EEG(fig3, ax, data, 1, 0, -np.max(np.abs(data)), np.max(np.abs(data)))

            if col < 3:
                if row == 0:
                    ylims = [-11,11]
                if row == 1 or row == 2:
                    ylims = [-6,6]
                if row == 3:
                    ylims = [-3.5, 3.5]
            else:
                if row == 0 or row == 1 or row == 3:
                    ylims = [-6,6]
                if row == 2:
                    ylims = [-3.5, 3.5]

            

            plot_EEG(fig3, ax, data, 0.6, 0, ylims[0], ylims[1])

            # ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Titles
            if row == 0:
                ax.set_title(f'{group} vs HC', fontsize=8)

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
    spacing_y = -0.08

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
                if col == 1:
                    group = 'ADMOD'
                if col == 2:
                    group = 'ADSEV'
                if col == 3:
                    group = 'F'
                if col == 4:
                    group = 'A'
                
                
                ax = fig.add_axes([current_left, current_bottom, 0.15, 0.15], frameon=False)
                if col == 2:
                    current_left += width + new_spacing_x
                else:
                    current_left += width + spacing_x
                # ax = fig.add_axes([0.05 + col * 0.19, 0.75 - row * 0.18, 0.15, 0.15])


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
                ylims = [-6, 6]
                if row == 1 and col == 0:
                    vmin = -4.720872348422581
                    vmax = 4.720872348422581
                # Plot EEG
                plot_simple_head_model(ax, 0.6, 0)
                plot_EEG(fig, ax, data, 0.6, 0, ylims[0], ylims[1])

                # ticks
                ax.set_xticks([])
                ax.set_yticks([])

                # Titles
                if row == 1:
                    ax.set_title(f'{group} vs HC', fontsize=8)

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
            current_bottom -= height + spacing_y

        fig.text(0.29, 0.95, 'DB1', ha='center', fontsize=8)
        fig.text(0.79, 0.95, 'DB2', ha='center', fontsize=8)
        linepos = [0.94, 0.94]
        line1 = mlines.Line2D([0.03, 0.56], linepos, color='black', linewidth=0.7)
        line2 = mlines.Line2D([0.63, 0.93], linepos, color='black', linewidth=0.7)
        fig.add_artist(line1)
        fig.add_artist(line2)

    plt.show()