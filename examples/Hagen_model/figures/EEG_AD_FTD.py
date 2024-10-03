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

EEG_AD_FTD_path = '/DATOS/pablomc/EEG_AD_FTD_results'
n_var = 1

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
    # Ears
    patches = []
    right_ear = mpatches.FancyBboxPatch([pos + radius + radius / 20, -radius/10],
                                        radius/50, radius/5,
        boxstyle=mpatches.BoxStyle("Round", pad=radius/20))
    patches.append(right_ear)

    left_ear = mpatches.FancyBboxPatch([pos -radius - radius / 20 - radius / 50,
                                        -radius / 10],radius / 50, radius / 5,
        boxstyle=mpatches.BoxStyle("Round", pad=radius/20))

    patches.append(left_ear)
    collection = PatchCollection(patches, facecolor='none',
                                 edgecolor='k', alpha=1.0, lw=2)
    ax.add_collection(collection)

    # Circumference of the head
    circ_npts = 100
    head_x = pos + radius * np.cos(np.linspace(0, 2 * np.pi, circ_npts))
    head_y = radius * np.sin(np.linspace(0, 2 * np.pi, circ_npts))

    ax.plot(head_x, head_y, 'k')
    ax.plot([radius])

    # Nose
    ax.plot([pos -radius / 10, pos, pos + radius  / 10], [radius,
            radius + radius/10, radius], 'k')

def plot_EEG(fig, Vax, data, radius, pos, vmin, vmax):
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
    Vax.contour(xi, yi, zi, 5, colors = "grey", zorder = 2,linewidths = 0.5,
                vmin = vmin,vmax = vmax)

    # Make a color bar
    # cbar = fig.colorbar(CS, ax=Vax)
    fig.colorbar(ScalarMappable(norm=CS.norm, cmap=CS.cmap), ax = Vax)

    # Add the EEG electrode positions
    Vax.scatter(x[:20], y[:20], marker = 'o', c = 'k', s = 2, zorder = 3)


if __name__ == "__main__":
    # Load features
    # POCTEP dataset
    emp_data_POCTEP_source = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_POCTEP_False.pkl'))
    emp_data_POCTEP_raw = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_POCTEP_True.pkl'))
    # OpenNEURO dataset
    emp_data_OpenNeuro = pd.read_pickle(os.path.join(EEG_AD_FTD_path, 'catch22', 'emp_data_OpenNEURO.pkl'))

    # Load LMER results
    lmer_feat = pickle.load(open(os.path.join(EEG_AD_FTD_path, 'catch22', 'lmer_feat.pkl'), 'rb'))
    lmer_preds = pickle.load(open(os.path.join(EEG_AD_FTD_path, 'catch22', 'lmer_preds.pkl'), 'rb'))

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


    # Fig. 3
    fig3 = plt.figure(figsize=(7.5, 4.), dpi=300)

    for row in range(4):
        for col in range(5):
            ax = fig3.add_axes([0.05 + col * 0.19, 0.72 - row * 0.23, 0.15, 0.2])

            if row == 0:
                feat = 18
            if row == 1:
                feat = 8
            if row == 2:
                feat = 4
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
            plot_simple_head_model(ax, 1, 0)
            # plot_EEG(fig3, ax, data, 1, 0, -np.max(np.abs(data)), np.max(np.abs(data)))

            if col < 3:
                if row == 0:
                    ylims = [-12,12]
                else:
                    ylims = [-6,6]
            else:
                ylims = [-6, 6]

            plot_EEG(fig3, ax, data, 1, 0, ylims[0], ylims[1])

            # ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Titles
            if row == 0:
                ax.set_title(f'{group}vsHC')

            # Labels
            if col == 0:
                if row == 0:
                    ax.set_ylabel(r'$rs\_range$')
                if row == 1:
                    ax.set_ylabel(r'$TransVar$')
                if row == 2:
                    ax.set_ylabel(r'$ami2$')
                if row == 3:
                    ax.set_ylabel(r'$dfa$')

    # Figs. 5 & 6
    for param in range(n_var):
        fig = plt.figure(figsize=(7.5, 5.), dpi=300)

        for row in range(5):
            for col in range(5):
                ax = fig.add_axes([0.05 + col * 0.19, 0.8 - row * 0.18, 0.15, 0.15])

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

                # Get lmer results
                if col < 3:
                    lmer_results = lmer_preds[n_var-1]['DB1'][all_confs[row]][param]
                else:
                    lmer_results = lmer_preds[n_var-1]['DB2'][all_confs[row]][param]

                data = []
                for elec in range(19):
                    p_value = lmer_results[f'{group}vsHC']['p.value'][elec]
                    z_score = lmer_results[f'{group}vsHC']['z.ratio'][elec]
                    if p_value < 0.01 and np.abs(z_score) > 2.5:
                        data.append(z_score)
                    else:
                        data.append(0)

                # Plot EEG
                plot_simple_head_model(ax, 1, 0)
                plot_EEG(fig3, ax, data, 1, 0, -np.max(np.abs(data)), np.max(np.abs(data)))

                # ticks
                ax.set_xticks([])
                ax.set_yticks([])

                # Titles
                if row == 0:
                    ax.set_title(f'{group}vsHC')

                # Labels
                if col == 0:
                    if row == 0:
                        ax.set_ylabel(r'$catch22$')
                    if row == 1:
                        ax.set_ylabel(r'$rs\_range$')
                    if row == 2:
                        ax.set_ylabel(r'$TransVar$')
                    if row == 3:
                        ax.set_ylabel(r'$ami2$')
                    if row == 4:
                        ax.set_ylabel(r'$dfa$')


    plt.show()