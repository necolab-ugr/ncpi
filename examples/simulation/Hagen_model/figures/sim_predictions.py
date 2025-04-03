import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

# Machine learning model
ML_model = 'MLP'

def hellinger_distance(p, q):
    """
    Compute the Hellinger Distance between two discrete probability distributions P and Q.

    Parameters:
    p (np.array): Probability distribution P.
    q (np.array): Probability distribution Q.

    Returns:
    float: Hellinger Distance between P and Q.
    """
    # Ensure the inputs are numpy arrays
    p = np.asarray(p)
    q = np.asarray(q)

    # Normalize the distributions to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute the Hellinger Distance
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

# Path to ML models trained based on a held-out dataset approach
ML_path = '/DATOS/pablomc/ML_models/held_out_data_models'

# Set the random seed
np.random.seed(0)

# Load predictions and actual parameters
all_methods = ['catch22', 'catch22_psp_1', 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
               'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', 'MD_hrv_classic_pnn40',
               'power_spectrum_parameterization_1']
all_preds = {}
all_theta = {}
for method in all_methods:
    try:
        with open(os.path.join(ML_path, ML_model, method, 'predictions'), 'rb') as file:
            all_preds[method] = np.array(pickle.load(file))
        with open(os.path.join(ML_path, 'datasets', method, 'held_out_dataset'), 'rb') as file:
            X_test, theta_test = pickle.load(file)
            all_theta[method] = np.array(theta_test)
    except:
        print('No data for method: ', method)

# Create a figure and set its properties
fig1 = plt.figure(figsize=(7.5, 6), dpi=300)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Define a colormap
cmap = plt.colormaps['pink']

# Plot predictions vs actual parameters
all_errors = {'E_I': {}, 'tau_syn_exc': {}, 'tau_syn_inh': {}, 'J_syn_ext': {}}
for row in range(4):
    for col in range(6):
        ax = fig1.add_axes([0.12 + col * 0.15, 0.84 - row * 0.15, 0.09, 0.1])
        method = all_methods[col]

        try:
            # Compute E/I ratio
            if row == 0:
                pred_param = (all_preds[method][:,0]/all_preds[method][:,2]) /\
                             (all_preds[method][:,1]/all_preds[method][:,3])
                actual_param = (all_theta[method][:,0]/all_theta[method][:,2]) /\
                               (all_theta[method][:,1]/all_theta[method][:,3])
            else:
                pred_param = all_preds[method][:, row+3]
                actual_param = all_theta[method][:, row+3]

            # Remove nan values
            nan_pos = np.where(np.isnan(pred_param))[0]
            if len(nan_pos) > 0:
                pred_param = np.delete(pred_param, nan_pos)
                actual_param = np.delete(actual_param, nan_pos)

            # Define the range of the parameter
            if row == 0:
                min_ = 0.1
                max_ = 8.
            elif row == 1:
                min_ = 0.1
                max_ = 2.
            elif row == 2:
                min_ = 1.
                max_ = 8.
            elif row == 3:
                min_ = 10.
                max_ = 40.

            param_range = np.linspace(min_, max_, 8)

            # Use the same number of samples for each bin
            min_number = 10**10
            for ii,param in enumerate(param_range[:-1]):
                pos = np.where((actual_param >= param_range[ii]) & (actual_param < param_range[ii+1]))[0]
                if len(pos) < min_number:
                    min_number = len(pos)

            # Plots
            all_loc = []
            all_minmax = []
            abs_error = []
            for ii,param in enumerate(param_range[:-1]):
                pos = np.where((actual_param >= param_range[ii]) & (actual_param < param_range[ii+1]))[0]
                loc = (param_range[ii] + param_range[ii+1]) / 2
                all_loc.append(loc)

                # Choose the same number of samples for each bin
                if len(pos) > min_number:
                    pos = np.random.choice(pos, min_number, replace=False)

                # print(f'Method: {method}, row: {row}, col: {col}, param: {param}, number os samples: {len(pos)}')

                if len(pos) == 0:
                    continue

                # # Clip the data between the 15 % and 85 % quantiles
                if row == 0 and col == 3:
                   pp1 = 15
                   pp2 = 85
                # Clip the data between the 5 % and 95 % quantiles
                else:
                    pp1 = 5
                    pp2 = 95

                q1, q3 = np.percentile(pred_param[pos], [pp1, pp2])
                clipped_data = pred_param[pos][(pred_param[pos] >= q1) & (pred_param[pos] <= q3)]
                all_minmax.append([np.min(clipped_data), np.max(clipped_data)])

                # Violin plot
                violin = ax.violinplot(clipped_data, positions=[loc], widths=(max_ - min_)/9., showextrema=False)

                for pc in violin['bodies']:
                    pc.set_facecolor(cmap(ii / len(param_range)))
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.8)
                    pc.set_linewidth(0.2)

                # Boxplot
                box = ax.boxplot(pred_param[pos], positions=[loc], showfliers=False,
                                 widths=(max_ - min_)/9., patch_artist=True,
                                 medianprops=dict(color='red', linewidth=0.8),
                                 whiskerprops=dict(color='black', linewidth=0.5),
                                 capprops=dict(color='black', linewidth=0.5),
                                 boxprops=dict(linewidth=0.5, facecolor=(0, 0, 0, 0)))

                for patch in box['boxes']:
                    patch.set_linewidth(0.2)

                # Compute absolute errors
                error = np.abs(pred_param[pos] - actual_param[pos])
                abs_error.extend(error)

            # Store the errors
            all_errors[list(all_errors.keys())[row]][method] = abs_error

            # ticks
            ax.set_xticks(all_loc[::2 if row < 3 else 3])
            ax.set_xticklabels(['%.1f' % x for x in all_loc[::2 if row < 3 else 3]])
            all_minmax = np.array(all_minmax)
            ax.set_yticks(np.linspace(np.min(all_minmax), np.max(all_minmax), 3))
            ax.set_yticklabels(['%.1f' % x for x in np.linspace(np.min(all_minmax), np.max(all_minmax), 3)])

            # x-/y-labels
            if row == 3:
                ax.set_xlabel('actual', fontsize = 8)
            if col == 0:
                ax.set_ylabel('predicted', fontsize = 8)
                ax.yaxis.set_label_coords(-0.5, 0.5)

            # titles
            if row == 0:
                if col == 0:
                    ax.set_title(r'$catch22$', fontsize = 8)
                elif col == 1:
                    ax.set_title(r'$catch22 + $'+  r'$1/f$'+' '+r'$slope$', fontsize = 8)
                elif col == 2:
                    ax.set_title(r'$dfa$', fontsize = 8)
                elif col == 3:
                    ax.set_title(r'$rs\ range$', fontsize = 8)
                elif col == 4:
                    ax.set_title(r'$high\ fluct.$', fontsize = 8)
                elif col == 5:
                    ax.set_title(r'$1/f$'+' '+r'$slope$', fontsize = 8)

            # limits
            ax.set_xlim([min_, max_])

        except:
            continue

# Show parameter labels
for row in range(4):
    ax = fig1.add_axes([0.01, 0.84 - row * 0.15, 0.04, 0.1])
    ax.axis('off')

    if row == 0:
        ax.text(0.5, 0.5, r'$E/I$', fontsize=8, ha='center')
    elif row == 1:
        ax.text(0.5, 0.5, r'$\tau_{syn}^{exc}$', fontsize=8, ha='center')
    elif row == 2:
        ax.text(0.5, 0.5, r'$\tau_{syn}^{inh}$', fontsize=8, ha='center')
    else:
        ax.text(0.5, 0.5, r'$J_{syn}^{ext}$', fontsize=8, ha='center')

# Plot errors
colors = ['#FFC0CB', '#FF69B4', '#00FF00', '#32CD32', '#228B22', '#006400']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=6)

for col in range(4):
    ax = fig1.add_axes([0.08 + col * 0.24, 0.07, 0.18, 0.18])
    all_hist = []

    # Define bins to compute histograms
    if col == 0:
        bins = np.linspace(0, 10, 15)
    elif col == 1:
        bins = np.linspace(0, 1.5, 15)
    elif col == 2:
        bins = np.linspace(0, 5, 15)
    else:
        bins = np.linspace(0, 40, 15)

    for ii,method in enumerate(all_methods):
        try:
            # titles
            if ii == 0:
                label = r'$catch22$'
            elif ii == 1:
                label = r'$ch22 + $'+' '+r'slp'
            elif ii == 2:
                label = r'$dfa$'
            elif ii == 3:
                label = r'$rs\ range$'
            elif ii == 4:
                label = r'$high\ fluct.$'
            elif ii == 5:
                label = r'$1/f$'+' '+r'$slope$'

            # Compute histogram
            hist, bin_edges = np.histogram(all_errors[list(all_errors.keys())[col]][method], bins=bins, density=True)

            # Smooth the histogram using a Gaussian filter
            smoothed_hist = gaussian_filter1d(hist, sigma=1)
            all_hist.append(smoothed_hist)
            ax.plot(bin_edges[:-1], smoothed_hist, label=label, color=colors[ii], alpha=0.4, linewidth = 1.5)

        except:
            continue

    # Compute pairwise Hellinger distances
    HD_ = []
    for x in range(6):
        for y in range(x+1, 6):
            HD_.append(hellinger_distance(all_hist[x], all_hist[y]))
            # print('Hellinger distance between', all_methods[x], 'and', all_methods[y], ':', HD_[-1])

    # Show distances
    # Between catch22 and catch22 + slope
    ax.text(0.2, 0.9 if col == 0 else 0.25, r'$D_{H,1}=$'+' %.2f' % HD_[0], fontsize=6, ha='center',
            transform=ax.transAxes, color = colors[1])
    # Between all the remaining methods
    ax.text(0.2, 0.15, r'$D_{H,2}=$'+' %.2f' % np.mean(HD_[9:]), fontsize=6, ha='center', transform=ax.transAxes,
            color = colors[5])
    # Between catch22 and all the remaining methods
    ax.text(0.2, 0.05, r'$D_{H,3}=$'+' %.2f' % np.mean(HD_[1:5]), fontsize=6, ha='center', transform=ax.transAxes,
            color = 'k')

    # legend
    if col == 0:
        ax.legend(loc='upper right', fontsize=6, handletextpad=0.2, borderpad=0.2, labelspacing=0.2)

    # labels
    if col == 0:
        ax.set_ylabel('probability density', fontsize=8)
    ax.set_xlabel('absolute error', fontsize=8)

    # titles
    if col == 0:
        ax.set_title(r'$E/I$', fontsize=10)
    elif col == 1:
        ax.set_title(r'$\tau_{syn}^{exc}$', fontsize=10)
    elif col == 2:
        ax.set_title(r'$\tau_{syn}^{inh}$', fontsize=10)
    else:
        ax.set_title(r'$J_{syn}^{ext}$', fontsize=10)

    # limits
    if col == 0:
        ax.set_xlim([0, 12])

# Plot letters
ax = fig1.add_axes([0., 0., 1., 1.])
ax.axis('off')
ax.text(0.005, 0.97, 'A', fontsize=12, fontweight='bold')
ax.text(0.01, 0.28, 'B', fontsize=12, fontweight='bold')

# Save the figure
plt.savefig(f'sim_predictions_{ML_model}.png', bbox_inches='tight')
# plt.show()