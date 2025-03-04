import os
import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

# Choose whether to compute posteriors and diagnostic metrics
compute_metrics = True

def create_white_to_color_cmap(color):
    """
    Create a colormap that transitions from white to the specified color. Alpha (transparency) is higher for white.

    Parameters
    ----------
    color : str
        Color to transition to. Must be a valid matplotlib color string.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Colormap that transitions from white to the specified color.

    """
    # Define the colormap from white to the specified color
    cmap = LinearSegmentedColormap.from_list('white_to_color', ['white', color])

    # Initialize the colormap's lookup table
    cmap._init()

    # Set alpha (transparency) gradient
    # Use the size of the lookup table (cmap._lut.shape[0]) to ensure compatibility
    cmap._lut[:, -1] = np.linspace(0, 0.6, cmap._lut.shape[0])

    return cmap


# List of methods
all_methods = ['catch22', 'catch22_psp_1', 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
               'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', 'MD_hrv_classic_pnn40',
               'power_spectrum_parameterization_1']

# Set the numpy seed
np.random.seed(0)

# Path to ML models trained based on a held-out dataset approach
ML_path = '/DATOS/pablomc/ML_models/held_out_data_models'

# Dictionaries to store posteriors and diagnostic metrics
all_post_samples = {}
all_theta = {}
z_score = {}
shrinkage = {}
PRE = {}

if compute_metrics:
    for method in all_methods:
        print(f'\n\n--- Method: {method}\n')

        # Initialize the dictionaries
        all_post_samples[method] = []
        all_theta[method] = []
        z_score[method] = []
        shrinkage[method] = []
        PRE[method] = []

        # Load density estimators and inference models
        try:
            # Load X and theta from the held-out dataset
            print('\n--- Loading the held-out dataset')
            with open(os.path.join(ML_path, 'datasets', method, 'held_out_dataset'), 'rb') as file:
                X, theta = pickle.load(file)

            # Calculate the E/I ratio for the held-out dataset
            theta_EI = np.zeros((theta.shape[0], 4))
            theta_EI[:, 0] = (theta[:, 0] / theta[:, 2]) / (theta[:, 1] / theta[:, 3])
            theta_EI[:, 1] = theta[:, 4]
            theta_EI[:, 2] = theta[:, 5]
            theta_EI[:, 3] = theta[:, 6]

            # Variance of the prior
            var_theta = np.var(theta_EI, axis=0)

            print('\n--- Loading SBI models')
            density_estimator = pickle.load(open(os.path.join(ML_path, 'SBI', method, 'density_estimator'), 'rb'))
            model = pickle.load(open(os.path.join(ML_path, 'SBI', method, 'model'), 'rb'))

            # Compute posteriors
            print('\n--- Computing posteriors')
            posterior = [model[i].build_posterior(density_estimator[i]) for i in range(len(density_estimator))]

            # Select some random samples
            print('\n--- Drawing samples from the posteriors')
            idx = np.random.choice(theta_EI.shape[0], 10**5, replace=False)

            # Draw samples from the posteriors
            for xx, sample in enumerate(idx):
                print(f'\r--- Sample {xx + 1}/{len(idx)}, ID: {sample}', end='', flush=True)
                # Observation
                x_o = torch.from_numpy(np.array(X[sample], dtype=np.float32))

                # Check if the observation is valid
                if torch.isnan(x_o).any():
                    print(f'\n--- Invalid observation: {x_o}')
                    continue

                # Posterior samples
                posterior_samples = [post.sample((5000,), x=x_o, show_progress_bars=False) for post in
                                     posterior]

                # Compute the average of the posterior samples
                avg_post_samples = torch.from_numpy(np.zeros((posterior_samples[0].shape)))
                for ii in range(len(posterior_samples)):
                    avg_post_samples += posterior_samples[ii]
                avg_post_samples /= len(posterior_samples)

                # Calculate E/I
                new_post_samples = torch.from_numpy(np.zeros((posterior_samples[0].shape[0], 4)))
                new_post_samples[:, 0] = (avg_post_samples[:, 0] / avg_post_samples[:, 2]) / \
                                         (avg_post_samples[:, 1] / avg_post_samples[:, 3])
                new_post_samples[:, 1] = avg_post_samples[:, 4]
                new_post_samples[:, 2] = avg_post_samples[:, 5]
                new_post_samples[:, 3] = avg_post_samples[:, 6]

                # Store theta and posterior samples
                all_post_samples[method].append(new_post_samples)
                all_theta[method].append(theta_EI[sample, :])

                # Calculate z-score
                z = np.abs( (np.mean(new_post_samples.numpy(),axis = 0) - theta_EI[sample, :]) /\
                            np.std(new_post_samples.numpy(),axis = 0) )
                z_score[method].append(z)

                # Calculate shrinkage
                s = np.ones(4) - np.var(new_post_samples.numpy(),axis = 0) / var_theta
                shrinkage[method].append(s)

                # Calculate PRE
                p = np.median(np.abs(new_post_samples.numpy() - theta_EI[sample, :]), axis = 0)
                PRE[method].append(p)

            # Convert to numpy
            all_theta[method] = np.array(all_theta[method])
            all_post_samples[method] = [post.numpy() for post in all_post_samples[method]]
            z_score[method] = np.array(z_score[method])
            shrinkage[method] = np.array(shrinkage[method])
            PRE[method] = np.array(PRE[method])

        except:
            print(f'\n--- Error loading SBI models for method {method}')
            continue

    # Create folder to save results
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the results
    print('Saving results to file')
    with open('data/all_post_samples.pkl', 'wb') as file:
        pickle.dump(all_post_samples, file)
    with open('data/all_theta.pkl', 'wb') as file:
        pickle.dump(all_theta, file)
    with open('data/z_score.pkl', 'wb') as file:
        pickle.dump(z_score, file)
    with open('data/shrinkage.pkl', 'wb') as file:
        pickle.dump(shrinkage, file)
    with open('data/PRE.pkl', 'wb') as file:
        pickle.dump(PRE, file)
else:
    print('Loading results from file')
    # Load the results
    with open('data/all_post_samples.pkl', 'rb') as file:
        all_post_samples = pickle.load(file)
    with open('data/all_theta.pkl', 'rb') as file:
        all_theta = pickle.load(file)
    with open('data/z_score.pkl', 'rb') as file:
        z_score = pickle.load(file)
    with open('data/shrinkage.pkl', 'rb') as file:
        shrinkage = pickle.load(file)
    with open('data/PRE.pkl', 'rb') as file:
        PRE = pickle.load(file)

# Pick some clusters to plot
method = 'power_spectrum_parameterization_1'
all_gt = np.array([[ 2.3, 1.2, 3.6, 23.],
                   [1., 0.6, 2., 35.]]) # Ground truth values
n_clusters = len(all_gt)
all_cluster_posterior = []
all_cluster_theta = []

# Find the samples that are close to the ground truth values
percentage = 0.25
for gt in all_gt:
    if len(all_theta[method]) > 0:
        pos = np.where(np.min(all_theta[method] >= gt * (1 - percentage), axis=1) & \
                       np.min(all_theta[method] < gt * (1 + percentage), axis=1))[0]
        if len(pos) > 0:
            print(f'Cluster {gt}: {len(pos)} samples found')
            all_cluster_theta.append(np.mean(all_theta[method][pos], axis=0))

            post = np.zeros((all_post_samples[method][0].shape))
            for ii in pos:
                post += all_post_samples[method][ii]
            post /= len(pos)
            all_cluster_posterior.append(post)

# Plots
# Create the figures and set their properties
fig1 = plt.figure(figsize=(7.5, 6), dpi=150)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Pairplot
lims = [[0, 2.5], [0, 2.], [0, 6.5], [20, 40]]
colors = ['blue', 'green', 'orchid']
for row in range(4):
    for col in np.arange(row,4):
        ax = fig1.add_axes([0.04 + col * 0.14, 0.84 - row * 0.14, 0.1, 0.1])

        try:
            # Diagonal: 1D histogram
            if row == col:
                for cluster in range(n_clusters):
                    hist, bin_edges = np.histogram(all_cluster_posterior[cluster][:, row], bins=50, density=True)
                    ax.plot(bin_edges[:-1], hist/np.max(hist),color = colors[cluster], alpha = 0.5, linewidth = 2.5)
                    ax.set_xlim(lims[row])

                    # Ground truth values
                    ax.plot([all_cluster_theta[cluster][row],all_cluster_theta[cluster][row]], [0,1], color = colors[cluster],
                            linewidth = 0.5, linestyle = '--')

                # x-labels
                if row == 0:
                    ax.set_xlabel(r'$E/I$')
                elif row == 1:
                    ax.set_xlabel(r'$\tau_{syn}^{exc}$')
                elif row == 2:
                    ax.set_xlabel(r'$\tau_{syn}^{inh}$')
                else:
                    ax.set_xlabel(r'$J_{syn}^{ext}$')

                # ticks
                ax.set_yticks([])


            # Upper triangle: 2D histogram
            elif row < col:
                for cluster in range(n_clusters):
                    hist, x_edges, y_edges = np.histogram2d(all_cluster_posterior[cluster][:, col],
                                                            all_cluster_posterior[cluster][:, row],
                                                            bins=50, density=True)

                    # Create a custom colormap for this cluster
                    cmap = create_white_to_color_cmap(colors[cluster])

                    # Plot with transparency
                    ax.pcolormesh(x_edges, y_edges, hist.T, shading='auto', cmap=cmap, vmin=0, vmax=np.max(hist))

                    # Ground truth values
                    ax.scatter([all_cluster_theta[cluster][col]], [all_cluster_theta[cluster][row]], s = 5.,
                               c = colors[cluster])

                # Set axes limits
                ax.set_xlim(lims[col])
                ax.set_ylim(lims[row])

        except:
            pass

# Z-scores versus posterior shrinkage
labels = [r'$E/I$',r'$\tau_{syn}^{exc}$',r'$\tau_{syn}^{inh}$',r'$J_{syn}^{ext}$']
colors = ['#808080', '#A9A9A9', '#D3D3D3', '#DCDCDC']
for row in range(3):
    for col in range(2):
        ax = fig1.add_axes([0.67 + col * 0.18, 0.8 - row * 0.21, 0.12, 0.13])

        try:
            method = all_methods[row * 2 + col]
            for param in range(4):
                ax.scatter(shrinkage[method][:,param], z_score[method][:,param], s = 0.1,
                           c = colors[param], label = labels[param])
        except:
            pass

        # x/y-labels
        if row == 2:
            ax.set_xlabel('shrinkage')
        if col == 0:
            ax.set_ylabel('z-score')

        # titles
        if (row * 2 + col) == 0:
            ax.set_title(r'$catch22$', fontsize = 8)
        elif (row * 2 + col) == 1:
            ax.set_title(r'$ch22 + $' + ' ' + r'slp', fontsize = 8)
        elif (row * 2 + col) == 2:
            ax.set_title(r'$dfa$', fontsize = 8)
        elif (row * 2 + col) == 3:
            ax.set_title(r'$rs\ range$', fontsize = 8)
        elif (row * 2 + col) == 4:
            ax.set_title(r'$high\ fluct.$', fontsize = 8)
        elif (row * 2 + col) == 5:
            ax.set_title(r'$1/f$' + ' ' + r'$slope$', fontsize = 8)

        # legend
        if row == 0 and col == 0:
            ax.legend(loc = 'upper left', fontsize=6, handletextpad=0.2, borderpad=0.2, labelspacing=0.2)

        # # limits
        # ax.set_xlim([0,1])
        # ax.set_ylim([0,5])

# Parameter recovery error (PRE)
colors = ['#FFC0CB', '#FF69B4', '#00FF00', '#32CD32', '#228B22', '#006400']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=6)

for col in range(4):
    ax = fig1.add_axes([0.08 + col * 0.24, 0.07, 0.18, 0.18])

    # Define bins to compute histograms
    if col == 0:
        bins = np.linspace(0, 5, 15)
    elif col == 1:
        bins = np.linspace(0, 5, 15)
    elif col == 2:
        bins = np.linspace(0, 5, 15)
    else:
        bins = np.linspace(0, 10, 15)

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
            hist, bin_edges = np.histogram(PRE[method][:,col], bins=bins, density=True)

            # Smooth the histogram using a Gaussian filter
            smoothed_hist = gaussian_filter1d(hist, sigma=1)
            ax.plot(bin_edges[:-1], smoothed_hist, label=label, color=colors[ii], alpha=0.4, linewidth = 1.5)

        except:
            continue

    # legend
    if col == 0:
        ax.legend(loc='upper right', fontsize=6, handletextpad=0.2, borderpad=0.2, labelspacing=0.2)

    # labels
    if col == 0:
        ax.set_ylabel('probability density', fontsize=8)
    ax.set_xlabel('PRE', fontsize=8)

    # titles
    if col == 0:
        ax.set_title(r'$E/I$', fontsize=10)
    elif col == 1:
        ax.set_title(r'$\tau_{syn}^{exc}$', fontsize=10)
    elif col == 2:
        ax.set_title(r'$\tau_{syn}^{inh}$', fontsize=10)
    else:
        ax.set_title(r'$J_{syn}^{ext}$', fontsize=10)

# Plot letters
ax = fig1.add_axes([0., 0., 1., 1.])
ax.axis('off')
ax.text(0.01, 0.97, 'A', fontsize=12, fontweight='bold')
ax.text(0.59, 0.97, 'B', fontsize=12, fontweight='bold')
ax.text(0.01, 0.28, 'C', fontsize=12, fontweight='bold')

# Save the figure
plt.savefig('SBI_results.png', bbox_inches='tight')
# plt.show()