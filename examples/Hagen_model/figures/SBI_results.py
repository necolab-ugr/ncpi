import os
import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import RepeatedKFold

# Choose whether to compute posteriors and diagnostic metrics
compute_metrics = False

# Choose whether to use a held-out dataset or folds from RepeatedKFold
use_held_out_data = True

# Number of random samples to draw from the posteriors
n_samples = 10**5

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
if use_held_out_data:
    ML_path = '/DATOS/pablomc/ML_models/held_out_data_models'
# Path to ML models trained based on a RepeatedKFold approach
else:
    ML_path = '/DATOS/pablomc/ML_models/4_var'

if compute_metrics:
    # Dictionaries to store posteriors and diagnostic metrics
    all_post_samples = {}
    all_theta = {}
    z_score = {}
    shrinkage = {}
    abs_error = {}
    PRE = {}

    for method in all_methods:
        print(f'\n\n--- Method: {method}\n')

        # Initialize dictionaries
        all_post_samples[method] = []
        all_theta[method] = []
        z_score[method] = []
        shrinkage[method] = []
        abs_error[method] = []
        PRE[method] = []

        # Load density estimators and inference models
        try:
            # Load X and theta from the held-out dataset
            if use_held_out_data:
                print('\n--- Loading the held-out dataset')
                with open(os.path.join(ML_path, 'datasets', method, 'held_out_dataset'), 'rb') as file:
                    X, theta = pickle.load(file)
            # Load X and theta from all folds of RepeatedKFold and concatenate them
            else:
                X = pickle.load(open(os.path.join('/DATOS/pablomc/data',method,'sim_X'),'rb'))
                theta = pickle.load(open(os.path.join('/DATOS/pablomc/data',method,'sim_theta'),'rb'))
                rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
                new_X = []
                new_theta = []
                for repeat_idx, (train_index, test_index) in enumerate(rkf.split(X)):
                    if repeat_idx == 0:
                        new_X = X[test_index]
                        new_theta = theta['data'][test_index]
                    else:
                        new_X = np.concatenate((new_X,X[test_index]),axis=0)
                        new_theta = np.concatenate((new_theta,theta['data'][test_index]),axis=0)
                X = new_X
                theta = new_theta

            # Calculate the E/I ratio
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
            print(f'\n--- Drawing {n_samples} samples from the posteriors')
            idx = np.random.choice(theta_EI.shape[0], n_samples, replace=False)

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
                if use_held_out_data:
                    posterior_samples = [post.sample((5000,), x=x_o, show_progress_bars=False) for post in
                                         posterior]
                else:
                    fold = int(10. * sample / theta_EI.shape[0])
                    posterior_samples = [posterior[fold].sample((5000,), x=x_o, show_progress_bars=False)]

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

                # Calculate absolute error
                e = np.abs( np.mean(new_post_samples.numpy(),axis = 0) - theta_EI[sample, :] )
                abs_error[method].append(e)

                # Calculate PRE for the smallest 25% of the differences between the posterior samples and the
                # ground truth
                diff = np.abs(new_post_samples.numpy() - theta_EI[sample, :])
                p = np.mean(np.sort(diff,axis = 0)[:int(0.25*diff.shape[0]),:],axis = 0)
                PRE[method].append(p)

            # Convert to numpy
            all_theta[method] = np.array(all_theta[method])
            all_post_samples[method] = [post.numpy() for post in all_post_samples[method]]
            z_score[method] = np.array(z_score[method])
            shrinkage[method] = np.array(shrinkage[method])
            abs_error[method] = np.array(abs_error[method])
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
    with open('data/abs_error.pkl', 'wb') as file:
        pickle.dump(abs_error, file)
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
    with open('data/abs_error.pkl', 'rb') as file:
        abs_error = pickle.load(file)
    with open('data/PRE.pkl', 'rb') as file:
        PRE = pickle.load(file)

# Pick 2 posteriors to plot
method = 'catch22'
pos = np.argsort(np.sum(z_score[method],axis = 1))

s0 = pos[0]
s1 = pos[1]
# Find the first posterior that is different to the first one
for i in np.arange(1,len(pos)):
    sel = True
    for param in range(4):
        diff = np.abs((all_theta[method][pos[i]][param] - all_theta[method][s0][param]) /\
                      np.max(all_theta[method][:,param]))
        if  diff < 0.05:
            sel = False
    if sel:
        s1 = pos[i]
        break

all_theta_plot = [all_theta[method][s0], all_theta[method][s1]]
all_posterior_plot = [all_post_samples[method][s0], all_post_samples[method][s1]]

# Plots
# Create the figures and set their properties
fig1 = plt.figure(figsize=(7.5, 6), dpi=150)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Pairplot
lims = [[0, 5], [0, 3.], [-4, 6.5], [10, 50]]
colors = ['blue', 'green']
for row in range(4):
    for col in np.arange(row,4):
        ax = fig1.add_axes([0.04 + col * 0.14, 0.84 - row * 0.14, 0.1, 0.1])

        try:
            # Diagonal: 1D histogram
            if row == col:
                for sample in range(2):
                    hist, bin_edges = np.histogram(all_posterior_plot[sample][:, row], bins=50, density=True)
                    # 1D smoothing
                    hist = gaussian_filter1d(hist, sigma=1)
                    ax.plot(bin_edges[:-1], hist/np.max(hist),color = colors[sample], alpha = 0.5, linewidth = 2.5)
                    ax.set_xlim(lims[row])

                    # Ground truth values
                    ax.plot([all_theta_plot[sample][row],all_theta_plot[sample][row]], [0,1], color = colors[sample],
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
                for sample in range(2):
                    hist, x_edges, y_edges = np.histogram2d(all_posterior_plot[sample][:, col],
                                                            all_posterior_plot[sample][:, row],
                                                            bins=50, density=True)

                    # Create a custom colormap for this sample
                    cmap = create_white_to_color_cmap(colors[sample])

                    # Plot with transparency
                    ax.pcolormesh(x_edges, y_edges, hist.T, shading='auto', cmap=cmap, vmin=0, vmax=np.max(hist))

                    # Ground truth values
                    ax.scatter([all_theta_plot[sample][col]], [all_theta_plot[sample][row]], s = 5.,
                               c = colors[sample])

                # Set axes limits
                ax.set_xlim(lims[col])
                ax.set_ylim(lims[row])

        except:
            pass

# Legend
ax = fig1.add_axes([0.04, 0.35, 0.2, 0.2])
ax.axis('off')
ax.plot([], [], color='blue', label='sample 1')
ax.plot([], [], color='green', label='sample 2')
# Add ground truth values
ax.plot([], [], color='black', linestyle='--', label='ground truth ('+r'$\theta_0$)')
ax.scatter([], [], color='black', s = 1., label='ground truth ('+r'$\theta_0$)')
ax.legend(loc='upper left', fontsize=8)

# Z-scores versus posterior shrinkage
labels = [r'$E/I$',r'$\tau_{syn}^{exc}$',r'$\tau_{syn}^{inh}$',r'$J_{syn}^{ext}$']
for row in range(3):
    for col in range(2):
        ax = fig1.add_axes([0.67 + col * 0.18, 0.8 - row * 0.21, 0.12, 0.13])

        try:
            method = all_methods[row * 2 + col]

            x = np.linspace(0., 1.05, 100)
            y = np.linspace(0., 10.05, 100)
            # hist, x_edges, y_edges = np.histogram2d(shrinkage[method].flatten(),
            #                                         z_score[method].flatten(),
            #                                         bins=(x, y), density=True)
            hist, x_edges, y_edges = np.histogram2d(shrinkage[method][:,0],
                                                    z_score[method][:,0],
                                                    bins=(x, y), density=True)

            # Low-pass filtering
            hist = gaussian_filter1d(hist, sigma=7, axis=0)
            hist = gaussian_filter1d(hist, sigma=7, axis=1)

            ax.pcolormesh(x_edges, y_edges, hist.T, shading='auto', cmap='Reds')

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

# Histograms of errors
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
        bins = np.linspace(0, 10, 15)
    else:
        bins = np.linspace(0, 30, 15)

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
            smoothed_hist = gaussian_filter1d(hist, sigma=2)
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
ax.text(0.01, 0.28, 'B', fontsize=12, fontweight='bold')
ax.text(0.59, 0.97, 'C', fontsize=12, fontweight='bold')

# Save the figure
plt.savefig('SBI_results.png', bbox_inches='tight')
# plt.show()