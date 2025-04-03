import os
import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import RepeatedKFold

# Choose whether to compute posteriors and diagnostic metrics
compute_metrics = True

# Choose whether to use a held-out dataset or folds from RepeatedKFold
use_held_out_data = True

# Number of random samples to draw from the posteriors
n_samples = 25000

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

# Set the seeds
np.random.seed(0)
torch.manual_seed(0)

# Path to ML models trained based on a held-out dataset approach
if use_held_out_data:
    ML_path = '/DATOS/pablomc/ML_models/held_out_data_models'
# Path to ML models trained based on a RepeatedKFold approach
else:
    ML_path = '/DATOS/pablomc/ML_models/4_var'

# Limits of histograms
lims = [[-15, 15], [-2, 5], [-2, 12], [0, 60]]

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
        # if method == 'catch22' or method == 'catch22_psp_1' or method == 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1':
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

            # Variance of the prior (due to the Law of Large Numbers, this variance will be very close to the
            # population variance)
            var_theta = np.var(theta_EI, axis=0)

            print('\n--- Loading SBI models')
            density_estimator = pickle.load(open(os.path.join(ML_path, 'SBI', method, 'density_estimator'), 'rb'))
            model = pickle.load(open(os.path.join(ML_path, 'SBI', method, 'model'), 'rb'))
            scaler = pickle.load(open(os.path.join(ML_path, 'SBI', method, 'scaler'), 'rb'))

            # Print info about the density estimator
            flow = density_estimator[0]  # Extract the Flow model
            transform = flow._transform  # Get the transform object

            # Extract the first MaskedAffineAutoregressiveTransform
            maf_transform = transform._transforms[1]  # First MAF layer

            # Extract the autoregressive network (MADE)
            made = maf_transform.autoregressive_net

            # Input features
            input_features = made.context_layer.in_features

            # Hidden features
            output_features = made.context_layer.out_features

            print(f"Number of input features: {input_features}")
            print(f"Number of hidden features: {output_features}")

            # Compute posteriors
            print('\n--- Computing posteriors')
            posterior = [model[i].build_posterior(density_estimator[i]) for i in range(len(density_estimator))]

            # Select some random samples
            print(f'\n--- Drawing {n_samples} samples from the held-out dataset')
            if method == 'catch22':
                idx = np.random.choice(theta_EI.shape[0], n_samples, replace=False)
            n_post_samples = 5000
            print(f'--- Sampling {n_post_samples} times the posteriors\n\n')

            # Draw samples from the posteriors
            for xx, sample in enumerate(idx):
                print(f'\r--- Sample {xx + 1}/{len(idx)}, ID: {sample}', end='', flush=True)

                # Features (observation)
                feat = X[sample]
                # Transform the features
                feat = scaler.transform(feat.reshape(1, -1))
                # Torch tensor
                x_o = torch.from_numpy(np.array(feat, dtype=np.float32))

                # Check if the observation is valid
                if torch.isnan(x_o).any():
                    print(f'\n--- Invalid observation: {x_o}')
                    continue

                # Posterior samples
                if use_held_out_data:
                    posterior_samples = [post.sample((n_post_samples,), x=x_o, show_progress_bars=False) for post in
                                         posterior]
                else:
                    fold = int(10. * sample / theta_EI.shape[0])
                    posterior_samples = [posterior[fold].sample((n_post_samples,), x=x_o, show_progress_bars=False)]

                # Calculate E/I
                new_post_samples = [np.zeros((posterior_samples[0].shape[0],
                                              4)) for _ in range(len(posterior_samples))]
                for ii in range(len(posterior_samples)):
                    new_post_samples[ii][:, 0] = (posterior_samples[ii][:, 0] / posterior_samples[ii][:, 2]) /\
                                                 (posterior_samples[ii][:, 1] / posterior_samples[ii][:, 3])
                    new_post_samples[ii][:, 1] = posterior_samples[ii][:, 4]
                    new_post_samples[ii][:, 2] = posterior_samples[ii][:, 5]
                    new_post_samples[ii][:, 3] = posterior_samples[ii][:, 6]

                # Average the sorted posterior samples across folds
                avg_post_samples = np.zeros((new_post_samples[0].shape))
                for ii in range(len(new_post_samples)):
                    avg_post_samples += np.sort(new_post_samples[ii], axis=0)
                avg_post_samples /= len(new_post_samples)

                # Store theta and posterior samples
                all_post_samples[method].append(avg_post_samples)
                all_theta[method].append(theta_EI[sample, :])

                # Diagnostic metrics
                z = np.zeros(4)
                s = np.zeros(4)
                e = np.zeros(4)
                p = np.zeros(4)
                try:
                    for param in range(4):
                        # Remove low-probability samples to prevent distributions with long tails
                        hist, bin_edges = np.histogram(avg_post_samples[:,param],
                                                       bins=np.linspace(lims[param][0], lims[param][1], 1000),
                                                       density=True)
                        hist = gaussian_filter1d(hist, sigma=5)
                        hist /= np.max(hist)
                        iii = bin_edges[np.where(hist > 0.5)[0]]
                        pos = np.where((avg_post_samples[:,param] > iii[0]) & (avg_post_samples[:,param] <= iii[-1]))[0]
                        param_post_samples = avg_post_samples[pos,param]

                        # Calculate z-score
                        z[param] = np.abs( (np.mean(param_post_samples) - theta_EI[sample, param]) /\
                                            np.std(param_post_samples) )

                        # Calculate shrinkage
                        s[param] = 1. - np.var(param_post_samples) / var_theta[param]

                        # # Debug
                        # print(f'--- Parameter {param}:')
                        # print(f'z-score: {z[param]}, shrinkage: {s[param]}')
                        # plt.plot(bin_edges[1:], hist, label='original',alpha=0.5, color='blue', linewidth=1.5)
                        # hist, bin_edges = np.histogram(param_post_samples,
                        #                                bins=np.linspace(lims[param][0], lims[param][1], 1000),
                        #                                density=True)
                        # hist = gaussian_filter1d(hist, sigma=5)
                        # hist /= np.max(hist)
                        # plt.plot(bin_edges[1:], hist, label='filtered',alpha=0.5, color='red', linewidth=1.5)
                        # plt.legend()
                        # plt.show()

                        # Calculate absolute error
                        e[param] = np.abs( np.mean(param_post_samples) - theta_EI[sample, param] )

                        # Calculate PRE for the smallest 25% of the differences between the posterior samples and the
                        # ground truth
                        diff = np.abs(param_post_samples - theta_EI[sample, param])
                        p[param] = np.mean(np.sort(diff)[:int(0.25*len(diff))])
                except:
                    pass

                z_score[method].append(z)
                shrinkage[method].append(s)
                abs_error[method].append(e)
                PRE[method].append(p)

            # Convert to numpy
            all_theta[method] = np.array(all_theta[method])
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

pos = np.argsort(np.sum(z_score[method] + (1 - shrinkage[method]),axis = 1))
s0 = pos[0]
s1 = pos[1]

# # Find a different sample
# for i in np.arange(1,len(pos)):
#     sel = True
#     for param in range(4):
#         diff = np.abs((all_theta[method][pos[i]][param] - all_theta[method][s0][param]) /\
#                       np.max(all_theta[method][:,param]))
#         if  diff < 0.1:
#             sel = False
#     if sel:
#         s1 = pos[i]
#         break

# np.random.seed(int(time.time()))
# s0 = np.random.randint(0, len(all_theta[method]))
# s1 = np.random.randint(0, len(all_theta[method]))

print(f'\n--- Posteriors to plot: {s0} and {s1}')

all_theta_plot = [all_theta[method][s0], all_theta[method][s1]]
all_posterior_plot = [all_post_samples[method][s0], all_post_samples[method][s1]]

# Plots
# Create the figures and set their properties
fig1 = plt.figure(figsize=(7.5, 6), dpi=150)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Pairplot
lims = [[-2, 5], [-1, 3.], [-1, 10], [-5, 70]]
colors = ['blue', 'green']
for row in range(4):
    for col in np.arange(row,4):
        ax = fig1.add_axes([0.04 + col * 0.14, 0.84 - row * 0.14, 0.1, 0.1])

        try:
            # Diagonal: 1D histogram
            if row == col:
                for sample in range(2):
                    hist, bin_edges = np.histogram(all_posterior_plot[sample][:, row],
                                                   bins= np.linspace(lims[row][0], lims[row][1], 50),
                                                   density=True)
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
                                                            bins=(np.linspace(lims[col][0], lims[col][1], 25),
                                                                  np.linspace(lims[row][0], lims[row][1], 25)),
                                                            density=True)

                    # Smoothing
                    hist = gaussian_filter1d(hist, sigma=1, axis=0)
                    hist = gaussian_filter1d(hist, sigma=1, axis=1)

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
        ax = fig1.add_axes([0.66 + col * 0.18, 0.8 - row * 0.21, 0.12, 0.13])

        try:
            method = all_methods[row * 2 + col]

            x = np.linspace(0., 1.05, 8)
            y = np.linspace(0., 10.05, 8)
            hist, x_edges, y_edges = np.histogram2d(shrinkage[method].flatten(),
                                                    z_score[method].flatten(),
                                                    bins=(x, y), density=True)

            # Low-pass filtering
            hist = gaussian_filter1d(hist, sigma=1, axis=0)
            hist = gaussian_filter1d(hist, sigma=1, axis=1)

            mesh = ax.pcolormesh(x_edges, y_edges, hist.T, shading='auto', cmap='Reds')

            # Add a colorbar
            plt.colorbar(mesh, ax=ax)

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
    all_hist = []

    # Define bins to compute histograms
    if col == 0:
        bins = np.linspace(0, 2., 15)
    elif col == 1:
        bins = np.linspace(0, 1.5, 15)
    elif col == 2:
        bins = np.linspace(0, 6., 15)
    else:
        bins = np.linspace(0, 40., 15)

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
            hist, bin_edges = np.histogram(abs_error[method][:,col], bins=bins, density=True)

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
    ax.text(0.2 if col > 0 else 0.26, 0.9 if col == 0 else 0.25, r'$D_{H,1}=$'+' %.2f' % HD_[0], fontsize=6, ha='center',
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

# Plot letters
ax = fig1.add_axes([0., 0., 1., 1.])
ax.axis('off')
ax.text(0.01, 0.97, 'A', fontsize=12, fontweight='bold')
ax.text(0.59, 0.97, 'B', fontsize=12, fontweight='bold')
ax.text(0.01, 0.28, 'C', fontsize=12, fontweight='bold')

# Save the figure
plt.savefig('SBI_results.png', bbox_inches='tight')
# plt.show()