import json
import sys
import os
import pickle
import pandas as pd
import scipy.signal as ss
import numpy as np
from matplotlib import pyplot as plt
import ncpi

# Parameters of LIF model simulations
sys.path.append(os.path.join(os.path.dirname(__file__), '../simulation/params'))

# Names of catch22 features
try:
    import pycatch22
    catch22_names = pycatch22.catch22_all([0])['names']
except:
    catch22_names = ['DN_HistogramMode_5',
                     'DN_HistogramMode_10',
                     'CO_f1ecac',
                     'CO_FirstMin_ac',
                     'CO_HistogramAMI_even_2_5',
                     'CO_trev_1_num',
                     'MD_hrv_classic_pnn40',
                     'SB_BinaryStats_mean_longstretch1',
                     'SB_TransitionMatrix_3ac_sumdiagcov',
                     'PD_PeriodicityWang_th0_01',
                     'CO_Embed2_Dist_tau_d_expfit_meandiff',
                     'IN_AutoMutualInfoStats_40_gaussian_fmmi',
                     'FC_LocalSimple_mean1_tauresrat',
                     'DN_OutlierInclude_p_001_mdrmd',
                     'DN_OutlierInclude_n_001_mdrmd',
                     'SP_Summaries_welch_rect_area_5_1',
                     'SB_BinaryStats_diff_longstretch0',
                     'SB_MotifThree_quantile_hh',
                     'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
                     'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
                     'SP_Summaries_welch_rect_centroid',
                     'FC_LocalSimple_mean3_stderr']

def get_spike_rate(times, transient, dt, tstop):
    """
    Compute the spike rate from spike times.

    Parameters
    ----------
    times : array
        Spike times.
    transient : float
        Transient time at the start of the simulation.
    dt : float
        Simulation time step or bin size.
    tstop : float
        Simulation stop time.

    Returns
    -------
    bins : array
        Time bins.
    hist : array
        Spike rate.
    """
    bins = np.arange(transient, tstop + dt, dt)
    hist, _ = np.histogram(times, bins=bins)
    return bins, hist.astype(float)

# Debug
compute_new_sim = False

# Random seed for numpy
np.random.seed(0)

# Number of repetitions of each simulation
trials = 6

# Configurations of parameters to simulate
# best_fit = [1.589, 2.020, -23.84, -8.441, 0.5, 0.5, 29.89]

# Changing J_ext
confs = [[1.589, 2.020, -23.84, -8.441, 0.5, 0.5, 28.],
          [1.589, 2.020, -23.84, -8.441, 0.5, 0.5, 30.],
          [1.589, 2.020, -23.84, -8.441, 0.5, 0.5, 32.]]

# Simulation outputs
spikes = [[] for _ in range(trials)]
CDMs = [[] for _ in range(trials)]

for trial in range(trials):
    for k,params in enumerate(confs):
        if compute_new_sim:
            print(f'\nTrial {trial+1}/{trials}, Configuration {k+1}/{len(confs)}')
            # Parameters of the model
            J_EE = params[0]
            J_IE = params[1]
            J_EI = params[2]
            J_II = params[3]
            tau_syn_E = params[4]
            tau_syn_I = params[5]
            J_ext = params[6]

            # Load LIF_params
            from network_params import LIF_params

            # Modify parameters
            LIF_params['J_YX'] = [[J_EE, J_IE], [J_EI, J_II]]
            LIF_params['tau_syn_YX'] = [[tau_syn_E, tau_syn_I],
                                        [tau_syn_E, tau_syn_I]]
            LIF_params['J_ext'] = J_ext

            # Create a Simulation object
            sim = ncpi.Simulation(param_folder='../simulation/params',
                                  python_folder='../simulation/python',
                                  output_folder='../simulation/output')

            # Save parameters to a pickle file
            with open(os.path.join('../simulation/output', 'network.pkl'), 'wb') as f:
                pickle.dump(LIF_params, f)

            # Run the simulation
            sim.simulate('simulation.py', 'simulation_params.py')

            # Load spike times
            with open(os.path.join('../simulation/output', 'times.pkl'), 'rb') as f:
                times = pickle.load(f)

            # Load gids
            with open(os.path.join('../simulation/output', 'gids.pkl'), 'rb') as f:
                gids = pickle.load(f)

            # Load tstop
            with open(os.path.join('../simulation/output', 'tstop.pkl'), 'rb') as f:
                tstop = pickle.load(f)

            # Load dt
            with open(os.path.join('../simulation/output', 'dt.pkl'), 'rb') as f:
                dt = pickle.load(f)

            # Load X and N_X
            with open(os.path.join('../simulation/output', 'network.pkl'), 'rb') as f:
                LIF_params = pickle.load(f)
                P_X = LIF_params['X']
                N_X = LIF_params['N_X']

            # Load the path to the multicompartment neuron network folder
            with open('../config.json', 'r') as config_file:
                config = json.load(config_file)
            multicompartment_neuron_network_path = config['multicompartment_neuron_network_path']
            # Simulation output
            output_path = os.path.join(multicompartment_neuron_network_path, 'output', 'adb947bfb931a5a8d09ad078a6d256b0')

            # Transient period
            from analysis_params import KernelParams
            transient = KernelParams.transient
            for X in P_X:
                gids[X] = gids[X][times[X] >= transient]
                times[X] = times[X][times[X] >= transient]

            # Compute the kernel
            print('Computing the kernel...')
            potential = ncpi.FieldPotential()
            biophys = ['set_Ih_linearized_hay2011', 'make_cell_uniform']
            H_YX = potential.create_kernel(multicompartment_neuron_network_path,
                                           output_path,
                                           KernelParams,
                                           biophys,
                                           dt,
                                           tstop,
                                           electrodeParameters=None,
                                           CDM=True)

            # Compute CDM
            probe = 'KernelApproxCurrentDipoleMoment'
            CDM_data = dict(EE=[], EI=[], IE=[], II=[])

            for X in P_X:
                for Y in P_X:
                    # Compute the firing rate
                    bins, spike_rate = get_spike_rate(times[X], transient, dt, tstop)
                    # Pick only the z-component of the CDM kernel
                    kernel = H_YX[f'{X}:{Y}'][probe][2, :]
                    # CDM
                    sig = np.convolve(spike_rate, kernel, 'same')
                    CDM_data[f'{X}{Y}'] = ss.decimate(sig,
                                                      q=10,
                                                      zero_phase=True)

            # Collect the simulation outputs
            spikes[trial].append([times, gids])
            CDMs[trial].append(CDM_data)

            # Save the simulation outputs to a pickle file
            if not os.path.exists('data'):
                os.makedirs('data')
            with open(f'data/output_{k}_{trial}.pkl', 'wb') as f:
                pickle.dump([times, gids, CDM_data, dt, tstop, transient, P_X, N_X], f)

        else:
            with open(f'data/output_{k}_{trial}.pkl', 'rb') as f:
                times, gids, CDM_data, dt, tstop, transient, P_X, N_X = pickle.load(f)
            spikes[trial].append([times, gids])
            CDMs[trial].append(CDM_data)


# Create a figure and set its properties
fig = plt.figure(figsize=(7.5, 6.), dpi=300)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# Time interval
T = [4000, 4100]

# Raster plot of the spike trains
colors = ['#1f77b4', '#ff7f0e']
for col in range(3):
    ax = fig.add_axes([0.1 + col * 0.3, 0.73, 0.25, 0.22])
    for i,X in enumerate(P_X):
        t = spikes[0][col][0][X] # pick the first trial
        gi = spikes[0][col][1][X]
        # gi = gi[t >= transient]
        # t = t[t >= transient]

        # Spikes
        ii = (t >= T[0]) & (t <= T[1])
        ax.plot(t[ii], gi[ii], '.', color = colors[i], markersize=0.5)

    ax.set_title(r'$J_{syn}^{ext}$ = %s nA' % confs[col][6])
    if col == 0:
        ax.set_ylabel('Neuron ID')
        ax.yaxis.set_label_coords(-0.22, 0.5)

        # Fake legend
        for j, Y in enumerate(P_X):
            ax.plot([], [], '.', color = colors[j], label=f'{Y}', markersize=4)
        ax.legend(loc=1, fontsize=8, labelspacing=0.2)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('')
    ax.axis('tight')
    ax.set_xticklabels([])
    ax.set_xticks([])


# Firing rates
for col in range(3):
    ax = fig.add_axes([0.1 + col * 0.3, 0.6, 0.25, 0.12])
    for i,X in enumerate(P_X):
        # Compute the firing rate
        bins, spike_rate = get_spike_rate(spikes[0][col][0][X], transient, dt, tstop)
        # Plot the firing rate
        bins = bins[:-1]
        ii = (bins >= T[0]) & (bins <= T[1])
        ax.plot(bins[ii], spike_rate[ii], color='C{}'.format(i),label=r'$\nu_\mathrm{%s}$' % X)

    if col == 0:
        ax.legend(loc=1)
        ax.set_ylabel(r'$\nu_X$ (spik./$\Delta t$)')
        ax.yaxis.set_label_coords(-0.22, 0.5)

    ax.axis('tight')
    ax.set_xticklabels([])
    ax.set_xticks([])

# CDMs
for col in range(3):
    ax = fig.add_axes([0.1 + col * 0.3, 0.47, 0.25, 0.12])
    CDM = CDMs[0][col]['EE'] + CDMs[0][col]['EI'] + CDMs[0][col]['IE'] + CDMs[0][col]['II']
    bins = np.arange(transient, tstop, dt)
    bins = bins[::10]  # to take into account the decimate ratio
    ii = (bins >= T[0]) & (bins <= T[1])
    ax.plot(bins[ii], CDM[ii], color='k')

    if col == 0:
        ax.set_ylabel(r'CDM ($P_z$)')
        ax.yaxis.set_label_coords(-0.22, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('t (ms)')
    ax.axis('tight')

    # Add scale
    y_max = np.max(CDM[ii])
    y_min = np.min(CDM[ii])
    scale = (y_max - y_min) / 5
    ax.plot([T[0] if col < 2 else T[0] + 50,T[0] if col < 2 else T[0] + 50],
             [y_min + scale, y_min], 'k')
    ax.text(T[0] + 1 if col < 2 else T[0] + 51,
            y_min + scale/4., r'$2^{%s}nAcm$' % np.round(np.log2(scale*10**(-4))), fontsize=8)

# Power spectra
ax = fig.add_axes([0.1, 0.07, 0.27, 0.3])
colors = ['C0', 'C1', 'C2']
for col in range(3):
    CDM = [CDMs[trial][col]['EE'] + CDMs[trial][col]['EI'] +
           CDMs[trial][col]['IE'] + CDMs[trial][col]['II'] for trial in range(trials)]
    f, Pxx = ss.welch(CDM, fs=1000./(10.*dt))
    # Trial-averaged power spectrum
    Pxx = np.mean(Pxx, axis=0)
    # Normalize the power spectrum
    Pxx = Pxx / np.sum(Pxx)
    f1 = f[f >= 10]
    f2 = f1[f1 <= 200]
    ax.semilogy(f2, Pxx[(f >= 10) & (f <= 200)], label=r'$J_{syn}^{ext}$ = %s nA' % confs[col][6],
                color=colors[col])
ax.legend(loc='lower left', fontsize=8, labelspacing=0.2)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Normalized power')

# Collect the CDMs
all_CDMs = []
IDs = []
epochs = []
for trial in range(trials):
    for k,params in enumerate(confs):
        all_CDMs.append(CDMs[trial][k]['EE'] + CDMs[trial][k]['EI'] + CDMs[trial][k]['IE'] + CDMs[trial][k]['II'])
        IDs.append(k)
        epochs.append(trial)

# Compute features
all_features = {}
all_methods = ['catch22', 'power_spectrum_parameterization']
for method in all_methods:
    print(f'\n\n--- Method: {method}')

    # Create a fake Pandas DataFrame (only Data and fs are relevant)
    df = pd.DataFrame({'ID': IDs,
                       'Group': IDs,
                       'Epoch': epochs,
                       'Sensor': np.zeros(len(IDs)),  # dummy sensor
                       'Data': all_CDMs})
    df.Recording = 'LFP'
    df.fs = 1000. / (10. * dt)

    # Compute features
    if method == 'catch22':
        features = ncpi.Features(method='catch22')
    elif method == 'power_spectrum_parameterization':
        # Parameters of the fooof algorithm
        fooof_setup_sim = {'peak_threshold': 1.,
                           'min_peak_height': 0.,
                           'max_n_peaks': 5,
                           'peak_width_limits': (10., 50.)}
        features = ncpi.Features(method='power_spectrum_parameterization',
                                 params={'fs': df.fs,
                                         'fmin': 5.,
                                         'fmax': 200.,
                                         'fooof_setup': fooof_setup_sim,
                                         'r_squared_th': 0.9})

    df = features.compute_features(df)

    # Keep only the aperiodic exponent
    if method == 'power_spectrum_parameterization':
        df['Features'] = df['Features'].apply(lambda x: x[1])

    # Append the feature dataframes to a list
    all_features[method] = df

# Features
colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightgrey']
for row in range(2):
    for col in range(2):
        ax = fig.add_axes([0.5 + col * 0.27, 0.24 - row * 0.16, 0.18, 0.13])

        if row == 0 and col == 0:
            feats = np.array(all_features['power_spectrum_parameterization']['Features'].tolist())
            ax.set_ylabel(r'$1/f$' + ' ' + r'$slope$')
        if row == 0 and col == 1:
            feats = np.array(all_features['catch22']['Features'].tolist())
            idx = catch22_names.index('SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1')
            feats = feats[:, idx]
            ax.set_ylabel(r'$dfa$')
        if row == 1 and col == 0:
            feats = np.array(all_features['catch22']['Features'].tolist())
            idx = catch22_names.index('SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1')
            feats = feats[:, idx]
            ax.set_ylabel(r'$rs\ range$')
        if row == 1 and col == 1:
            feats = np.array(all_features['catch22']['Features'].tolist())
            idx = catch22_names.index('MD_hrv_classic_pnn40')
            feats = feats[:, idx]
            ax.set_ylabel(r'$high\ fluct.$')

        # Rearrange the features
        feats_plot = np.zeros((trials,len(confs)))
        for conf in range(len(confs)):
            feats_plot[:,conf] = feats[np.array(IDs) == conf]

        # Plot the average values
        ax.plot(np.arange(len(confs)), np.mean(feats_plot, axis=0), color=colors[row*2 + col])

        # Plot the variance
        ax.fill_between(np.arange(len(confs)), np.mean(feats_plot, axis=0) - np.std(feats_plot, axis=0),
                        np.mean(feats_plot, axis=0) + np.std(feats_plot, axis=0), color=colors[row*2 + col],
                        alpha=0.3)

        # Labels
        if row == 1:
            ax.set_xlabel(r'$J_{syn}^{ext}$ (nA)')
            ax.set_xticks(np.arange(3))
            ax.set_xticklabels([f'{confs[i][6]}' for i in range(3)])
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])



# Plot letters
ax = fig.add_axes([0., 0., 1., 1.])
ax.axis('off')
ax.text(0.01, 0.97, 'A', fontsize=12, fontweight='bold')
ax.text(0.01, 0.37, 'B', fontsize=12, fontweight='bold')
ax.text(0.4, 0.37, 'C', fontsize=12, fontweight='bold')

# Save the figure
plt.savefig('example_full_pipeline.png', bbox_inches='tight')
# plt.show()
