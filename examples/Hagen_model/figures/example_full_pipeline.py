import json
import sys
import os
import pickle

import pandas as pd
import scipy.signal as ss
import numpy as np
from matplotlib import pyplot as plt

# ccpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ccpi

# Parameters of LIF model simulations
sys.path.append(os.path.join(os.path.dirname(__file__), '../simulation/params'))

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
compute_new_sim = True

# Number of repetitions of the each simulation
trials = 5

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
            sim = ccpi.Simulation(param_folder='../simulation/params',
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

            # Compute the kernel
            print('Computing the kernel...')
            potential = ccpi.FieldPotential()
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
fig = plt.figure(figsize=(7., 6.), dpi=300)
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
        gi = gi[t >= transient]
        t = t[t >= transient]

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
        ax.set_ylabel(r'$\nu_X$ (spikes/$\Delta t$)')
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
        ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('t (ms)')
    ax.axis('tight')

    # Add scale
    y_max = np.max(CDM[ii])
    y_min = np.min(CDM[ii])
    scale = (y_max - y_min) / 5
    ax.plot([T[0], T[0]], [y_min + scale, y_min], 'k')
    ax.text(T[0] + 1, y_min + scale/2., r'$2^{%s}nAcm$' % np.round(np.log2(scale*10**(-4))), fontsize=8)

# Power spectra
ax = fig.add_axes([0.1, 0.07, 0.27, 0.3])
colors = ['C0', 'C1', 'C2']
for col in range(3):
    CDM = [CDMs[trial][col]['EE'] + CDMs[trial][col]['EI'] +
           CDMs[trial][col]['IE'] + CDMs[trial][col]['II'] for trial in range(trials)]
    f, Pxx = ss.welch(CDM, fs=1000./(10.*dt), nperseg=250)
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

# Compute features and predictions
pred_data = [{} for trial in range(trials)]
for trial in range(trials):
    for method in ['catch22', 'power_spectrum_parameterization', 'fEI']:
        print(f'Computing features and predictions for {method} and trial {trial+1}/{trials}')
        # Create a Pandas DataFrame for computing features
        df = pd.DataFrame({'ID': np.arange(3),
                           'Group': np.arange(3),
                           'Epoch': np.zeros(3),
                           'Sensor': np.zeros(3),  # dummy sensor
                           'Data': [CDMs[trial][col]['EE'] + CDMs[trial][col]['EI'] +
                                    CDMs[trial][col]['IE'] + CDMs[trial][col]['II'] for col in range(3)]})
        df.Recording = 'LFP'
        df.fs = 1000. / (10. * dt)

        # Create a Features object
        if method == 'catch22':
            features = ccpi.Features(method=method)
        elif method == 'power_spectrum_parameterization':
            # Parameters of the fooof algorithm
            fooof_setup_sim = {'peak_threshold': 1.,
                               'min_peak_height': 0.,
                               'max_n_peaks': 2,
                               'peak_width_limits': (10., 50.)}
            features = ccpi.Features(method='power_spectrum_parameterization',
                                     params={'fs': df.fs,
                                             'fmin': 5.,
                                             'fmax': 200.,
                                             'fooof_setup': fooof_setup_sim,
                                             'r_squared_th': 0.9})
        elif method == 'fEI':
            features = ccpi.Features(method='fEI',
                                     params={'fs': df.fs,
                                             'fmin': 8.,
                                             'fmax': 30.,
                                             'fEI_folder': '../../../ccpi/Matlab'})
        # Compute features
        df = features.compute_features(df)

        # Keep only the aperiodic exponent
        if method == 'power_spectrum_parameterization':
            df['Features'] = df['Features'].apply(lambda x: x[1])

        # Transfer prediction model and scaler to the data folder
        pickle.dump(pickle.load(open(os.path.join('../data', method, 'model'), 'rb')),
                    open('data/model.pkl', 'wb'))
        pickle.dump(pickle.load(open(os.path.join('../data', method, 'scaler'), 'rb')),
                    open('data/scaler.pkl', 'wb'))

        # Create the Inference object
        model = 'MLPRegressor'
        inference = ccpi.Inference(model=model)

        # Predict the parameters from the features of the empirical data
        predictions = inference.predict(np.array(df['Features'].tolist()))

        # Append the predictions to the DataFrame
        pd_preds = pd.DataFrame({'Predictions': predictions})
        df = pd.concat([df, pd_preds], axis=1)

        # Append the predictions to the list
        pred_data[trial][method] = df

# Plot predictions
confs = np.array(confs)
colors = ['powderblue', 'salmon', 'blueviolet']
for row in range(2):
    for col in range(2):
        ax = fig.add_axes([0.45 + col * 0.2, 0.1 + row * 0.15, 0.15, 0.1])

        for i,method in enumerate(['catch22', 'power_spectrum_parameterization','fEI']):
            predictions = np.array([pred_data[trial][method]['Predictions'].tolist() for trial in range(trials)])
            if row == 1 and col == 0:
                preds = (predictions[:,:,0] / predictions[:,:,2]) / (predictions[:,:,1] / predictions[:,:,3])
                ax.set_title(r'$E/I$')
            elif row == 1 and col == 1:
                preds = predictions[:,:,4]
                ax.set_title(r'$\tau_{syn}^{exc}$ (ms)')
            elif row == 0 and col == 0:
                preds = predictions[:,:,5]
                ax.set_title(r'$\tau_{syn}^{inh}$ (ms)')
            elif row == 0 and col == 1:
                preds = predictions[:,:,6]
                ax.set_title(r'$J_{syn}^{ext}$ (nA)')

            # Plot predicted values
            ax.plot(np.mean(preds, axis = 0), color=colors[i])
            # Plot standard deviation
            ax.fill_between(np.arange(3), np.mean(preds, axis = 0) - np.std(preds, axis = 0),
                            np.mean(preds, axis = 0) + np.std(preds, axis = 0), color=colors[i], alpha=0.3)

            if row == 0:
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels([r'$J_{syn}^{ext}$ = %s nA' % confs[ii][6] for ii in range(3)], rotation=25,
                                   fontsize=8)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

# Create a fake legend
ax = fig.add_axes([0.82, 0.2, 0.15, 0.15])
ax.axis('off')
labels = [r'$catch22$',r'$1/f$'+' '+r'$slope$',r'$fE/I$']
for i,method in enumerate(['catch22', 'power_spectrum_parameterization','fEI']):
    ax.plot([0], [0], color=colors[i], label=labels[i])
ax.legend(loc='upper left', fontsize=8, labelspacing=0.2)

# Plot letters
ax = fig.add_axes([0., 0., 1., 1.])
ax.axis('off')
ax.text(0.01, 0.97, 'A', fontsize=12, fontweight='bold')
ax.text(0.01, 0.37, 'B', fontsize=12, fontweight='bold')
ax.text(0.4, 0.37, 'C', fontsize=12, fontweight='bold')

plt.show()
