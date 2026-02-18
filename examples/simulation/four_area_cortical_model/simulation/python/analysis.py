import os
import pickle
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
from importlib import util
import ncpi
from ncpi import tools

# Choose to either download files and precomputed outputs used in simulations of the reference multicompartment neuron
# network model (True) or load them from a local path (False)
zenodo_dw_mult = True

# Zenodo URL that contains the data (used if zenodo_dw_mult is True)
zenodo_URL_mult = "https://zenodo.org/api/records/15429373"

# Zenodo directory where the data is stored (must be an absolute path to correctly load morphologies in neuron)
zenodo_dir = os.path.expandvars(os.path.expanduser(
    os.path.join("$HOME", "multicompartment_neuron_network")
))

# Download data
if zenodo_dw_mult:
    print('\n--- Downloading data.')
    start_time = time.time()
    tools.download_zenodo_record(zenodo_URL_mult, download_dir=zenodo_dir)
    end_time = time.time()
    print(f"All files downloaded in {(end_time - start_time) / 60:.2f} minutes.")


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
        Simulation time step.
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


def get_mean_spike_rate(times, transient, tstop):
    """
    Compute the mean firing rate.

    Parameters
    ----------
    times : array
        Spike times.
    transient : float
        Transient time at the start of the simulation.
    tstop : float
        Simulation stop time.

    Returns
    -------
    float
        Mean firing rate.
    """
    return times.size / (tstop - transient) * 1000


if __name__ == "__main__":
    # Read the script file path from sys.argv[1]
    script_path = sys.argv[1]

    # Add the directory containing the script to the Python path
    script_dir = os.path.dirname(script_path)
    sys.path.append(script_dir)

    # Import the script as a module
    module_name = os.path.basename(script_path).replace('.py', '')
    spec = util.spec_from_file_location(module_name, script_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Transient time
    transient = module.KernelParams.transient

    # load tstop
    with open(os.path.join(sys.argv[2], 'tstop.pkl'), 'rb') as f:
        tstop = pickle.load(f)

    # load dt
    with open(os.path.join(sys.argv[2], 'dt.pkl'), 'rb') as f:
        dt = pickle.load(f)

    # Load spike times
    with open(os.path.join(sys.argv[2], 'times.pkl'), 'rb') as f:
        times = pickle.load(f)

    # Load gids
    with open(os.path.join(sys.argv[2], 'gids.pkl'), 'rb') as f:
        gids = pickle.load(f)

    # Load X and N_X
    with open(os.path.join(sys.argv[2], 'network.pkl'), 'rb') as f:
        LIF_params = pickle.load(f)
        P_X = LIF_params['X']
        N_X = LIF_params['N_X']
        areas = LIF_params['areas']

    # Simulation output from the multicompartment neuron network model
    output_path = os.path.join(zenodo_dir, 'output',
                               'adb947bfb931a5a8d09ad078a6d256b0')

    # Path to the data files of the multicompartment neuron models
    multicompartment_neuron_network_path = zenodo_dir

    # Compute the kernel (shared across areas)
    print('Computing the kernel...')
    potential = ncpi.FieldPotential()
    biophys = ['set_Ih_linearized_hay2011', 'make_cell_uniform']
    H_YX = potential.create_kernel(multicompartment_neuron_network_path,
                                   module.KernelParams,
                                   biophys,
                                   dt,
                                   tstop,
                                   output_sim_path=output_path,
                                   electrodeParameters=module.KernelParams.electrodeParameters,
                                   CDM=True)

    # Analysis containers
    area_cdm = {}
    area_spectra = {}

    # Time interval for plots
    T = [4000, 4100]

    # Prepare single figure: rows for spikes, rates, CDM, spectra (each row split by area)
    n_areas = len(areas)
    fig = plt.figure(figsize=(4 * n_areas, 12), dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(4, n_areas, height_ratios=[1.3, 1.0, 1.0, 1.0])

    ax_raster = [fig.add_subplot(gs[0, i]) for i in range(n_areas)]
    ax_rate = [fig.add_subplot(gs[1, i]) for i in range(n_areas)]
    ax_cdm = [fig.add_subplot(gs[2, i]) for i in range(n_areas)]
    ax_psd = [fig.add_subplot(gs[3, i]) for i in range(n_areas)]

    for area_idx, area in enumerate(areas):
        # Filter transient
        area_times = {}
        area_gids = {}
        for i, X in enumerate(P_X):
            t = times[area][X]
            g = gids[area][X]
            mask = t >= transient
            area_times[X] = t[mask]
            area_gids[X] = g[mask]

        # Spikes and firing rates
        for i, X in enumerate(P_X):
            mean_spike_rate = get_mean_spike_rate(area_times[X], transient, tstop)
            t = area_times[X]
            gi = area_gids[X]
            ii = (t >= T[0]) & (t <= T[1])
            ax_raster[area_idx].plot(t[ii], gi[ii], '.',
                                     mfc='C{}'.format(i),
                                     mec='w',
                                     label=r'%s: %.2f s$^{-1}$' % (
                                         X, mean_spike_rate / N_X[i])
                                     )
            bins, spike_rate = get_spike_rate(area_times[X], transient, dt, tstop)
            bins = bins[:-1]
            ii = (bins >= T[0]) & (bins <= T[1])
            ax_rate[area_idx].plot(bins[ii], spike_rate[ii], color='C{}'.format(i),
                                   label=r'%s' % X)

        ax_raster[area_idx].set_title(f'{area} (spikes)', fontsize=9)
        ax_raster[area_idx].set_ylabel('gid')
        ax_raster[area_idx].set_xticklabels([])
        ax_raster[area_idx].legend(loc='upper right', fontsize=7)
        ax_raster[area_idx].axis('tight')

        ax_rate[area_idx].set_title(f'{area} (rate)', fontsize=9)
        ax_rate[area_idx].set_xlabel('t (ms)')
        ax_rate[area_idx].set_ylabel(r'$\nu_X$ (spikes/$\Delta t$)')
        if area_idx == 0:
            ax_rate[area_idx].legend(loc='upper right', fontsize=7)
        ax_rate[area_idx].axis('tight')

        # Compute CDM (z-component) using FieldPotential helper
        probe = 'KernelApproxCurrentDipoleMoment'
        cdm_signals = potential.compute_cdm_from_kernels(
            H_YX,
            spike_times=area_times,
            dt=dt,
            tstop=tstop,
            transient=transient,
            probe=probe,
            component=2,
            mode='same',
            scale=dt / 1000.0,
        )

        cdm_data = dict(EE=[], EI=[], IE=[], II=[])
        for X in P_X:
            for Y in P_X:
                key = f'{Y}:{X}'
                cdm_data[f'{X}{Y}'] = ss.decimate(
                    cdm_signals[key],
                    q=10,
                    zero_phase=True
                )

        area_cdm[area] = cdm_data

        # Total CDM for spectra
        cdm_total = (
            np.asarray(cdm_data['EE']) +
            np.asarray(cdm_data['EI']) +
            np.asarray(cdm_data['IE']) +
            np.asarray(cdm_data['II'])
        )

        # Match reference snippet: treat each trial as an entry; here we have one "trial"
        cdm_trials = [cdm_total]
        fs = 1000.0 / (10.0 * dt)
        if cdm_total.size < 2:
            freqs = np.array([])
            psd = np.array([])
            psd_norm = np.array([])
        else:
            freqs, psd = ss.welch(cdm_trials, fs=fs)
            # Trial-averaged power spectrum
            psd = np.mean(psd, axis=0)
            # Normalize the power spectrum
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd

        area_spectra[area] = dict(freqs=freqs, psd=psd, psd_norm=psd_norm)

        # Plot CDM time series (same interval as spikes/rates)
        dt_dec = dt * 10
        time = np.arange(0, cdm_total.size) * dt_dec
        mask_t = (time >= T[0]) & (time <= T[1])
        time_win = time[mask_t]
        cdm_win = cdm_total[mask_t]
        ax_cdm[area_idx].plot(time_win, cdm_win, color='C0')
        ax_cdm[area_idx].set_title(f'{area} CDM (z)', fontsize=9)
        ax_cdm[area_idx].set_xlabel('t (ms)')
        ax_cdm[area_idx].set_ylabel('CDM (a.u.)')
        ax_cdm[area_idx].set_xlim(T[0], T[1])

        # Add scale bar for CDM
        if cdm_win.size > 0:
            # Add scale
            y_max = np.max(cdm_win)
            y_min = np.min(cdm_win)
            scale = (y_max - y_min) / 5
            ax_cdm[area_idx].plot([T[0], T[0]],
                    [y_min + scale, y_min], 'k')
            ax_cdm[area_idx].text(T[0] + 1,
                    y_min + scale / 4., r'$2^{%s}nAcm$' % np.round(np.log2(scale * 10 ** (-4))),
                                  fontsize=8)

        ax_cdm[area_idx].axis('tight')
        ax_cdm[area_idx].set_yticks([])
        ax_cdm[area_idx].set_ylabel('')

        # Plot spectra (one panel per area)
        if freqs.size > 0:
            mask = (freqs >= 20) & (freqs <= 200)
            ax_psd[area_idx].semilogy(freqs[mask], psd_norm[mask], color='C1')
        ax_psd[area_idx].set_title(f'{area} spectrum', fontsize=9)
        ax_psd[area_idx].set_xlabel('f (Hz)')
        ax_psd[area_idx].set_ylabel('PSD (a.u./Hz)')
        ax_psd[area_idx].set_xlim(20, 200)
        ax_psd[area_idx].axis('tight')

    plt.show()
