import importlib
import json
import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import ncpi

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

def zscore(data, ch, time):
    """
    Compute the z score using the maximum value of all channels instead of
    the standard deviation of the sample.

    Parameters
    ----------
    data : list
        List of data arrays.
    ch : int
        Channel to normalize.
    time : list
        Time array to normalize.

    Returns
    -------
    tuple
        Maximum value and normalized data.
    """
    tr_data = np.array(data)[:,time]
    tr_data -= np.mean(tr_data,axis = 1).reshape(-1,1)
    return (np.max(np.abs(tr_data)),tr_data[ch] /np.max(np.abs(tr_data)))


if __name__ == "__main__":
    # Read the script file path from sys.argv[1]
    script_path = sys.argv[1]

    # Add the directory containing the script to the Python path
    script_dir = os.path.dirname(script_path)
    sys.path.append(script_dir)

    # Import the script as a module
    module_name = os.path.basename(script_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Transient time
    transient = module.KernelParams.transient

    # load tstop
    with open(os.path.join(sys.argv[2],'tstop.pkl'), 'rb') as f:
        tstop = pickle.load(f)

    # load dt
    with open(os.path.join(sys.argv[2],'dt.pkl'), 'rb') as f:
        dt = pickle.load(f)

    # Load spike times
    with open(os.path.join(sys.argv[2],'times.pkl'), 'rb') as f:
        times = pickle.load(f)

    # Load gids
    with open(os.path.join(sys.argv[2],'gids.pkl'), 'rb') as f:
        gids = pickle.load(f)

    # Load X and N_X
    with open(os.path.join(sys.argv[2],'network.pkl'), 'rb') as f:
        LIF_params = pickle.load(f)
        P_X = LIF_params['X']
        N_X = LIF_params['N_X']

    # Load the path to the multicompartment neuron network folder
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    multicompartment_neuron_network_path = config['multicompartment_neuron_network_path']
    # Simulation output
    output_path = os.path.join(multicompartment_neuron_network_path, 'output','adb947bfb931a5a8d09ad078a6d256b0')

    # Compute the kernel
    print('Computing the kernel...')
    potential = ncpi.FieldPotential()
    biophys = ['set_Ih_linearized_hay2011','make_cell_uniform']
    H_YX = potential.create_kernel(multicompartment_neuron_network_path,
                                   output_path,
                                   module.KernelParams,
                                   biophys,
                                   dt,
                                   tstop,
                                   electrodeParameters=module.KernelParams.electrodeParameters,
                                   CDM=True)

    # Compute LFP signals
    probe = 'GaussCylinderPotential'
    LFP_data = dict(EE=[], EI=[], IE=[], II=[])

    for X in P_X:
        for Y in P_X:
            # Compute the firing rate
            bins, spike_rate = get_spike_rate(times[X], transient, dt, tstop)
            n_ch = H_YX[f'{X}:{Y}'][probe].shape[0]
            for ch in range(n_ch):
                # LFP kernel at electrode 'ch'
                kernel = H_YX[f'{X}:{Y}'][probe][ch, :]
                # LFP signal
                sig = np.convolve(spike_rate, kernel, 'same')
                # Decimate signal (x10)
                LFP_data[f'{X}{Y}'].append(ss.decimate(
                    sig,
                    q=10,
                    zero_phase=True))

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

    # Plot spikes and firing rates
    fig = plt.figure(figsize=[6,5], dpi=300)
    ax1 = fig.add_axes([0.15,0.45,0.75,0.5])
    ax2 = fig.add_axes([0.15,0.08,0.75,0.3])
    # Time interval
    T = [4000, 4100]

    for i, Y in enumerate(P_X):
        #  Compute the mean firing rate
        mean_spike_rate = get_mean_spike_rate(times[Y], transient, tstop)

        t = times[Y]
        gi = gids[Y]
        gi = gi[t >= transient]
        t = t[t >= transient]

        # Spikes
        ii = (t >= T[0]) & (t <= T[1])
        ax1.plot(t[ii], gi[ii], '.',
                 mfc='C{}'.format(i),
                 mec='w',
                 label=r'$\langle \nu_\mathrm{%s} \rangle =%.2f$ s$^{-1}$' % (
                    Y, mean_spike_rate / N_X[i])
                 )
    ax1.legend(loc=1)
    ax1.axis('tight')
    ax1.set_xticklabels([])
    ax1.set_ylabel('gid', labelpad=0)

    # Rates
    for i, Y in enumerate(P_X):
        # Compute the firing rate
        bins, spike_rate = get_spike_rate(times[Y], transient, dt, tstop)
        # Plot the firing rate
        bins = bins[:-1]
        ii = (bins >= T[0]) & (bins <= T[1])
        ax2.plot(bins[ii], spike_rate[ii], color='C{}'.format(i),
                 label=r'$\nu_\mathrm{%s}$' % Y)

    ax2.axis('tight')
    ax2.set_xlabel('t (ms)', labelpad=0)
    ax2.set_ylabel(r'$\nu_X$ (spikes/$\Delta t$)', labelpad=0)

    # Plot kernels and LFP/CDM data
    # Create figure and panels
    fig1 = plt.figure(figsize=[7, 6], dpi=150)
    fig2 = plt.figure(figsize=[7, 6], dpi=150)
    ax1 = []
    ax2 = []

    # First row
    for k in range(4):
        ax1.append(fig1.add_axes([0.1 + k * 0.22, 0.4, 0.18, 0.5]))
        ax2.append(fig2.add_axes([0.1 + k * 0.22, 0.4, 0.18, 0.5]))
    # Second row
    for k in range(4):
        ax1.append(fig1.add_axes([0.1 + k * 0.22, 0.1, 0.18, 0.2]))
        ax2.append(fig2.add_axes([0.1 + k * 0.22, 0.1, 0.18, 0.2]))

    # Time arrays
    dt *= 10  # take into account the decimate ratio
    bins = bins[::10]  # take into account the decimate ratio
    time = np.arange(-module.KernelParams.tau, module.KernelParams.tau + dt, dt)
    T = [4000, 4100]
    ii = (bins >= T[0]) & (bins <= T[1])
    iii = np.where(bins >= T[0] + np.diff(T)[0] / 2)[0][0]

    # LFP probe
    probe = 'GaussCylinderPotential'
    k = 0
    for X in P_X:
        for Y in P_X:
            n_ch = H_YX[f'{X}:{Y}'][probe].shape[0]
            for ch in range(n_ch):
                # Decimate first
                dec_kernel = ss.decimate(H_YX[f'{X}:{Y}'][probe],
                                         q=10, zero_phase=True)
                # Z-scored kernel from 0 to 1/2 of tau
                maxk, norm_ker = zscore(dec_kernel, ch,
                                        np.arange(int(time.shape[0] / 2),
                                                  int(3 * time.shape[0] / 4)))
                # Z-scored LFP signal from T[0] to T[1]
                maxs, norm_sig = zscore(LFP_data[f'{X}{Y}'], ch, ii)
                # Plot data stacked in the Z-axis
                ax1[k].plot(time[int(time.shape[0] / 2):int(3 * time.shape[0] / 4)],
                            norm_ker - ch)
                ax2[k].plot(bins[ii], norm_sig - ch)
            ax1[k].set_title(f'H_{X}:{Y}')
            ax2[k].set_title(f'H_{X}:{Y}')
            if k == 0:
                ax1[k].set_yticks(np.arange(0, -n_ch, -1))
                ax1[k].set_yticklabels(['ch. ' + str(ch) for ch in np.arange(1, n_ch + 1)])
                ax2[k].set_yticks(np.arange(0, -n_ch, -1))
                ax2[k].set_yticklabels(['ch. ' + str(ch) for ch in np.arange(1, n_ch + 1)])
            else:
                ax1[k].set_yticks([])
                ax1[k].set_yticklabels([])
                ax2[k].set_yticks([])
                ax2[k].set_yticklabels([])
            ax1[k].set_xlabel(r'$tau_{ms}$')
            ax2[k].set_xlabel('t (ms)')

            # Add scales
            ax1[k].plot([time[int(0.59 * time.shape[0])], time[int(0.59 * time.shape[0])]],
                        [0, -1], linewidth=2., color='k')
            sexp = np.round(np.log2(maxk))
            ax1[k].text(time[int(0.6 * time.shape[0])], -0.5, r'$2^{%s}mV$' % sexp)
            # ax2[k].plot([bins[iii],bins[iii]],
            #              [0,-1],linewidth = 2., color = 'k')
            # sexp = np.round(np.log2(maxs))
            # ax2[k].text(bins[iii],-0.5,r'$2^{%s}mV$' % sexp)
            k += 1

    # Current dipole moment
    probe = 'KernelApproxCurrentDipoleMoment'
    k = 0
    for X in P_X:
        for Y in P_X:
            # Pick only the z-component of the CDM kernel.
            # (* 1E-4 : nAum --> nAcm unit conversion)
            dec_kernel = ss.decimate([1E-4 * H_YX[f'{X}:{Y}'][probe][2]],
                                     q=10, zero_phase=True)
            maxk, norm_ker = zscore(dec_kernel, 0,
                                    np.arange(int(time.shape[0] / 2),
                                              int(3 * time.shape[0] / 4)))
            # Z-scored CDM signal
            maxs, norm_sig = zscore([1E-4 * CDM_data[f'{X}{Y}']], 0, ii)
            # Plot data
            ax1[k + 4].plot(time[int(time.shape[0] / 2):int(3 * time.shape[0] / 4)], norm_ker)
            ax2[k + 4].plot(bins[ii], norm_sig)

            if k == 0:
                ax1[k + 4].set_yticks([0])
                ax1[k + 4].set_yticklabels([r'$P_z$'])
                ax2[k + 4].set_yticks([0])
                ax2[k + 4].set_yticklabels([r'$P_z$'])
            else:
                ax1[k + 4].set_yticks([])
                ax1[k + 4].set_yticklabels([])
                ax2[k + 4].set_yticks([])
                ax2[k + 4].set_yticklabels([])
            ax1[k + 4].set_xlabel(r'$tau_{ms}$')
            ax2[k + 4].set_xlabel('t (ms)')

            # Add scales
            ax1[k + 4].plot([time[int(0.59 * time.shape[0])], time[int(0.59 * time.shape[0])]],
                            [0, -1], linewidth=2., color='k')
            sexp = np.round(np.log2(maxk))
            ax1[k + 4].text(time[int(0.6 * time.shape[0])], -0.5, r'$2^{%s}nAcm$' % sexp)
            ax2[k + 4].plot([bins[iii + 5], bins[iii + 5]],
                            [0, -1], linewidth=2., color='k')
            sexp = np.round(np.log2(maxs))
            ax2[k + 4].text(bins[iii], -0.5, r'$2^{%s}nAcm$' % sexp)
            k += 1

    plt.show()
