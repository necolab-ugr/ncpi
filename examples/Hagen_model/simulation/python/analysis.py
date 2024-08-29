import importlib
import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

def get_spike_rate(times, transient, dt, tstop):
    bins = np.arange(transient, tstop + dt, dt)
    hist, _ = np.histogram(times, bins=bins)
    return bins, hist.astype(float)

def get_mean_spike_rate(times, transient, tstop):
    return times.size / (tstop - transient) * 1000


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
    transient = module.transient

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
        X = LIF_params['X']
        N_X = LIF_params['N_X']

    # Plot spikes and firing rates
    fig = plt.figure(figsize=[6,5], dpi=300)
    ax1 = fig.add_axes([0.15,0.45,0.75,0.5])
    ax2 = fig.add_axes([0.15,0.08,0.75,0.3])
    # Time interval
    T = [4000, 4100]

    for i, Y in enumerate(X):
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
    for i, Y in enumerate(X):
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

    plt.show()