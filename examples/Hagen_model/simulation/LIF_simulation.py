import hashlib
import json
import os
import pickle
from time import time

import numpy as np
import scipy.signal as ss
import nest

# LIF network class
import LIF_network

def get_size(start_path = '.'):
    """
    Walk all subdirectories, summing file sizes.

    Parameters
    ----------
    start_path : str
        Path to the directory to start the search.

    Returns
    -------
    total_size : int
        Total size of all files in the directory.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

class LIF_simulation:
    """
    Class for running LIF simulations.
    """

    def __init__(self, params=None):
        """
        Constructor.

        Parameters
        ----------
        params : list
            List of 7 elements with the parameters of the LIF model.
        """
        # Best fit parameters
        if params is None:
            self.params = [1.589, 2.020, -23.84, -8.441, 0.5, 0.5, 29.89]
        else:
            # Check that params is a list of 7 elements
            if len(params) != 7:
                raise ValueError('params must have 7 elements')
            self.params = params

    def simulate(self, n_threads=8):
        """
        Run a LIF simulation with the parameters given in the constructor.

        Parameters
        ----------
        n_threads : int
            Number of threads to use in the simulation.
        """

        # Parameters of the model
        J_EE = self.params[0]
        J_IE = self.params[1]
        J_EI = self.params[2]
        J_II = self.params[3]
        tau_syn_E = self.params[4]
        tau_syn_I = self.params[5]
        J_ext = self.params[6]

        print('Simulating with parameters:')
        print('J_EE:', J_EE)
        print('J_IE:', J_IE)
        print('J_EI:', J_EI)
        print('J_II:', J_II)
        print('tau_syn_E:', tau_syn_E)
        print('tau_syn_I:', tau_syn_I)
        print('J_ext:', J_ext)

        # Create hash
        js_0 = json.dumps([J_EE, J_IE, J_EI, J_II,
                           tau_syn_E, tau_syn_I,
                           J_ext],
                          sort_keys=True).encode()
        folder = hashlib.md5(js_0).hexdigest()

        # Create folders to save simulation data
        if not os.path.isdir('LIF_simulations'):
            os.mkdir('LIF_simulations')
        if not os.path.isdir('LIF_simulations/' + folder):
            os.mkdir('LIF_simulations/' + folder)

        # Create the LIF_network object
        LIF_net = LIF_network.LIF_network()

        # Set the number of threads
        LIF_net.local_num_threads = n_threads

        # Load/Create the kernel
        try:
            LIF_net.H_YX = pickle.load(open('LIF_simulations/H_YX', 'rb'))
        except (FileNotFoundError, IOError):
            print("\nKernel not found. Computing kernel...\n", end=' ', flush=True)
            LIF_net.create_kernel()
            # Save kernel to file
            pickle.dump(LIF_net.H_YX, open('LIF_simulations/H_YX', 'wb'))

        # Modify parameters
        LIF_net.LIF_params['J_YX'] = [[J_EE, J_IE], [J_EI, J_II]]
        LIF_net.LIF_params['tau_syn_YX'] = [[tau_syn_E, tau_syn_I],
                                            [tau_syn_E, tau_syn_I]]
        LIF_net.LIF_params['J_ext'] = J_ext

        # Create the LIF network
        LIF_net.create_LIF_network()

        # Simulation
        print('Simulating...\n', end=' ', flush=True)
        tac = time()
        LIF_net.simulate(tstop=LIF_net.tstop)
        toc = time()
        print(f'The simulation took {toc - tac} seconds.\n', end=' ', flush=True)

        # Mean spike/firing rates
        lif_mean_nu_X = dict()  # mean spike rates
        lif_nu_X = dict()  # binned firing rate

        for i, X in enumerate(LIF_net.LIF_params['X']):
            times = nest.GetStatus(
                LIF_net.spike_recorders[X])[0][
                'events']['times']
            times = times[times >= LIF_net.TRANSIENT]

            lif_mean_nu_X[X] = LIF_net.get_mean_spike_rate(times)
            bins, lif_nu_X[X] = LIF_net.get_spike_rate(times)

        # Compute LFP signals
        probe = 'GaussCylinderPotential'
        LFP_data = dict(EE = [],EI = [],IE = [],II = [])

        for X in LIF_net.LIF_params['X']:
            for Y in LIF_net.LIF_params['X']:
                n_ch = LIF_net.H_YX[f'{X}:{Y}'][probe].shape[0]
                for ch in range(n_ch):
                    # LFP kernel at electrode 'ch'
                    kernel = LIF_net.H_YX[f'{X}:{Y}'][probe][ch,:]
                    # LFP signal
                    sig = np.convolve(lif_nu_X[X], kernel, 'same')
                    # Decimate signal (x10)
                    LFP_data[f'{X}{Y}'].append(ss.decimate(
                                                sig,
                                                q=10,
                                                zero_phase=True))

        # Compute CDM
        probe = 'KernelApproxCurrentDipoleMoment'
        CDM_data = dict(EE=[], EI=[], IE=[], II=[])

        for X in LIF_net.LIF_params['X']:
            for Y in LIF_net.LIF_params['X']:
                # Pick only the z-component of the CDM kernel
                kernel = LIF_net.H_YX[f'{X}:{Y}'][probe][2, :]
                # CDM
                sig = np.convolve(lif_nu_X[X], kernel, 'same')
                CDM_data[f'{X}{Y}'] = ss.decimate(sig,
                                                  q=10,
                                                  zero_phase=True)

        # Save simulation parameters
        print('Saving data...\n', end=' ', flush=True)
        pickle.dump(LIF_net.LIF_params, open(
            'LIF_simulations/' + folder + '/LIF_params', 'wb'))
        pickle.dump(LIF_net.TRANSIENT, open(
            'LIF_simulations/' + folder + '/TRANSIENT', 'wb'))
        pickle.dump(LIF_net.dt, open(
            'LIF_simulations/' + folder + '/dt', 'wb'))
        pickle.dump(LIF_net.tstop, open(
            'LIF_simulations/' + folder + '/tstop', 'wb'))
        pickle.dump(LIF_net.tau, open(
            'LIF_simulations/' + folder + '/tau', 'wb'))

        # Save LFP
        pickle.dump(LFP_data,open(
                            'LIF_simulations/'+folder+'/LFP_data','wb'))

        # Save CDM
        pickle.dump(CDM_data['EE'] + \
                    CDM_data['EI'] + \
                    CDM_data['IE'] + \
                    CDM_data['II'],
                    open('LIF_simulations/' + folder + '/CDM_data', 'wb'))
        pickle.dump(CDM_data, open(
                        'LIF_simulations/' + folder + '/CDM_data_all', 'wb'))

        # Save mean spike rates and binned firing rates
        pickle.dump(lif_mean_nu_X,open(
                        'LIF_simulations/'+folder+'/lif_mean_nu_X','wb'))
        pickle.dump([bins, lif_nu_X],open(
                        'LIF_simulations/'+folder+'/lif_nu_X','wb'))

        # Save spike times and gids
        for i, Y in enumerate(LIF_net.LIF_params['X']):
            pickle.dump(nest.GetStatus(LIF_net.spike_recorders[Y])[0]['events']['times'],
                        open('LIF_simulations/'+folder+'/times_'+Y,'wb'))
            pickle.dump(nest.GetStatus(LIF_net.spike_recorders[Y])[0]['events']['senders'],
                        open('LIF_simulations/'+folder+'/gids_'+Y,'wb'))

        print('Done!\n', end=' ', flush=True)

        # # Check size and remove simulations with large files
        # th = 10  # MB
        # os.chdir('LIF_simulations')
        # print("\nFolder size: %s MB\n" % str(get_size(folder) / (2 ** 20)))
        # if get_size(folder) / (2 ** 20) > th:
        #     os.system('rm -r ' + folder)
        # os.chdir('..')





