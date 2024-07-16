#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import nest
import os
import scipy.signal as ss
import pickle
import numpy as np
import LIF_network
import json
import hashlib


class Simulation:
    '''
    This class is used to run the simulation of the LIF network.
    '''
    def __init__(self, params, multi_compartment_model_path):
        # check size of array
        if len(params) != 7:
            raise ValueError('The number of parameters should be 7.')
        self.params = params

        # Assert that multi_compartment_model_path is a string
        if not isinstance(multi_compartment_model_path, str):
            raise TypeError("multi_compartment_model_path must be a string")

        self.multi_compartment_model_path = multi_compartment_model_path

    def run(self):
        '''
        This function runs the simulation.
        '''
        # create folder to save new simulations
        if not os.path.isdir('LIF_simulations'):
            os.mkdir('LIF_simulations')

        # Get parameters
        J_EE = self.params[0]
        J_IE = self.params[1]
        J_EI = self.params[2]
        J_II = self.params[3]
        tau_syn_E = self.params[4]
        tau_syn_I = self.params[5]
        J_ext = self.params[6]

        # Create hash
        js_0 = json.dumps([J_EE, J_IE, J_EI, J_II,
                           tau_syn_E, tau_syn_I,
                           J_ext],
                          sort_keys=True).encode()
        folder = hashlib.md5(js_0).hexdigest()

        # Create folder
        if not os.path.isdir('LIF_simulations/' +\
                             folder):
            os.mkdir('LIF_simulations/' + folder)

        # create the LIF_network object
        LIF_net = LIF_network.LIF_network()

        # load/create kernel
        try:
            LIF_net.H_YX = pickle.load(open(
                'LIF_simulations/H_YX', 'rb'))
        except (FileNotFoundError, IOError):
            print("\nKernel not found." \
                  " Computing kernel...\n",
                  end=' ', flush=True)
            LIF_net.create_kernel(self.multi_compartment_model_path)
            # Save kernel to file
            pickle.dump(LIF_net.H_YX, open(
                'LIF_simulations/H_YX', 'wb'))

        # Modify parameters
        LIF_net.LIF_params['J_YX'] = [[J_EE, J_IE], [J_EI, J_II]]
        LIF_net.LIF_params['tau_syn_YX'] = [[tau_syn_E, tau_syn_I],
                                            [tau_syn_E, tau_syn_I]]
        LIF_net.LIF_params['J_ext'] = J_ext

        # create the LIF network
        LIF_net.create_LIF_network()

        # Simulation
        print('Simulating...\n', end=' ', flush=True)
        tac = time()
        LIF_net.simulate(tstop=LIF_net.tstop)
        toc = time()
        print(f'The simulation took {toc - tac} seconds.\n',
              end=' ', flush=True)

        # mean spike/firing rates
        lif_mean_nu_X = dict()  # mean spike rates
        lif_nu_X = dict()  # binned firing rate

        for i, X in enumerate(LIF_net.LIF_params['X']):
            times = nest.GetStatus(
                LIF_net.spike_recorders[X])[0][
                'events']['times']
            times = times[times >= LIF_net.TRANSIENT]

            lif_mean_nu_X[X] = LIF_net.get_mean_spike_rate(times)
            bins, lif_nu_X[X] = LIF_net.get_spike_rate(times)

        # # compute LFP signals
        # probe = 'GaussCylinderPotential'
        # LFP_data = dict(EE = [],EI = [],IE = [],II = [])
        #
        # for X in LIF_net.LIF_params['X']:
        #     for Y in LIF_net.LIF_params['X']:
        #         n_ch = LIF_net.H_YX[f'{X}:{Y}'][probe].shape[0]
        #         for ch in range(n_ch):
        #             # LFP kernel at electrode 'ch'
        #             kernel = LIF_net.H_YX[f'{X}:{Y}'][probe][ch,:]
        #             # LFP signal
        #             sig = np.convolve(lif_nu_X[X], kernel, 'same')
        #             # Decimate signal (x10)
        #             LFP_data[f'{X}{Y}'].append(ss.decimate(
        #                                         sig,
        #                                         q=10,
        #                                         zero_phase=True))

        # compute CDM
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

        # Save simulation data to file
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

        # pickle.dump(LFP_data,open(
        #                     'LIF_simulations/'+folder+'/LFP_data','wb'))
        pickle.dump(CDM_data['EE'] + \
                    CDM_data['EI'] + \
                    CDM_data['IE'] + \
                    CDM_data['II'],
                    open('LIF_simulations/' + folder + '/CDM_data', 'wb'))
        pickle.dump(lif_mean_nu_X,open(
                        'LIF_simulations/'+folder+'/lif_mean_nu_X','wb'))
        # pickle.dump([bins, lif_nu_X],open(
        #                 'LIF_simulations/'+folder+'/lif_nu_X','wb'))
        #
        # for i, Y in enumerate(LIF_net.LIF_params['X']):
        #     pickle.dump(
        # nest.GetStatus(LIF_net.spike_recorders[Y])[0]['events']['times'],
        #                 open('LIF_simulations/'+folder+'/times_'+Y,'wb'))
        #     pickle.dump(
        # nest.GetStatus(LIF_net.spike_recorders[Y])[0]['events']['senders'],
        #                 open('LIF_simulations/'+folder+'/gids_'+Y,'wb'))

        print('Done!\n', end=' ', flush=True)

        # # Check size and remove simulations with large files
        # th = 10  # MB
        # os.chdir('LIF_simulations')
        # print("\nFolder size: %s MB\n" % str(get_size(
        #     folder) / (2 ** 20)))
        # if get_size(folder) / (2 ** 20) > th:
        #     os.system('rm -r ' + folder)
        # os.chdir('..')

        return folder
