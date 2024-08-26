import os
import numpy as np
import h5py
from copy import deepcopy
import time
import neuron
import nest
import json
from lfpykernels import KernelApprox,\
                        GaussCylinderPotential,\
                        KernelApproxCurrentDipoleMoment
import network_methods as methods
import network_parameters as params

class LIF_network(object):
    """
    Class to create and simulate a LIF network model.
    """

    def __init__(self):
        # recompile mod files if needed
        mech_loaded = neuron.load_mechanisms('mod')
        if not mech_loaded:
            os.system('cd mod && nrnivmodl && cd -')
            neuron.load_mechanisms('mod')

        # parameters of the simulation
        self.TRANSIENT = 2000
        self.dt = 0.0625
        self.tstop = 12000.0

        # time lag relative to spike for kernel predictions
        self.tau = 100

        # parameters of the multicompartment neuron network
        self.MC_params = {
            'weight_EE': 0.00015,
            'weight_IE': 0.000125,
            'weight_EI': 0.0045,
            'weight_II': 0.002,
            'weight_scaling': 1.0,
            'biophys': 'lin',
            'i_syn': True,
            'n_ext': [465, 160],
            'g_eff': True,
            'perseg_Vrest': False}

        # best fit params of the LIF network model
        self.LIF_params = dict(
            X = ['E', 'I'],
            N_X=[8192, 1024],
            C_m_X=[289.1, 110.7],
            tau_m_X=[10., 10.],
            E_L_X=[-65., -65.],
            C_YX=[[0.2, 0.2], [0.2, 0.2]],
            J_YX=[[1.589, 2.020], [-23.84, -8.441]],
            delay_YX = [[2.520, 1.714], [1.585, 1.149]],
            tau_syn_YX = [[0.5, 0.5], [0.5, 0.5]],
            n_ext=[465, 160],
            nu_ext=40.,
            J_ext=29.89,
            model='iaf_psc_exp',
            dt=2**-4)

        # Number of threads for the LIF network model simulations
        self.local_num_threads = 8

    def create_kernel(self):
        """
        Create kernels from multicompartment neuron network descriptions.
        """
        # kernel container
        self.H_YX = dict()

        # define biophysical membrane properties
        set_biophys = [methods.set_Ih_linearized_hay2011,
                       methods.make_cell_uniform]

        # synaptic weights
        weights = [[self.MC_params['weight_EE'],
                    self.MC_params['weight_IE']],
                   [self.MC_params['weight_EI'],
                    self.MC_params['weight_II']]]

        # class RecExtElectrode/PointSourcePotential parameters:
        electrodeParameters = params.electrodeParameters.copy()
        for key in ['r', 'n', 'N', 'method']:
            del electrodeParameters[key]

        # predictor assuming planar disk source elements convolved with Gaussian
        # along z-axis
        gauss_cyl_potential = GaussCylinderPotential(
            cell=None,
            z=electrodeParameters['z'],
            sigma=electrodeParameters['sigma'],
            R=params.populationParameters['pop_args']['radius'],
            sigma_z=params.populationParameters['pop_args']['scale'],
            )

        # set up recording of current dipole moments.
        # The class KernelApproxCurrentDipoleMoment accounts only for contributions
        # along the vertical z-axis as other components cancel with rotational symmetry
        current_dipole_moment = KernelApproxCurrentDipoleMoment(cell=None)

        # compute average firing rate of presynaptic populations X
        # Path to the folders containing the simulation data
        with open('../config.json', 'r') as config_file:
            config = json.load(config_file)
        OUTPUTPATH = os.path.join(config['multicompartment_neuron_network_path'],
                                  'output/adb947bfb931a5a8d09ad078a6d256b0')
        mean_nu_X = methods.compute_mean_nu_X(params, OUTPUTPATH,
                                              self.tstop,
                                              TRANSIENT=self.TRANSIENT)

        # compute a kernel for each combination of (X,Y)
        for i, (X, N_X) in enumerate(zip(params.population_names,
                                         params.population_sizes)):
            for j, (Y, N_Y, morphology) in enumerate(zip(params.population_names,
                                                         params.population_sizes,
                                                         params.morphologies)):

                # extract median soma voltages from actual network simulation and
                # assume this value corresponds to Vrest.
                with h5py.File(os.path.join(OUTPUTPATH, 'somav.h5'
                                            ), 'r') as f:
                    Vrest = np.median(f[Y][()][:, 200:])

                cellParameters = deepcopy(params.cellParameters)
                cellParameters.update(dict(
                    morphology=morphology,
                    custom_fun=set_biophys,
                    custom_fun_args=[dict(Vrest=Vrest), dict(Vrest=Vrest)],
                ))

                # some inputs must be lists
                synapseParameters = [
                    dict(weight=weights[ii][j],
                         syntype='Exp2Syn',
                         **params.synapseParameters[ii][j])
                    for ii in range(len(params.population_names))]
                synapsePositionArguments = [
                    params.synapsePositionArguments[ii][j]
                    for ii in range(len(params.population_names))]

                # create kernel approximator object
                kernel = KernelApprox(
                    X=params.population_names,
                    Y=Y,
                    N_X=np.array(params.population_sizes),
                    N_Y=N_Y,
                    C_YX=np.array(params.connectionProbability[i]),
                    cellParameters=cellParameters,
                    populationParameters=params.populationParameters['pop_args'],
                    multapseFunction=params.multapseFunction,
                    multapseParameters=[
                        params.multapseArguments[ii][j] for ii in range(
                                                len(params.population_names))],
                    delayFunction=params.delayFunction,
                    delayParameters=[
                        params.delayArguments[ii][j] for ii in range(
                                                len(params.population_names))],
                    synapseParameters=synapseParameters,
                    synapsePositionArguments=synapsePositionArguments,
                    extSynapseParameters=params.extSynapseParameters,
                    nu_ext=1000. / params.netstim_interval,
                    n_ext=self.MC_params['n_ext'][j],
                    nu_X=mean_nu_X,
                )

                # get kernel
                self.H_YX['{}:{}'.format(Y, X)] = kernel.get_kernel(
                    probes=[gauss_cyl_potential, current_dipole_moment],
                    Vrest=Vrest, dt=self.dt, X=X, t_X=self.TRANSIENT,
                    tau=self.tau, g_eff=self.MC_params['g_eff'])

    def create_LIF_network(self):
        """
        create network nodes and connections.
        """

        nest.ResetKernel()
        nest.SetKernelStatus(
            dict(
                local_num_threads=self.local_num_threads,
                rng_seed=int(1+time.localtime().tm_mon *\
                             time.localtime().tm_mday *\
                             time.localtime().tm_hour *\
                             time.localtime().tm_min *\
                             time.localtime().tm_sec *\
                             np.random.rand(1)[0]),
                resolution=self.LIF_params['dt'],
                tics_per_ms=1000 /
                self.LIF_params['dt']))

        # neurons
        self.neurons = {}
        for (X, N, C_m, tau_m, E_L, (tau_syn_ex, tau_syn_in)
             ) in zip(self.LIF_params['X'], self.LIF_params['N_X'],
                      self.LIF_params['C_m_X'], self.LIF_params['tau_m_X'],
                      self.LIF_params['E_L_X'], self.LIF_params['tau_syn_YX']):
            net_params = dict(
                C_m=C_m,
                tau_m=tau_m,
                E_L=E_L,
                V_reset=E_L,
                tau_syn_ex=tau_syn_ex,
                tau_syn_in=tau_syn_in
            )
            print('Creating population %s, tau_syn_ex = %s, tau_syn_in = %s\n' % (
                                                        X,tau_syn_ex,tau_syn_in),
                                                        end=' ', flush=True)
            self.neurons[X] = nest.Create(self.LIF_params['model'],
                                          N, net_params)

        # poisson generators
        self.poisson = {}
        for X, n_ext in zip(self.LIF_params['X'], self.LIF_params['n_ext']):
            self.poisson[X] = nest.Create(
                'poisson_generator', 1, dict(
                    rate=self.LIF_params['nu_ext'] * n_ext))

        # spike recorders
        self.spike_recorders = {}
        for X in self.LIF_params['X']:
            self.spike_recorders[X] = nest.Create('spike_recorder', 1)

        # connections
        for i, X in enumerate(self.LIF_params['X']):
            # recurrent connections
            for j, Y in enumerate(self.LIF_params['X']):
                conn_spec = dict(
                    rule='pairwise_bernoulli',
                    p=self.LIF_params['C_YX'][i][j],
                )
                print('Connecting %s with %s with weight %s\n' % (X,Y,
                                                self.LIF_params['J_YX'][i][j]),
                                                end=' ', flush=True)
                syn_spec = dict(
                    synapse_model='static_synapse',
                    weight=nest.math.redraw(
                        nest.random.normal(
                            mean=self.LIF_params['J_YX'][i][j],
                            std=abs(self.LIF_params['J_YX'][i][j]) * 0.1,
                        ),
                        min=0. if self.LIF_params['J_YX'][i][j] >= 0 else np.NINF,
                        max=np.Inf if self.LIF_params['J_YX'][i][j] >= 0 else 0.,
                    ),

                    delay=nest.math.redraw(
                        nest.random.normal(
                            mean=self.LIF_params['delay_YX'][i][j],
                            std=self.LIF_params['delay_YX'][i][j] * 0.5,
                        ),
                        min=0.3,
                        max=np.Inf,
                    )
                )

                nest.Connect(
                    self.neurons[X],
                    self.neurons[Y],
                    conn_spec,
                    syn_spec)

            # poisson generators
            nest.Connect(
                self.poisson[X],
                self.neurons[X],
                'all_to_all',
                dict(
                    weight=self.LIF_params['J_ext']))

            # recorders
            nest.Connect(self.neurons[X], self.spike_recorders[X])

    def simulate(self, tstop=6000):
        """Instantiate and run simulation"""
        nest.Simulate(tstop)

    def get_spike_rate(self,times):
        bins = (np.arange(self.TRANSIENT / self.dt, self.tstop / self.dt + 1)
                * self.dt - self.dt / 2)
        hist, _ = np.histogram(times, bins=bins)
        return bins, hist.astype(float)

    def get_mean_spike_rate(self,times):
        return times.size /  (self.tstop - self.TRANSIENT) * 1000
