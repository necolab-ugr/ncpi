import importlib
import os
import pickle
import sys
import nest
import time
import numpy as np

class LIF_network(object):
    """
    Class to create and simulate a LIF network model.
    """

    def __init__(self, LIF_params):
        # Parameters of the LIF network model
        self.LIF_params = LIF_params

    def create_LIF_network(self, local_num_threads, dt):
        """
        Create network nodes and connections.
        """

        nest.ResetKernel()
        nest.SetKernelStatus(
            dict(
                local_num_threads=local_num_threads,
                rng_seed=int(1+time.localtime().tm_mon *\
                             time.localtime().tm_mday *\
                             time.localtime().tm_hour *\
                             time.localtime().tm_min *\
                             time.localtime().tm_sec *\
                             np.random.rand(1)[0]),
                resolution=dt,
                tics_per_ms=1000 / dt))

        # Neurons
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

        # Poisson generators
        self.poisson = {}
        for X, n_ext in zip(self.LIF_params['X'], self.LIF_params['n_ext']):
            self.poisson[X] = nest.Create(
                'poisson_generator', 1, dict(
                    rate=self.LIF_params['nu_ext'] * n_ext))

        # Spike recorders
        self.spike_recorders = {}
        for X in self.LIF_params['X']:
            self.spike_recorders[X] = nest.Create('spike_recorder', 1)

        # Connections
        for i, X in enumerate(self.LIF_params['X']):
            # Recurrent connections
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

            # Poisson generators
            nest.Connect(
                self.poisson[X],
                self.neurons[X],
                'all_to_all',
                dict(
                    weight=self.LIF_params['J_ext']))

            # Recorders
            nest.Connect(self.neurons[X], self.spike_recorders[X])

    def simulate(self, tstop):
        """Instantiate and run simulation"""
        nest.Simulate(tstop)


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

    # Simulation time
    tstop = module.tstop

    # Number of threads
    local_num_threads = module.local_num_threads

    # Simulation time step
    dt = module.dt

    # Load network parameters
    with open(os.path.join(sys.argv[2],'network.pkl'), 'rb') as f:
        LIF_params = pickle.load(f)

    # Create the LIF network model
    network = LIF_network(LIF_params)
    network.create_LIF_network(local_num_threads, dt)

    # Simulation
    print('Simulating...\n', end=' ', flush=True)
    tac = time.time()
    network.simulate(tstop)
    toc = time.time()
    print(f'The simulation took {toc - tac} seconds.\n', end=' ', flush=True)

    # Get spike times
    times = dict()
    for i, X in enumerate(network.LIF_params['X']):
        times[X] = nest.GetStatus(network.spike_recorders[X])[0]['events']['times']

    # Get gids
    gids = dict()
    for i, X in enumerate(network.LIF_params['X']):
        gids[X] = nest.GetStatus(network.spike_recorders[X])[0]['events']['senders']

    # Save spike times
    with open(os.path.join(sys.argv[2],'times.pkl'), 'wb') as f:
        pickle.dump(times, f)

    # Save gids
    with open(os.path.join(sys.argv[2],'gids.pkl'), 'wb') as f:
        pickle.dump(gids, f)

    # Save tstop
    with open(os.path.join(sys.argv[2],'tstop.pkl'), 'wb') as f:
        pickle.dump(tstop, f)

    # Save dt
    with open(os.path.join(sys.argv[2],'dt.pkl'), 'wb') as f:
        pickle.dump(dt, f)
