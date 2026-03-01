import os
import pickle
import sys
import nest
import time
import numpy as np
from importlib import util


class LIF_network(object):
    """
    Class to create and simulate a four-area LIF network model.
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
                rng_seed=int(1 + time.localtime().tm_mon *
                             time.localtime().tm_mday *
                             time.localtime().tm_hour *
                             time.localtime().tm_min *
                             time.localtime().tm_sec *
                             np.random.rand(1)[0]),
                resolution=dt,
                tics_per_ms=1000 / dt))

        areas = self.LIF_params['areas']
        pops = self.LIF_params['X']

        # Neurons
        self.neurons = {area: {} for area in areas}
        for area in areas:
            for (X, N, C_m, tau_m, E_L, (tau_syn_ex, tau_syn_in)) in zip(
                self.LIF_params['X'],
                self.LIF_params['N_X'],
                self.LIF_params['C_m_X'],
                self.LIF_params['tau_m_X'],
                self.LIF_params['E_L_X'],
                self.LIF_params['tau_syn_YX'],
            ):
                net_params = dict(
                    C_m=C_m,
                    tau_m=tau_m,
                    E_L=E_L,
                    V_reset=E_L,
                    tau_syn_ex=tau_syn_ex,
                    tau_syn_in=tau_syn_in,
                )
                print(
                    '\n[CREATE] %s_%s | N=%d | tau_syn_ex=%s ms | tau_syn_in=%s ms | J_ext=%s nA'
                    % (area, X, N, tau_syn_ex, tau_syn_in, self.LIF_params['J_ext']),
                    flush=True,
                )
                self.neurons[area][X] = nest.Create(
                    self.LIF_params['model'],
                    N,
                    net_params,
                )

        # Poisson generators
        self.poisson = {area: {} for area in areas}
        for area in areas:
            for X, n_ext in zip(pops, self.LIF_params['n_ext']):
                self.poisson[area][X] = nest.Create(
                    'poisson_generator', 1, dict(
                        rate=self.LIF_params['nu_ext'] * n_ext))

        # Spike recorders
        self.spike_recorders = {area: {} for area in areas}
        for area in areas:
            for X in pops:
                self.spike_recorders[area][X] = nest.Create('spike_recorder', 1)

        # Connections
        for area in areas:
            for i, X in enumerate(pops):
                # Recurrent connections
                for j, Y in enumerate(pops):
                    conn_spec = dict(
                        rule='pairwise_bernoulli',
                        p=self.LIF_params['C_YX'][i][j],
                    )
                    print('\nConnecting %s_%s with %s_%s with weight %s\n' % (
                        area, X, area, Y, self.LIF_params['J_YX'][i][j]
                    ), end=' ', flush=True)
                    syn_spec = dict(
                        synapse_model='static_synapse',
                        weight=nest.math.redraw(
                            nest.random.normal(
                                mean=self.LIF_params['J_YX'][i][j],
                                std=abs(self.LIF_params['J_YX'][i][j]) * 0.1,
                            ),
                            min=0. if self.LIF_params['J_YX'][i][j] >= 0 else -np.inf,
                            max=np.inf if self.LIF_params['J_YX'][i][j] >= 0 else 0.,
                        ),

                        delay=nest.math.redraw(
                            nest.random.normal(
                                mean=self.LIF_params['delay_YX'][i][j],
                                std=self.LIF_params['delay_YX'][i][j] * 0.5,
                            ),
                            min=0.3,
                            max=np.inf,
                        )
                    )
                    nest.Connect(
                        self.neurons[area][X],
                        self.neurons[area][Y],
                        conn_spec,
                        syn_spec,
                    )

                # Poisson generators representing external drive (other brain areas, subcortical structures and background noise)
                nest.Connect(
                    self.poisson[area][X],
                    self.neurons[area][X],
                    'all_to_all',
                    dict(weight=self.LIF_params['J_ext']),
                )

                # Recorders
                nest.Connect(self.neurons[area][X], self.spike_recorders[area][X])

        # Inter-area excitatory-only connections (E->E and E->I)
        inter = self.LIF_params.get('inter_area', {})
        if inter:
            C_inter = inter.get('C_YX')
            J_inter = inter.get('J_YX')
            delay_inter = inter.get('delay_YX')
            # Inter-area variability
            weight_cv = 0.10
            delay_cv = 0.30
            min_delay = 1.0

            for pre_area in areas:
                for post_area in areas:
                    if pre_area == post_area:
                        continue
                    for i, pre in enumerate(pops):
                        for j, post in enumerate(pops):
                            if C_inter[i][j] <= 0.0 or J_inter[i][j] == 0.0:
                                continue
                            conn_spec = dict(
                                rule='pairwise_bernoulli',
                                p=C_inter[i][j],
                            )
                            print('\nConnecting %s_%s with %s_%s (inter-area) weight %s\n' % (
                                pre_area, pre, post_area, post, J_inter[i][j]
                            ), end=' ', flush=True)
                            syn_spec = dict(
                                synapse_model='static_synapse',
                                weight=nest.math.redraw(
                                    nest.random.normal(
                                        mean=J_inter[i][j],
                                        std=abs(J_inter[i][j]) * weight_cv,
                                    ),
                                    min=0.0 if J_inter[i][j] >= 0 else -np.inf,
                                    max=np.inf if J_inter[i][j] >= 0 else 0.0,
                                ),
                                delay=nest.math.redraw(
                                    nest.random.normal(
                                        mean=delay_inter[i][j],
                                        std=delay_inter[i][j] * delay_cv,
                                    ),
                                    min=min_delay,
                                    max=np.inf,
                                ),
                            )
                            nest.Connect(
                                self.neurons[pre_area][pre],
                                self.neurons[post_area][post],
                                conn_spec,
                                syn_spec,
                            )

    def simulate(self, tstop):
        """Instantiate and run simulation"""
        nest.Simulate(tstop)


def _simulate_with_progress(tstop, dt):
    if tstop <= 0:
        return
    # Use 1-second chunks (aligned to dt) to reduce simulation-loop overhead.
    step = max(dt, 1000.0)
    step = max(dt, round(step / dt) * dt)
    sim_time = 0.0
    while sim_time < tstop:
        remaining = tstop - sim_time
        this_step = step if remaining > step else remaining
        if this_step <= 0:
            break
        nest.Simulate(this_step)
        sim_time += this_step
        pct = int(min(100, round((sim_time / tstop) * 100)))
        print(f"PROGRESS:{pct}", flush=True)


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

    # Simulation time
    tstop = module.tstop

    # Number of threads
    local_num_threads = module.local_num_threads

    # Simulation time step
    dt = module.dt

    # Load network parameters
    with open(os.path.join(sys.argv[2], 'network.pkl'), 'rb') as f:
        LIF_params = pickle.load(f)

    # Create the LIF network model
    network = LIF_network(LIF_params)
    network.create_LIF_network(local_num_threads, dt)

    # Simulation
    print('\nSimulating...\n', end=' ', flush=True)
    tac = time.time()
    _simulate_with_progress(tstop, dt)
    toc = time.time()
    print(f'The simulation took {toc - tac} seconds.\n', end=' ', flush=True)

    areas = LIF_params['areas']
    pops = LIF_params['X']

    # Get spike times
    times = {area: {} for area in areas}
    for area in areas:
        for X in pops:
            times[area][X] = nest.GetStatus(
                network.spike_recorders[area][X]
            )[0]['events']['times']

    # Get gids
    gids = {area: {} for area in areas}
    for area in areas:
        for X in pops:
            gids[area][X] = nest.GetStatus(
                network.spike_recorders[area][X]
            )[0]['events']['senders']

    # Save spike times
    with open(os.path.join(sys.argv[2], 'times.pkl'), 'wb') as f:
        pickle.dump(times, f)

    # Save gids
    with open(os.path.join(sys.argv[2], 'gids.pkl'), 'wb') as f:
        pickle.dump(gids, f)

    # Save tstop
    with open(os.path.join(sys.argv[2], 'tstop.pkl'), 'wb') as f:
        pickle.dump(tstop, f)

    # Save dt
    with open(os.path.join(sys.argv[2], 'dt.pkl'), 'wb') as f:
        pickle.dump(dt, f)
