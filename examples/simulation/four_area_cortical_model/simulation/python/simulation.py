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

    def create_LIF_network(self, local_num_threads, dt, rng_seed=None):
        """
        Create network nodes and connections.
        """

        if rng_seed is None:
            rng_seed = int(1 + time.localtime().tm_mon *
                           time.localtime().tm_mday *
                           time.localtime().tm_hour *
                           time.localtime().tm_min *
                           time.localtime().tm_sec *
                           np.random.rand(1)[0])

        nest.ResetKernel()
        nest.SetKernelStatus(
            dict(
                local_num_threads=local_num_threads,
                rng_seed=int(rng_seed),
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
                print('Creating population %s\n' % f"{area}_{X}", end=' ', flush=True)
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
                    print('Connecting %s with %s \n' % (f"{area}_{X}", f"{area}_{Y}"), end=' ', flush=True)
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
            if C_inter is None or J_inter is None or delay_inter is None:
                raise ValueError("inter_area must define C_YX, J_YX and delay_YX.")
            C_inter = np.asarray(C_inter, dtype=float)
            J_inter = np.asarray(J_inter, dtype=float)
            delay_inter = np.asarray(delay_inter, dtype=float)
            if C_inter.ndim != 2 or J_inter.ndim != 2 or delay_inter.ndim != 2:
                raise ValueError("inter_area matrices must be 2D.")
            if C_inter.shape != J_inter.shape or C_inter.shape != delay_inter.shape:
                raise ValueError("inter_area matrices C_YX, J_YX and delay_YX must share the same shape.")

            n_pops = len(pops)
            n_areas = len(areas)
            full_size = n_pops * n_areas
            if C_inter.shape == (n_pops, n_pops):
                inter_mode = "population"
            elif C_inter.shape == (full_size, full_size):
                inter_mode = "area_population"
            else:
                raise ValueError(
                    "Unsupported inter_area matrix shape. Use either "
                    f"{n_pops}x{n_pops} (population-level) or {full_size}x{full_size} "
                    "(full area-population connectivity)."
                )

            # Inter-area variability
            weight_cv = 0.10
            delay_cv = 0.30
            min_delay = 1.0

            for pre_area_idx, pre_area in enumerate(areas):
                for post_area_idx, post_area in enumerate(areas):
                    if pre_area == post_area:
                        continue
                    for i, pre in enumerate(pops):
                        for j, post in enumerate(pops):
                            if inter_mode == "population":
                                p_conn = float(C_inter[i][j])
                                j_weight = float(J_inter[i][j])
                                j_delay = float(delay_inter[i][j])
                            else:
                                src_idx = pre_area_idx * n_pops + i
                                tgt_idx = post_area_idx * n_pops + j
                                p_conn = float(C_inter[src_idx][tgt_idx])
                                j_weight = float(J_inter[src_idx][tgt_idx])
                                j_delay = float(delay_inter[src_idx][tgt_idx])

                            if p_conn <= 0.0 or j_weight == 0.0:
                                continue
                            conn_spec = dict(
                                rule='pairwise_bernoulli',
                                p=p_conn,
                            )
                            print(
                                'Connecting %s with %s \n' % (f"{pre_area}_{pre}", f"{post_area}_{post}"),
                                end=' ',
                                flush=True,
                            )
                            syn_spec = dict(
                                synapse_model='static_synapse',
                                weight=nest.math.redraw(
                                    nest.random.normal(
                                        mean=j_weight,
                                        std=abs(j_weight) * weight_cv,
                                    ),
                                    min=0.0 if j_weight >= 0 else -np.inf,
                                    max=np.inf if j_weight >= 0 else 0.0,
                                ),
                                delay=nest.math.redraw(
                                    nest.random.normal(
                                        mean=j_delay,
                                        std=j_delay * delay_cv,
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
    total_segments = int(np.ceil(tstop / step))
    total_segments = max(1, total_segments)
    segment_idx = 0
    sim_time = 0.0
    while sim_time < tstop:
        remaining = tstop - sim_time
        this_step = step if remaining > step else remaining
        if this_step <= 0:
            break
        nest.Simulate(this_step)
        sim_time += this_step
        segment_idx += 1
        print(f"SIM_SEGMENT {segment_idx}/{total_segments}", flush=True)


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

    # Optional fixed NumPy seed for reproducible randomization
    numpy_seed = getattr(module, "numpy_seed", None)
    rng_seed = None
    if numpy_seed is not None:
        numpy_seed = int(numpy_seed)
        if numpy_seed < 0:
            raise ValueError("numpy_seed must be a non-negative integer.")
        np.random.seed(numpy_seed)
        rng_seed = int(np.random.randint(1, 2**31 - 1))

    # Load network parameters
    with open(os.path.join(sys.argv[2], 'network.pkl'), 'rb') as f:
        LIF_params = pickle.load(f)

    # Create the LIF network model
    network = LIF_network(LIF_params)
    network.create_LIF_network(local_num_threads, dt, rng_seed=rng_seed)

    # Simulation
    print('Simulating...\n', end=' ', flush=True)
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
