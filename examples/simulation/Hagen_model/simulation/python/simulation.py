import os
import pickle
import sys
import nest
import time
import numpy as np
from importlib import util

class LIF_network(object):
    """
    Class to create and simulate a LIF network model.
    """

    def __init__(self, LIF_params):
        # Parameters of the LIF network model
        self.LIF_params = LIF_params

    def create_LIF_network(self, local_num_threads, dt, rng_seed=None):
        """
        Create network nodes and connections.
        """

        if rng_seed is None:
            rng_seed = int(1+time.localtime().tm_mon *\
                           time.localtime().tm_mday *\
                           time.localtime().tm_hour *\
                           time.localtime().tm_min *\
                           time.localtime().tm_sec *\
                           np.random.rand(1)[0])

        nest.ResetKernel()
        nest.SetKernelStatus(
            dict(
                local_num_threads=local_num_threads,
                rng_seed=int(rng_seed),
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
            print('Creating population %s\n' % X, end=' ', flush=True)
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
                print('Connecting %s with %s \n' % (X, Y), end=' ', flush=True)
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


def _simulate_with_progress(tstop, dt, chunk_ms):
    if tstop <= 0:
        return
    # Use chunks aligned to dt to reduce simulation-loop overhead while keeping progress updates.
    step = max(dt, float(chunk_ms))
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


def _resolve_chunking_config(module):
    """Read optional chunking controls from the simulation params module."""
    simulate_in_chunks = getattr(module, "simulate_in_chunks", True)
    chunk_ms = getattr(module, "simulation_chunk_ms", 1000.0)

    simulate_in_chunks = bool(simulate_in_chunks)
    try:
        chunk_ms = float(chunk_ms)
    except (TypeError, ValueError) as exc:
        raise TypeError("simulation_chunk_ms must be convertible to float.") from exc

    if chunk_ms <= 0:
        raise ValueError(f"simulation_chunk_ms must be positive, got {chunk_ms}.")

    return simulate_in_chunks, chunk_ms


def _check_spike_output_size(times, gids, lif_params, tstop, max_mean_population_spike_rate_hz):
    """Reject simulations with excessive population firing before writing spike pickles."""
    duration_seconds = float(tstop) / 1000.0
    if duration_seconds <= 0:
        raise ValueError(f"tstop must be positive, got {tstop}.")

    population_sizes = dict(zip(lif_params['X'], lif_params['N_X']))
    spike_counts = {X: int(np.asarray(times[X]).size) for X in lif_params['X']}
    mean_rates_hz = {
        X: spike_counts[X] / (duration_seconds * float(population_sizes[X]))
        for X in lif_params['X']
    }
    raw_spike_bytes = sum(
        np.asarray(times[X]).nbytes + np.asarray(gids[X]).nbytes
        for X in lif_params['X']
    )

    print(
        (
            f"Spike counts: {spike_counts}; "
            f"mean population rates Hz: {mean_rates_hz}; "
            f"raw spike array bytes: {raw_spike_bytes} "
            f"({raw_spike_bytes / 1024 ** 2:.2f} MiB)."
        ),
        flush=True,
    )

    if max_mean_population_spike_rate_hz is None:
        return

    max_mean_population_spike_rate_hz = float(max_mean_population_spike_rate_hz)
    too_high = {
        X: rate
        for X, rate in mean_rates_hz.items()
        if rate > max_mean_population_spike_rate_hz
    }
    if too_high:
        raise RuntimeError(
            (
                "Skipping spike pickle output because mean population firing rate "
                f"exceeds {max_mean_population_spike_rate_hz:.3g} Hz: {too_high}."
            )
        )


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

    simulate_in_chunks, simulation_chunk_ms = _resolve_chunking_config(module)
    max_mean_population_spike_rate_hz = getattr(
        module, "max_mean_population_spike_rate_hz", 50.0
    )

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
    with open(os.path.join(sys.argv[2],'network.pkl'), 'rb') as f:
        LIF_params = pickle.load(f)

    # Create the LIF network model
    network = LIF_network(LIF_params)
    network.create_LIF_network(local_num_threads, dt, rng_seed=rng_seed)

    # Simulation
    print('Simulating...\n', end=' ', flush=True)
    tac = time.time()
    if simulate_in_chunks:
        _simulate_with_progress(tstop, dt, simulation_chunk_ms)
    else:
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

    try:
        _check_spike_output_size(
            times,
            gids,
            network.LIF_params,
            tstop,
            max_mean_population_spike_rate_hz,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr, flush=True)
        sys.exit(2)

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
