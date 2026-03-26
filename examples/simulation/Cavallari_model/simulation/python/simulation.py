import os
import pickle
import sys
import nest
import time
import numpy as np
from importlib import util
from pathlib import Path


MODULE_CANDIDATES = (
    ("cavallari_module",),
    ("build", "libcavallari_module.so"),
    ("build", "src", "cavallari_module.so"),
)


class LIF_network(object):
    """
    Class to create and simulate a LIF network model.
    """

    def __init__(self, LIF_params, tstop):
        # Parameters of the LIF network model
        self.LIF_params = LIF_params
        self.tstop = tstop

    def _ensure_custom_model_available(self):
        if "iaf_bw_2003" in nest.node_models:
            return

        model_root = Path(__file__).resolve().parents[2] / "neuron_model"
        for rel_parts in MODULE_CANDIDATES:
            if len(rel_parts) == 1 and rel_parts[0] == "cavallari_module":
                try:
                    nest.Install(rel_parts[0])
                except Exception:
                    pass
                else:
                    if "iaf_bw_2003" in nest.node_models:
                        return
                continue

            candidate = model_root.joinpath(*rel_parts)
            if not candidate.exists():
                continue
            nest.Install(str(candidate.resolve()))
            if "iaf_bw_2003" in nest.node_models:
                return

        install_script = model_root / "install.sh"
        raise FileNotFoundError(
            "The custom NEST model 'iaf_bw_2003' is not available. "
            f"Build it first with `{install_script}`."
        )

    def _build_external_rate_profiles(self, dt):
        network_params = self.LIF_params["network_params"]
        time_array = np.arange(dt, self.tstop - dt, dt, dtype=float)
        if time_array.size == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty

        ou_sigma = float(network_params["OU_sigma"])
        ou_tau = float(network_params["OU_tau"])
        ou_x = np.zeros(time_array.size, dtype=float)
        if ou_tau <= 0:
            raise ValueError("OU_tau must be positive.")

        ou_sigma_bis = ou_sigma * np.sqrt(2.0 / ou_tau)
        ou_sqrtdt = np.sqrt(dt)
        for idx in range(time_array.size - 1):
            ou_x[idx + 1] = (
                ou_x[idx]
                + dt * (-ou_x[idx] / ou_tau)
                + ou_sigma_bis * ou_sqrtdt * np.random.randn()
            )

        v_signal = (
            float(network_params["A_ext"]) * np.sin(2.0 * np.pi * float(network_params["f_ext"]) * time_array / 1000.0)
            + float(network_params["v_0"])
        )
        # Keep the original two-stream construction. The cortical-cortical
        # stream is clipped because Poisson generator rates must be non-negative.
        thalamic_rate = 800.0 * v_signal
        cortical_rate = 800.0 * np.clip(ou_x, a_min=0.0, a_max=None)
        return time_array, thalamic_rate, cortical_rate

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
                tics_per_ms=1000 / dt,
                overwrite_files=True))
        self._ensure_custom_model_available()

        # Neurons
        self.neurons = {}
        for X, N in zip(self.LIF_params["X"], self.LIF_params["N_X"]):
            net_params = self.LIF_params["neuron_params"][X]
            print("Creating population %s\n" % X, end=" ", flush=True)
            self.neurons[X] = nest.Create(self.LIF_params["model"],
                                          N, net_params)

        # External inputs
        rate_times, thalamic_rate, cortical_rate = self._build_external_rate_profiles(dt)
        self.thalamic_input = nest.Create("inhomogeneous_poisson_generator", 1)
        self.cortical_input = nest.Create("inhomogeneous_poisson_generator", 1)
        nest.SetStatus(
            self.thalamic_input,
            dict(
                rate_times=rate_times,
                rate_values=thalamic_rate))
        nest.SetStatus(
            self.cortical_input,
            dict(
                rate_times=rate_times,
                rate_values=cortical_rate))
        self.external_rate_profile = {
            "thalamic": dict(
                rate_times=rate_times.copy(),
                rate_values=thalamic_rate.copy(),
            ),
            "cortical": dict(
                rate_times=rate_times.copy(),
                rate_values=cortical_rate.copy(),
            ),
        }

        # Spike recorders
        self.spike_recorders = {}
        for X in self.LIF_params["X"]:
            self.spike_recorders[X] = nest.Create("spike_recorder", 1)

        # Connections
        for i, Y in enumerate(self.LIF_params["X"]):
            # Recurrent connections
            for j, X in enumerate(self.LIF_params["X"]):
                conn_spec = dict(
                    rule="pairwise_bernoulli",
                    p=self.LIF_params["C_YX"][i][j],
                    allow_autapses=False,
                    allow_multapses=False,
                )
                print("Connecting %s with %s \n" % (X, Y), end=" ", flush=True)
                syn_spec = dict(
                    synapse_model="static_synapse",
                    weight=self.LIF_params["g_YX"][i][j],
                    delay=self.LIF_params["delay_YX"][i][j],
                )

                nest.Connect(
                    self.neurons[X],
                    self.neurons[Y],
                    conn_spec,
                    syn_spec)

            # Recorders
            nest.Connect(self.neurons[Y], self.spike_recorders[Y])

        # External inputs keep the original thalamic/cortical split, but
        # connect directly to neurons since each target already receives an
        # independent Poisson train from the generator.
        network_params = self.LIF_params["network_params"]
        nest.Connect(
            self.thalamic_input,
            self.neurons["E"],
            "all_to_all",
            dict(weight=network_params["g_th_exc_external"]))
        nest.Connect(
            self.cortical_input,
            self.neurons["E"],
            "all_to_all",
            dict(weight=network_params["g_cc_exc_external"]))
        nest.Connect(
            self.thalamic_input,
            self.neurons["I"],
            "all_to_all",
            dict(weight=network_params["g_th_inh_external"]))
        nest.Connect(
            self.cortical_input,
            self.neurons["I"],
            "all_to_all",
            dict(weight=network_params["g_cc_inh_external"]))

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


if __name__ == "__main__":
    # Read the script file path from sys.argv[1]
    script_path = sys.argv[1]

    # Add the directory containing the script to the Python path
    script_dir = os.path.dirname(script_path)
    sys.path.append(script_dir)

    # Import the script as a module
    module_name = os.path.basename(script_path).replace(".py", "")
    spec = util.spec_from_file_location(module_name, script_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Simulation time
    tstop = float(module.tstop)

    # Number of threads
    local_num_threads = int(module.local_num_threads)

    # Simulation time step
    dt = float(module.dt)

    simulate_in_chunks, simulation_chunk_ms = _resolve_chunking_config(module)

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
    with open(os.path.join(sys.argv[2], "network.pkl"), "rb") as f:
        LIF_params = pickle.load(f)

    # Create the LIF network model
    network = LIF_network(LIF_params, tstop)
    network.create_LIF_network(local_num_threads, dt, rng_seed=rng_seed)

    # Simulation
    print("Simulating...\n", end=" ", flush=True)
    tac = time.time()
    if simulate_in_chunks:
        _simulate_with_progress(tstop, dt, simulation_chunk_ms)
    else:
        network.simulate(tstop)
    toc = time.time()
    print(f"The simulation took {toc - tac} seconds.\n", end=" ", flush=True)

    # Get spike times
    times = dict()
    for i, X in enumerate(network.LIF_params["X"]):
        times[X] = np.asarray(
            nest.GetStatus(network.spike_recorders[X])[0]["events"]["times"])

    # Get gids
    gids = dict()
    for i, X in enumerate(network.LIF_params["X"]):
        gids[X] = np.asarray(
            nest.GetStatus(network.spike_recorders[X])[0]["events"]["senders"])

    # Save spike times
    with open(os.path.join(sys.argv[2], "times.pkl"), "wb") as f:
        pickle.dump(times, f)

    # Save gids
    with open(os.path.join(sys.argv[2], "gids.pkl"), "wb") as f:
        pickle.dump(gids, f)

    # Save external rate profiles
    with open(os.path.join(sys.argv[2], "external_rate_profile.pkl"), "wb") as f:
        pickle.dump(network.external_rate_profile, f)

    # Save tstop
    with open(os.path.join(sys.argv[2], "tstop.pkl"), "wb") as f:
        pickle.dump(tstop, f)

    # Save dt
    with open(os.path.join(sys.argv[2], "dt.pkl"), "wb") as f:
        pickle.dump(dt, f)
