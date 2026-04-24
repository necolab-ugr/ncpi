import os
import pickle
import sys
import time
from importlib import util
from pathlib import Path
import nest
import numpy as np


MODULE_CANDIDATES = (
    # Try the installed module name first, then common local build artifacts.
    ("cavallari_module",),
    ("build", "libcavallari_module.so"),
    ("build", "src", "cavallari_module.so"),
)


def _node_ids(nodes):
    """Return plain integer node ids from a NEST NodeCollection."""
    return list(nodes.tolist())


def _default_analysis_params():
    """Default state variables recorded from the full E/I populations."""
    return {
        "To_be_measured": ["V_m", "g_ex", "g_in"],
    }


class network:
    """Reproduce the original pablomc88 `network_1.py` workflow with NEST 3.8.

    The `network_1.py` reference was taken from
    https://github.com/pablomc88/EEG_proxy_from_network_point_neurons.

    This model preserves the original population-level setup while adapting the
    external input pathway for current NEST APIs. Thalamic and cortical drives
    are implemented as inhomogeneous Poisson generators connected directly to
    the excitatory and inhibitory populations. Generator rates are scaled to
    preserve the effective drive previously delivered through parrot-neuron
    relays.
    """

    def __init__(
        self,
        Network_params,
        Neuron_params,
        Simulation_params,
        External_input_params,
        Analysis_params,
    ):
        """Initialize the network from parameter dictionaries."""
        self.N_exc = Network_params["N_exc"]
        self.N_inh = Network_params["N_inh"]
        self.P = Network_params["P"]
        self.extent = Network_params["extent"]
        self.exc_exc_recurrent = Network_params["exc_exc_recurrent"]
        self.exc_inh_recurrent = Network_params["exc_inh_recurrent"]
        self.inh_inh_recurrent = Network_params["inh_inh_recurrent"]
        self.inh_exc_recurrent = Network_params["inh_exc_recurrent"]
        self.th_exc_external = Network_params["th_exc_external"]
        self.th_inh_external = Network_params["th_inh_external"]
        self.cc_exc_external = Network_params["cc_exc_external"]
        self.cc_inh_external = Network_params["cc_inh_external"]

        self.excitatory_cell_params = {
            "V_th": Neuron_params[0]["V_th"],
            "V_reset": Neuron_params[0]["V_reset"],
            "t_ref": Neuron_params[0]["t_ref"],
            "g_L": Neuron_params[0]["g_L"],
            "C_m": Neuron_params[0]["C_m"],
            "E_ex": Neuron_params[0]["E_ex"],
            "E_in": Neuron_params[0]["E_in"],
            "E_L": Neuron_params[0]["E_L"],
            "tau_rise_AMPA": Neuron_params[0]["tau_rise_AMPA"],
            "tau_decay_AMPA": Neuron_params[0]["tau_decay_AMPA"],
            "tau_rise_GABA_A": Neuron_params[0]["tau_rise_GABA_A"],
            "tau_decay_GABA_A": Neuron_params[0]["tau_decay_GABA_A"],
            "tau_m": Neuron_params[0]["tau_m"],
            "I_e": Neuron_params[0]["I_e"],
        }
        self.inhibitory_cell_params = {
            "V_th": Neuron_params[1]["V_th"],
            "V_reset": Neuron_params[1]["V_reset"],
            "t_ref": Neuron_params[1]["t_ref"],
            "g_L": Neuron_params[1]["g_L"],
            "C_m": Neuron_params[1]["C_m"],
            "E_ex": Neuron_params[1]["E_ex"],
            "E_in": Neuron_params[1]["E_in"],
            "E_L": Neuron_params[1]["E_L"],
            "tau_rise_AMPA": Neuron_params[1]["tau_rise_AMPA"],
            "tau_decay_AMPA": Neuron_params[1]["tau_decay_AMPA"],
            "tau_rise_GABA_A": Neuron_params[1]["tau_rise_GABA_A"],
            "tau_decay_GABA_A": Neuron_params[1]["tau_decay_GABA_A"],
            "tau_m": Neuron_params[1]["tau_m"],
            "I_e": Neuron_params[1]["I_e"],
        }

        self.simtime = Simulation_params["simtime"]
        self.simstep = Simulation_params["simstep"]
        self.num_threads = Simulation_params["num_threads"]
        self.toMemory = Simulation_params["toMemory"]

        self.v_0 = External_input_params["v_0"]
        self.A_ext = External_input_params["A_ext"]
        self.f_ext = External_input_params["f_ext"]
        self.OU_sigma = External_input_params["OU_sigma"]
        self.OU_tau = External_input_params["OU_tau"]

        self.To_be_measured = Analysis_params["To_be_measured"]

        self.external_rate_profile = {}

    def _ensure_custom_model_available(self):
        """Install or locate the custom Cavallari NEST neuron model."""
        if "iaf_bw_2003" in nest.node_models:
            return

        model_root = Path(__file__).resolve().parents[2] / "neuron_model"
        for rel_parts in MODULE_CANDIDATES:
            if rel_parts == ("cavallari_module",):
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

    def _random_positions(self, n_nodes):
        """Sample uniformly distributed 2D positions within the network extent."""
        return [
            [
                np.random.uniform(-self.extent / 2.0, self.extent / 2.0),
                np.random.uniform(-self.extent / 2.0, self.extent / 2.0),
            ]
            for _ in range(int(n_nodes))
        ]

    def _create_nodes(self, model, n_nodes):
        """Create positioned NEST nodes for a population or input generator."""
        positions = nest.spatial.free(
            pos=self._random_positions(n_nodes),
            extent=[self.extent, self.extent],
        )
        return nest.Create(model, int(n_nodes), positions=positions)

    def create_network(self, rng_seed=None, results_dir=None):
        """Initialize NEST, create populations, inputs, synapses, and recorders."""
        if rng_seed is None:
            rng_seed = int(
                1
                + time.localtime().tm_mon
                * time.localtime().tm_mday
                * time.localtime().tm_hour
                * time.localtime().tm_min
                * time.localtime().tm_sec
                * np.random.rand(1)[0]
            )

        if results_dir is None:
            results_dir = Path(__file__).resolve().parents[1] / "results"
        else:
            results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        nest.ResetKernel()
        nest.SetKernelStatus(
            {
                "local_num_threads": int(self.num_threads),
                "resolution": float(self.simstep),
                "tics_per_ms": 1000 / float(self.simstep),
                "data_path": str(results_dir.resolve()),
                "rng_seed": int(rng_seed),
            }
        )
        self._ensure_custom_model_available()

        nest.CopyModel("iaf_bw_2003", "exc_cell", self.excitatory_cell_params)
        nest.CopyModel("iaf_bw_2003", "inh_cell", self.inhibitory_cell_params)
        nest.CopyModel("inhomogeneous_poisson_generator", "thalamocortical_input")
        nest.CopyModel("inhomogeneous_poisson_generator", "cortical_input")

        self.exc = self._create_nodes("exc_cell", self.N_exc)
        self.inh = self._create_nodes("inh_cell", self.N_inh)
        self.thalamo = self._create_nodes("thalamocortical_input", 1)
        self.cort = self._create_nodes("cortical_input", 1)

        time_array = np.arange(self.simstep, self.simtime - self.simstep, self.simstep)
        OU_sigma_bis = self.OU_sigma * np.sqrt(2.0 / self.OU_tau)
        OU_sqrtdt = np.sqrt(self.simstep)
        OU_x = np.zeros(len(time_array), dtype=float)
        for i in range(len(time_array) - 1):
            OU_x[i + 1] = (
                OU_x[i]
                + self.simstep * (-OU_x[i] / self.OU_tau)
                + OU_sigma_bis * OU_sqrtdt * np.random.randn()
            )

        v_signal = self.A_ext * np.sin(2.0 * np.pi * self.f_ext * time_array / 1000.0) + self.v_0
        th_rate_values = 800.0 * v_signal
        cc_rate_values = 800.0 * OU_x
        nest.SetStatus(self.thalamo, {"rate_times": time_array, "rate_values": th_rate_values})
        nest.SetStatus(self.cort, {"rate_times": time_array, "rate_values": cc_rate_values})
        self.external_rate_profile = {
            "thalamic": {"rate_times": time_array.copy(), "rate_values": th_rate_values.copy()},
            "cortical": {"rate_times": time_array.copy(), "rate_values": cc_rate_values.copy()},
        }

        recurrent_pairs = (
            (self.exc, self.exc, "exc_cell", "exc_cell", self.exc_exc_recurrent),
            (self.exc, self.inh, "exc_cell", "inh_cell", self.exc_inh_recurrent),
            (self.inh, self.inh, "inh_cell", "inh_cell", self.inh_inh_recurrent),
            (self.inh, self.exc, "inh_cell", "exc_cell", self.inh_exc_recurrent),
        )
        for source_nodes, target_nodes, _, _, weight in recurrent_pairs:
            nest.Connect(
                source_nodes,
                target_nodes,
                {
                    "rule": "pairwise_bernoulli",
                    "p": float(self.P),
                    "allow_autapses": False,
                    "allow_multapses": False,
                },
                {
                    "synapse_model": "static_synapse",
                    "weight": float(weight),
                    "delay": 1.0,
                },
            )

        nest.Connect(
            self.thalamo,
            self.exc,
            "all_to_all",
            {"synapse_model": "static_synapse", "weight": self.th_exc_external, "delay": 1.0},
        )
        nest.Connect(
            self.thalamo,
            self.inh,
            "all_to_all",
            {"synapse_model": "static_synapse", "weight": self.th_inh_external, "delay": 1.0},
        )
        nest.Connect(
            self.cort,
            self.exc,
            "all_to_all",
            {"synapse_model": "static_synapse", "weight": self.cc_exc_external, "delay": 1.0},
        )
        nest.Connect(
            self.cort,
            self.inh,
            "all_to_all",
            {"synapse_model": "static_synapse", "weight": self.cc_inh_external, "delay": 1.0},
        )

        record_to = "memory" if self.toMemory else "ascii"
        if len(self.To_be_measured) > 0:
            nest.CopyModel(
                "multimeter",
                "RecordingNode",
                {
                    "interval": float(self.simstep),
                    "record_from": list(self.To_be_measured),
                    "record_to": record_to,
                },
            )
        nest.CopyModel(
            "spike_recorder",
            "SpikesRecorder",
            {
                "record_to": record_to,
            },
        )

        self.subthreshold_recorders = []
        if len(self.To_be_measured) > 0:
            rec_exc = nest.Create("RecordingNode")
            nest.Connect(rec_exc, self.exc)
            self.subthreshold_recorders.append(rec_exc)

            rec_inh = nest.Create("RecordingNode")
            nest.Connect(rec_inh, self.inh)
            self.subthreshold_recorders.append(rec_inh)

        self.spikes = []

        sp_exc = nest.Create("SpikesRecorder")
        nest.Connect(self.exc, sp_exc)
        self.spikes.append(sp_exc)

        sp_inh = nest.Create("SpikesRecorder")
        nest.Connect(self.inh, sp_inh)
        self.spikes.append(sp_inh)

    def simulate(self, tstop):
        """Instantiate and run simulation"""
        nest.Simulate(tstop)


def _simulate_with_progress(tstop, dt, chunk_ms):
    """Run NEST in dt-aligned chunks and print segment progress."""
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


def _resolve_population_names(lif_params):
    """Resolve excitatory/inhibitory population labels from saved LIF params."""
    raw_names = lif_params.get("X")
    if not isinstance(raw_names, (list, tuple)) or len(raw_names) < 2:
        return "E", "I"
    exc_name = str(raw_names[0])
    inh_name = str(raw_names[1])
    if exc_name == inh_name:
        raise ValueError(
            "lif_params['X'] must contain two distinct population names (exc, inh)."
        )
    return exc_name, inh_name


def _resolve_neuron_param_pair(lif_params, exc_name, inh_name):
    """Return excitatory/inhibitory neuron parameter dicts from flexible key layouts."""
    neuron_params = lif_params.get("neuron_params")
    if not isinstance(neuron_params, dict):
        raise KeyError("lif_params['neuron_params'] must be a dict.")

    exc_params = neuron_params.get(exc_name)
    inh_params = neuron_params.get(inh_name)
    if isinstance(exc_params, dict) and isinstance(inh_params, dict):
        return exc_params, inh_params

    exc_params = neuron_params.get("E")
    inh_params = neuron_params.get("I")
    if isinstance(exc_params, dict) and isinstance(inh_params, dict):
        return exc_params, inh_params

    dict_values = [value for value in neuron_params.values() if isinstance(value, dict)]
    if len(dict_values) >= 2:
        return dict_values[0], dict_values[1]

    available_keys = sorted(str(key) for key in neuron_params.keys())
    raise KeyError(
        "Unable to resolve excitatory/inhibitory neuron parameter dictionaries from "
        f"lif_params['neuron_params'] keys: {available_keys}"
    )


def _check_spike_output_size(times, gids, lif_params, tstop, max_mean_population_spike_rate_hz):
    """Reject simulations with excessive population firing before writing spike pickles."""
    duration_seconds = float(tstop) / 1000.0
    if duration_seconds <= 0:
        raise ValueError(f"tstop must be positive, got {tstop}.")

    raw_population_names = lif_params.get("X")
    if isinstance(raw_population_names, (list, tuple)) and raw_population_names:
        population_names = [str(name) for name in raw_population_names]
    else:
        population_names = [str(name) for name in times.keys()]

    missing_times = [name for name in population_names if name not in times]
    missing_gids = [name for name in population_names if name not in gids]
    if missing_times or missing_gids:
        raise KeyError(
            "Missing population spike payloads. "
            f"missing times={missing_times}, missing gids={missing_gids}, "
            f"available times keys={sorted(str(key) for key in times.keys())}, "
            f"available gids keys={sorted(str(key) for key in gids.keys())}."
        )

    raw_population_sizes = lif_params.get("N_X")
    if not isinstance(raw_population_sizes, (list, tuple)) or len(raw_population_sizes) < len(population_names):
        raise ValueError(
            "lif_params['N_X'] must provide one population size per entry in lif_params['X']."
        )
    population_sizes = {
        name: float(raw_population_sizes[idx])
        for idx, name in enumerate(population_names)
    }
    spike_counts = {name: int(np.asarray(times[name]).size) for name in population_names}
    mean_rates_hz = {
        name: spike_counts[name] / (duration_seconds * float(population_sizes[name]))
        for name in population_names
    }
    raw_spike_bytes = sum(
        np.asarray(times[name]).nbytes + np.asarray(gids[name]).nbytes
        for name in population_names
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


def _default_network_payload(lif_params, module):
    """Build Cavallari network constructor inputs from stored LIF parameters."""
    network_params = lif_params["network_params"]
    exc_name, inh_name = _resolve_population_names(lif_params)
    exc_params, inh_params = _resolve_neuron_param_pair(lif_params, exc_name, inh_name)
    neuron_params = [exc_params, inh_params]
    simulation_params = {
        "simtime": float(module.tstop),
        "simstep": float(module.dt),
        "num_threads": int(module.local_num_threads),
        "toMemory": True,
    }
    external_input_params = {
        "v_0": network_params["v_0"],
        "A_ext": network_params["A_ext"],
        "f_ext": network_params["f_ext"],
        "OU_sigma": network_params["OU_sigma"],
        "OU_tau": network_params["OU_tau"],
    }
    return (
        network_params,
        neuron_params,
        simulation_params,
        external_input_params,
        _default_analysis_params(),
    )


def _aggregate_state_chunk(events, E_ex, E_in):
    """Aggregate recorded state events into population-summed currents by time."""
    times = np.asarray(events["times"], dtype=float)
    if times.size == 0:
        empty_float = np.array([], dtype=float)
        empty_int = np.array([], dtype=int)
        return {
            "times": empty_float,
            "AMPA": empty_float,
            "GABA": empty_float,
            "Vm": empty_float,
            "counts": empty_int,
        }

    V_m = np.asarray(events["V_m"], dtype=float)
    g_ex = np.asarray(events["g_ex"], dtype=float)
    g_in = np.asarray(events["g_in"], dtype=float)

    unique_times, inverse = np.unique(times, return_inverse=True)
    return {
        "times": unique_times,
        "AMPA": np.bincount(inverse, weights=-(g_ex * (V_m - E_ex))),
        "GABA": np.bincount(inverse, weights=-(g_in * (V_m - E_in))),
        "Vm": np.bincount(inverse, weights=V_m),
        "counts": np.bincount(inverse).astype(int),
    }


def _append_state_chunks(chunks, payload):
    """Append a non-empty aggregated state payload to chunk accumulators."""
    if payload["times"].size == 0:
        return
    for key in ("times", "AMPA", "GABA", "Vm", "counts"):
        chunks[key].append(payload[key])


def _finalize_state_chunks(chunks, interval):
    """Concatenate aggregated state chunks and attach recording metadata."""
    if not chunks["times"]:
        empty_float = np.array([], dtype=float)
        empty_int = np.array([], dtype=int)
        return {
            "times": empty_float,
            "AMPA": empty_float,
            "GABA": empty_float,
            "Vm": empty_float,
            "counts": empty_int,
            "interval": interval,
            "aggregation": "population_sum",
        }

    return {
        "times": np.concatenate(chunks["times"]),
        "AMPA": np.concatenate(chunks["AMPA"]),
        "GABA": np.concatenate(chunks["GABA"]),
        "Vm": np.concatenate(chunks["Vm"]),
        "counts": np.concatenate(chunks["counts"]),
        "interval": interval,
        "aggregation": "population_sum",
    }


def _finalize_spikes(acc):
    """Concatenate per-population spike arrays from one or more chunks."""
    finalized = {}
    for key, parts in acc.items():
        if not parts:
            finalized[key] = np.array([], dtype=float)
        else:
            finalized[key] = np.concatenate(parts)
    return finalized


def _extract_population_spikes(data_s, population_names):
    """Build per-population spike time and gid payloads from recorder events."""
    if not isinstance(population_names, (list, tuple)) or len(population_names) < 2:
        raise ValueError("population_names must contain two entries (exc, inh).")
    if not isinstance(data_s, list) or len(data_s) < 2:
        raise ValueError("data_s must contain two spike-recorder event payloads.")

    exc_name = str(population_names[0])
    inh_name = str(population_names[1])

    def _events(idx):
        payload = data_s[idx]
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, (list, tuple)) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return first
        return {}

    exc_events = _events(0)
    inh_events = _events(1)

    times = _finalize_spikes(
        {
            exc_name: [np.asarray(exc_events.get("times", []), dtype=float)],
            inh_name: [np.asarray(inh_events.get("times", []), dtype=float)],
        }
    )
    gids = _finalize_spikes(
        {
            exc_name: [np.asarray(exc_events.get("senders", []), dtype=int)],
            inh_name: [np.asarray(inh_events.get("senders", []), dtype=int)],
        }
    )
    return times, gids


def _load_module(module_path):
    """Load a Python simulation-parameter file as a module."""
    script_dir = os.path.dirname(module_path)
    sys.path.append(script_dir)
    module_name = os.path.basename(module_path).replace(".py", "")
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    # Load run-time parameters from the simulation params file.
    module = _load_module(sys.argv[1])
    simulate_in_chunks, simulation_chunk_ms = _resolve_chunking_config(module)
    max_mean_population_spike_rate_hz = getattr(
        module, "max_mean_population_spike_rate_hz", 50.0
    )

    # Optionally seed NumPy and derive the NEST RNG seed from it.
    numpy_seed = getattr(module, "numpy_seed", None)
    rng_seed = None
    if numpy_seed is not None:
        numpy_seed = int(numpy_seed)
        if numpy_seed < 0:
            raise ValueError("numpy_seed must be a non-negative integer.")
        np.random.seed(numpy_seed)
        rng_seed = int(np.random.randint(1, 2**31 - 1))

    # Load the sampled or configured network parameters.
    output_dir = sys.argv[2]
    with open(os.path.join(output_dir, "network.pkl"), "rb") as f:
        lif_params = pickle.load(f)
    population_names = list(_resolve_population_names(lif_params))

    # Convert the stored payload into the constructor format used here.
    (
        network_params,
        neuron_params,
        simulation_params,
        external_input_params,
        analysis_params,
    ) = _default_network_payload(lif_params, module)

    # Create populations, inputs, recurrent synapses, and recorders.
    net = network(
        network_params,
        neuron_params,
        simulation_params,
        external_input_params,
        analysis_params,
    )
    nest_results_dir = os.path.join(output_dir, "nest_output")
    net.create_network(rng_seed=rng_seed, results_dir=nest_results_dir)

    # Store the planned simulation intervals for reproducibility metadata.
    total_time = float(module.tstop)
    if simulate_in_chunks:
        step = max(float(module.dt), float(simulation_chunk_ms))
        step = max(float(module.dt), round(step / float(module.dt)) * float(module.dt))
        interval_edges = np.arange(0.0, total_time, step)
        intervals = [min(step, total_time - start) for start in interval_edges]
    else:
        intervals = [total_time]

    # Run the simulation, optionally with progress updates.
    print("Simulating...\n", end=" ", flush=True)
    tic = time.time()
    if simulate_in_chunks:
        _simulate_with_progress(total_time, float(module.dt), simulation_chunk_ms)
    else:
        net.simulate(total_time)
    toc = time.time()
    print(f"The simulation took {toc - tic} seconds.\n", end=" ", flush=True)

    # Collect state recorder events after simulation.
    data_v = []
    for i in range(2):
        if len(net.To_be_measured) > 0:
            data_v.append(nest.GetStatus(net.subthreshold_recorders[i], keys="events"))
        else:
            data_v.append([])

    # Collect spike recorder events after simulation.
    data_s = []
    for i in range(2):
        data_s.append(nest.GetStatus(net.spikes[i], keys="events"))

    # Format spike times and gids by population.
    times, gids = _extract_population_spikes(data_s, population_names)

    # Preserve population ids needed by downstream analysis.
    population_ids = {
        "pop_ex": _node_ids(net.exc),
        "pop_in": _node_ids(net.inh),
        "pop_thalamo": _node_ids(net.thalamo),
        "pop_cc": _node_ids(net.cort),
    }

    # Aggregate excitatory state variables into the proxy input payload.
    E_ex = neuron_params[0]["E_ex"]
    E_in = neuron_params[0]["E_in"]
    exc_state_chunks = {key: [] for key in ("times", "AMPA", "GABA", "Vm", "counts")}
    if len(net.To_be_measured) > 0:
        exc_chunk = _aggregate_state_chunk(data_v[0][0], E_ex, E_in)
    else:
        exc_chunk = _aggregate_state_chunk({"times": []}, E_ex, E_in)
    _append_state_chunks(exc_state_chunks, exc_chunk)
    exc_state_payload = _finalize_state_chunks(exc_state_chunks, float(module.dt))

    # Abort before writing large spike files if firing rates are unexpectedly high.
    try:
        _check_spike_output_size(
            times,
            gids,
            lif_params,
            total_time,
            max_mean_population_spike_rate_hz,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr, flush=True)
        sys.exit(2)

    # Save simulation outputs for downstream proxy computation and diagnostics.
    with open(os.path.join(output_dir, "times.pkl"), "wb") as f:
        pickle.dump(times, f)
    with open(os.path.join(output_dir, "gids.pkl"), "wb") as f:
        pickle.dump(gids, f)
    with open(os.path.join(output_dir, "exc_state_events.pkl"), "wb") as f:
        pickle.dump(exc_state_payload, f)
    with open(os.path.join(output_dir, "recording_metadata.pkl"), "wb") as f:
        pickle.dump(
            {
                "To_be_measured": analysis_params["To_be_measured"],
                "intervals": intervals,
                "population_ids": population_ids,
                "spike_streams": list(population_names),
            },
            f,
        )
    with open(os.path.join(output_dir, "external_rate_profile.pkl"), "wb") as f:
        pickle.dump(net.external_rate_profile, f)
    with open(os.path.join(output_dir, "tstop.pkl"), "wb") as f:
        pickle.dump(total_time, f)
    with open(os.path.join(output_dir, "dt.pkl"), "wb") as f:
        pickle.dump(float(module.dt), f)
