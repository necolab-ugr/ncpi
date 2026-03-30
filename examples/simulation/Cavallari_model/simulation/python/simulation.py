import os
import pickle
import sys
import time
from importlib import util
from pathlib import Path
import nest
import numpy as np


MODULE_CANDIDATES = (
    ("cavallari_module",),
    ("build", "libcavallari_module.so"),
    ("build", "src", "cavallari_module.so"),
)


def _node_ids(nodes):
    return list(nodes.tolist())


def _update_dicts(dict1, dict2):
    merged = dict(dict1)
    merged.update(dict2)
    return merged


def _default_analysis_params():
    return {
        "To_be_measured": ["V_m", "g_ex", "g_in"],
        "cells_to_analyze": [i for i in range(10)],
    }


class network:
    """NEST 3.8 reproduction of the original pablomc88 `network_1.py` flow.

    External thalamic and cortical drives are implemented here as direct
    inhomogeneous Poisson inputs to excitatory and inhibitory populations,
    with generator rates scaled to replace the original parrot-neuron relay.
    """

    def __init__(
        self,
        Network_params,
        Neuron_params,
        Simulation_params,
        External_input_params,
        Analysis_params,
    ):
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
        self.cells_to_analyze = Analysis_params["cells_to_analyze"]

        self.external_rate_profile = {}
        self.selected_spike_recorders = {}

    def _ensure_custom_model_available(self):
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
        return [
            [
                np.random.uniform(-self.extent / 2.0, self.extent / 2.0),
                np.random.uniform(-self.extent / 2.0, self.extent / 2.0),
            ]
            for _ in range(int(n_nodes))
        ]

    def _create_nodes(self, model, n_nodes):
        positions = nest.spatial.free(
            pos=self._random_positions(n_nodes),
            extent=[self.extent, self.extent],
        )
        return nest.Create(model, int(n_nodes), positions=positions)

    def create_network(self, rng_seed=None):
        np.random.seed(int(time.time()))
        if rng_seed is None:
            rng_seed = int((time.time() * 100) % (2 ** 32))

        results_dir = Path(__file__).resolve().parents[1] / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        nest.ResetKernel()
        nest.SetKernelStatus(
            {
                "local_num_threads": int(self.num_threads),
                "resolution": float(self.simstep),
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

    def simulate_network(self, interval):
        if len(self.To_be_measured) > 0:
            self.subthreshold_recorders = []

            rec_exc = nest.Create("RecordingNode")
            nest.Connect(rec_exc, self.exc)
            self.subthreshold_recorders.append(rec_exc)

            rec_inh = nest.Create("RecordingNode")
            nest.Connect(rec_inh, self.inh)
            self.subthreshold_recorders.append(rec_inh)

        self.spikes = []
        self.mult_exc = nest.Create("spike_recorder", 1)
        self.mult_inh = nest.Create("spike_recorder", 1)

        sp_exc = nest.Create("SpikesRecorder")
        nest.Connect(self.exc, sp_exc)
        self.spikes.append(sp_exc)
        if self.cells_to_analyze:
            exc_ids = _node_ids(self.exc)
            selected_exc = [exc_ids[i] for i in self.cells_to_analyze if i < len(exc_ids)]
            if selected_exc:
                nest.Connect(nest.NodeCollection(selected_exc), self.mult_exc)

        sp_inh = nest.Create("SpikesRecorder")
        nest.Connect(self.inh, sp_inh)
        self.spikes.append(sp_inh)
        if self.cells_to_analyze:
            inh_ids = _node_ids(self.inh)
            selected_inh = [inh_ids[i] for i in self.cells_to_analyze if i < len(inh_ids)]
            if selected_inh:
                nest.Connect(nest.NodeCollection(selected_inh), self.mult_inh)

        print("\n--- Simulation ---\n")
        nest.SetKernelStatus({"print_time": True})
        nest.Simulate(float(interval))

        n_rec = 2
        data_v = []
        for i in range(n_rec):
            if len(self.To_be_measured) > 0:
                data_v.append(nest.GetStatus(self.subthreshold_recorders[i], keys="events"))
            else:
                data_v.append([])

        data_s = []
        for i in range(n_rec):
            data_s.append(nest.GetStatus(self.spikes[i], keys="events"))

        senders_v = []
        for i in range(n_rec):
            if len(self.To_be_measured) > 0:
                senders_v.append(np.asarray(data_v[i][0]["senders"]))
            else:
                senders_v.append([])

        senders_s = []
        for i in range(n_rec):
            senders_s.append(np.asarray(data_s[i][0]["senders"]))

        pop_ex = _node_ids(self.exc)
        pop_in = _node_ids(self.inh)
        pop_thalamo = _node_ids(self.thalamo)
        pop_cc = _node_ids(self.cort)

        return [
            self.simtime,
            data_v,
            data_s,
            senders_v,
            senders_s,
            pop_ex,
            pop_in,
            pop_thalamo,
            pop_cc,
            self.mult_exc,
            self.mult_inh,
        ]


def _resolve_chunking_config(module):
    simulate_in_chunks = bool(getattr(module, "simulate_in_chunks", True))
    chunk_ms = float(getattr(module, "simulation_chunk_ms", 1000.0))
    if chunk_ms <= 0:
        raise ValueError("simulation_chunk_ms must be positive.")
    return simulate_in_chunks, chunk_ms


def _default_network_payload(lif_params, module):
    network_params = lif_params["network_params"]
    neuron_params = [lif_params["neuron_params"]["E"], lif_params["neuron_params"]["I"]]
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


def _aggregate_exc_state_chunk(events, E_ex, E_in):
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
    if payload["times"].size == 0:
        return
    for key in ("times", "AMPA", "GABA", "Vm", "counts"):
        chunks[key].append(payload[key])


def _finalize_state_chunks(chunks, interval):
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


def _append_spikes(times_acc, gids_acc, events):
    event_map = {
        "E": events[0][0],
        "I": events[1][0],
    }
    for key, payload in event_map.items():
        times_acc[key].append(np.asarray(payload["times"], dtype=float))
        gids_acc[key].append(np.asarray(payload["senders"], dtype=int))


def _finalize_spikes(acc):
    finalized = {}
    for key, parts in acc.items():
        if not parts:
            finalized[key] = np.array([], dtype=float)
        else:
            finalized[key] = np.concatenate(parts)
    return finalized


def _load_module(module_path):
    script_dir = os.path.dirname(module_path)
    sys.path.append(script_dir)
    module_name = os.path.basename(module_path).replace(".py", "")
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    module = _load_module(sys.argv[1])
    simulate_in_chunks, simulation_chunk_ms = _resolve_chunking_config(module)

    numpy_seed = getattr(module, "numpy_seed", None)
    rng_seed = None
    if numpy_seed is not None:
        numpy_seed = int(numpy_seed)
        if numpy_seed < 0:
            raise ValueError("numpy_seed must be a non-negative integer.")
        np.random.seed(numpy_seed)
        rng_seed = int(np.random.randint(1, 2**31 - 1))

    output_dir = sys.argv[2]
    with open(os.path.join(output_dir, "network.pkl"), "rb") as f:
        lif_params = pickle.load(f)

    (
        network_params,
        neuron_params,
        simulation_params,
        external_input_params,
        analysis_params,
    ) = _default_network_payload(lif_params, module)

    net = network(
        network_params,
        neuron_params,
        simulation_params,
        external_input_params,
        analysis_params,
    )
    net.create_network(rng_seed=rng_seed)

    total_time = float(module.tstop)
    if simulate_in_chunks:
        interval_edges = np.arange(0.0, total_time, float(simulation_chunk_ms))
        intervals = [min(float(simulation_chunk_ms), total_time - start) for start in interval_edges]
    else:
        intervals = [total_time]

    exc_state_chunks = {key: [] for key in ("times", "AMPA", "GABA", "Vm", "counts")}
    times_acc = {key: [] for key in ("E", "I")}
    gids_acc = {key: [] for key in ("E", "I")}
    population_ids = {}

    E_ex = neuron_params[0]["E_ex"]
    E_in = neuron_params[0]["E_in"]
    print("Simulating...\n", end=" ", flush=True)
    tic = time.time()
    for segment_idx, interval in enumerate(intervals, start=1):
        result = net.simulate_network(interval)
        if not population_ids:
            population_ids = {
                "pop_ex": result[5],
                "pop_in": result[6],
                "pop_thalamo": result[7],
                "pop_cc": result[8],
            }
        _append_spikes(times_acc, gids_acc, result[2])
        exc_chunk = _aggregate_exc_state_chunk(result[1][0][0], E_ex, E_in)
        _append_state_chunks(exc_state_chunks, exc_chunk)
        print(f"SIM_SEGMENT {segment_idx}/{len(intervals)}", flush=True)

    toc = time.time()
    print(f"The simulation took {toc - tic} seconds.\n", end=" ", flush=True)

    times = _finalize_spikes(times_acc)
    gids = _finalize_spikes(gids_acc)
    exc_state_payload = _finalize_state_chunks(exc_state_chunks, float(module.dt))

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
                "cells_to_analyze": analysis_params["cells_to_analyze"],
                "intervals": intervals,
                "population_ids": population_ids,
                "spike_streams": ["E", "I"],
            },
            f,
        )
    with open(os.path.join(output_dir, "external_rate_profile.pkl"), "wb") as f:
        pickle.dump(net.external_rate_profile, f)
    with open(os.path.join(output_dir, "tstop.pkl"), "wb") as f:
        pickle.dump(total_time, f)
    with open(os.path.join(output_dir, "dt.pkl"), "wb") as f:
        pickle.dump(float(module.dt), f)
