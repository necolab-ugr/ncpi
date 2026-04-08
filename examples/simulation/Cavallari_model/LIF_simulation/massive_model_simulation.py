import argparse
import copy
import csv
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import util
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import pycatch22
from ncpi import FieldPotential, Features, tools

# --- Batch/run controls ---
# Total parameter sets.
TOTAL_SIMULATIONS = 10**6
# Timeout for one point-neuron simulation.
SIMULATION_TIMEOUT_SECONDS = 240
# Threads for the simulation subprocess.
LOCAL_NUM_THREADS = 56
# Abort before writing huge spike pickles.
MAX_MEAN_POPULATION_SPIKE_RATE_HZ = 50.0
# Use a fixed integer for reproducible sampling.
RANDOM_SEED = 0

# --- Debug plotting ---
PLOT_PARAMETER_SAMPLES = False
PLOT_EACH_SIMULATION = False

# --- Signal/transient ---
# Choose either "proxy" or "kernel".
SIGNAL_METHOD = "proxy"
# Drop initialization transient from analyses.
TRANSIENT_MS = 500.0

# --- Spectral validity gates ---
STRONG_PEAK_RELATIVE_THRESHOLD = 0.9
STRONG_PEAK_POWER_THRESHOLD = 0.4

# --- Pairwise spike-train correlation gates ---
SPIKE_TRAIN_SAMPLE_NEURONS = 1000
SPIKE_TRAIN_CORRELATION_BIN_MS = 10.0
# Minimum non-constant binned spike trains.
MIN_PAIRWISE_CORRELATION_USABLE_NEURONS = 50
PAIRWISE_CORRELATION_MEAN_ABS_MAX = 0.2
PAIRWISE_CORRELATION_ABS_P95_MAX = 0.5

# --- Interspike-interval gates ---
ISI_SAMPLE_NEURONS = 1000
# Minimum neurons with >=2 post-transient spikes.
MIN_ISI_USABLE_NEURONS = 50
# Median across-neuron mean ISI bounds, ms.
ISI_MEDIAN_MEAN_MS_LIMITS = (2.0, 5000.0)
# Median ISI coefficient-of-variation bounds.
ISI_MEDIAN_CV_LIMITS = (0.5, 3.0)

# Choose to either download files and precomputed outputs used in simulations of the
# reference multicompartment neuron network model (True) or load them from a local path (False)
zenodo_dw_mult = False

# Zenodo URL that contains the data (used if zenodo_dw_mult is True)
zenodo_URL_mult = "https://zenodo.org/api/records/15429373"

# Zenodo directory where the MC data is stored (must be an absolute path to correctly load morphologies in NEURON)
zenodo_dir = os.path.expandvars(os.path.expanduser(
    os.path.join("$HOME", "multicompartment_neuron_network")
))

# Directory with Cavallari-adapted MC output simulations
multi_output_path = os.path.join(zenodo_dir, "output_Cavallari")

# Biophysical mechanisms to include when computing kernels from the multicompartment model.
MULTI_BIOPHYS = ["set_Ih_linearized_hay2011", "make_cell_uniform"]

# Paths for the simulation and analysis
SCRIPT_DIR = os.path.dirname(__file__)
PARAMS_DIR = os.path.join(SCRIPT_DIR, "params")
PYTHON_DIR = os.path.join(SCRIPT_DIR, "python")
SIMULATION_SCRIPT = os.path.join(PYTHON_DIR, "simulation.py")
MC_ANALYSIS_PARAMS = os.path.join(os.path.dirname(SCRIPT_DIR), "MC_simulation", "analysis_params.py")
DEFAULT_BASE_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "simulation_output")
OUTPUT_ROOT = DEFAULT_BASE_OUTPUT_ROOT

# Parameter ranges are expressed as multipliers of the current Cavallari baseline.
CONDUCTANCE_SCALE_RANGE = (0.5, 2.0)
SYNAPTIC_TIME_SCALE_RANGE = (0.5, 2.0)
EXT_INPUT_SCALE_RANGE = (0.5, 4.0)

CATCH22_NAMES = pycatch22.catch22_all([0])["names"]
SIM_THETA_PARAMETERS = [
    "g_EE",
    "g_IE",
    "g_EI",
    "g_II",
    "tau_syn_AMPA_scale",
    "tau_syn_GABA_scale",
    "ext_input_scale",
]
SIM_PLOT_PARAMETERS = [
    "g_EE",
    "g_IE",
    "g_EI",
    "g_II",
    "E_tau_decay_AMPA",
    "E_tau_decay_GABA_A",
    "I_tau_decay_AMPA",
    "I_tau_decay_GABA_A",
    "ext_input_scale",
]
SIGNAL_DATA_METHODS = {"kernel": "CDM", "proxy": "proxy"}
SIM_DATA_METHODS = ("CDM", "proxy", "specparam", "catch22")


def _specparam_params(fs: float) -> Dict[str, Any]:
    """Build the specparam parameter dict."""
    try:
        fs = float(fs)
    except (TypeError, ValueError) as exc:
        raise TypeError("`fs` must be convertible to float.") from exc

    if fs <= 0:
        raise ValueError(f"`fs` must be positive, got {fs}.")

    specparam_setup_emp = {
        "peak_threshold": 1.0,
        "min_peak_height": 0.0,
        "max_n_peaks": 5,
        "peak_width_limits": (10.0, 50.0),
    }

    return {
        "fs": fs,
        "freq_range": (5.0, 200.0),
        "specparam_model": dict(specparam_setup_emp),
        "metric_thresholds": {"gof_rsquared": 0.9},
        "metric_policy": "reject",
    }


def load_module(module_path, module_name):
    """Load a Python module from a file path."""
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    """Parse command-line arguments for the batch runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Cavallari massive model simulation either as a standalone job "
            "or as one batch in a SLURM array."
        )
    )
    parser.add_argument(
        "--batch-id",
        type=int,
        default=None,
        help="Zero-based batch index. Defaults to SLURM_ARRAY_TASK_ID or 0.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Total number of batches. Defaults to SLURM_ARRAY_TASK_COUNT or 1.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Directory where batch outputs should be stored. "
            f"Defaults to {DEFAULT_BASE_OUTPUT_ROOT}."
        ),
    )
    args = parser.parse_args()

    if args.batch_id is None:
        slurm_batch_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        args.batch_id = int(slurm_batch_id) if slurm_batch_id is not None else 0

    if args.num_batches is None:
        slurm_num_batches = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        args.num_batches = int(slurm_num_batches) if slurm_num_batches is not None else 1

    if args.num_batches < 1:
        raise ValueError(f"`num_batches` must be >= 1, got {args.num_batches}.")
    if args.batch_id < 0:
        raise ValueError(f"`batch_id` must be >= 0, got {args.batch_id}.")
    if args.batch_id >= args.num_batches:
        raise ValueError(
            f"`batch_id` must be smaller than `num_batches`; got {args.batch_id} and {args.num_batches}."
        )
    if SIGNAL_METHOD not in SIGNAL_DATA_METHODS:
        raise ValueError(
            f"`SIGNAL_METHOD` must be one of {tuple(SIGNAL_DATA_METHODS.keys())}, got {SIGNAL_METHOD!r}."
        )

    return args


def resolve_base_output_root(cli_output_root=None):
    """Resolve the base output directory from CLI or defaults."""
    output_root = cli_output_root or DEFAULT_BASE_OUTPUT_ROOT
    return os.path.abspath(os.path.expanduser(os.path.expandvars(output_root)))


def configure_output_directories(batch_id, num_batches, base_output_root=None):
    """Set the global output directory for the current batch."""
    global OUTPUT_ROOT

    resolved_base_output_root = resolve_base_output_root(base_output_root)

    if num_batches == 1:
        OUTPUT_ROOT = resolved_base_output_root
    else:
        OUTPUT_ROOT = os.path.join(
            resolved_base_output_root, f"batch_{batch_id:04d}"
        )


def select_batch_parameter_sets(sampled_parameter_sets, batch_id, num_batches):
    """Split sampled parameters across batches and return this batch."""
    batch_indices = np.array_split(np.arange(len(sampled_parameter_sets), dtype=int), num_batches)
    return [
        (int(global_index), sampled_parameter_sets[int(global_index)])
        for global_index in batch_indices[batch_id]
    ]


def active_data_methods(signal_method):
    """Return the export methods used by the selected signal workflow."""
    return (SIGNAL_DATA_METHODS[signal_method], "specparam", "catch22")


def ensure_output_directories(signal_method):
    """Create the output folders used by this simulation run."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for method in active_data_methods(signal_method):
        os.makedirs(os.path.join(OUTPUT_ROOT, "data", method), exist_ok=True)


def download_multicompartment_data():
    """Download the multicompartment reference data when enabled."""
    if not zenodo_dw_mult:
        return

    print("\n--- Downloading data.")
    start_time = time.time()
    tools.download_zenodo_record(zenodo_URL_mult, download_dir=zenodo_dir)
    end_time = time.time()
    print(f"All files downloaded in {(end_time - start_time) / 60:.2f} minutes.")


def sample_parameters(rng, base_network_params, base_neuron_params):
    """Sample one parameter set from the configured Cavallari ranges."""
    while True:
        sampled = {}
        for name, key in (
            ("g_EE", "exc_exc_recurrent"),
            ("g_IE", "exc_inh_recurrent"),
            ("g_EI", "inh_exc_recurrent"),
            ("g_II", "inh_inh_recurrent"),
        ):
            scale = float(rng.uniform(*CONDUCTANCE_SCALE_RANGE))
            sampled[name] = float(base_network_params[key]) * scale

        sampled["tau_syn_AMPA_scale"] = float(rng.uniform(*SYNAPTIC_TIME_SCALE_RANGE))
        sampled["tau_syn_GABA_scale"] = float(rng.uniform(*SYNAPTIC_TIME_SCALE_RANGE))
        sampled["ext_input_scale"] = float(rng.uniform(*EXT_INPUT_SCALE_RANGE))

        gaba_slower_than_ampa = all(
            neuron_params["tau_decay_GABA_A"] * sampled["tau_syn_GABA_scale"]
            >= neuron_params["tau_decay_AMPA"] * sampled["tau_syn_AMPA_scale"]
            for neuron_params in base_neuron_params
        )
        inhibitory_abs_min = min(abs(sampled["g_EI"]), abs(sampled["g_II"]))
        excitatory_bounded = (
            2.0 * sampled["g_EE"] <= inhibitory_abs_min
            and 2.0 * sampled["g_IE"] <= inhibitory_abs_min
        )

        if (
            gaba_slower_than_ampa
            and excitatory_bounded
            and sampled["g_IE"] > sampled["g_EE"]
        ):
            return sampled


def build_lif_parameters(base_network_params, base_neuron_params, sampled_params):
    """Build Cavallari LIF network parameters for one sampled configuration."""
    network_params = copy.deepcopy(base_network_params)
    neuron_params = copy.deepcopy(base_neuron_params)

    network_params["exc_exc_recurrent"] = sampled_params["g_EE"]
    network_params["exc_inh_recurrent"] = sampled_params["g_IE"]
    network_params["inh_exc_recurrent"] = sampled_params["g_EI"]
    network_params["inh_inh_recurrent"] = sampled_params["g_II"]
    network_params["th_exc_external"] *= sampled_params["ext_input_scale"]
    network_params["th_inh_external"] *= sampled_params["ext_input_scale"]

    neuron_params[0]["tau_decay_AMPA"] *= sampled_params["tau_syn_AMPA_scale"]
    neuron_params[1]["tau_decay_AMPA"] *= sampled_params["tau_syn_AMPA_scale"]
    neuron_params[0]["tau_decay_GABA_A"] *= sampled_params["tau_syn_GABA_scale"]
    neuron_params[1]["tau_decay_GABA_A"] *= sampled_params["tau_syn_GABA_scale"]

    return {
        "X": ["E", "I"],
        "N_X": [network_params["N_exc"], network_params["N_inh"]],
        "model": "iaf_bw_2003",
        "neuron_params": {
            "E": neuron_params[0],
            "I": neuron_params[1],
        },
        "network_params": network_params,
    }


def build_plot_parameter_values(sampled_params, base_neuron_params):
    """Build derived parameter values to show in diagnostic plots."""
    values = {name: float(sampled_params[name]) for name in SIM_THETA_PARAMETERS}
    values["E_tau_decay_AMPA"] = (
        float(base_neuron_params[0]["tau_decay_AMPA"])
        * float(sampled_params["tau_syn_AMPA_scale"])
    )
    values["E_tau_decay_GABA_A"] = (
        float(base_neuron_params[0]["tau_decay_GABA_A"])
        * float(sampled_params["tau_syn_GABA_scale"])
    )
    values["I_tau_decay_AMPA"] = (
        float(base_neuron_params[1]["tau_decay_AMPA"])
        * float(sampled_params["tau_syn_AMPA_scale"])
    )
    values["I_tau_decay_GABA_A"] = (
        float(base_neuron_params[1]["tau_decay_GABA_A"])
        * float(sampled_params["tau_syn_GABA_scale"])
    )
    return values


def write_runtime_simulation_params(path, base_simulation_params):
    """Write runtime simulation parameters to a temporary Python file."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"tstop = {float(base_simulation_params.tstop)!r}\n")
        handle.write(f"local_num_threads = {int(LOCAL_NUM_THREADS)!r}\n")
        handle.write(f"dt = {float(base_simulation_params.dt)!r}\n")
        handle.write("simulate_in_chunks = False\n")
        handle.write(
            f"max_mean_population_spike_rate_hz = {float(MAX_MEAN_POPULATION_SPIKE_RATE_HZ)!r}\n"
        )


def run_simulation(lif_params, base_simulation_params, output_dir):
    """Run one simulation and report its runtime status."""
    with open(os.path.join(output_dir, "network.pkl"), "wb") as handle:
        pickle.dump(lif_params, handle)

    runtime_params_path = os.path.join(output_dir, "simulation_params_runtime.py")
    write_runtime_simulation_params(runtime_params_path, base_simulation_params)

    cmd = [sys.executable, SIMULATION_SCRIPT, runtime_params_path, output_dir]
    started = time.time()
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=SIMULATION_TIMEOUT_SECONDS,
            cwd=SCRIPT_DIR,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "runtime_seconds": time.time() - started}
    except subprocess.CalledProcessError as exc:
        status = "high_spike_rate" if exc.returncode == 2 else "failed"
        return {
            "status": status,
            "runtime_seconds": time.time() - started,
            "returncode": exc.returncode,
        }

    return {"status": "completed", "runtime_seconds": time.time() - started}


def load_simulation_outputs(output_dir, signal_method):
    """Load the core output files produced by a simulation."""
    outputs = {}
    filenames = ["times.pkl", "gids.pkl", "tstop.pkl", "dt.pkl", "network.pkl"]
    if signal_method == "proxy":
        filenames.append("exc_state_events.pkl")

    for filename in filenames:
        path = os.path.join(output_dir, filename)
        with open(path, "rb") as handle:
            outputs[filename[:-4]] = pickle.load(handle)
    return outputs


def apply_transient_filter(times, gids, transient):
    """Discard spikes that occur before the transient period ends."""
    filtered_times = {}
    filtered_gids = {}
    for population in times:
        keep = np.asarray(times[population]) >= transient
        filtered_times[population] = np.asarray(times[population])[keep]
        filtered_gids[population] = np.asarray(gids[population])[keep]
    return filtered_times, filtered_gids


def mean_firing_rate_hz(spike_times, population_size, transient, tstop):
    """Compute the mean firing rate in Hz after the transient."""
    duration_ms = float(tstop) - float(transient)
    if duration_ms <= 0:
        return np.nan
    return float(np.asarray(spike_times).size) * 1000.0 / (duration_ms * population_size)


def build_population_neuron_ids(populations, population_sizes):
    """Return inferred NEST neuron ids for the simulated populations."""
    neuron_ids = {}
    next_gid = 1
    for population in populations:
        size = int(population_sizes[population])
        neuron_ids[population] = np.arange(next_gid, next_gid + size, dtype=int)
        next_gid += size
    return neuron_ids


def _sample_neuron_ids(rng, candidate_ids, sample_size):
    """Sample neuron ids without replacement from a candidate pool."""
    candidate_ids = np.asarray(candidate_ids, dtype=int)
    if candidate_ids.size == 0:
        return candidate_ids
    sample_size = min(int(sample_size), candidate_ids.size)
    return np.asarray(rng.choice(candidate_ids, size=sample_size, replace=False), dtype=int)


def _spikes_for_sample(times, gids, populations, sampled_ids):
    """Return sampled spike rows and times for selected neuron ids."""
    sampled_ids = np.asarray(sampled_ids, dtype=int)
    if sampled_ids.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    max_gid = int(np.max(sampled_ids))
    lookup = np.full(max_gid + 1, -1, dtype=int)
    lookup[sampled_ids] = np.arange(sampled_ids.size, dtype=int)

    row_parts = []
    time_parts = []
    for population in populations:
        population_gids = np.asarray(gids[population], dtype=int)
        if population_gids.size == 0:
            continue
        population_times = np.asarray(times[population], dtype=float)
        in_bounds = (population_gids >= 0) & (population_gids <= max_gid)
        if not np.any(in_bounds):
            continue

        bounded_gids = population_gids[in_bounds]
        rows = lookup[bounded_gids]
        keep = rows >= 0
        if np.any(keep):
            row_parts.append(rows[keep])
            time_parts.append(population_times[in_bounds][keep])

    if not row_parts:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(row_parts), np.concatenate(time_parts)


def compute_pairwise_correlation_summary(
    times,
    gids,
    populations,
    sampled_ids,
    transient,
    tstop,
):
    """Compute binned spike-train pairwise correlations for sampled neurons."""
    sampled_ids = np.asarray(sampled_ids, dtype=int)
    summary = {
        "sampled_neurons": int(sampled_ids.size),
        "usable_neurons": 0,
        "pair_count": 0,
        "bin_ms": float(SPIKE_TRAIN_CORRELATION_BIN_MS),
        "mean": np.nan,
        "mean_abs": np.nan,
        "median": np.nan,
        "abs_p95": np.nan,
        "min": np.nan,
        "max": np.nan,
    }
    if sampled_ids.size < 2:
        return summary

    bin_edges = np.arange(
        float(transient),
        float(tstop) + float(SPIKE_TRAIN_CORRELATION_BIN_MS),
        float(SPIKE_TRAIN_CORRELATION_BIN_MS),
    )
    if bin_edges.size < 2:
        return summary

    n_bins = bin_edges.size - 1
    counts = np.zeros((sampled_ids.size, n_bins), dtype=np.float32)
    rows, spike_times = _spikes_for_sample(times, gids, populations, sampled_ids)
    if rows.size:
        columns = np.searchsorted(bin_edges, spike_times, side="right") - 1
        keep = (
            (spike_times >= float(transient))
            & (spike_times < float(tstop))
            & (columns >= 0)
            & (columns < n_bins)
        )
        if np.any(keep):
            np.add.at(counts, (rows[keep], columns[keep]), 1.0)

    active = np.var(counts, axis=1) > 0.0
    usable = counts[active]
    summary["usable_neurons"] = int(usable.shape[0])
    if usable.shape[0] < 2:
        return summary

    corr_matrix = np.corrcoef(usable)
    pair_indices = np.triu_indices_from(corr_matrix, k=1)
    correlations = np.asarray(corr_matrix[pair_indices], dtype=float)
    correlations = correlations[np.isfinite(correlations)]
    summary["pair_count"] = int(correlations.size)
    if correlations.size == 0:
        return summary

    summary.update(
        {
            "mean": float(np.mean(correlations)),
            "mean_abs": float(np.mean(np.abs(correlations))),
            "median": float(np.median(correlations)),
            "abs_p95": float(np.percentile(np.abs(correlations), 95.0)),
            "min": float(np.min(correlations)),
            "max": float(np.max(correlations)),
        }
    )
    return summary


def compute_isi_summary(times, gids, populations, sampled_ids, transient, tstop):
    """Compute per-neuron interspike-interval summaries for sampled neurons."""
    sampled_ids = np.asarray(sampled_ids, dtype=int)
    summary = {
        "sampled_neurons": int(sampled_ids.size),
        "usable_neurons": 0,
        "cv_usable_neurons": 0,
        "total_intervals": 0,
        "median_mean_ms": np.nan,
        "p05_mean_ms": np.nan,
        "p95_mean_ms": np.nan,
        "median_cv": np.nan,
        "mean_cv": np.nan,
    }
    if sampled_ids.size == 0:
        return summary

    rows, spike_times = _spikes_for_sample(times, gids, populations, sampled_ids)
    if rows.size == 0:
        return summary

    keep = (
        np.isfinite(spike_times)
        & (spike_times >= float(transient))
        & (spike_times < float(tstop))
    )
    rows = rows[keep]
    spike_times = spike_times[keep]
    if rows.size == 0:
        return summary

    order = np.lexsort((spike_times, rows))
    rows = rows[order]
    spike_times = spike_times[order]
    boundaries = np.flatnonzero(np.diff(rows)) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, rows.size]

    mean_isis = []
    cvs = []
    total_intervals = 0
    for start, stop in zip(starts, stops):
        if stop - start < 2:
            continue
        intervals = np.diff(spike_times[start:stop])
        intervals = intervals[np.isfinite(intervals) & (intervals > 0.0)]
        if intervals.size == 0:
            continue
        total_intervals += int(intervals.size)
        mean_isi = float(np.mean(intervals))
        mean_isis.append(mean_isi)
        if intervals.size > 1 and mean_isi > 0.0:
            cvs.append(float(np.std(intervals, ddof=1) / mean_isi))

    if not mean_isis:
        return summary

    mean_isis = np.asarray(mean_isis, dtype=float)
    summary.update(
        {
            "usable_neurons": int(mean_isis.size),
            "cv_usable_neurons": int(len(cvs)),
            "total_intervals": int(total_intervals),
            "median_mean_ms": float(np.median(mean_isis)),
            "p05_mean_ms": float(np.percentile(mean_isis, 5.0)),
            "p95_mean_ms": float(np.percentile(mean_isis, 95.0)),
        }
    )
    if cvs:
        cvs = np.asarray(cvs, dtype=float)
        summary["median_cv"] = float(np.median(cvs))
        summary["mean_cv"] = float(np.mean(cvs))
    return summary


def compute_spike_train_qc(times, gids, populations, population_neuron_ids, transient, tstop, rng):
    """Compute random-neuron spike-train correlation and ISI QC summaries."""
    all_neuron_ids = np.concatenate(
        [np.asarray(population_neuron_ids[population], dtype=int) for population in populations]
    )
    correlation_ids = _sample_neuron_ids(
        rng,
        all_neuron_ids,
        SPIKE_TRAIN_SAMPLE_NEURONS,
    )
    remaining_ids = np.setdiff1d(all_neuron_ids, correlation_ids, assume_unique=False)
    isi_pool = remaining_ids if remaining_ids.size > 0 else all_neuron_ids
    isi_ids = _sample_neuron_ids(rng, isi_pool, ISI_SAMPLE_NEURONS)

    return {
        "pairwise_correlation": compute_pairwise_correlation_summary(
            times,
            gids,
            populations,
            correlation_ids,
            transient,
            tstop,
        ),
        "isi": compute_isi_summary(
            times,
            gids,
            populations,
            isi_ids,
            transient,
            tstop,
        ),
    }


def format_elapsed_time(total_seconds):
    """Convert elapsed seconds to hours and days."""
    total_seconds = float(total_seconds)
    return total_seconds / 3600.0, total_seconds / 86400.0


def get_spike_rate(times, transient, dt, tstop):
    """Bin spike times into a simple rate histogram."""
    bins = np.arange(transient, tstop + dt, dt)
    hist, _ = np.histogram(times, bins=bins)
    return bins[:-1], hist.astype(float)


class KernelParamsAdapter:
    """Adapter matching the Cavallari example's kernel-parameter access pattern."""

    def __init__(self, kernel_params):
        self._kernel_params = kernel_params

    def __getattr__(self, name):
        return getattr(self._kernel_params, name)

    @staticmethod
    def _current_post_index():
        try:
            frame = sys._getframe().f_back
        except ValueError:
            frame = None
        while frame is not None:
            if frame.f_code.co_name == "create_kernel" and "j" in frame.f_locals:
                return frame.f_locals["j"]
            frame = frame.f_back
        return None

    @property
    def extSynapseParameters(self):
        ext_params = self._kernel_params.extSynapseParameters
        post_idx = self._current_post_index()
        if post_idx is not None and isinstance(ext_params, (list, tuple)):
            return ext_params[post_idx]
        return ext_params

    @property
    def netstim_interval(self):
        interval = self._kernel_params.netstim_interval
        post_idx = self._current_post_index()
        if post_idx is not None and isinstance(interval, (list, tuple, np.ndarray)):
            return float(np.asarray(interval, dtype=float)[post_idx])
        if isinstance(interval, (list, tuple)):
            return np.asarray(interval, dtype=float)
        return interval


def load_kernel_params():
    """Load Cavallari-adapted MC analysis parameters."""
    module = load_module(MC_ANALYSIS_PARAMS, "cavallari_mc_analysis_params_massive")
    return KernelParamsAdapter(module.KernelParams)


def compute_kernel(potential, kernel_params, dt, tstop):
    """Compute the field-potential kernel used for CDM reconstruction."""
    print("Computing the kernel...")
    return potential.create_kernel(
        zenodo_dir,
        kernel_params,
        MULTI_BIOPHYS,
        dt,
        tstop,
        output_sim_path=multi_output_path,
        electrodeParameters=None,
        CDM=True,
    )


def is_cdm_components(signal_data):
    return isinstance(signal_data, dict) and all(key in signal_data for key in ("EE", "EI", "IE", "II"))


def wrap_cdm_components(cdm_components, transient, dt, populations, decimation_factor=10):
    component_order = [f"{pre}{post}" for pre in populations for post in populations]
    signal = np.sum(
        np.vstack([np.asarray(cdm_components[key], dtype=float) for key in component_order]),
        axis=0,
    )
    return {
        "signal": signal,
        "times": float(transient) + np.arange(signal.size) * float(dt) * float(decimation_factor),
        "components": cdm_components,
        "decimation_factor": decimation_factor,
        "source": "kernel",
    }


def compute_cdm(potential, kernels, spike_times, dt, tstop, transient, populations):
    """Compute CDM components and their summed signal."""
    cdm_signals = potential.compute_cdm_lfp_from_kernels(
        kernels,
        spike_times=spike_times,
        dt=dt,
        tstop=tstop,
        transient=transient,
        probe="KernelApproxCurrentDipoleMoment",
        component=2,
        mode="same",
        scale=dt / 1000.0,
    )

    cdm_components = {}
    for pre in populations:
        for post in populations:
            key = f"{pre}{post}"
            source_key = f"{post}:{pre}"
            cdm_components[key] = np.asarray(
                ss.decimate(cdm_signals[source_key], q=10, zero_phase=True)
            )

    return wrap_cdm_components(cdm_components, transient, dt, populations)


def get_proxy_signal(proxy_data):
    if isinstance(proxy_data, dict) and "signal" in proxy_data:
        return np.asarray(proxy_data["signal"], dtype=float)
    if isinstance(proxy_data, dict) and "components" in proxy_data:
        return get_proxy_signal(proxy_data["components"])
    if is_cdm_components(proxy_data):
        return np.sum(
            np.vstack([np.asarray(proxy_data[key], dtype=float) for key in ("EE", "EI", "IE", "II")]),
            axis=0,
        )
    return np.asarray(proxy_data, dtype=float)


def get_feature_signal(signal_method, signal_data):
    if signal_method == "kernel" and isinstance(signal_data, dict) and "components" in signal_data:
        return get_proxy_signal(signal_data["components"])
    return get_proxy_signal(signal_data)


def get_proxy_times(proxy_data):
    if isinstance(proxy_data, dict) and "times" in proxy_data:
        return np.asarray(proxy_data["times"], dtype=float)
    raise KeyError("Signal payload does not contain an explicit time axis.")


def trim_signal_to_transient(signal_data, transient):
    """Discard signal samples before the configured transient time."""
    if not isinstance(signal_data, dict) or "signal" not in signal_data or "times" not in signal_data:
        return signal_data
    signal = get_proxy_signal(signal_data)
    times = get_proxy_times(signal_data)
    keep = times >= float(transient)
    trimmed = dict(signal_data)
    trimmed["signal"] = signal[keep]
    trimmed["times"] = times[keep]
    return trimmed


def _extract_exc_currents(exc_state_events):
    """Extract the population-summed excitatory AMPA/GABA currents."""
    times = np.asarray(exc_state_events["times"], dtype=float)
    if {"AMPA", "GABA", "Vm"}.issubset(exc_state_events):
        return (
            times,
            np.asarray(exc_state_events["AMPA"], dtype=float),
            np.asarray(exc_state_events["GABA"], dtype=float),
            np.asarray(exc_state_events["Vm"], dtype=float),
        )
    raise KeyError("Expected aggregated excitatory-state payload with AMPA/GABA/Vm entries.")


def compute_proxy(exc_state_events, sim_step):
    """Compute the Cavallari LFP proxy exactly as in example_model_simulation."""
    times, ampa_current, gaba_current, _ = _extract_exc_currents(exc_state_events)
    if times.size == 0:
        raise ValueError("No excitatory-state samples are available for proxy computation.")

    signal = np.abs(ampa_current) + np.abs(gaba_current)
    start_time_pos = int(TRANSIENT_MS / float(sim_step))
    if start_time_pos >= signal.size:
        raise ValueError("Proxy normalization window starts after the end of the signal.")

    norm_window = signal[start_time_pos:]
    norm_std = np.std(norm_window)
    if np.isclose(norm_std, 0.0):
        raise ValueError("Proxy normalization failed because the post-500 ms signal is constant.")

    signal = (signal - np.mean(norm_window)) / norm_std
    return {
        "signal": signal,
        "times": times,
        "lfp_index": 5,
    }


def decimate_proxy(proxy_data, decimation_factor=10):
    """Decimate the proxy as in the Cavallari example."""
    signal = get_proxy_signal(proxy_data)
    times = get_proxy_times(proxy_data)
    if signal.size == 0:
        raise ValueError("Cannot decimate an empty proxy signal.")
    if decimation_factor <= 1:
        return {
            "signal": signal,
            "times": times,
            "lfp_index": proxy_data.get("lfp_index", 5),
            "decimation_factor": decimation_factor,
        }

    decimated_signal = ss.decimate(signal, decimation_factor)
    decimated_times = times[::decimation_factor][: decimated_signal.size]
    return {
        "signal": decimated_signal,
        "times": decimated_times,
        "lfp_index": proxy_data.get("lfp_index", 5),
        "decimation_factor": decimation_factor,
    }


def compute_signal_data(signal_method, potential, kernel_cache, outputs, filtered_times, transient, populations):
    """Compute the selected signal workflow for one simulation."""
    if signal_method == "kernel":
        signal_data = compute_cdm(
            potential,
            kernel_cache,
            filtered_times,
            outputs["dt"],
            outputs["tstop"],
            transient,
            populations,
        )
        signal = get_feature_signal(signal_method, signal_data)
        times = get_proxy_times(signal_data)
        signal_dt = float(outputs["dt"]) * float(signal_data.get("decimation_factor", 1))
        return signal_data, signal, times, signal_dt

    proxy_data = trim_signal_to_transient(
        decimate_proxy(compute_proxy(outputs["exc_state_events"], outputs["dt"])),
        transient,
    )
    signal = get_feature_signal(signal_method, proxy_data)
    times = get_proxy_times(proxy_data)
    signal_dt = float(outputs["dt"]) * float(proxy_data.get("decimation_factor", 1))
    return proxy_data, signal, times, signal_dt


def compute_catch22_features(feature_extractor, signal):
    """Compute normalized catch22 features from a signal."""
    x = np.asarray(signal, dtype=float).squeeze()
    sigma = float(np.std(x))
    if not np.isfinite(sigma) or sigma == 0.0:
        values = np.full(len(CATCH22_NAMES), np.nan)
    else:
        values = np.asarray(feature_extractor.catch22((x - np.mean(x)) / sigma), dtype=float)
    return {name: float(values[idx]) for idx, name in enumerate(CATCH22_NAMES)}


def _normalize_peak_array(peaks):
    """Normalize peak output to a consistent `(n, 3)` array."""
    peaks = np.asarray(peaks, dtype=float)
    if peaks.ndim == 1 and peaks.size == 3:
        return peaks.reshape(1, 3)
    if peaks.size == 0:
        return np.empty((0, 3), dtype=float)
    return peaks


def _sort_peaks_by_frequency(peaks):
    """Sort peaks by center frequency to keep debug output ordering stable."""
    peaks = _normalize_peak_array(peaks)
    if peaks.shape[0] <= 1:
        return peaks
    order = np.argsort(peaks[:, 0], kind="mergesort")
    return peaks[order]


def _select_peak_from_all(peaks, select_peak, freq_range):
    """Select one peak from the full peak list using the configured rule."""
    selected_peaks = np.empty((0, 3), dtype=float)
    if peaks.shape[0] == 0 or select_peak == "all":
        return selected_peaks

    cfs = peaks[:, 0]
    pws = peaks[:, 1]

    if select_peak == "max_pw":
        if not np.all(np.isnan(pws)):
            idx = int(np.nanargmax(pws))
            selected_peaks = peaks[idx:idx + 1]
    elif select_peak == "max_cf_in_range":
        fmin, fmax = map(float, freq_range)
        mask = (cfs >= fmin) & (cfs <= fmax)
        if np.any(mask):
            idx = int(np.nanargmax(cfs[mask]))
            selected_peaks = peaks[mask][idx:idx + 1]
    else:
        raise ValueError("select_peak must be one of {'all', 'max_pw', 'max_cf_in_range'}")

    return selected_peaks


def _extract_model_peaks(fit_model):
    """Extract the peak table from a fitted specparam model."""
    if fit_model is None:
        return np.empty((0, 3), dtype=float)
    try:
        return _sort_peaks_by_frequency(fit_model.results.params.periodic.params)
    except Exception:
        return np.empty((0, 3), dtype=float)


def _plot_specparam_debug_spectrum(ax, freqs, power_spectrum, fit_model, freq_range):
    """Plot the PSD and fitted model using specparam's native plot helper."""
    freqs = np.asarray(freqs, dtype=float)
    power_spectrum = np.asarray(power_spectrum, dtype=float)
    mask = (
        np.isfinite(freqs)
        & np.isfinite(power_spectrum)
        & (freqs >= float(freq_range[0]))
        & (freqs <= float(freq_range[1]))
    )
    if fit_model is None:
        if np.any(mask):
            ax.plot(freqs[mask], power_spectrum[mask], color="k", linewidth=1.2)
        ax.set_xlim(freq_range)
        return

    try:
        fit_model.plot(
            plot_peaks="shade",
            ax=ax,
            freqs=freqs[mask] if np.any(mask) else None,
            power_spectrum=power_spectrum[mask] if np.any(mask) else None,
            freq_range=freq_range,
            plt_log=False,
            add_legend=True,
            peak_kwargs={"color": "green"},
        )
    except TypeError:
        plt.sca(ax)
        fit_model.plot(
            plot_peaks="shade",
            freqs=freqs[mask] if np.any(mask) else None,
            power_spectrum=power_spectrum[mask] if np.any(mask) else None,
            freq_range=freq_range,
            plt_log=False,
            add_legend=True,
            peak_kwargs={"color": "green"},
        )


def compute_specparam_features(feature_extractor, signal):
    """Compute specparam summary features and peak counts for a signal."""
    select_peak = feature_extractor.params.get("select_peak", "max_pw")
    freq_range = feature_extractor.params.get("freq_range", (5.0, 200.0))
    try:
        result = feature_extractor.specparam(
            sample=np.asarray(signal, dtype=float),
            select_peak="all",
        )
    except Exception:
        result = {
            "aperiodic_params": np.array([np.nan, np.nan]),
            "peak_cf": np.nan,
            "peak_pw": np.nan,
            "peak_bw": np.nan,
            "n_peaks": 0,
            "selected_peaks": np.empty((0, 3)),
            "all_peaks": np.empty((0, 3)),
            "metrics": {"gof_rsquared": np.nan},
            "valid": False,
        }
    aperiodic = np.asarray(result.get("aperiodic_params", np.array([np.nan, np.nan])), dtype=float)
    slope = float(aperiodic[1]) if aperiodic.size > 1 else np.nan
    peaks = _sort_peaks_by_frequency(result.get("all_peaks", np.empty((0, 3))))
    selected_peaks = _select_peak_from_all(peaks, select_peak, freq_range)
    if selected_peaks.shape[0] == 1:
        peak_frequency, peak_power, peak_bandwidth = map(float, selected_peaks[0])
    else:
        peak_frequency = np.nan
        peak_power = np.nan
        peak_bandwidth = np.nan
    strongest_peak_power = np.nan
    if peaks.size:
        peak_powers = peaks[:, 1]
        strongest_peak_power = np.nanmax(peak_powers)
        if np.isfinite(strongest_peak_power):
            strong_peak_count = int(
                np.sum(peak_powers >= STRONG_PEAK_RELATIVE_THRESHOLD * strongest_peak_power)
            )
        else:
            strong_peak_count = 0
    else:
        strong_peak_count = 0
    return {
        "slope": slope,
        "n_peaks": int(result.get("n_peaks", peaks.shape[0])),
        "peak_frequency": peak_frequency,
        "peak_power": peak_power,
        "peak_bandwidth": peak_bandwidth,
        "all_peaks": peaks,
        "strong_peak_count": strong_peak_count,
        "strongest_peak_power": float(strongest_peak_power),
        "fit_valid": bool(result.get("valid", True)),
        "gof_rsquared": float(result.get("metrics", {}).get("gof_rsquared", np.nan)),
    }


def fit_specparam_for_plot(signal, fs):
    """Compute a Welch PSD and fit specparam for debug plotting."""
    from specparam import SpectralModel

    x = np.asarray(signal, dtype=float).squeeze()
    nperseg = int(0.5 * float(fs))
    nperseg = max(1, min(nperseg, x.shape[0]))
    freqs, power_spectrum = ss.welch(x, fs=float(fs), nperseg=nperseg)
    specparam_params = _specparam_params(fs)
    freq_range = specparam_params["freq_range"]

    mask = (
        np.isfinite(freqs)
        & np.isfinite(power_spectrum)
        & (freqs > 0)
        & (power_spectrum > 0)
    )
    freqs = freqs[mask]
    power_spectrum = power_spectrum[mask]
    if freqs.size < 5:
        return freqs, power_spectrum, None, freq_range

    fmin = max(float(freq_range[0]), float(np.min(freqs)))
    fmax = min(float(freq_range[1]), float(np.max(freqs)))
    if fmin >= fmax:
        return freqs, power_spectrum, None, freq_range

    fm = SpectralModel(**dict(specparam_params["specparam_model"]))
    fm.fit(freqs, power_spectrum, [fmin, fmax])
    return freqs, power_spectrum, fm, freq_range


def format_spike_train_qc_lines(spike_train_qc):
    """Build compact debug-text lines for spike-train QC metrics."""
    if spike_train_qc is None:
        return ["Spike-train QC: not computed"]

    pairwise = spike_train_qc["pairwise_correlation"]
    isi = spike_train_qc["isi"]
    return [
        (
            "Pairwise corr: "
            f"usable={pairwise['usable_neurons']}/{pairwise['sampled_neurons']}, "
            f"mean_abs={pairwise['mean_abs']:.4f}, abs_p95={pairwise['abs_p95']:.4f}"
        ),
        (
            "ISI: "
            f"usable={isi['usable_neurons']}/{isi['sampled_neurons']}, "
            f"median_mean={isi['median_mean_ms']:.4f} ms, "
            f"median_cv={isi['median_cv']:.4f}"
        ),
    ]


def plot_simulation_debug(
    simulation_index,
    plot_params,
    times,
    gids,
    populations,
    signal_total,
    signal_time,
    signal_dt,
    signal_label,
    transient,
    tstop,
    firing_rate_e,
    firing_rate_i,
    specparam_result,
    spike_train_qc,
    is_valid,
    invalid_reasons=None,
):
    """Plot debug views for one simulation and annotate extracted features."""
    if not PLOT_EACH_SIMULATION:
        return

    fig = plt.figure(figsize=(10, 12), dpi=150, constrained_layout=True)
    grid = fig.add_gridspec(4, 2)
    raster_ax = fig.add_subplot(grid[0, :])
    rate_ax = fig.add_subplot(grid[1, :])
    signal_ax = fig.add_subplot(grid[2, 0])
    psd_ax = fig.add_subplot(grid[2, 1])
    spec_left_ax = fig.add_subplot(grid[3, 0])
    spec_right_ax = fig.add_subplot(grid[3, 1])

    raster_colors = {"E": "C0", "I": "C1"}
    time_window = [4000.0, 4200.0]

    for population in populations:
        t = np.asarray(times[population])
        g = np.asarray(gids[population])
        mask = (t >= time_window[0]) & (t <= time_window[1])
        raster_ax.plot(
            t[mask],
            g[mask],
            ".",
            ms=1.2,
            color=raster_colors.get(population, "0.3"),
            label=population,
        )

    title = f"Simulation {simulation_index} | valid={is_valid}"
    if not is_valid and invalid_reasons:
        title += f" | reason={', '.join(invalid_reasons)}"
    raster_ax.set_title(title)
    raster_ax.set_ylabel("Neuron ID")
    raster_ax.set_xlim(time_window)
    raster_ax.legend(loc="upper right")

    for population in populations:
        bins, spike_rate = get_spike_rate(times[population], transient, signal_dt / 10.0, tstop)
        mask = (bins >= time_window[0]) & (bins <= time_window[1])
        rate_ax.plot(
            bins[mask],
            spike_rate[mask],
            color=raster_colors.get(population, "0.3"),
            label=f"{population} rate",
        )
    rate_ax.set_ylabel("Spikes / dt")
    rate_ax.set_xlim(time_window)
    rate_ax.legend(loc="upper right")

    signal_mask = (signal_time >= time_window[0]) & (signal_time <= time_window[1])
    signal_ax.plot(signal_time[signal_mask], signal_total[signal_mask], color="k", linewidth=0.9)
    signal_ax.set_ylabel(signal_label)
    signal_ax.set_xlabel("Time (ms)")
    signal_ax.set_xlim(time_window)

    fs = 1000.0 / signal_dt
    try:
        freqs, power, fit_model, freq_range = fit_specparam_for_plot(signal_total, fs)
    except Exception:
        freqs, power = ss.welch(signal_total, fs=fs)
        fit_model = None
        freq_range = _specparam_params(fs)["freq_range"]
    _plot_specparam_debug_spectrum(psd_ax, freqs, power, fit_model, freq_range)
    psd_ax.set_ylabel("Power")
    psd_ax.set_xlabel("Frequency (Hz)")
    psd_ax.set_title(f"{signal_label} spectrum + specparam fit")

    spec_left_ax.axis("off")
    spec_right_ax.axis("off")
    all_peaks = _extract_model_peaks(fit_model)
    if all_peaks.shape[0] == 0:
        all_peaks = _normalize_peak_array(specparam_result.get("all_peaks", np.empty((0, 3))))

    peak_lines = [f"Detected peaks in plotted fit: {all_peaks.shape[0]}"]
    if all_peaks.size:
        for peak_index, (peak_cf, peak_pw, peak_bw) in enumerate(all_peaks, start=1):
            peak_lines.append(
                f"Peak {peak_index}: cf={peak_cf:.4f} Hz, pw={peak_pw:.4f}, bw={peak_bw:.4f} Hz"
            )
    else:
        peak_lines.append("Peak list: none")

    parameter_lines = [f"{name}: {float(plot_params[name]):.4f}" for name in SIM_PLOT_PARAMETERS]

    summary_text = "\n".join(
        [
            f"Slope: {specparam_result['slope']:.4f}",
            f"Selected peak frequency: {specparam_result['peak_frequency']:.4f} Hz",
            f"Selected peak power: {specparam_result['peak_power']:.4f}",
            f"Selected peak bandwidth: {specparam_result['peak_bandwidth']:.4f} Hz",
            f"Mean firing rate E: {firing_rate_e:.4f} Hz",
            f"Mean firing rate I: {firing_rate_i:.4f} Hz",
            f"Peaks >= {int(100 * STRONG_PEAK_RELATIVE_THRESHOLD)}% of strongest: {specparam_result['strong_peak_count']}",
            f"Specparam fit valid: {specparam_result['fit_valid']}",
            *format_spike_train_qc_lines(spike_train_qc),
            *peak_lines,
        ]
    )
    parameter_text = "\n".join(
        [
            "Sampled parameters:",
            *parameter_lines,
        ]
    )
    spec_left_ax.text(0.02, 0.95, summary_text, va="top", ha="left", fontsize=11)
    spec_right_ax.text(0.02, 0.95, parameter_text, va="top", ha="left", fontsize=11)

    plt.show()
    plt.close(fig)


def evaluate_validity(firing_rate_e, firing_rate_i, specparam_features, spike_train_qc=None):
    """Evaluate whether a simulation passes the acceptance rules."""
    reasons = []
    if not np.isfinite(firing_rate_e) or not np.isfinite(firing_rate_i):
        reasons.append("non_finite_firing_rate")
    if not np.isfinite(specparam_features["slope"]):
        reasons.append("non_finite_slope")
    elif specparam_features["slope"] < 0.0:
        reasons.append("negative_slope")
    if firing_rate_i > 40.0:
        reasons.append("I_rate_above_40")
    if firing_rate_e > 20.0:
        reasons.append("E_rate_above_20")
    if firing_rate_i < 1.5 * firing_rate_e:
        reasons.append("I_rate_less_than_1p5x_E")
    if firing_rate_i > 10.0 * firing_rate_e:
        reasons.append("I_rate_more_than_10x_E")
    if (
        specparam_features["strong_peak_count"] > 2
        and specparam_features["strongest_peak_power"] > STRONG_PEAK_POWER_THRESHOLD
    ):
        reasons.append("more_than_two_strong_peaks")
    if not specparam_features["fit_valid"]:
        reasons.append("specparam_fit_rejected")
    if spike_train_qc is None:
        reasons.append("missing_spike_train_qc")
    else:
        pairwise = spike_train_qc["pairwise_correlation"]
        if pairwise["usable_neurons"] < MIN_PAIRWISE_CORRELATION_USABLE_NEURONS:
            reasons.append("pairwise_correlation_too_few_neurons")
        elif not np.isfinite(pairwise["mean_abs"]) or not np.isfinite(pairwise["abs_p95"]):
            reasons.append("non_finite_pairwise_correlation")
        else:
            if pairwise["mean_abs"] > PAIRWISE_CORRELATION_MEAN_ABS_MAX:
                reasons.append("pairwise_correlation_mean_abs_above_0p2")
            if pairwise["abs_p95"] > PAIRWISE_CORRELATION_ABS_P95_MAX:
                reasons.append("pairwise_correlation_abs_p95_above_0p5")

        isi = spike_train_qc["isi"]
        if isi["usable_neurons"] < MIN_ISI_USABLE_NEURONS:
            reasons.append("isi_too_few_neurons")
        elif not np.isfinite(isi["median_mean_ms"]):
            reasons.append("non_finite_isi")
        else:
            isi_low, isi_high = ISI_MEDIAN_MEAN_MS_LIMITS
            if isi["median_mean_ms"] < isi_low or isi["median_mean_ms"] > isi_high:
                reasons.append("isi_median_mean_out_of_range")
            cv_low, cv_high = ISI_MEDIAN_CV_LIMITS
            if np.isfinite(isi["median_cv"]) and (
                isi["median_cv"] < cv_low or isi["median_cv"] > cv_high
            ):
                reasons.append("isi_median_cv_out_of_range")
    return len(reasons) == 0, reasons


def flatten_summary_row(
    sample_index,
    simulation_index,
    batch_id,
    sampled_params,
    runtime_info,
    signal_method,
    firing_rate_e=None,
    firing_rate_i=None,
    catch22_features=None,
    specparam_features=None,
    spike_train_qc=None,
    valid=None,
    valid_reasons=None,
):
    """Flatten simulation results into a CSV-friendly row."""
    row = {
        "sample_index": sample_index,
        "simulation_index": simulation_index,
        "batch_id": batch_id,
        "signal_method": signal_method,
        "status": runtime_info["status"],
        "runtime_seconds": runtime_info.get("runtime_seconds", np.nan),
        "valid": valid,
        "valid_reasons": ";".join(valid_reasons or []),
        "firing_rate_E_hz": firing_rate_e,
        "firing_rate_I_hz": firing_rate_i,
    }
    row.update(sampled_params)
    if catch22_features is not None:
        row.update({f"catch22_{name}": value for name, value in catch22_features.items()})
    if specparam_features is not None:
        row.update(
            {
                "specparam_slope": specparam_features["slope"],
                "specparam_peak_frequency": specparam_features["peak_frequency"],
                "specparam_peak_power": specparam_features["peak_power"],
                "specparam_strong_peak_count": specparam_features["strong_peak_count"],
                "specparam_fit_valid": specparam_features["fit_valid"],
                "specparam_gof_rsquared": specparam_features["gof_rsquared"],
            }
        )
    if spike_train_qc is not None:
        pairwise = spike_train_qc["pairwise_correlation"]
        isi = spike_train_qc["isi"]
        row.update(
            {
                "pairwise_corr_sampled_neurons": pairwise["sampled_neurons"],
                "pairwise_corr_usable_neurons": pairwise["usable_neurons"],
                "pairwise_corr_pair_count": pairwise["pair_count"],
                "pairwise_corr_bin_ms": pairwise["bin_ms"],
                "pairwise_corr_mean": pairwise["mean"],
                "pairwise_corr_mean_abs": pairwise["mean_abs"],
                "pairwise_corr_median": pairwise["median"],
                "pairwise_corr_abs_p95": pairwise["abs_p95"],
                "pairwise_corr_min": pairwise["min"],
                "pairwise_corr_max": pairwise["max"],
                "isi_sampled_neurons": isi["sampled_neurons"],
                "isi_usable_neurons": isi["usable_neurons"],
                "isi_cv_usable_neurons": isi["cv_usable_neurons"],
                "isi_total_intervals": isi["total_intervals"],
                "isi_median_mean_ms": isi["median_mean_ms"],
                "isi_p05_mean_ms": isi["p05_mean_ms"],
                "isi_p95_mean_ms": isi["p95_mean_ms"],
                "isi_median_cv": isi["median_cv"],
                "isi_mean_cv": isi["mean_cv"],
            }
        )
    return row


def save_csv(rows, path):
    """Write a list of dictionaries to CSV."""
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_pickle(payload, path):
    """Serialize an object to a pickle file."""
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def build_theta_row(sampled_params):
    """Build the theta vector for one accepted sample."""
    return np.asarray(
        [sampled_params[name] for name in SIM_THETA_PARAMETERS],
        dtype=np.float32,
    )


def build_catch22_row(catch22_result):
    """Build the catch22 feature vector for export."""
    return np.asarray(
        [catch22_result[name] for name in CATCH22_NAMES],
        dtype=np.float32,
    )


def build_specparam_row(specparam_result):
    """Build the specparam feature vector for export."""
    return np.asarray(
        [
            specparam_result["slope"],
            specparam_result["peak_frequency"],
            specparam_result["peak_power"],
        ],
        dtype=np.float32,
    )


def build_cdm_row(cdm_total):
    """Build the normalized CDM trace used for export, as in Hagen."""
    cdm_total = np.asarray(cdm_total, dtype=float).squeeze()
    if cdm_total.ndim != 1 or cdm_total.size <= 500:
        return None

    trimmed = cdm_total[500:]
    sigma = float(np.std(trimmed))
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None

    normalized = (trimmed - np.mean(trimmed)) / sigma
    return np.asarray(normalized, dtype=np.float32)


def build_proxy_row(proxy_data):
    """Build the Cavallari proxy trace used for export."""
    signal = np.asarray(proxy_data["signal"], dtype=float).squeeze()
    if signal.ndim != 1 or signal.size == 0:
        return None
    return signal.astype(np.float32)


def build_signal_row(signal_method, signal_data, signal):
    """Build the exported trace row for the selected signal method."""
    if signal_method == "kernel":
        return build_cdm_row(signal)
    return build_proxy_row(signal_data)


def initialize_sim_data_exports(signal_method):
    """Initialize in-memory export containers for accepted samples."""
    return {
        method: {"theta": [], "X": []}
        for method in active_data_methods(signal_method)
    }


def append_sim_data_export(exports, method, theta_row, x_row):
    """Append one feature row to the export buffers."""
    if x_row is None:
        return
    exports[method]["theta"].append(np.asarray(theta_row, dtype=np.float32))
    exports[method]["X"].append(np.asarray(x_row, dtype=np.float32))


def save_sim_data_exports(exports):
    """Save accumulated simulation datasets to disk."""
    for method, payload in exports.items():
        if not payload["theta"] or not payload["X"]:
            continue

        theta = {
            "parameters": list(SIM_THETA_PARAMETERS),
            "data": np.stack(payload["theta"]).astype(np.float32),
        }
        X = np.stack(payload["X"]).astype(np.float32)
        method_dir = os.path.join(OUTPUT_ROOT, "data", method)
        save_pickle(theta, os.path.join(method_dir, "sim_theta"))
        save_pickle(X, os.path.join(method_dir, "sim_X"))


def plot_parameter_samples(sampled_parameter_sets, base_neuron_params):
    """Plot the sampled parameter combinations for this batch."""
    if not PLOT_PARAMETER_SAMPLES or not sampled_parameter_sets:
        return

    print("\nPlotting sampled parameter combinations before running simulations.", flush=True)

    parameter_names = list(SIM_PLOT_PARAMETERS)
    plot_parameter_sets = [
        build_plot_parameter_values(params, base_neuron_params)
        for params in sampled_parameter_sets
    ]
    sampled_values = np.array(
        [[params[name] for name in parameter_names] for params in plot_parameter_sets],
        dtype=float,
    )

    n_params = len(parameter_names)
    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=(2.0 * n_params, 2.0 * n_params),
        dpi=150,
        constrained_layout=True,
    )

    for row_idx in range(n_params):
        for col_idx in range(n_params):
            ax = axes[row_idx, col_idx]
            if row_idx == col_idx:
                ax.hist(sampled_values[:, col_idx], bins=20, color="0.75", edgecolor="0.35")
            elif row_idx > col_idx:
                ax.scatter(
                    sampled_values[:, col_idx],
                    sampled_values[:, row_idx],
                    s=8,
                    alpha=0.35,
                    color="C0",
                    linewidths=0.0,
                )
            else:
                ax.axis("off")
                continue

            if row_idx == n_params - 1:
                ax.set_xlabel(parameter_names[col_idx], fontsize=8)
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(parameter_names[row_idx], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    plt.show()
    plt.close(fig)


def main():
    """Run the full massive-model simulation batch workflow."""
    args = parse_args()
    configure_output_directories(args.batch_id, args.num_batches, args.output_root)
    ensure_output_directories(SIGNAL_METHOD)
    print(f"Using output root: {OUTPUT_ROOT}", flush=True)

    if SIGNAL_METHOD == "kernel":
        download_multicompartment_data()

    rng = np.random.default_rng(RANDOM_SEED)

    network_params = load_module(
        os.path.join(PARAMS_DIR, "network_params.py"),
        "cavallari_network_params_massive",
    )
    simulation_params = load_module(
        os.path.join(PARAMS_DIR, "simulation_params.py"),
        "cavallari_simulation_params_massive",
    )

    population_sizes = {
        "E": int(network_params.Network_params["N_exc"]),
        "I": int(network_params.Network_params["N_inh"]),
    }
    populations = ["E", "I"]
    population_neuron_ids = build_population_neuron_ids(populations, population_sizes)

    kernel_params = None
    potential = None
    kernel_cache = None
    if SIGNAL_METHOD == "kernel":
        kernel_params = load_kernel_params()
        potential = FieldPotential()

    catch22_features = Features(method="catch22", params={"normalize": True})
    signal_fs = 1000.0 / (10.0 * float(simulation_params.dt))
    specparam_features = Features(method="specparam", params=_specparam_params(signal_fs))

    all_simulation_rows = []
    valid_detail_rows = []
    sim_data_exports = initialize_sim_data_exports(SIGNAL_METHOD)
    sampled_parameter_sets = [
        sample_parameters(rng, network_params.Network_params, network_params.Neuron_params)
        for _ in range(TOTAL_SIMULATIONS)
    ]
    batch_parameter_sets = select_batch_parameter_sets(
        sampled_parameter_sets, args.batch_id, args.num_batches
    )
    plot_parameter_samples(
        [params for _, params in batch_parameter_sets],
        network_params.Neuron_params,
    )
    batch_start_time = time.time()

    print(
        (
            f"Running batch {args.batch_id + 1}/{args.num_batches} with "
            f"{len(batch_parameter_sets)} assigned simulations."
        ),
        flush=True,
    )

    for local_simulation_index, (global_simulation_index, sampled_params) in enumerate(
        batch_parameter_sets,
        start=1,
    ):
        simulation_index = global_simulation_index + 1
        lif_params = build_lif_parameters(
            network_params.Network_params,
            network_params.Neuron_params,
            sampled_params,
        )

        temp_output_dir = tempfile.mkdtemp(prefix="cavallari_massive_", dir=OUTPUT_ROOT)
        banner_message = (
            f"###### Batch {args.batch_id + 1}/{args.num_batches} | "
            f"simulation {local_simulation_index}/{len(batch_parameter_sets)} "
            f"(global {simulation_index}/{TOTAL_SIMULATIONS}) | "
            f"valid so far {len(valid_detail_rows)} ######"
        )
        banner_border = "#" * len(banner_message)
        print(
            f"\n{banner_border}\n{banner_message}\n{banner_border}",
            flush=True,
        )

        runtime_info = run_simulation(lif_params, simulation_params, temp_output_dir)
        if runtime_info["status"] != "completed":
            all_simulation_rows.append(
                flatten_summary_row(
                    sample_index=None,
                    simulation_index=simulation_index,
                    batch_id=args.batch_id,
                    sampled_params=sampled_params,
                    runtime_info=runtime_info,
                    signal_method=SIGNAL_METHOD,
                    valid=False,
                    valid_reasons=[runtime_info["status"]],
                )
            )
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            continue

        try:
            outputs = load_simulation_outputs(temp_output_dir, SIGNAL_METHOD)
            transient = TRANSIENT_MS

            filtered_times, filtered_gids = apply_transient_filter(
                outputs["times"], outputs["gids"], transient
            )

            firing_rate_e = mean_firing_rate_hz(
                filtered_times["E"], population_sizes["E"], transient, outputs["tstop"]
            )
            firing_rate_i = mean_firing_rate_hz(
                filtered_times["I"], population_sizes["I"], transient, outputs["tstop"]
            )

            if SIGNAL_METHOD == "kernel" and kernel_cache is None:
                # The Cavallari example uses fixed kernel settings, so the kernel can be
                # reused across samples with the same dt/tstop configuration.
                kernel_cache = compute_kernel(
                    potential, kernel_params, outputs["dt"], outputs["tstop"]
                )

            signal_data, signal, signal_time, signal_dt = compute_signal_data(
                SIGNAL_METHOD,
                potential,
                kernel_cache,
                outputs,
                filtered_times,
                transient,
                populations,
            )

            catch22_result = compute_catch22_features(catch22_features, signal)
            specparam_result = compute_specparam_features(specparam_features, signal)
            spike_train_qc = compute_spike_train_qc(
                filtered_times,
                filtered_gids,
                populations,
                population_neuron_ids,
                transient,
                outputs["tstop"],
                rng,
            )
            is_valid, invalid_reasons = evaluate_validity(
                firing_rate_e, firing_rate_i, specparam_result, spike_train_qc
            )

            sample_index = len(valid_detail_rows) if is_valid else None
            summary_row = flatten_summary_row(
                sample_index=sample_index,
                simulation_index=simulation_index,
                batch_id=args.batch_id,
                sampled_params=sampled_params,
                runtime_info=runtime_info,
                signal_method=SIGNAL_METHOD,
                firing_rate_e=firing_rate_e,
                firing_rate_i=firing_rate_i,
                catch22_features=catch22_result,
                specparam_features=specparam_result,
                spike_train_qc=spike_train_qc,
                valid=is_valid,
                valid_reasons=invalid_reasons,
            )
            all_simulation_rows.append(summary_row)

            signal_data_method = SIGNAL_DATA_METHODS[SIGNAL_METHOD]
            plot_simulation_debug(
                simulation_index=simulation_index,
                plot_params=build_plot_parameter_values(
                    sampled_params,
                    network_params.Neuron_params,
                ),
                times=filtered_times,
                gids=filtered_gids,
                populations=populations,
                signal_total=signal,
                signal_time=signal_time,
                signal_dt=signal_dt,
                signal_label=signal_data_method,
                transient=transient,
                tstop=outputs["tstop"],
                firing_rate_e=firing_rate_e,
                firing_rate_i=firing_rate_i,
                specparam_result=specparam_result,
                spike_train_qc=spike_train_qc,
                is_valid=is_valid,
                invalid_reasons=invalid_reasons,
            )

            if is_valid:
                theta_row = build_theta_row(sampled_params)
                append_sim_data_export(
                    sim_data_exports,
                    signal_data_method,
                    theta_row,
                    build_signal_row(SIGNAL_METHOD, signal_data, signal),
                )
                append_sim_data_export(
                    sim_data_exports,
                    "catch22",
                    theta_row,
                    build_catch22_row(catch22_result),
                )
                append_sim_data_export(
                    sim_data_exports,
                    "specparam",
                    theta_row,
                    build_specparam_row(specparam_result),
                )
                valid_detail_rows.append(
                    {
                        "sample_index": sample_index,
                        "simulation_index": simulation_index,
                        "batch_id": args.batch_id,
                        "signal_method": SIGNAL_METHOD,
                        "parameters": sampled_params,
                        "firing_rates_hz": {"E": firing_rate_e, "I": firing_rate_i},
                        "spike_counts": {
                            "E": int(filtered_times["E"].size),
                            "I": int(filtered_times["I"].size),
                        },
                        "catch22": catch22_result,
                        "specparam": {
                            "slope": specparam_result["slope"],
                            "peak_frequency": specparam_result["peak_frequency"],
                            "peak_power": specparam_result["peak_power"],
                        },
                        "spike_train_qc": spike_train_qc,
                        "data_dirs": {
                            method: os.path.join(OUTPUT_ROOT, "data", method)
                            for method in active_data_methods(SIGNAL_METHOD)
                        },
                    }
                )
                print(
                    f"Accepted sample {sample_index + 1}/{len(batch_parameter_sets)} "
                    f"(batch {args.batch_id + 1}, simulation {simulation_index}, "
                    f"runtime {runtime_info['runtime_seconds']:.2f}s).",
                    flush=True,
                )
            else:
                print(
                    f"Rejected simulation {simulation_index} in batch {args.batch_id + 1}: "
                    f"{', '.join(invalid_reasons)}.",
                    flush=True,
                )
        finally:
            shutil.rmtree(temp_output_dir, ignore_errors=True)

    save_sim_data_exports(sim_data_exports)
    save_csv(all_simulation_rows, os.path.join(OUTPUT_ROOT, "all_simulations_summary.csv"))
    save_csv(
        [row for row in all_simulation_rows if row.get("valid")],
        os.path.join(OUTPUT_ROOT, "valid_samples_summary.csv"),
    )
    save_pickle(valid_detail_rows, os.path.join(OUTPUT_ROOT, "valid_samples_details.pkl"))
    batch_elapsed_seconds = time.time() - batch_start_time

    print(
        (
            f"\nFinished batch {args.batch_id + 1}/{args.num_batches}: "
            f"{len(batch_parameter_sets)} simulations assigned."
        ),
        flush=True,
    )
    if len(valid_detail_rows) < len(batch_parameter_sets):
        print(
            f"Valid simulations in this batch: {len(valid_detail_rows)}/{len(batch_parameter_sets)}.",
            flush=True,
        )
    if not PLOT_PARAMETER_SAMPLES and not PLOT_EACH_SIMULATION:
        total_hours, total_days = format_elapsed_time(batch_elapsed_seconds)
        print(
            f"Total simulation batch time: {total_hours:.2f} hours ({total_days:.2f} days).",
            flush=True,
        )


if __name__ == "__main__":
    main()
