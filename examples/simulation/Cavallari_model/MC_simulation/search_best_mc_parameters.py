#!/usr/bin/env python3
"""Search MC scaling factors that best match the baseline Cavallari LIF model.

The matching targets follow the usage in Hagen et al. (2022, Fig. 14), where
the point-neuron network is compared to the reference network through mean
population firing rates and population-rate power spectra. Here the baseline
Cavallari LIF simulation provides those target statistics, and the
multicompartment (MC) Cavallari model is searched over two scalar factors:

* ``recurrent_scale`` scales the four recurrent MC connection weights.
* ``th_scale`` scales the two thalamic/external MC synaptic weights.

The script:
1. runs the baseline Cavallari LIF network,
2. extracts post-transient E/I binned spike-count traces and their normalized spectra,
3. evaluates a bounded grid search on the MC model,
4. writes scalar summaries and spectra for the targets and best MC match.
"""

from __future__ import annotations
import csv
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import h5py
import ncpi
import numpy as np
import scipy.signal as ss


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parents[3]
LIF_DIR = ROOT_DIR.parent / "LIF_simulation"
LIF_PARAMS_DIR = LIF_DIR / "params"
LIF_PYTHON_DIR = LIF_DIR / "python"
LIF_SIMULATION_SCRIPT = LIF_PYTHON_DIR / "simulation.py"
LIF_NETWORK_PARAMS = LIF_PARAMS_DIR / "network_params.py"
LIF_SIMULATION_PARAMS = LIF_PARAMS_DIR / "simulation_params.py"
MC_ANALYSIS_PARAMS = ROOT_DIR / "analysis_params.py"
MC_SIMULATION_SCRIPT = ROOT_DIR / "example_model_simulation.py"

DEFAULT_RESULTS_ROOT = ROOT_DIR / "parameter_search_results"
POPULATION_ORDER = ("E", "I")
POPULATION_SIZE_KEYS = {"E": "N_exc", "I": "N_inh"}
RECURRENT_WEIGHT_KEYS = ("weight_EE", "weight_IE", "weight_EI", "weight_II")
THALAMIC_WEIGHT_KEYS = ("th_exc_external", "th_inh_external")

# Tunable parameters
MAX_SIMULATIONS = 100
RANDOM_SEED = 0
RECURRENT_SCALE_RANGE = (0.5, 10.0)
TH_SCALE_RANGE = (0.5, 100.0)
TRANSIENT_MS = 500.0
TSTOP_MS = None
FREQ_RANGE = (5.0, 200.0)
PSD_BIN_WIDTH_HZ = 10.0
RATE_WEIGHT = 1.0
PSD_WEIGHT = 1.0
LIF_LOCAL_NUM_THREADS = 64
MC_PROCESSES = 64
LIF_TIMEOUT_SECONDS = 300.0
MC_TIMEOUT_SECONDS = 7200.0
OUTPUT_DIR = None
MC_MODEL_ROOT = os.path.join(os.path.expanduser("~"), "multicompartment_neuron_network")
KEEP_TEMPDIRS = False


def load_module(module_path: Path, module_name: str):
    """Load a Python module from a filesystem path."""
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_settings():
    """Collect and validate the tunable parameters declared at the top of the file."""
    settings = SimpleNamespace(
        max_simulations=int(MAX_SIMULATIONS),
        random_seed=int(RANDOM_SEED),
        recurrent_scale_range=tuple(float(v) for v in RECURRENT_SCALE_RANGE),
        th_scale_range=tuple(float(v) for v in TH_SCALE_RANGE),
        transient_ms=float(TRANSIENT_MS),
        tstop_ms=None if TSTOP_MS is None else float(TSTOP_MS),
        freq_range=tuple(float(v) for v in FREQ_RANGE),
        psd_bin_width_hz=float(PSD_BIN_WIDTH_HZ),
        rate_weight=float(RATE_WEIGHT),
        psd_weight=float(PSD_WEIGHT),
        lif_local_num_threads=None if LIF_LOCAL_NUM_THREADS is None else int(LIF_LOCAL_NUM_THREADS),
        mc_processes=int(MC_PROCESSES),
        lif_timeout_seconds=float(LIF_TIMEOUT_SECONDS),
        mc_timeout_seconds=float(MC_TIMEOUT_SECONDS),
        output_dir=OUTPUT_DIR,
        mc_model_root=str(MC_MODEL_ROOT),
        keep_tempdirs=bool(KEEP_TEMPDIRS),
    )

    if settings.max_simulations < 1:
        raise ValueError("`max_simulations` must be >= 1.")
    if settings.transient_ms < 0:
        raise ValueError("`transient_ms` must be >= 0.")
    if settings.mc_processes < 1:
        raise ValueError("`mc_processes` must be >= 1.")
    if settings.rate_weight < 0 or settings.psd_weight < 0:
        raise ValueError("Objective weights must be non-negative.")
    if settings.psd_bin_width_hz <= 0:
        raise ValueError("`psd_bin_width_hz` must be positive.")

    for name, value_range in (
        ("recurrent_scale_range", settings.recurrent_scale_range),
        ("th_scale_range", settings.th_scale_range),
        ("freq_range", settings.freq_range),
    ):
        if len(value_range) != 2 or value_range[1] < value_range[0]:
            raise ValueError(f"`{name}` must be ordered as (min, max).")

    return settings


def resolve_output_dir(cli_output_dir: str | None) -> Path:
    """Resolve and create the search output directory."""
    if cli_output_dir:
        output_dir = Path(cli_output_dir).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = (DEFAULT_RESULTS_ROOT / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(payload, path: Path):
    """Serialize a JSON payload to disk."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_runtime_lif_params(path: Path, base_module, tstop_ms: float, local_threads: int):
    """Write an override params file when the default LIF params are not used."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"tstop = {float(tstop_ms)!r}\n")
        handle.write(f"local_num_threads = {int(local_threads)!r}\n")
        handle.write(f"dt = {float(base_module.dt)!r}\n")
        handle.write(
            f"simulate_in_chunks = {bool(getattr(base_module, 'simulate_in_chunks', True))!r}\n"
        )
        if hasattr(base_module, "simulation_chunk_ms"):
            handle.write(
                f"simulation_chunk_ms = {float(getattr(base_module, 'simulation_chunk_ms'))!r}\n"
            )
        if hasattr(base_module, "numpy_seed"):
            handle.write(f"numpy_seed = {int(getattr(base_module, 'numpy_seed'))!r}\n")
        if hasattr(base_module, "max_mean_population_spike_rate_hz"):
            handle.write(
                "max_mean_population_spike_rate_hz = "
                f"{float(getattr(base_module, 'max_mean_population_spike_rate_hz'))!r}\n"
            )


def build_baseline_lif_params():
    """Build the baseline Cavallari LIF parameter payload."""
    network_module = load_module(LIF_NETWORK_PARAMS, "cavallari_lif_network_params_search")
    network_params = deepcopy(network_module.Network_params)
    neuron_params = deepcopy(network_module.Neuron_params)
    return {
        "X": list(POPULATION_ORDER),
        "N_X": [network_params["N_exc"], network_params["N_inh"]],
        "model": "iaf_bw_2003",
        "neuron_params": {
            "E": neuron_params[0],
            "I": neuron_params[1],
        },
        "network_params": network_params,
    }


def load_pickle(path: Path):
    """Load a pickle file."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_lif_target_simulation(args, output_dir: Path):
    """Run the baseline LIF simulation and load its outputs."""
    sim_params = load_module(LIF_SIMULATION_PARAMS, "cavallari_lif_sim_params_search")
    lif_params = build_baseline_lif_params()
    tstop_ms = float(args.tstop_ms) if args.tstop_ms is not None else float(sim_params.tstop)
    local_threads = (
        int(args.lif_local_num_threads)
        if args.lif_local_num_threads is not None
        else int(sim_params.local_num_threads)
    )

    use_default_params = (
        args.tstop_ms is None
        and (
            args.lif_local_num_threads is None
            or int(args.lif_local_num_threads) == int(sim_params.local_num_threads)
        )
    )

    if use_default_params:
        param_folder = LIF_PARAMS_DIR
        param_filename = "simulation_params.py"
    else:
        runtime_params_path = output_dir / "simulation_params_runtime.py"
        write_runtime_lif_params(runtime_params_path, sim_params, tstop_ms, local_threads)
        param_folder = output_dir
        param_filename = runtime_params_path.name

    with (output_dir / "network.pkl").open("wb") as handle:
        pickle.dump(lif_params, handle)

    started = time.time()
    sim = ncpi.Simulation(
        param_folder=str(param_folder),
        python_folder=str(LIF_PYTHON_DIR),
        output_folder=str(output_dir),
    )
    sim.simulate("simulation.py", param_filename)
    runtime_seconds = time.time() - started

    outputs = {
        "times": load_pickle(output_dir / "times.pkl"),
        "gids": load_pickle(output_dir / "gids.pkl"),
        "dt": float(load_pickle(output_dir / "dt.pkl")),
        "tstop": float(load_pickle(output_dir / "tstop.pkl")),
        "network": lif_params,
        "runtime_seconds": runtime_seconds,
    }
    return outputs


def build_population_sizes_from_network(network_params: dict):
    """Return population sizes keyed by population label."""
    return {
        population: int(network_params[POPULATION_SIZE_KEYS[population]])
        for population in POPULATION_ORDER
    }


def get_spike_rate(times, transient, dt, tstop):
    """Bin spike times exactly as in the Cavallari LIF example script."""
    bins = np.arange(transient, tstop + dt, dt)
    hist, _ = np.histogram(times, bins=bins)
    return bins, hist.astype(float)


def compute_power_spectrum_like_lif_example(signal, fs_hz):
    """Compute a Welch spectrum and normalize total power as in the LIF example."""
    signal = np.asarray(signal, dtype=float)
    freqs_hz, power_spectrum = ss.welch(signal, fs=float(fs_hz))
    power_sum = float(np.sum(power_spectrum))
    if power_sum > 0.0:
        power_spectrum = power_spectrum / power_sum
    return np.asarray(freqs_hz, dtype=float), np.asarray(power_spectrum, dtype=float)


def compute_histogram_rate_metrics(spike_histogram, dt_ms: float):
    """Compute count-trace summary statistics from a binned spike histogram."""
    spike_histogram = np.asarray(spike_histogram, dtype=float)
    mean_spikes_per_dt = float(np.mean(spike_histogram))

    if spike_histogram.size < 2:
        return {
            "mean_spikes_per_dt": mean_spikes_per_dt,
            "freqs_hz": np.array([], dtype=float),
            "psd": np.array([], dtype=float),
            "dominant_frequency_hz": float("nan"),
        }

    fs_hz = 1000.0 / float(dt_ms)
    freqs_hz, psd = compute_power_spectrum_like_lif_example(spike_histogram, fs_hz)

    if freqs_hz.size == 0:
        dominant_frequency_hz = float("nan")
    else:
        dominant_frequency_hz = float(freqs_hz[int(np.argmax(psd))])

    return {
        "mean_spikes_per_dt": mean_spikes_per_dt,
        "freqs_hz": np.asarray(freqs_hz, dtype=float),
        "psd": np.asarray(psd, dtype=float),
        "dominant_frequency_hz": dominant_frequency_hz,
    }


def compute_lif_population_metrics(times, transient_ms: float, tstop_ms: float, dt_ms: float):
    """Compute E/I count-trace metrics from LIF spike-time outputs."""
    metrics = {}
    for population in POPULATION_ORDER:
        spike_times = np.asarray(times[population], dtype=float)
        bins, histogram = get_spike_rate(spike_times, transient_ms, dt_ms, tstop_ms)
        metrics[population] = compute_histogram_rate_metrics(
            histogram,
            dt_ms=dt_ms,
        )
        metrics[population]["bins_ms"] = np.asarray(bins[:-1], dtype=float)
    return metrics


def compute_mc_population_metrics(spikes_path: Path, transient_ms: float, tstop_ms: float, dt_ms: float):
    """Compute E/I count-trace metrics from the MC ``spikes.h5`` output."""
    metrics = {}

    with h5py.File(spikes_path, "r") as handle:
        for population in POPULATION_ORDER:
            all_spike_times = []
            group = handle[population]
            times_dataset = group["times"]
            for spike_train in times_dataset:
                spike_times = np.asarray(spike_train, dtype=float)
                if spike_times.size == 0:
                    continue
                all_spike_times.append(spike_times)

            if all_spike_times:
                population_spike_times = np.concatenate(all_spike_times)
            else:
                population_spike_times = np.array([], dtype=float)

            bins, histogram = get_spike_rate(
                population_spike_times, transient_ms, dt_ms, tstop_ms
            )

            metrics[population] = compute_histogram_rate_metrics(
                histogram,
                dt_ms=dt_ms,
            )
            metrics[population]["bins_ms"] = np.asarray(bins[:-1], dtype=float)

    return metrics


def restrict_spectrum(freqs_hz, psd, freq_range):
    """Restrict a spectrum to the configured comparison band."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    psd = np.asarray(psd, dtype=float)
    keep = (
        np.isfinite(freqs_hz)
        & np.isfinite(psd)
        & (freqs_hz >= float(freq_range[0]))
        & (freqs_hz <= float(freq_range[1]))
    )
    return freqs_hz[keep], psd[keep]


def bin_spectrum(freqs_hz, psd, freq_range, bin_width_hz):
    """Average a spectrum into coarse frequency bins for scoring."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if freqs_hz.size == 0 or psd.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    bin_width_hz = float(bin_width_hz)
    fmin = float(freq_range[0])
    fmax = float(freq_range[1])
    edges = np.arange(fmin, fmax + bin_width_hz, bin_width_hz, dtype=float)
    if edges.size < 2:
        edges = np.array([fmin, fmax], dtype=float)
    elif edges[-1] < fmax:
        edges = np.append(edges, fmax)

    binned_freqs = []
    binned_psd = []
    for left_edge, right_edge in zip(edges[:-1], edges[1:]):
        if right_edge == edges[-1]:
            keep = (freqs_hz >= left_edge) & (freqs_hz <= right_edge)
        else:
            keep = (freqs_hz >= left_edge) & (freqs_hz < right_edge)
        if not np.any(keep):
            continue
        binned_freqs.append(0.5 * (left_edge + right_edge))
        binned_psd.append(float(np.mean(psd[keep])))

    return np.asarray(binned_freqs, dtype=float), np.asarray(binned_psd, dtype=float)


def summarize_metrics(metrics, freq_range):
    """Convert metrics to a JSON-friendly summary."""
    summary = {}
    for population, values in metrics.items():
        freqs_hz, psd = restrict_spectrum(values["freqs_hz"], values["psd"], freq_range)
        summary[population] = {
            "mean_spikes_per_dt": float(values["mean_spikes_per_dt"]),
            "dominant_frequency_hz": float(values["dominant_frequency_hz"]),
            "num_spectrum_points": int(freqs_hz.size),
        }
    return summary


def save_metrics_arrays(path: Path, metrics, freq_range):
    """Store spectra and means as a compact ``npz`` file."""
    payload = {}
    for population, values in metrics.items():
        freqs_hz, psd = restrict_spectrum(values["freqs_hz"], values["psd"], freq_range)
        payload[f"{population}_freqs_hz"] = freqs_hz
        payload[f"{population}_psd"] = psd
        payload[f"{population}_bins_ms"] = np.asarray(values.get("bins_ms", np.array([], dtype=float)))
        payload[f"{population}_mean_spikes_per_dt"] = np.array([values["mean_spikes_per_dt"]], dtype=float)
    np.savez(path, **payload)


def interpolate_candidate_psd(target_freqs_hz, candidate_freqs_hz, candidate_psd):
    """Interpolate a candidate PSD onto the target frequency grid."""
    target_freqs_hz = np.asarray(target_freqs_hz, dtype=float)
    candidate_freqs_hz = np.asarray(candidate_freqs_hz, dtype=float)
    candidate_psd = np.asarray(candidate_psd, dtype=float)
    if target_freqs_hz.size == 0 or candidate_freqs_hz.size == 0:
        return np.array([], dtype=float)
    return np.interp(
        target_freqs_hz,
        candidate_freqs_hz,
        candidate_psd,
        left=np.nan,
        right=np.nan,
    )


def score_metrics(
    target_metrics,
    candidate_metrics,
    freq_range,
    psd_bin_width_hz,
    rate_weight: float,
    psd_weight: float,
):
    """Score one candidate against the baseline LIF count-trace targets."""
    rate_terms = []
    psd_terms = []

    for population in POPULATION_ORDER:
        target_mean = float(target_metrics[population]["mean_spikes_per_dt"])
        candidate_mean = float(candidate_metrics[population]["mean_spikes_per_dt"])
        denom = max(abs(target_mean), 1e-6)
        rate_terms.append(((candidate_mean - target_mean) / denom) ** 2)

        target_freqs_hz, target_psd = restrict_spectrum(
            target_metrics[population]["freqs_hz"],
            target_metrics[population]["psd"],
            freq_range,
        )
        candidate_freqs_hz, candidate_psd = restrict_spectrum(
            candidate_metrics[population]["freqs_hz"],
            candidate_metrics[population]["psd"],
            freq_range,
        )
        target_freqs_hz, target_psd = bin_spectrum(
            target_freqs_hz,
            target_psd,
            freq_range,
            psd_bin_width_hz,
        )
        candidate_freqs_hz, candidate_psd = bin_spectrum(
            candidate_freqs_hz,
            candidate_psd,
            freq_range,
            psd_bin_width_hz,
        )
        candidate_psd_interp = interpolate_candidate_psd(
            target_freqs_hz, candidate_freqs_hz, candidate_psd
        )

        keep = (
            np.isfinite(target_psd)
            & np.isfinite(candidate_psd_interp)
            & (target_psd >= 0.0)
            & (candidate_psd_interp >= 0.0)
        )
        if np.any(keep):
            log_target = np.log10(target_psd[keep] + 1e-18)
            log_candidate = np.log10(candidate_psd_interp[keep] + 1e-18)
            psd_terms.append(float(np.mean((log_candidate - log_target) ** 2)))
        else:
            psd_terms.append(float("inf"))

    rate_error = float(np.mean(rate_terms)) if rate_terms else float("inf")
    psd_error = float(np.mean(psd_terms)) if psd_terms else float("inf")
    total_score = float(rate_weight) * rate_error + float(psd_weight) * psd_error
    return {
        "rate_error": rate_error,
        "psd_error": psd_error,
        "total_score": total_score,
    }


def sample_search_candidates(rng, max_simulations: int, recurrent_scale_range, th_scale_range):
    """Sample MC search candidates uniformly, following the massive simulation style."""
    candidates = []
    for _ in range(int(max_simulations)):
        candidates.append(
            (
                float(rng.uniform(*recurrent_scale_range)),
                float(rng.uniform(*th_scale_range)),
            )
        )
    return candidates


def write_mc_runtime_analysis_params(path: Path, mc_analysis_params_path: Path, recurrent_scale: float, th_scale: float, tstop_ms: float, transient_ms: float):
    """Write a temporary ``analysis_params.py`` for a copied MC example script."""
    wrapper_code = f"""#!/usr/bin/env python3
import importlib.util

ANALYSIS_PARAMS_PATH = {str(mc_analysis_params_path)!r}
RECURRENT_SCALE = {float(recurrent_scale)!r}
TH_SCALE = {float(th_scale)!r}
TSTOP_MS = {float(tstop_ms)!r}
TRANSIENT_MS = {float(transient_ms)!r}

spec = importlib.util.spec_from_file_location("analysis_params_base", ANALYSIS_PARAMS_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

params = module.KernelParams
params.MC_params = dict(params.MC_params)
for key in {RECURRENT_WEIGHT_KEYS!r}:
    params.MC_params[key] *= RECURRENT_SCALE
for key in {THALAMIC_WEIGHT_KEYS!r}:
    params.MC_params[key] *= TH_SCALE

weight_scale = float(getattr(params, "conductance_weight_scale", 1.0))
params.MC_params["weight_matrix"] = [
    [abs(float(params.MC_params["weight_EE"])) * weight_scale,
     abs(float(params.MC_params["weight_IE"])) * weight_scale],
    [abs(float(params.MC_params["weight_EI"])) * weight_scale,
     abs(float(params.MC_params["weight_II"])) * weight_scale],
]

params.networkParameters = dict(params.networkParameters)
params.networkParameters["tstop"] = TSTOP_MS
params.transient = TRANSIENT_MS
params.extSynapseParameters = [
    dict(
        params.extSynapseParameters[0],
        weight=abs(float(params.MC_params["th_exc_external"])) * weight_scale,
    ),
    dict(
        params.extSynapseParameters[1],
        weight=abs(float(params.MC_params["th_inh_external"])) * weight_scale,
    ),
]
params.netstim_interval = [
    1000.0 / (800.0 * params.MC_params["v_0"]),
    1000.0 / (800.0 * params.MC_params["v_0"]),
]
KernelParams = params
"""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(wrapper_code)


def run_mc_candidate(args, recurrent_scale: float, th_scale: float, tstop_ms: float):
    """Run one MC candidate and return its metrics."""
    mc_model_root = Path(args.mc_model_root).expanduser().resolve()
    mc_output_dir = mc_model_root / "output_Cavallari"
    spikes_path = mc_output_dir / "spikes.h5"

    temp_root = tempfile.mkdtemp(prefix="mc_search_", dir=str(ROOT_DIR))
    temp_root_path = Path(temp_root)
    runtime_script_path = temp_root_path / "example_model_simulation.py"
    runtime_analysis_params_path = temp_root_path / "analysis_params.py"
    shutil.copy2(MC_SIMULATION_SCRIPT, runtime_script_path)
    write_mc_runtime_analysis_params(
        runtime_analysis_params_path,
        MC_ANALYSIS_PARAMS,
        recurrent_scale,
        th_scale,
        tstop_ms,
        args.transient_ms,
    )

    if int(args.mc_processes) > 1:
        cmd = [
            "mpirun",
            "--use-hwthread-cpus",
            "-np",
            str(int(args.mc_processes)),
            "python",
            str(runtime_script_path),
        ]
    else:
        cmd = [sys.executable, str(runtime_script_path)]

    started = time.time()
    status = "completed"
    returncode = 0
    env = os.environ.copy()
    pythonpath_entries = [str(PROJECT_ROOT), str(temp_root_path)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=float(args.mc_timeout_seconds),
            cwd=str(temp_root_path),
            env=env,
        )
    except subprocess.TimeoutExpired:
        status = "timeout"
    except subprocess.CalledProcessError as exc:
        status = "failed"
        returncode = int(exc.returncode)
    runtime_seconds = time.time() - started

    metrics = None
    if status == "completed" and spikes_path.exists():
        mc_analysis_module = load_module(
            MC_ANALYSIS_PARAMS,
            f"cavallari_mc_analysis_params_runtime_{int(time.time() * 1e6)}",
        )
        population_sizes = {
            population: int(
                mc_analysis_module.KernelParams.population_sizes[
                    POPULATION_ORDER.index(population)
                ]
            )
            for population in POPULATION_ORDER
        }
        metrics = compute_mc_population_metrics(
            spikes_path=spikes_path,
            transient_ms=float(args.transient_ms),
            tstop_ms=float(tstop_ms),
            dt_ms=float(mc_analysis_module.KernelParams.networkParameters["dt"]),
        )
    elif status == "completed":
        status = "missing_output"

    if not args.keep_tempdirs:
        shutil.rmtree(temp_root_path, ignore_errors=True)

    return {
        "status": status,
        "returncode": returncode,
        "runtime_seconds": runtime_seconds,
        "metrics": metrics,
        "mc_output_dir": str(mc_output_dir),
    }


def scalarize_candidate_row(simulation_index: int, recurrent_scale: float, th_scale: float, runtime_info: dict, score_info: dict | None):
    """Build a flat result row for the CSV output."""
    row = {
        "simulation_index": int(simulation_index),
        "recurrent_scale": float(recurrent_scale),
        "th_scale": float(th_scale),
        "status": runtime_info["status"],
        "runtime_seconds": float(runtime_info["runtime_seconds"]),
        "returncode": int(runtime_info.get("returncode", 0)),
    }

    metrics = runtime_info.get("metrics")
    if metrics is not None:
        for population in POPULATION_ORDER:
            row[f"{population}_mean_spikes_per_dt"] = float(
                metrics[population]["mean_spikes_per_dt"]
            )
            row[f"{population}_dominant_frequency_hz"] = float(
                metrics[population]["dominant_frequency_hz"]
            )
    else:
        for population in POPULATION_ORDER:
            row[f"{population}_mean_spikes_per_dt"] = float("nan")
            row[f"{population}_dominant_frequency_hz"] = float("nan")

    if score_info is not None:
        row["rate_error"] = float(score_info["rate_error"])
        row["psd_error"] = float(score_info["psd_error"])
        row["total_score"] = float(score_info["total_score"])
    else:
        row["rate_error"] = float("inf")
        row["psd_error"] = float("inf")
        row["total_score"] = float("inf")

    return row


def write_results_csv(path: Path, rows):
    """Write the full MC search table."""
    fieldnames = [
        "simulation_index",
        "recurrent_scale",
        "th_scale",
        "status",
        "runtime_seconds",
        "returncode",
        "E_mean_spikes_per_dt",
        "I_mean_spikes_per_dt",
        "E_dominant_frequency_hz",
        "I_dominant_frequency_hz",
        "rate_error",
        "psd_error",
        "total_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run the target extraction and MC parameter search."""
    args = build_settings()
    output_dir = resolve_output_dir(args.output_dir)

    write_json(
        {
            "max_simulations": int(args.max_simulations),
            "random_seed": int(args.random_seed),
            "recurrent_scale_range": list(args.recurrent_scale_range),
            "th_scale_range": list(args.th_scale_range),
            "transient_ms": float(args.transient_ms),
            "tstop_ms": None if args.tstop_ms is None else float(args.tstop_ms),
            "freq_range_hz": list(args.freq_range),
            "psd_bin_width_hz": float(args.psd_bin_width_hz),
            "welch_convention": "scipy.signal.welch(signal, fs=...) with total-power normalization",
            "rate_weight": float(args.rate_weight),
            "psd_weight": float(args.psd_weight),
            "lif_local_num_threads": args.lif_local_num_threads,
            "mc_processes": int(args.mc_processes),
            "mc_model_root": str(Path(args.mc_model_root).expanduser().resolve()),
        },
        output_dir / "search_config.json",
    )

    lif_output_dir = output_dir / "lif_target_run"
    lif_output_dir.mkdir(parents=True, exist_ok=True)
    # Progress marker for the fixed target-generation stage.
    print("Running baseline LIF target simulation...", flush=True)
    lif_outputs = run_lif_target_simulation(args, lif_output_dir)
    target_metrics = compute_lif_population_metrics(
        times=lif_outputs["times"],
        transient_ms=float(args.transient_ms),
        tstop_ms=float(lif_outputs["tstop"]),
        dt_ms=float(lif_outputs["dt"]),
    )

    target_summary = {
        "simulation_runtime_seconds": float(lif_outputs["runtime_seconds"]),
        "tstop_ms": float(lif_outputs["tstop"]),
        "dt_ms": float(lif_outputs["dt"]),
        "transient_ms": float(args.transient_ms),
        "metrics": summarize_metrics(target_metrics, args.freq_range),
    }
    write_json(target_summary, output_dir / "lif_target_summary.json")
    save_metrics_arrays(output_dir / "lif_target_metrics.npz", target_metrics, args.freq_range)

    rng = np.random.default_rng(args.random_seed)
    candidates = sample_search_candidates(
        rng=rng,
        max_simulations=int(args.max_simulations),
        recurrent_scale_range=args.recurrent_scale_range,
        th_scale_range=args.th_scale_range,
    )
    # Progress marker for the bounded MC search phase.
    print(
        (
            f"Evaluating {len(candidates)} MC candidates "
            f"sampled uniformly with seed {args.random_seed}..."
        ),
        flush=True,
    )

    results_rows = []
    best_result = None
    best_metrics = None

    for simulation_index, (recurrent_scale, th_scale) in enumerate(candidates, start=1):
        # Progress marker for each candidate evaluation.
        print(
            (
                f"[{simulation_index}/{len(candidates)}] "
                f"recurrent_scale={recurrent_scale:.6g}, th_scale={th_scale:.6g}"
            ),
            flush=True,
        )
        runtime_info = run_mc_candidate(
            args,
            recurrent_scale=recurrent_scale,
            th_scale=th_scale,
            tstop_ms=float(lif_outputs["tstop"]),
        )

        score_info = None
        if runtime_info["metrics"] is not None:
            score_info = score_metrics(
                target_metrics=target_metrics,
                candidate_metrics=runtime_info["metrics"],
                freq_range=args.freq_range,
                psd_bin_width_hz=float(args.psd_bin_width_hz),
                rate_weight=float(args.rate_weight),
                psd_weight=float(args.psd_weight),
            )
            # Progress marker with the candidate objective terms after a successful run.
            print(
                (
                    f"  status={runtime_info['status']} "
                    f"score={score_info['total_score']:.6g} "
                    f"rate_error={score_info['rate_error']:.6g} "
                    f"psd_error={score_info['psd_error']:.6g}"
                ),
                flush=True,
            )
        else:
            # Progress marker when the simulation fails or times out.
            print(
                f"  status={runtime_info['status']} returncode={runtime_info['returncode']}",
                flush=True,
            )

        row = scalarize_candidate_row(
            simulation_index,
            recurrent_scale,
            th_scale,
            runtime_info,
            score_info,
        )
        results_rows.append(row)

        if score_info is not None and np.isfinite(score_info["total_score"]):
            is_better = best_result is None or score_info["total_score"] < best_result["total_score"]
            if is_better:
                best_result = dict(row)
                best_metrics = runtime_info["metrics"]

    write_results_csv(output_dir / "mc_search_results.csv", results_rows)

    if best_result is None or best_metrics is None:
        write_json(
            {
                "status": "no_valid_candidate",
                "evaluated_candidates": len(results_rows),
            },
            output_dir / "best_result.json",
        )
        raise RuntimeError("No valid MC candidate completed successfully.")

    best_summary = {
        "status": "ok",
        "best_candidate": {
            "simulation_index": int(best_result["simulation_index"]),
            "recurrent_scale": float(best_result["recurrent_scale"]),
            "th_scale": float(best_result["th_scale"]),
            "runtime_seconds": float(best_result["runtime_seconds"]),
            "rate_error": float(best_result["rate_error"]),
            "psd_error": float(best_result["psd_error"]),
            "total_score": float(best_result["total_score"]),
        },
        "metrics": summarize_metrics(best_metrics, args.freq_range),
        "target_metrics": summarize_metrics(target_metrics, args.freq_range),
    }
    write_json(best_summary, output_dir / "best_result.json")
    save_metrics_arrays(output_dir / "best_mc_metrics.npz", best_metrics, args.freq_range)

    print("\nBest candidate", flush=True)
    print(
        (
            f"recurrent_scale={best_result['recurrent_scale']:.6g}, "
            f"th_scale={best_result['th_scale']:.6g}, "
            f"score={best_result['total_score']:.6g}"
        ),
        flush=True,
    )
    print(f"Outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
