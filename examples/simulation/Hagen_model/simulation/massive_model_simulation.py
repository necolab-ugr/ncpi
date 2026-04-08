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

# Global tunable parameters for the simulation batch
TOTAL_SIMULATIONS = 10**6
SIMULATION_TIMEOUT_SECONDS = 60
LOCAL_NUM_THREADS = 56
MAX_MEAN_POPULATION_SPIKE_RATE_HZ = 50.0
PLOT_PARAMETER_SAMPLES = False
PLOT_EACH_SIMULATION = False
RANDOM_SEED = 0
STRONG_PEAK_RELATIVE_THRESHOLD = 0.9
STRONG_PEAK_POWER_THRESHOLD = 0.4

# Choose to either download files and precomputed outputs used in simulations of the
# reference multicompartment neuron network model (True) or load them from a local path (False)
zenodo_dw_mult = False

# Zenodo URL that contains the data (used if zenodo_dw_mult is True)
zenodo_URL_mult = "https://zenodo.org/api/records/15429373"

# Zenodo directory where the MC data is stored (must be an absolute path to correctly load morphologies in NEURON)
zenodo_dir = os.path.expandvars(os.path.expanduser(
    os.path.join("$HOME", "multicompartment_neuron_network")
))

# Directory with some MC output simulations
multi_output_path = os.path.join(
    zenodo_dir, "output", "adb947bfb931a5a8d09ad078a6d256b0"
)

# Biophysical mechanisms to include when computing kernels from the multicompartment model.
MULTI_BIOPHYS = ["set_Ih_linearized_hay2011", "make_cell_uniform"]

# Paths for the simulation and analysis
SCRIPT_DIR = os.path.dirname(__file__)
PARAMS_DIR = os.path.join(SCRIPT_DIR, "params")
PYTHON_DIR = os.path.join(SCRIPT_DIR, "python")
SIMULATION_SCRIPT = os.path.join(PYTHON_DIR, "simulation.py")
DEFAULT_BASE_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "simulation_output")
OUTPUT_ROOT = DEFAULT_BASE_OUTPUT_ROOT

# Ranges for the parameters to sample in the simulations
PARAMETER_RANGES = {
    "J_EE": (0.5, 4.),
    "J_IE": (0.5, 4.),
    "J_EI": (-40.0, -1.0),
    "J_II": (-40.0, -1.0),
    "tau_syn_E": (0.1, 2.),
    "tau_syn_I": (0.1, 8.),
    "J_ext": (10.0, 50.0),
}

CATCH22_NAMES = pycatch22.catch22_all([0])["names"]
SIM_THETA_PARAMETERS = [
    "J_EE",
    "J_IE",
    "J_EI",
    "J_II",
    "tau_syn_E",
    "tau_syn_I",
    "J_ext",
]
SIM_DATA_METHODS = ("CDM", "specparam", "catch22")


def _specparam_params(fs: float) -> Dict[str, Any]:
    """Build the specparam parameter dict."""
    
    # --- Input validation ---
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
            "Run the Hagen massive model simulation either as a standalone job "
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


def ensure_output_directories():
    """Create the output folders used by this simulation run."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for method in SIM_DATA_METHODS:
        os.makedirs(os.path.join(OUTPUT_ROOT, "data", method), exist_ok=True)


def download_multicompartment_data():
    """Download the multicompartment reference data when enabled."""
    if not zenodo_dw_mult:
        return

    print('\n--- Downloading data.')
    start_time = time.time()
    tools.download_zenodo_record(zenodo_URL_mult, download_dir=zenodo_dir)
    end_time = time.time()
    print(f"All files downloaded in {(end_time - start_time) / 60:.2f} minutes.")


def sample_parameters(rng):
    """Sample one valid parameter set from the configured ranges."""
    while True:
        sampled = {}
        for name, (low, high) in PARAMETER_RANGES.items():
            sampled[name] = float(rng.uniform(low, high))

        inhibitory_abs_min = min(abs(sampled["J_EI"]), abs(sampled["J_II"]))
        excitatory_bounded = (
            2.0 * sampled["J_EE"] <= inhibitory_abs_min
            and 2.0 * sampled["J_IE"] <= inhibitory_abs_min
        )

        if excitatory_bounded and sampled["J_IE"] > sampled["J_EE"]:
            return sampled


def build_lif_parameters(base_params, sampled_params):
    """Build LIF network parameters for one sampled configuration."""
    lif_params = copy.deepcopy(base_params)
    lif_params["J_YX"] = [
        [sampled_params["J_EE"], sampled_params["J_IE"]],
        [sampled_params["J_EI"], sampled_params["J_II"]],
    ]
    lif_params["tau_syn_YX"] = [
        [sampled_params["tau_syn_E"], sampled_params["tau_syn_I"]],
        [sampled_params["tau_syn_E"], sampled_params["tau_syn_I"]],
    ]
    lif_params["J_ext"] = sampled_params["J_ext"]
    return lif_params


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


def load_simulation_outputs(output_dir):
    """Load the core output files produced by a simulation."""
    outputs = {}
    for filename in ("times.pkl", "gids.pkl", "tstop.pkl", "dt.pkl", "network.pkl"):
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


def format_elapsed_time(total_seconds):
    """Convert elapsed seconds to hours and days."""
    total_seconds = float(total_seconds)
    return total_seconds / 3600.0, total_seconds / 86400.0


def get_spike_rate(times, transient, dt, tstop):
    """Bin spike times into a simple rate histogram."""
    bins = np.arange(transient, tstop + dt, dt)
    hist, _ = np.histogram(times, bins=bins)
    return bins[:-1], hist.astype(float)


def compute_kernel(potential, kernel_params, dt, tstop):
    """Compute the field-potential kernel used for CDM reconstruction."""
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
    for post in populations:
        for pre in populations:
            key = f"{post}{pre}"
            source_key = f"{pre}:{post}"
            cdm_components[key] = np.asarray(
                ss.decimate(cdm_signals[source_key], q=10, zero_phase=True)
            )

    cdm_total = np.sum(np.vstack(list(cdm_components.values())), axis=0)
    cdm_dt = dt * 10.0
    cdm_time = transient + np.arange(cdm_total.size) * cdm_dt
    return cdm_components, cdm_total, cdm_time, cdm_dt


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


def plot_simulation_debug(
    simulation_index,
    sampled_params,
    times,
    gids,
    populations,
    cdm_total,
    cdm_time,
    cdm_dt,
    transient,
    tstop,
    firing_rate_e,
    firing_rate_i,
    specparam_result,
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
    cdm_ax = fig.add_subplot(grid[2, 0])
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
        bins, spike_rate = get_spike_rate(times[population], transient, cdm_dt / 10.0, tstop)
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

    cdm_mask = (cdm_time >= time_window[0]) & (cdm_time <= time_window[1])
    cdm_ax.plot(cdm_time[cdm_mask], cdm_total[cdm_mask], color="k", linewidth=0.9)
    cdm_ax.set_ylabel("CDM")
    cdm_ax.set_xlabel("Time (ms)")
    cdm_ax.set_xlim(time_window)

    fs = 1000.0 / cdm_dt
    try:
        freqs, power, fit_model, freq_range = fit_specparam_for_plot(cdm_total, fs)
    except Exception:
        freqs, power = ss.welch(cdm_total, fs=fs)
        fit_model = None
        freq_range = _specparam_params(fs)["freq_range"]
    _plot_specparam_debug_spectrum(psd_ax, freqs, power, fit_model, freq_range)
    psd_ax.set_ylabel("Power")
    psd_ax.set_xlabel("Frequency (Hz)")
    psd_ax.set_title("CDM spectrum + specparam fit")

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

    parameter_lines = [
        f"{name}: {float(value):.4f}"
        for name, value in sampled_params.items()
    ]

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


def evaluate_validity(firing_rate_e, firing_rate_i, specparam_features):
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
    if firing_rate_i > 5.0 * firing_rate_e:
        reasons.append("I_rate_more_than_5x_E")
    if (
        specparam_features["strong_peak_count"] > 2
        and specparam_features["strongest_peak_power"] > STRONG_PEAK_POWER_THRESHOLD
    ):
        reasons.append("more_than_two_strong_peaks")
    if not specparam_features["fit_valid"]:
        reasons.append("specparam_fit_rejected")
    return len(reasons) == 0, reasons


def flatten_summary_row(
    sample_index,
    simulation_index,
    batch_id,
    sampled_params,
    runtime_info,
    firing_rate_e=None,
    firing_rate_i=None,
    catch22_features=None,
    specparam_features=None,
    valid=None,
    valid_reasons=None,
):
    """Flatten simulation results into a CSV-friendly row."""
    row = {
        "sample_index": sample_index,
        "simulation_index": simulation_index,
        "batch_id": batch_id,
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
    """Build the normalized CDM trace used for export."""
    cdm_total = np.asarray(cdm_total, dtype=float).squeeze()
    if cdm_total.ndim != 1 or cdm_total.size <= 500:
        return None

    trimmed = cdm_total[500:]
    sigma = float(np.std(trimmed))
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None

    normalized = (trimmed - np.mean(trimmed)) / sigma
    return np.asarray(normalized, dtype=np.float32)


def initialize_sim_data_exports():
    """Initialize in-memory export containers for accepted samples."""
    return {
        method: {"theta": [], "X": []}
        for method in SIM_DATA_METHODS
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


def plot_parameter_samples(sampled_parameter_sets):
    """Plot the sampled parameter combinations for this batch."""
    if not PLOT_PARAMETER_SAMPLES or not sampled_parameter_sets:
        return

    print("\nPlotting sampled parameter combinations before running simulations.", flush=True)

    parameter_names = list(PARAMETER_RANGES.keys())
    sampled_values = np.array(
        [[params[name] for name in parameter_names] for params in sampled_parameter_sets],
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
    ensure_output_directories()
    print(f"Using output root: {OUTPUT_ROOT}", flush=True)
    download_multicompartment_data()

    rng = np.random.default_rng(RANDOM_SEED)

    network_params = load_module(
        os.path.join(PARAMS_DIR, "network_params.py"),
        "hagen_network_params_massive",
    )
    simulation_params = load_module(
        os.path.join(PARAMS_DIR, "simulation_params.py"),
        "hagen_simulation_params_massive",
    )
    analysis_params = load_module(
        os.path.join(PARAMS_DIR, "analysis_params.py"),
        "hagen_analysis_params_massive",
    )

    transient = analysis_params.KernelParams.transient
    populations = list(network_params.LIF_params["X"])
    population_sizes = {
        pop: int(size) for pop, size in zip(populations, network_params.LIF_params["N_X"])
    }

    potential = FieldPotential()
    kernel_cache = None

    catch22_features = Features(method="catch22", params={"normalize": True})
    specparam_params = _specparam_params(1000.0 / (10.0 * float(simulation_params.dt)))
    specparam_features = Features(method="specparam", params=specparam_params)

    all_simulation_rows = []
    valid_detail_rows = []
    sim_data_exports = initialize_sim_data_exports()
    sampled_parameter_sets = [sample_parameters(rng) for _ in range(TOTAL_SIMULATIONS)]
    batch_parameter_sets = select_batch_parameter_sets(
        sampled_parameter_sets, args.batch_id, args.num_batches
    )
    plot_parameter_samples([params for _, params in batch_parameter_sets])
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
        lif_params = build_lif_parameters(network_params.LIF_params, sampled_params)

        temp_output_dir = tempfile.mkdtemp(prefix="hagen_massive_", dir=OUTPUT_ROOT)
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
                    valid=False,
                    valid_reasons=[runtime_info["status"]],
                )
            )
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            continue

        try:
            outputs = load_simulation_outputs(temp_output_dir)
            filtered_times, filtered_gids = apply_transient_filter(
                outputs["times"], outputs["gids"], transient
            )

            firing_rate_e = mean_firing_rate_hz(
                filtered_times["E"], population_sizes["E"], transient, outputs["tstop"]
            )
            firing_rate_i = mean_firing_rate_hz(
                filtered_times["I"], population_sizes["I"], transient, outputs["tstop"]
            )

            if kernel_cache is None:
                # The example pipeline uses fixed kernel settings, so the kernel can be
                # reused across samples with the same dt/tstop configuration.
                kernel_cache = compute_kernel(
                    potential, analysis_params.KernelParams, outputs["dt"], outputs["tstop"]
                )

            cdm_components, cdm_total, cdm_time, cdm_dt = compute_cdm(
                potential,
                kernel_cache,
                filtered_times,
                outputs["dt"],
                outputs["tstop"],
                transient,
                populations,
            )

            catch22_result = compute_catch22_features(catch22_features, cdm_total)
            specparam_result = compute_specparam_features(specparam_features, cdm_total)
            is_valid, invalid_reasons = evaluate_validity(
                firing_rate_e, firing_rate_i, specparam_result
            )

            sample_index = len(valid_detail_rows) if is_valid else None
            summary_row = flatten_summary_row(
                sample_index=sample_index,
                simulation_index=simulation_index,
                batch_id=args.batch_id,
                sampled_params=sampled_params,
                runtime_info=runtime_info,
                firing_rate_e=firing_rate_e,
                firing_rate_i=firing_rate_i,
                catch22_features=catch22_result,
                specparam_features=specparam_result,
                valid=is_valid,
                valid_reasons=invalid_reasons,
            )
            all_simulation_rows.append(summary_row)

            plot_simulation_debug(
                simulation_index=simulation_index,
                sampled_params=sampled_params,
                times=filtered_times,
                gids=filtered_gids,
                populations=populations,
                cdm_total=cdm_total,
                cdm_time=cdm_time,
                cdm_dt=cdm_dt,
                transient=transient,
                tstop=outputs["tstop"],
                firing_rate_e=firing_rate_e,
                firing_rate_i=firing_rate_i,
                specparam_result=specparam_result,
                is_valid=is_valid,
                invalid_reasons=invalid_reasons,
            )

            if is_valid:
                theta_row = build_theta_row(sampled_params)
                append_sim_data_export(
                    sim_data_exports,
                    "CDM",
                    theta_row,
                    build_cdm_row(cdm_total),
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
                        "data_dirs": {
                            method: os.path.join(OUTPUT_ROOT, "data", method)
                            for method in SIM_DATA_METHODS
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
