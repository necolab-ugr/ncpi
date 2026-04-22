import ast
import importlib
import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
from werkzeug.serving import make_server

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - exercised only when Playwright is absent.
    PlaywrightError = None
    sync_playwright = None


REPO_ROOT = Path(__file__).resolve().parents[3]
WEBUI_DIR = REPO_ROOT / "webui"
CAVALLARI_EXAMPLE_DIR = REPO_ROOT / "examples" / "simulation" / "Cavallari_model" / "LIF_simulation"
CAVALLARI_NEURON_MODEL_DIR = CAVALLARI_EXAMPLE_DIR.parent / "neuron_model"
CAVALLARI_RELATIVE_MARGIN = 0.10
CAVALLARI_ALTERNATE_VECTOR_SCALES = {
    "N_X": [1.10, 0.90],
    "C_m_X": [0.90, 1.10],
    "tau_m_X": [1.10, 0.90],
}
CAVALLARI_ALTERNATE_RECURRENT_SCALES = {
    "exc_inh_recurrent": 1.10,
    "inh_exc_recurrent": 0.90,
}
CAVALLARI_RECURRENT_GRID_KEYS = (
    "exc_exc_recurrent",
    "exc_inh_recurrent",
    "inh_exc_recurrent",
    "inh_inh_recurrent",
)
CAVALLARI_RUNTIME_CONTROL_KEYS = {"local_num_threads"}
TEST_SIMULATION_TSTOP_MS = 5000.0
TEST_SIMULATION_DT_MS = 0.2
PLAYWRIGHT_SHOW_BROWSER_SLOW_MO_MS = 750
_TERMINAL_REPORTER = None
_CAPTURE_MANAGER = None


def _write_progress_line(text):
    """Write a progress line through pytest's terminal reporter or stderr."""
    if _TERMINAL_REPORTER is not None:
        _TERMINAL_REPORTER.write_line(text)
        return
    sys.__stderr__.write(f"{text}\n")
    sys.__stderr__.flush()


def _log_test_progress(message):
    """Emit a namespaced progress message for this test module."""
    text = f"[test_cavallari] {message}"
    if _CAPTURE_MANAGER is not None:
        with _CAPTURE_MANAGER.global_and_fixture_disabled():
            _write_progress_line(text)
        return
    _write_progress_line(text)


def _ensure_import_paths():
    """Ensure the repository and WebUI directories are importable during the tests."""
    for entry in (str(REPO_ROOT), str(WEBUI_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _reload_webui_app():
    """Reload the WebUI app module with fresh module state for the current test module."""
    _ensure_import_paths()
    for module_name in ("webui.app", "compute_utils", "tmp_paths"):
        if module_name in sys.modules:
            del sys.modules[module_name]
    module = importlib.import_module("webui.app")
    module.refresh_tmp_paths()
    return module


def _set_fast_simulation_defaults(module):
    """Reduce default simulation runtime parameters for faster WebUI simulation tests."""
    for defaults_name in ("HAGEN_DEFAULTS", "CAVALLARI_DEFAULTS", "FOUR_AREA_DEFAULTS"):
        defaults = getattr(module, defaults_name, None)
        if isinstance(defaults, dict):
            defaults["tstop"] = TEST_SIMULATION_TSTOP_MS
            defaults["dt"] = TEST_SIMULATION_DT_MS


@contextmanager
def _playwright_temp_environment():
    # Keep Playwright browser profile paths short to avoid Chromium socket/path limits.
    """Temporarily shorten Playwright temp paths to avoid Chromium path-length issues."""
    temp_root = Path("/tmp/pw")
    temp_root.mkdir(parents=True, exist_ok=True)

    original_env = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
    original_tempdir = tempfile.tempdir
    try:
        short_tmp = str(temp_root)
        os.environ["TMPDIR"] = short_tmp
        os.environ["TMP"] = short_tmp
        os.environ["TEMP"] = short_tmp
        tempfile.tempdir = short_tmp
        yield short_tmp
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = original_tempdir


def _form_data_from_page(page):
    """Read the current run-simulation form values directly from the page."""
    return page.evaluate(
        """
        () => {
            const form = document.querySelector('#run-simulation-form');
            const data = {};
            for (const [key, value] of new FormData(form).entries()) {
                data[key] = value;
            }
            return data;
        }
        """
    )


def _normalized_form_data_from_page(page):
    """Collect browser-normalized FormData without leaving the current page."""
    return page.evaluate(
        """
        () => {
            const form = document.querySelector('#run-simulation-form');
            let blocked = false;
            const blockSubmission = (event) => {
                event.preventDefault();
                blocked = true;
            };
            form.addEventListener('submit', blockSubmission, { once: true });
            form.requestSubmit();

            if (!blocked) {
                throw new Error('Expected test helper to block form submission during normalization.');
            }

            const data = {};
            for (const [key, value] of new FormData(form).entries()) {
                data[key] = value;
            }
            return data;
        }
        """
    )


def _apply_fast_simulation_runtime_to_page(page):
    """Override the browser-side runtime fields so tests avoid the long JS preset values."""
    _cavallari_param_input(page, "tstop").fill(str(TEST_SIMULATION_TSTOP_MS))
    _cavallari_param_input(page, "dt").fill(str(TEST_SIMULATION_DT_MS))


def _wait_for_job_completion(base_url, job_id, timeout_seconds=1800):
    """Poll the WebUI job endpoint until the job finishes or times out."""
    status_url = f"{base_url}/status/{job_id}"
    deadline = time.time() + timeout_seconds
    last_payload = None
    last_status = None
    last_progress = None
    last_report_time = 0.0

    while time.time() < deadline:
        with urllib.request.urlopen(status_url) as response:
            last_payload = json.load(response)
        current_status = last_payload.get("status")
        current_progress = last_payload.get("progress")
        now = time.time()
        if (
            current_status != last_status
            or current_progress != last_progress
            or now - last_report_time >= 10.0
        ):
            _log_test_progress(
                f"webui job {job_id}: status={current_status}, progress={current_progress}"
            )
            last_status = current_status
            last_progress = current_progress
            last_report_time = now
        if last_payload["status"] in {"finished", "failed", "cancelled"}:
            return last_payload
        time.sleep(1.5)

    raise TimeoutError(f"Timed out waiting for simulation job {job_id}. Last payload: {last_payload}")


def _download_simulation_zip(base_url, job_id, destination):
    """Download the simulation-results ZIP for a completed WebUI job."""
    download_url = f"{base_url}/download_results/{job_id}?computation_type=simulation"
    with urllib.request.urlopen(download_url) as response:
        destination.write_bytes(response.read())
    return destination


def _extract_zip(zip_path, destination):
    """Extract a ZIP archive into the requested destination directory."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    return destination


def _load_output_pickle(output_dir, filename):
    """Load a simulation output pickle file from a result directory."""
    with open(output_dir / filename, "rb") as handle:
        return pickle.load(handle)


def _coerce_trial_payloads(payload):
    """Normalize single-trial and repeated-trial payloads to a list."""
    return payload if isinstance(payload, list) else [payload]


def _mean_firing_rate_by_bin_dataframe_from_payload(times, network, tstop, bin_width_ms=100.0):
    """Convert one simulation payload into a binned mean firing-rate DataFrame."""
    populations = list(network["X"])
    population_sizes = {
        str(population): int(size)
        for population, size in zip(populations, network["N_X"])
    }
    bin_edges = np.arange(0.0, tstop + bin_width_ms, bin_width_ms, dtype=float)
    if bin_edges.size < 2 or bin_edges[-1] < tstop:
        bin_edges = np.append(bin_edges, tstop)
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]
    bin_width_seconds = (bin_ends - bin_starts) / 1000.0

    rows = []
    for population in populations:
        population_key = str(population)
        spike_times = np.asarray(times[population], dtype=float)
        spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
        mean_firing_rate_hz = spike_counts / population_sizes[population_key] / bin_width_seconds
        rows.extend(
            {
                "population": population_key,
                "bin_start_ms": float(start_ms),
                "bin_end_ms": float(end_ms),
                "mean_firing_rate_hz": float(rate_hz),
            }
            for start_ms, end_ms, rate_hz in zip(bin_starts, bin_ends, mean_firing_rate_hz)
        )

    return pd.DataFrame(rows).sort_values(["population", "bin_start_ms"], kind="mergesort").reset_index(drop=True)


def _mean_firing_rate_by_bin_dataframes(output_dir, bin_width_ms=100.0):
    """Build per-trial mean firing-rate DataFrames from a simulation output directory."""
    times_payloads = _coerce_trial_payloads(_load_output_pickle(output_dir, "times.pkl"))
    network_payloads = _coerce_trial_payloads(_load_output_pickle(output_dir, "network.pkl"))
    tstop_payloads = _coerce_trial_payloads(_load_output_pickle(output_dir, "tstop.pkl"))

    trial_count = len(times_payloads)
    if len(network_payloads) != trial_count or len(tstop_payloads) != trial_count:
        raise AssertionError(
            "Inconsistent repeated-trial payload sizes across times.pkl, network.pkl, and tstop.pkl."
        )

    return [
        _mean_firing_rate_by_bin_dataframe_from_payload(
            times,
            network,
            float(tstop),
            bin_width_ms=bin_width_ms,
        )
        for times, network, tstop in zip(times_payloads, network_payloads, tstop_payloads)
    ]


def _prepare_cavallari_python_reference_environment(output_root):
    """Create the standalone Cavallari reference-run folder structure and inputs."""
    neuron_model_target = output_root / "neuron_model"
    if neuron_model_target.exists():
        return
    shutil.copytree(CAVALLARI_NEURON_MODEL_DIR, neuron_model_target)


def _run_python_cavallari_reference(app_module, form_data, output_dir, expected_trial=None):
    """Run the direct Python Cavallari reference simulation for comparison with the WebUI output."""
    params_dir = output_dir.parent / "params"
    python_dir = output_dir.parent / "python"
    params_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_label = output_dir.parent.name
    _log_test_progress(f"preparing python reference for {trial_label} in {output_dir}")

    (params_dir / "network_params.py").write_text(
        app_module._build_cavallari_network_params(form_data),
        encoding="utf-8",
    )
    (params_dir / "simulation_params.py").write_text(
        app_module._build_simulation_params(form_data, app_module.CAVALLARI_DEFAULTS),
        encoding="utf-8",
    )

    shutil.copy2(CAVALLARI_EXAMPLE_DIR / "python" / "simulation.py", python_dir / "simulation.py")
    app_module._enforce_simulation_chunk_seconds(str(python_dir / "simulation.py"), chunk_ms=1000.0)

    network_payload = (
        expected_trial["network_pickle"]
        if expected_trial is not None
        else app_module._build_cavallari_lif_params(form_data)
    )
    with open(output_dir / "network.pkl", "wb") as handle:
        pickle.dump(network_payload, handle)

    import ncpi

    simulation = ncpi.Simulation(
        param_folder=str(params_dir),
        python_folder=str(python_dir),
        output_folder=str(output_dir),
    )
    _log_test_progress(f"running python simulation for {trial_label}")
    simulation.simulate("simulation.py", "simulation_params.py")
    _log_test_progress(f"completed python reference for {trial_label}")

    return output_dir


def _run_python_cavallari_reference_dataframes(app_module, form_data, output_root, bin_width_ms=100.0):
    """Run the direct Python Cavallari reference simulation and summarize its outputs."""
    _, expanded_forms = app_module._expand_simulation_forms("cavallari", form_data)
    expected_trials = _independent_expected_cavallari_trials(app_module, form_data)
    total_trials = len(expanded_forms)
    assert len(expected_trials) == total_trials

    _prepare_cavallari_python_reference_environment(output_root)

    trial_dataframes = []
    for trial_index, (expanded_form, expected_trial) in enumerate(zip(expanded_forms, expected_trials)):
        combo_index = int(expanded_form.get("sim_combo_index", trial_index))
        repeat_index = int(expanded_form.get("sim_repeat_index", 0))
        _log_test_progress(
            f"python reference trial {trial_index + 1}/{total_trials} "
            f"(configuration={combo_index}, repetition={repeat_index})"
        )
        output_dir = _run_python_cavallari_reference(
            app_module,
            expanded_form,
            output_root / f"trial_{trial_index}" / "output",
            expected_trial=expected_trial,
        )
        dataframes = _mean_firing_rate_by_bin_dataframes(output_dir, bin_width_ms=bin_width_ms)
        assert len(dataframes) == 1
        trial_dataframes.append(dataframes[0])

    return trial_dataframes


def _independent_grid_range_candidates(spec, key):
    """Parse a numeric grid-range specification without relying on WebUI helper code."""
    parts = [part.strip() for part in str(spec).split(":")]
    if len(parts) != 3:
        raise ValueError(f"Invalid independent grid range for {key}: {spec!r}")
    start = float(parts[0])
    stop = float(parts[1])
    step = float(parts[2])
    if step == 0:
        raise ValueError(f"Invalid independent grid range for {key}: step cannot be zero.")

    values = []
    current = start
    eps = abs(step) * 1e-9 + 1e-12
    if step > 0:
        while current <= stop + eps:
            values.append(round(current, 12))
            current += step
    else:
        while current >= stop - eps:
            values.append(round(current, 12))
            current += step
    return values


def _independent_parse_cavallari_value(raw_value, default):
    """Parse a Cavallari form value independently from the WebUI implementation."""
    if raw_value is None:
        return default
    text = str(raw_value).strip()
    if text == "":
        return default
    if isinstance(default, str):
        return text
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(text))
    if isinstance(default, float):
        return float(text)
    return ast.literal_eval(text)


def _independent_parse_cavallari_grid_candidates(raw_value, default, key):
    """Parse Cavallari grid candidates independently from the WebUI implementation."""
    if raw_value is None:
        return [default]
    text = str(raw_value).strip()
    if text == "":
        return [default]
    if not text.lower().startswith("grid="):
        return [_independent_parse_cavallari_value(text, default)]

    spec = text[5:].strip()
    if isinstance(default, (int, float)) and ":" in spec and not any(ch in spec for ch in "[]{}(),"):
        candidates = _independent_grid_range_candidates(spec, key)
    else:
        parsed = ast.literal_eval(spec)
        candidates = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]

    if isinstance(default, int) and not isinstance(default, bool):
        return [int(float(candidate)) for candidate in candidates]
    if isinstance(default, float):
        return [float(candidate) for candidate in candidates]
    return candidates


def _independent_values_equal(left, right, tol=1e-12):
    """Compare nested values recursively while tolerating tiny floating-point differences."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_independent_values_equal(left[key], right[key], tol=tol) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_independent_values_equal(lv, rv, tol=tol) for lv, rv in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tol
    return left == right


def _expected_generated_simulation_params(app_module, *, tstop, local_num_threads, dt, numpy_seed):
    """Build the exact simulation_params.py namespace expected from the WebUI."""
    params = {
        "tstop": tstop,
        "local_num_threads": local_num_threads,
        "dt": dt,
        "numpy_seed": numpy_seed,
    }
    spike_rate_limit = getattr(app_module, "WEBUI_MAX_MEAN_POPULATION_SPIKE_RATE_HZ", None)
    if spike_rate_limit is not None:
        params["max_mean_population_spike_rate_hz"] = float(spike_rate_limit)
    return params


def _independent_expected_cavallari_trials(app_module, form_data):
    """Derive the expected Cavallari trial configurations directly from submitted form data."""
    defaults = app_module.CAVALLARI_DEFAULTS
    run_mode = str(form_data.get("sim_run_mode", "single")).strip().lower() or "single"
    repetitions = int(float(str(form_data.get("sim_repetitions", "1")).strip() or "1"))

    grid_keys = [
        key for key in app_module.CAVALLARI_GRID_KEYS
        if key in defaults
    ]
    candidate_lists = {
        key: _independent_parse_cavallari_grid_candidates(form_data.get(key), defaults[key], key)
        for key in grid_keys
    }
    numpy_seed = None
    if form_data.get("sim_use_numpy_seed") is not None:
        numpy_seed = int(float(str(form_data.get("sim_numpy_seed", "0")).strip() or "0"))

    if run_mode == "single":
        base_values = {
            key: candidate_lists[key][0]
            for key in grid_keys
        }
        combos = [base_values]
    elif run_mode == "grid":
        ordered_keys = list(candidate_lists)
        combos = [
            {
                key: value
                for key, value in zip(ordered_keys, combo_values)
            }
            for combo_values in product(*(candidate_lists[key] for key in ordered_keys))
        ]
    else:
        raise ValueError(f"Unsupported independent Cavallari run mode: {run_mode}")

    expected_trials = []
    for combo_index, values in enumerate(combos):
        n_x = [int(values["N_X"][0]), int(values["N_X"][1])]
        network_params = {
            "N_exc": n_x[0],
            "N_inh": n_x[1],
            "P": float(values["P"]),
            "extent": float(values["extent"]),
            "exc_exc_recurrent": float(values["exc_exc_recurrent"]),
            "exc_inh_recurrent": float(values["exc_inh_recurrent"]),
            "inh_exc_recurrent": float(values["inh_exc_recurrent"]),
            "inh_inh_recurrent": float(values["inh_inh_recurrent"]),
            "th_exc_external": float(values["th_exc_external"]),
            "th_inh_external": float(values["th_inh_external"]),
            "cc_exc_external": float(values["cc_exc_external"]),
            "cc_inh_external": float(values["cc_inh_external"]),
            "v_0": float(values["v_0"]),
            "A_ext": float(values["A_ext"]),
            "f_ext": float(values["f_ext"]),
            "OU_sigma": float(values["OU_sigma"]),
            "OU_tau": float(values["OU_tau"]),
        }
        excitatory_cell_params = {
            "V_th": float(values["V_th_X"][0]),
            "V_reset": float(values["V_reset_X"][0]),
            "t_ref": float(values["t_ref_X"][0]),
            "g_L": float(values["g_L_X"][0]),
            "C_m": float(values["C_m_X"][0]),
            "E_ex": float(values["E_ex_X"][0]),
            "E_in": float(values["E_in_X"][0]),
            "E_L": float(values["E_L_X"][0]),
            "tau_rise_AMPA": float(values["tau_rise_AMPA_X"][0]),
            "tau_decay_AMPA": float(values["tau_decay_AMPA_X"][0]),
            "tau_rise_GABA_A": float(values["tau_rise_GABA_A_X"][0]),
            "tau_decay_GABA_A": float(values["tau_decay_GABA_A_X"][0]),
            "tau_m": float(values["tau_m_X"][0]),
            "I_e": float(values["I_e_X"][0]),
        }
        inhibitory_cell_params = {
            "V_th": float(values["V_th_X"][1]),
            "V_reset": float(values["V_reset_X"][1]),
            "t_ref": float(values["t_ref_X"][1]),
            "g_L": float(values["g_L_X"][1]),
            "C_m": float(values["C_m_X"][1]),
            "E_ex": float(values["E_ex_X"][1]),
            "E_in": float(values["E_in_X"][1]),
            "E_L": float(values["E_L_X"][1]),
            "tau_rise_AMPA": float(values["tau_rise_AMPA_X"][1]),
            "tau_decay_AMPA": float(values["tau_decay_AMPA_X"][1]),
            "tau_rise_GABA_A": float(values["tau_rise_GABA_A_X"][1]),
            "tau_decay_GABA_A": float(values["tau_decay_GABA_A_X"][1]),
            "tau_m": float(values["tau_m_X"][1]),
            "I_e": float(values["I_e_X"][1]),
        }

        for repeat_index in range(repetitions):
            expected_trials.append({
                "combo_index": combo_index,
                "repeat_index": repeat_index,
                "network_file": {
                    "Network_params": network_params,
                    "excitatory_cell_params": excitatory_cell_params,
                    "inhibitory_cell_params": inhibitory_cell_params,
                    "Neuron_params": [excitatory_cell_params, inhibitory_cell_params],
                },
                "network_pickle": {
                    "X": list(values["X"]),
                    "N_X": n_x,
                    "model": values["model"],
                    "neuron_params": {
                        "E": excitatory_cell_params,
                        "I": inhibitory_cell_params,
                    },
                    "network_params": network_params,
                },
                "simulation": _expected_generated_simulation_params(
                    app_module,
                    tstop=float(values["tstop"]),
                    local_num_threads=int(values["local_num_threads"]),
                    dt=float(values["dt"]),
                    numpy_seed=numpy_seed,
                ),
            })
    return expected_trials


def _load_generated_python_namespace(py_path):
    """Execute a generated Python parameter file and return its public namespace."""
    namespace = {}
    exec(py_path.read_text(encoding="utf-8"), {}, namespace)
    return {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__")
    }


def _capture_webui_generated_param_files(app_module, monkeypatch, capture_root):
    """Patch the backend runner so each generated parameter-file pair is snapshotted."""
    original_run_process_with_progress = app_module._run_process_with_progress
    captured_param_dirs = []

    def wrapped_run_process_with_progress(cmd, cwd, job_status, job_id, estimate_seconds, **kwargs):
        """Snapshot generated parameter files before delegating to the real process runner."""
        params_dir = Path(cwd) / "params"
        snapshot_dir = capture_root / f"run_{len(captured_param_dirs)}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(params_dir / "network_params.py", snapshot_dir / "network_params.py")
        shutil.copy2(params_dir / "simulation_params.py", snapshot_dir / "simulation_params.py")
        captured_param_dirs.append(snapshot_dir)
        return original_run_process_with_progress(cmd, cwd, job_status, job_id, estimate_seconds, **kwargs)

    monkeypatch.setattr(app_module, "_run_process_with_progress", wrapped_run_process_with_progress)
    return captured_param_dirs


def _assert_cavallari_generated_parameter_files_match_expected(app_module, form_data, captured_param_dirs):
    """Verify that generated Cavallari parameter files match the independently derived expectations."""
    expected_trials = _independent_expected_cavallari_trials(app_module, form_data)
    assert len(captured_param_dirs) == len(expected_trials), (
        f"Captured {len(captured_param_dirs)} WebUI parameter-file snapshots, "
        f"expected {len(expected_trials)}."
    )

    for trial_index, (param_dir, expected_trial) in enumerate(zip(captured_param_dirs, expected_trials), start=1):
        actual_network_namespace = _load_generated_python_namespace(param_dir / "network_params.py")
        actual_network_file = {
            "Network_params": actual_network_namespace["Network_params"],
            "excitatory_cell_params": actual_network_namespace["excitatory_cell_params"],
            "inhibitory_cell_params": actual_network_namespace["inhibitory_cell_params"],
            "Neuron_params": actual_network_namespace["Neuron_params"],
        }
        actual_simulation = _load_generated_python_namespace(param_dir / "simulation_params.py")

        assert _independent_values_equal(actual_network_file, expected_trial["network_file"]), (
            f"Unexpected WebUI-generated network parameters for trial {trial_index}: "
            f"actual={actual_network_file!r}, expected={expected_trial['network_file']!r}"
        )
        assert _independent_values_equal(actual_simulation, expected_trial["simulation"]), (
            f"Unexpected WebUI-generated simulation parameters for trial {trial_index}: "
            f"actual={actual_simulation!r}, expected={expected_trial['simulation']!r}"
        )


def _navigate_to_cavallari_form(page, live_webui_server):
    """Navigate to the Cavallari simulation form and wait for its controls to finish loading."""
    page.goto(live_webui_server, wait_until="domcontentloaded")

    page.locator("a[href='/simulation']").first.click()
    page.wait_for_url(f"{live_webui_server}/simulation", wait_until="domcontentloaded")

    page.locator("a[href='/simulation/new_sim']").first.click()
    page.wait_for_url(f"{live_webui_server}/simulation/new_sim", wait_until="domcontentloaded")

    page.locator("a[href='/simulation/new_sim/cavallari']").first.click()
    page.wait_for_url(f"{live_webui_server}/simulation/new_sim/cavallari", wait_until="domcontentloaded")
    page.wait_for_function(
        """
        () => {
            const container = document.querySelector('#parameter-container');
            const nXInput = document.querySelector('.param-input[data-param="N_X"]');
            const recurrentInput = document.querySelector('.param-input[data-param="exc_exc_recurrent"]');
            return Boolean(
                container &&
                container.querySelector('.parameter-section') &&
                nXInput &&
                nXInput.parentElement?.querySelector('.single-array-controls') &&
                recurrentInput &&
                recurrentInput.parentElement?.querySelector('.grid-parameter-controls')
            );
        }
        """
    )


def _cavallari_param_input(page, param_name):
    """Return the locator for a Cavallari parameter input field."""
    return page.locator(f'.param-input[data-param="{param_name}"]').first


def _cavallari_single_array_rows(page, param_name):
    """Return the single-value array input rows for a Cavallari parameter."""
    input_locator = _cavallari_param_input(page, param_name)
    input_locator.wait_for(state="attached")
    return input_locator.locator(
        "xpath=following-sibling::*[contains(@class, 'single-array-controls')][1]"
    ).locator('[data-single-array-row="1"] input')


def _cavallari_grid_rows(page, param_name):
    """Return the grid-control rows for a Cavallari parameter."""
    input_locator = _cavallari_param_input(page, param_name)
    input_locator.wait_for(state="attached")
    return input_locator.locator(
        "xpath=following-sibling::*[contains(@class, 'grid-parameter-controls')][1]"
    ).locator('[data-grid-row="1"]')


def _fill_cavallari_scalar_param(page, param_name, value):
    """Fill a scalar Cavallari parameter input."""
    _cavallari_param_input(page, param_name).fill(str(value))


def _fill_cavallari_array_param(page, param_name, values):
    """Fill all leaf inputs for an array-valued Cavallari parameter."""
    rows = _cavallari_single_array_rows(page, param_name)
    page.wait_for_function(
        """
        ([selector, expectedCount]) => {
            const input = document.querySelector(selector);
            if (!input || !input.parentElement) {
                return false;
            }
            return input.parentElement.querySelectorAll('.single-array-controls [data-single-array-row="1"] input').length === expectedCount;
        }
        """,
        arg=[f'.param-input[data-param="{param_name}"]', len(values)],
    )
    assert rows.count() == len(values), f"Unexpected leaf count for {param_name}: {rows.count()}"
    for idx, value in enumerate(values):
        rows.nth(idx).fill(str(value))


def _fill_cavallari_grid_scalar_param(page, param_name, start, step, end):
    """Fill a scalar Cavallari grid-definition row."""
    row = _cavallari_grid_rows(page, param_name).first
    row.wait_for(state="visible")
    row.locator('[data-grid-role="start"]').fill(str(start))
    row.locator('[data-grid-role="step"]').fill(str(step))
    row.locator('[data-grid-role="end"]').fill(str(end))


def _scale_cavallari_parameter(value, factor):
    """Scale a Cavallari parameter value while preserving integer semantics when needed."""
    scaled = float(value) * float(factor)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(round(scaled))
    return round(scaled, 12)


def _build_cavallari_alternate_single_values(defaults):
    """Build alternate single-run Cavallari values for parameter-override tests."""
    alternate = {
        param_name: [
            _scale_cavallari_parameter(value, factor)
            for value, factor in zip(defaults[param_name], factors)
        ]
        for param_name, factors in CAVALLARI_ALTERNATE_VECTOR_SCALES.items()
    }
    for param_name, factor in CAVALLARI_ALTERNATE_RECURRENT_SCALES.items():
        alternate[param_name] = _scale_cavallari_parameter(defaults[param_name], factor)
    return alternate


def _apply_cavallari_single_parameter_overrides(page, overrides):
    """Apply the requested single-run Cavallari parameter overrides in the browser form."""
    for param_name, value in overrides.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            flat_values = np.ravel(value).tolist()
            _fill_cavallari_array_param(page, param_name, flat_values)
            continue
        _fill_cavallari_scalar_param(page, param_name, value)


def _expected_cavallari_form_values(app_module, overrides=None):
    """Build the expected normalized Cavallari form values for a submission."""
    expected_values = {
        key: app_module.CAVALLARI_DEFAULTS[key]
        for key in app_module.CAVALLARI_GRID_KEYS
        if key in app_module.CAVALLARI_DEFAULTS
    }
    if overrides:
        expected_values.update(overrides)
    return expected_values


def _assert_cavallari_form_values(app_module, form_data, overrides=None, ignored_keys=None):
    """Verify that the normalized submitted Cavallari form values match expectations."""
    ignored_keys = set(ignored_keys or ())
    ignored_keys.update(CAVALLARI_RUNTIME_CONTROL_KEYS)
    expected_values = _expected_cavallari_form_values(app_module, overrides=overrides)
    normalized_form_values = app_module._normalize_simulation_form_values(form_data, expected_values)
    for key, expected_value in expected_values.items():
        if key in ignored_keys or key not in form_data:
            continue
        actual_value = normalized_form_values[key]
        assert app_module._values_equal_for_grid(actual_value, expected_value), (
            f"Unexpected normalized value for {key}: raw={form_data[key]!r}, parsed={actual_value!r}, expected={expected_value!r}"
        )


def _expanded_cavallari_trials(app_module, form_data):
    """Expand Cavallari form data into normalized per-run trial configurations."""
    run_mode, expanded_forms = app_module._expand_simulation_forms("cavallari", form_data)
    defaults = app_module._simulation_grid_defaults("cavallari")
    normalized_trials = [
        app_module._normalize_simulation_form_values(form, defaults)
        for form in expanded_forms
    ]
    return run_mode, expanded_forms, normalized_trials


def _recurrent_signature(values):
    """Build a comparable signature for Cavallari recurrent-weight parameters."""
    return tuple(float(values[key]) for key in CAVALLARI_RECURRENT_GRID_KEYS)


def _build_cavallari_recurrent_margin_candidates(defaults):
    """Build boundary candidate values for the Cavallari recurrent-weight grid sweep."""
    candidates = {}
    for key in CAVALLARI_RECURRENT_GRID_KEYS:
        default_value = defaults[key]
        candidates[key] = (
            _scale_cavallari_parameter(default_value, 1.0 - CAVALLARI_RELATIVE_MARGIN),
            _scale_cavallari_parameter(default_value, 1.0 + CAVALLARI_RELATIVE_MARGIN),
        )
    expected_signatures = [
        tuple(float(value) for value in combo_values)
        for combo_values in product(*(candidates[key] for key in CAVALLARI_RECURRENT_GRID_KEYS))
    ]
    return candidates, expected_signatures


def _run_cavallari_webui_job(live_webui_server, pytestconfig, configure_page):
    """Use Playwright to submit a Cavallari simulation job and wait for completion."""
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=PLAYWRIGHT_SHOW_BROWSER_SLOW_MO_MS if show_browser else 0,
                )
            except PlaywrightError as exc:
                pytest.skip(f"Playwright Chromium is not available: {exc}")

            try:
                page = browser.new_page()
                _log_test_progress(f"opening Cavallari form at {live_webui_server}")
                _navigate_to_cavallari_form(page, live_webui_server)
                _apply_fast_simulation_runtime_to_page(page)
                _log_test_progress("configuring Cavallari form")
                configure_page(page)
                form_data = _normalized_form_data_from_page(page)
                _log_test_progress("submitting WebUI simulation")

                page.locator("button[type='submit']").click()
                page.wait_for_url("**/job_status/*", wait_until="domcontentloaded")
                job_id = Path(urlparse(page.url).path).name
                _log_test_progress(f"submitted WebUI simulation as job {job_id}")
                status_payload = _wait_for_job_completion(live_webui_server, job_id)
                assert status_payload["status"] == "finished", status_payload
                _log_test_progress(f"WebUI job {job_id} finished successfully")
            finally:
                browser.close()

    return form_data, job_id


def _assert_cavallari_webui_matches_python_reference(
    app_module,
    live_webui_server,
    job_id,
    form_data,
    tmp_path,
    expected_trial_count,
    bin_width_ms=100.0,
):
    """Compare Cavallari WebUI outputs against the direct Python reference outputs."""
    _log_test_progress(f"downloading WebUI results for job {job_id}")
    ui_zip_path = _download_simulation_zip(
        live_webui_server,
        job_id,
        tmp_path / "webui_simulation_results.zip",
    )
    ui_output_dir = _extract_zip(ui_zip_path, tmp_path / "webui_results")
    _log_test_progress("computing firing-rate summaries from WebUI output")
    webui_firing_rate_dfs = _mean_firing_rate_by_bin_dataframes(ui_output_dir, bin_width_ms=bin_width_ms)
    _log_test_progress("running direct Python reference simulations")
    python_firing_rate_dfs = _run_python_cavallari_reference_dataframes(
        app_module,
        form_data,
        tmp_path / "python_reference",
        bin_width_ms=bin_width_ms,
    )

    assert len(webui_firing_rate_dfs) == expected_trial_count
    assert len(python_firing_rate_dfs) == expected_trial_count

    _log_test_progress(f"comparing {expected_trial_count} WebUI/Python firing-rate summaries")
    for trial_index, (webui_df, python_df) in enumerate(zip(webui_firing_rate_dfs, python_firing_rate_dfs), start=1):
        _log_test_progress(f"comparing trial {trial_index}/{expected_trial_count}")
        pd.testing.assert_frame_equal(webui_df, python_df)
    _log_test_progress("WebUI and Python outputs match")


@pytest.fixture(scope="module")
def webui_app_module():
    """Provide a freshly reloaded WebUI app module for the current test module."""
    if importlib.util.find_spec("nest") is None:
        pytest.skip("NEST is required for the Cavallari web UI simulation test.")
    if sync_playwright is None:
        pytest.skip("Playwright is required for the web UI simulation tests.")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("NCPI_WEBUI_SESSION_ID", f"pytest_webui_simulation_{uuid.uuid4().hex}")
    monkeypatch.delenv("NCPI_WEBUI_SESSION_ROOT", raising=False)
    module = _reload_webui_app()
    _set_fast_simulation_defaults(module)
    module._clear_simulation_output_folder_all_files()
    yield module
    module._clear_simulation_output_folder_all_files()
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def _attach_test_cavallari_terminal_reporter(pytestconfig):
    """Attach pytest terminal-reporting plugins so progress logs remain visible."""
    global _TERMINAL_REPORTER, _CAPTURE_MANAGER
    previous_reporter = _TERMINAL_REPORTER
    previous_capture_manager = _CAPTURE_MANAGER
    _TERMINAL_REPORTER = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    _CAPTURE_MANAGER = pytestconfig.pluginmanager.get_plugin("capturemanager")
    try:
        yield
    finally:
        _TERMINAL_REPORTER = previous_reporter
        _CAPTURE_MANAGER = previous_capture_manager


@pytest.fixture()
def live_webui_server(webui_app_module, pytestconfig):
    """Serve the WebUI app on a temporary local HTTP server for a single test."""
    server_port = int(pytestconfig.getoption("webui_port") or 0)
    server = make_server("127.0.0.1", server_port, webui_app_module.app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.mark.slow
def test_cavallari_default_configuration_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig, monkeypatch
):
    """Verify that cavallari default configuration matches direct Python run."""
    _log_test_progress("starting default Cavallari WebUI vs Python comparison")
    webui_app_module._clear_simulation_output_folder_all_files()
    captured_param_dirs = _capture_webui_generated_param_files(
        webui_app_module,
        monkeypatch,
        tmp_path / "captured_webui_params",
    )
    form_data, job_id = _run_cavallari_webui_job(
        live_webui_server,
        pytestconfig,
        lambda page: page.locator("#sim-use-numpy-seed").check(),
    )
    run_mode, expanded_forms, _ = _expanded_cavallari_trials(webui_app_module, form_data)
    assert run_mode == "single"
    assert len(expanded_forms) == 1
    _assert_cavallari_form_values(webui_app_module, form_data)
    _assert_cavallari_generated_parameter_files_match_expected(webui_app_module, form_data, captured_param_dirs)

    _assert_cavallari_webui_matches_python_reference(
        webui_app_module,
        live_webui_server,
        job_id,
        form_data,
        tmp_path,
        expected_trial_count=1,
    )


@pytest.mark.slow
def test_cavallari_alternate_configuration_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig, monkeypatch
):
    """Verify that cavallari alternate configuration matches direct Python run."""
    _log_test_progress("starting alternate-parameter Cavallari WebUI vs Python comparison")
    webui_app_module._clear_simulation_output_folder_all_files()
    alternate_values = _build_cavallari_alternate_single_values(webui_app_module.CAVALLARI_DEFAULTS)
    captured_param_dirs = _capture_webui_generated_param_files(
        webui_app_module,
        monkeypatch,
        tmp_path / "captured_webui_params",
    )

    def configure_page(page):
        """Apply the test-specific form changes before submitting the WebUI job."""
        page.locator("#sim-use-numpy-seed").check()
        _apply_cavallari_single_parameter_overrides(page, alternate_values)

    form_data, job_id = _run_cavallari_webui_job(
        live_webui_server,
        pytestconfig,
        configure_page,
    )
    run_mode, expanded_forms, _ = _expanded_cavallari_trials(webui_app_module, form_data)
    assert run_mode == "single"
    assert len(expanded_forms) == 1
    _assert_cavallari_form_values(webui_app_module, form_data, overrides=alternate_values)
    _assert_cavallari_generated_parameter_files_match_expected(webui_app_module, form_data, captured_param_dirs)

    _assert_cavallari_webui_matches_python_reference(
        webui_app_module,
        live_webui_server,
        job_id,
        form_data,
        tmp_path,
        expected_trial_count=1,
    )


@pytest.mark.slow
def test_cavallari_alternate_configuration_with_three_repetitions_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig, monkeypatch
):
    """Verify that cavallari alternate configuration with three repetitions matches direct Python run."""
    _log_test_progress("starting alternate-parameter Cavallari comparison with 3 repetitions")
    webui_app_module._clear_simulation_output_folder_all_files()
    alternate_values = _build_cavallari_alternate_single_values(webui_app_module.CAVALLARI_DEFAULTS)
    captured_param_dirs = _capture_webui_generated_param_files(
        webui_app_module,
        monkeypatch,
        tmp_path / "captured_webui_params",
    )

    def configure_page(page):
        """Apply the test-specific form changes before submitting the WebUI job."""
        page.locator("#sim-use-numpy-seed").check()
        _apply_cavallari_single_parameter_overrides(page, alternate_values)
        page.locator("#sim-repetitions").fill("3")

    form_data, job_id = _run_cavallari_webui_job(
        live_webui_server,
        pytestconfig,
        configure_page,
    )
    run_mode, expanded_forms, normalized_trials = _expanded_cavallari_trials(webui_app_module, form_data)
    assert run_mode == "single"
    assert len(expanded_forms) == 3
    _assert_cavallari_form_values(webui_app_module, form_data, overrides=alternate_values)
    assert all(trial == normalized_trials[0] for trial in normalized_trials[1:])
    _assert_cavallari_generated_parameter_files_match_expected(webui_app_module, form_data, captured_param_dirs)

    _assert_cavallari_webui_matches_python_reference(
        webui_app_module,
        live_webui_server,
        job_id,
        form_data,
        tmp_path,
        expected_trial_count=3,
    )


@pytest.mark.slow
def test_cavallari_recurrent_weight_grid_sweep_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig, monkeypatch
):
    """Verify that cavallari recurrent weight grid sweep matches direct Python run."""
    _log_test_progress("starting Cavallari recurrent-weight grid sweep WebUI vs Python comparison")
    webui_app_module._clear_simulation_output_folder_all_files()
    recurrent_candidates, expected_recurrent_signatures = _build_cavallari_recurrent_margin_candidates(
        webui_app_module.CAVALLARI_DEFAULTS
    )
    captured_param_dirs = _capture_webui_generated_param_files(
        webui_app_module,
        monkeypatch,
        tmp_path / "captured_webui_params",
    )

    def configure_page(page):
        """Apply the test-specific form changes before submitting the WebUI job."""
        page.locator("#sim-use-numpy-seed").check()
        page.locator("input[name='sim_run_mode'][value='grid']").check()
        for param_name in CAVALLARI_RECURRENT_GRID_KEYS:
            start, end = recurrent_candidates[param_name]
            _fill_cavallari_grid_scalar_param(
                page,
                param_name,
                start=float(start),
                step=float(end) - float(start),
                end=float(end),
            )

    form_data, job_id = _run_cavallari_webui_job(
        live_webui_server,
        pytestconfig,
        configure_page,
    )
    run_mode, expanded_forms, normalized_trials = _expanded_cavallari_trials(webui_app_module, form_data)
    assert run_mode == "grid"
    assert len(expanded_forms) == 2 ** 4
    _assert_cavallari_form_values(
        webui_app_module,
        form_data,
        ignored_keys=set(CAVALLARI_RECURRENT_GRID_KEYS),
    )

    grid_metadata = webui_app_module._build_simulation_grid_metadata("cavallari", run_mode, expanded_forms)
    assert grid_metadata is not None
    assert grid_metadata["changed_keys"] == list(CAVALLARI_RECURRENT_GRID_KEYS)
    assert grid_metadata["trial_count"] == 2 ** 4
    assert grid_metadata["configuration_count"] == 2 ** 4
    assert grid_metadata["repetitions_per_configuration"] == 1
    assert {
        _recurrent_signature(trial)
        for trial in normalized_trials
    } == set(expected_recurrent_signatures)
    _assert_cavallari_generated_parameter_files_match_expected(webui_app_module, form_data, captured_param_dirs)

    _assert_cavallari_webui_matches_python_reference(
        webui_app_module,
        live_webui_server,
        job_id,
        form_data,
        tmp_path,
        expected_trial_count=2 ** 4,
    )
