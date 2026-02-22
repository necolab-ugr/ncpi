import tempfile
import os
import shutil
import subprocess
import ast
import threading
import glob
import base64
from collections import deque

# Folder for temporary files > 5 GB
# Set BEFORE any flask imports if possible
def _resolve_tempdir():
    candidates = []
    env_tmp = os.environ.get('TMPDIR')
    if env_tmp:
        candidates.append(env_tmp)
    candidates.extend(['/home/necolab/tmp', '/tmp'])
    for path in candidates:
        if os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK):
            return path
    return tempfile.gettempdir()

_tempdir = _resolve_tempdir()
tempfile.tempdir = _tempdir
os.environ['TMPDIR'] = _tempdir

# Temporary folder for uploaded files of forms
temp_uploaded_files = 'temp_uploaded_files'

from flask import Flask, render_template, request, jsonify, url_for, redirect, send_file, after_this_request, flash
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
import io
import pandas as pd
import numpy as np
import compute_utils

# Main app object
app = Flask(__name__)

# Set secret key for sessions (necessary to show alert messages)
app.secret_key = '602e6444-80b2-431c-b26c-b6cda2ac9c09'

# In-memory thread pool
executor = ThreadPoolExecutor(max_workers=5) 

# Dictionary to store job progress/results (job_id: status_dict)
# NOTE: This dictionary is volatile and will reset if the server restarts.
job_status = {}
MAX_OUTPUT_LINES = 200


HAGEN_DEFAULTS = {
    "tstop": 12000.0,
    "dt": 2 ** -4,
    "local_num_threads": 64,
    "X": ["E", "I"],
    "N_X": [8192, 1024],
    "C_m_X": [289.1, 110.7],
    "tau_m_X": [10.0, 10.0],
    "E_L_X": [-65.0, -65.0],
    "C_YX": [[0.2, 0.2], [0.2, 0.2]],
    "J_YX": [[1.589, 2.020], [-23.84, -8.441]],
    "delay_YX": [[2.520, 1.714], [1.585, 1.149]],
    "tau_syn_YX": [[0.5, 0.5], [0.5, 0.5]],
    "n_ext": [465, 160],
    "nu_ext": 40.0,
    "J_ext": 29.89,
    "model": "iaf_psc_exp",
}

FOUR_AREA_DEFAULTS = {
    "tstop": 12000.0,
    "dt": 2 ** -4,
    "local_num_threads": 64,
    "areas": ["frontal", "parietal", "temporal", "occipital"],
    "X": ["E", "I"],
    "N_X": [8192, 1024],
    "C_m_X": [289.1, 110.7],
    "tau_m_X": [10.0, 10.0],
    "E_L_X": [-65.0, -65.0],
    "C_YX": [[0.2, 0.2], [0.2, 0.2]],
    "J_EE": 1.589,
    "J_IE": 2.020,
    "J_EI": -23.84,
    "J_II": -8.441,
    "delay_YX": [[2.520, 1.714], [1.585, 1.149]],
    "tau_syn_YX": [[0.5, 0.5], [0.5, 0.5]],
    "n_ext": [465, 160],
    "nu_ext": 40.0,
    "J_ext": 29.89,
    "model": "iaf_psc_exp",
    "inter_area_scale": 0.15,
    "inter_area_p": 0.02,
    "inter_area_delay": 10.0,
}


def _get_form_value(form, key):
    value = form.get(key)
    if value is None:
        return None
    value = value.strip()
    return value if value != "" else None


def _parse_literal(form, key, default):
    value = _get_form_value(form, key)
    if value is None:
        return default
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid literal for '{key}': {value}") from exc


def _parse_float(form, key, default):
    value = _get_form_value(form, key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for '{key}': {value}") from exc


def _parse_int(form, key, default):
    value = _get_form_value(form, key)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Invalid int for '{key}': {value}") from exc


def _parse_str(form, key, default):
    value = _get_form_value(form, key)
    if value is None:
        return default
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _format_value(value):
    return repr(value)


def _ensure_sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"'{name}' must be a list or tuple.")


def _ensure_length(value, name, expected):
    _ensure_sequence(value, name)
    if len(value) != expected:
        raise ValueError(f"'{name}' must have length {expected}.")


def _ensure_matrix(value, name, rows, cols):
    _ensure_sequence(value, name)
    if len(value) != rows:
        raise ValueError(f"'{name}' must have {rows} rows.")
    for idx, row in enumerate(value):
        _ensure_sequence(row, f"{name}[{idx}]")
        if len(row) != cols:
            raise ValueError(f"'{name}' row {idx} must have length {cols}.")


def _estimate_duration_seconds(form, defaults, model_type):
    tstop = _parse_float(form, "tstop", defaults["tstop"])
    dt = _parse_float(form, "dt", defaults["dt"])
    if dt <= 0:
        return 120.0

    steps = tstop / dt

    try:
        n_x = _parse_literal(form, "N_X", defaults["N_X"])
        if isinstance(n_x, (list, tuple)) and len(n_x) > 0:
            total_neurons = float(sum(n_x))
        else:
            total_neurons = float(sum(defaults["N_X"]))
    except Exception:
        total_neurons = float(sum(defaults["N_X"]))

    areas_count = 1.0
    if model_type == "four_area":
        try:
            areas = _parse_literal(form, "areas", defaults["areas"])
            if isinstance(areas, (list, tuple)) and len(areas) > 0:
                areas_count = float(len(areas))
            else:
                areas_count = float(len(defaults["areas"]))
        except Exception:
            areas_count = float(len(defaults["areas"]))

    work_units = steps * total_neurons * areas_count
    rate = 1.0e7
    estimated = work_units / rate
    return max(120.0, min(3600.0, estimated))


def _run_process_with_progress(cmd, cwd, job_status, job_id, estimate_seconds):
    output_lines = []
    progress_seen = threading.Event()
    output_buffer = deque(maxlen=MAX_OUTPUT_LINES)

    if job_id in job_status:
        job_status[job_id].setdefault("output", "")

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    start = time.time()

    def _maybe_update_progress(line):
        if "PROGRESS:" not in line:
            return
        marker_index = line.find("PROGRESS:")
        if marker_index == -1:
            return
        value = line[marker_index + len("PROGRESS:"):].strip()
        try:
            pct = int(float(value))
        except ValueError:
            return
        pct = max(0, min(99, pct))
        job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), pct)
        progress_seen.set()
        if pct >= 5:
            elapsed = time.time() - start
            est_total = elapsed / (pct / 100.0)
            job_status[job_id]["estimated_time_remaining"] = start + est_total

    def _reader():
        if process.stdout is None:
            return
        for line in iter(process.stdout.readline, ""):
            output_lines.append(line)
            _maybe_update_progress(line)
            output_buffer.append(line)
            if job_id in job_status:
                job_status[job_id]["output"] = "".join(output_buffer)
        process.stdout.close()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    initial_estimate = estimate_seconds
    while True:
        if process.poll() is not None:
            break
        time.sleep(1)

    reader_thread.join(timeout=2)
    output_text = "".join(output_lines).strip()
    return process.returncode, output_text


def _build_simulation_params(form, defaults):
    tstop = _parse_float(form, "tstop", defaults["tstop"])
    dt = _parse_float(form, "dt", defaults["dt"])
    local_num_threads = _parse_int(form, "local_num_threads", defaults["local_num_threads"])
    return "\n".join([
        "# Simulation time",
        f"tstop = {tstop}",
        "",
        "# Number of threads for the LIF network model simulations",
        f"local_num_threads = {local_num_threads}",
        "",
        "# Simulation time step",
        f"dt = {dt}",
        "",
    ])


def _build_hagen_network_params(form):
    X = _parse_literal(form, "X", HAGEN_DEFAULTS["X"])
    _ensure_length(X, "X", 2)
    N_X = _parse_literal(form, "N_X", HAGEN_DEFAULTS["N_X"])
    C_m_X = _parse_literal(form, "C_m_X", HAGEN_DEFAULTS["C_m_X"])
    tau_m_X = _parse_literal(form, "tau_m_X", HAGEN_DEFAULTS["tau_m_X"])
    E_L_X = _parse_literal(form, "E_L_X", HAGEN_DEFAULTS["E_L_X"])
    C_YX = _parse_literal(form, "C_YX", HAGEN_DEFAULTS["C_YX"])
    J_YX = _parse_literal(form, "J_YX", HAGEN_DEFAULTS["J_YX"])
    delay_YX = _parse_literal(form, "delay_YX", HAGEN_DEFAULTS["delay_YX"])
    tau_syn_YX = _parse_literal(form, "tau_syn_YX", HAGEN_DEFAULTS["tau_syn_YX"])
    n_ext = _parse_literal(form, "n_ext", HAGEN_DEFAULTS["n_ext"])
    nu_ext = _parse_float(form, "nu_ext", HAGEN_DEFAULTS["nu_ext"])
    J_ext = _parse_float(form, "J_ext", HAGEN_DEFAULTS["J_ext"])
    model = _parse_str(form, "model", HAGEN_DEFAULTS["model"])

    _ensure_length(N_X, "N_X", 2)
    _ensure_length(C_m_X, "C_m_X", 2)
    _ensure_length(tau_m_X, "tau_m_X", 2)
    _ensure_length(E_L_X, "E_L_X", 2)
    _ensure_matrix(C_YX, "C_YX", 2, 2)
    _ensure_matrix(J_YX, "J_YX", 2, 2)
    _ensure_matrix(delay_YX, "delay_YX", 2, 2)
    _ensure_matrix(tau_syn_YX, "tau_syn_YX", 2, 2)
    _ensure_length(n_ext, "n_ext", 2)

    return "\n".join([
        "# Best fit params of the LIF network model",
        "LIF_params = dict(",
        f"    X={_format_value(X)},",
        f"    N_X={_format_value(N_X)},",
        f"    C_m_X={_format_value(C_m_X)},",
        f"    tau_m_X={_format_value(tau_m_X)},",
        f"    E_L_X={_format_value(E_L_X)},",
        f"    C_YX={_format_value(C_YX)},",
        f"    J_YX={_format_value(J_YX)},",
        f"    delay_YX={_format_value(delay_YX)},",
        f"    tau_syn_YX={_format_value(tau_syn_YX)},",
        f"    n_ext={_format_value(n_ext)},",
        f"    nu_ext={nu_ext},",
        f"    J_ext={J_ext},",
        f"    model={_format_value(model)})",
        "",
    ])


def _build_four_area_network_params(form):
    areas = _parse_literal(form, "areas", FOUR_AREA_DEFAULTS["areas"])
    _ensure_length(areas, "areas", 4)
    X = _parse_literal(form, "X", FOUR_AREA_DEFAULTS["X"])
    N_X = _parse_literal(form, "N_X", FOUR_AREA_DEFAULTS["N_X"])
    C_m_X = _parse_literal(form, "C_m_X", FOUR_AREA_DEFAULTS["C_m_X"])
    tau_m_X = _parse_literal(form, "tau_m_X", FOUR_AREA_DEFAULTS["tau_m_X"])
    E_L_X = _parse_literal(form, "E_L_X", FOUR_AREA_DEFAULTS["E_L_X"])
    C_YX = _parse_literal(form, "C_YX", FOUR_AREA_DEFAULTS["C_YX"])
    J_EE = _parse_float(form, "J_EE", FOUR_AREA_DEFAULTS["J_EE"])
    J_IE = _parse_float(form, "J_IE", FOUR_AREA_DEFAULTS["J_IE"])
    J_EI = _parse_float(form, "J_EI", FOUR_AREA_DEFAULTS["J_EI"])
    J_II = _parse_float(form, "J_II", FOUR_AREA_DEFAULTS["J_II"])
    J_YX = _parse_literal(form, "J_YX", [[J_EE, J_IE], [J_EI, J_II]])
    delay_YX = _parse_literal(form, "delay_YX", FOUR_AREA_DEFAULTS["delay_YX"])
    tau_syn_YX = _parse_literal(form, "tau_syn_YX", FOUR_AREA_DEFAULTS["tau_syn_YX"])
    n_ext = _parse_literal(form, "n_ext", FOUR_AREA_DEFAULTS["n_ext"])
    nu_ext = _parse_float(form, "nu_ext", FOUR_AREA_DEFAULTS["nu_ext"])
    J_ext = _parse_float(form, "J_ext", FOUR_AREA_DEFAULTS["J_ext"])
    model = _parse_str(form, "model", FOUR_AREA_DEFAULTS["model"])
    inter_area_scale = _parse_float(
        form, "inter_area_scale", FOUR_AREA_DEFAULTS["inter_area_scale"]
    )
    inter_area_p = _parse_float(form, "inter_area_p", FOUR_AREA_DEFAULTS["inter_area_p"])
    inter_area_delay = _parse_float(
        form, "inter_area_delay", FOUR_AREA_DEFAULTS["inter_area_delay"]
    )

    inter_area_C = _parse_literal(
        form, "inter_area.C_YX", [[inter_area_p, inter_area_p], [0.0, 0.0]]
    )
    inter_area_J = _parse_literal(
        form,
        "inter_area.J_YX",
        [[J_EE * inter_area_scale, J_IE * inter_area_scale], [0.0, 0.0]],
    )
    inter_area_delay_YX = _parse_literal(
        form, "inter_area.delay_YX", [[inter_area_delay, inter_area_delay], [0.0, 0.0]]
    )

    _ensure_length(X, "X", 2)
    _ensure_length(N_X, "N_X", 2)
    _ensure_length(C_m_X, "C_m_X", 2)
    _ensure_length(tau_m_X, "tau_m_X", 2)
    _ensure_length(E_L_X, "E_L_X", 2)
    _ensure_matrix(C_YX, "C_YX", 2, 2)
    _ensure_matrix(J_YX, "J_YX", 2, 2)
    _ensure_matrix(delay_YX, "delay_YX", 2, 2)
    _ensure_matrix(tau_syn_YX, "tau_syn_YX", 2, 2)
    _ensure_length(n_ext, "n_ext", 2)
    _ensure_matrix(inter_area_C, "inter_area.C_YX", 2, 2)
    _ensure_matrix(inter_area_J, "inter_area.J_YX", 2, 2)
    _ensure_matrix(inter_area_delay_YX, "inter_area.delay_YX", 2, 2)

    return "\n".join([
        "# Parameters defining a four-area cortical network model in which the Hagen et al. local LIF microcircuit is",
        "# replicated and coupled across four cortical areas. Local network parameters match the Hagen model.",
        "",
        f"areas = {_format_value(areas)}",
        "",
        "# Base local parameters (Hagen model)",
        f"J_EE = {J_EE}",
        f"J_IE = {J_IE}",
        f"J_EI = {J_EI}",
        f"J_II = {J_II}",
        "",
        "# Inter-area (long-range) excitatory connectivity parameters",
        f"inter_area_scale = {inter_area_scale}",
        f"inter_area_p = {inter_area_p}",
        f"inter_area_delay = {inter_area_delay}",
        "",
        "LIF_params = dict(",
        f"    areas=areas,",
        f"    X={_format_value(X)},",
        f"    N_X={_format_value(N_X)},",
        f"    C_m_X={_format_value(C_m_X)},",
        f"    tau_m_X={_format_value(tau_m_X)},",
        f"    E_L_X={_format_value(E_L_X)},",
        f"    C_YX={_format_value(C_YX)},",
        f"    J_YX={_format_value(J_YX)},",
        f"    delay_YX={_format_value(delay_YX)},",
        f"    tau_syn_YX={_format_value(tau_syn_YX)},",
        f"    n_ext={_format_value(n_ext)},",
        f"    nu_ext={nu_ext},",
        f"    # The external drives reflects inputs from other brain areas, subcortical structures and background noise",
        f"    J_ext={J_ext},",
        f"    model={_format_value(model)},",
        f"    # Inter-area excitatory-only connections (E->E and E->I); no inhibitory cortico-cortical connections",
        f"    inter_area=dict(",
        f"        C_YX={_format_value(inter_area_C)},",
        f"        J_YX={_format_value(inter_area_J)},",
        f"        delay_YX={_format_value(inter_area_delay_YX)},",
        f"    ),",
        ")",
        "",
    ])


# Main dashboard page loading
@app.route("/")
def dashboard():
    simulation_data_dir = os.path.join(tempfile.gettempdir(), "simulation_data")
    if os.path.isdir(simulation_data_dir):
        simulation_pkl_files = sorted(
            f for f in os.listdir(simulation_data_dir)
            if f.endswith(".pkl") and os.path.isfile(os.path.join(simulation_data_dir, f))
        )
    else:
        simulation_pkl_files = []
    temp_dir = tempfile.gettempdir()
    fp_dirs = [
        os.path.join("/tmp", "field_potential_proxy"),
        os.path.join("/tmp", "field_potential_kernel"),
        os.path.join("/tmp", "field_potential_meeg"),
        os.path.join(temp_dir, "field_potential_proxy"),
        os.path.join(temp_dir, "field_potential_kernel"),
        os.path.join(temp_dir, "field_potential_meeg"),
    ]
    field_potential_files = []
    for fp_dir in fp_dirs:
        if not os.path.isdir(fp_dir):
            continue
        for root, _, files in os.walk(fp_dir):
            for name in files:
                if name.endswith(".pkl"):
                    field_potential_files.append(name)
    analysis_data_dir = os.path.join(tempfile.gettempdir(), "analysis_data")
    if os.path.isdir(analysis_data_dir):
        analysis_data_files = sorted(
            f for f in os.listdir(analysis_data_dir)
            if (f.endswith(".pkl") or f.endswith(".pickle"))
            and os.path.isfile(os.path.join(analysis_data_dir, f))
        )
    else:
        analysis_data_files = []
    return render_template(
        "0.dashboard.html",
        simulation_pkl_files=simulation_pkl_files,
        has_simulation_pkl=bool(simulation_pkl_files),
        field_potential_files=sorted(set(field_potential_files)),
        has_field_potential_data=bool(field_potential_files),
        analysis_data_files=analysis_data_files,
        has_analysis_data=bool(analysis_data_files),
    )

# Simulation configuration page
@app.route("/simulation")
def simulation():
    return render_template("1.simulation.html")

@app.route("/upload_sim")
def upload_sim():
    simulation_data_dir = os.path.join(tempfile.gettempdir(), "simulation_data")
    if os.path.isdir(simulation_data_dir):
        simulation_pkl_files = sorted(
            f for f in os.listdir(simulation_data_dir)
            if f.endswith(".pkl") and os.path.isfile(os.path.join(simulation_data_dir, f))
        )
    else:
        simulation_pkl_files = []
    return render_template(
        "1.1.upload_sim.html",
        simulation_pkl_files=simulation_pkl_files,
        has_simulation_pkl=bool(simulation_pkl_files),
    )

@app.route("/upload_sim_files", methods=["POST"])
def upload_sim_files():
    files = request.files.getlist("simulation_files")
    uploaded_files = [f for f in files if f and f.filename]

    if len(uploaded_files) == 0:
        flash('No files uploaded, please try again.', 'error')
        return redirect(request.referrer or url_for('upload_sim'))

    simulation_data_dir = os.path.join(tempfile.gettempdir(), "simulation_data")
    os.makedirs(simulation_data_dir, exist_ok=True)

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        if not filename:
            continue
        file.save(os.path.join(simulation_data_dir, filename))

    return redirect(url_for('upload_sim'))

@app.route("/clear_simulation_data", methods=["POST"])
def clear_simulation_data():
    simulation_data_dir = os.path.join(tempfile.gettempdir(), "simulation_data")
    if os.path.isdir(simulation_data_dir):
        for name in os.listdir(simulation_data_dir):
            if not name.endswith(".pkl"):
                continue
            path = os.path.join(simulation_data_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    return redirect(url_for('dashboard'))

@app.route("/clear_field_potential_data", methods=["POST"])
def clear_field_potential_data():
    temp_dir = tempfile.gettempdir()
    fp_dirs = [
        os.path.join("/tmp", "field_potential_proxy"),
        os.path.join("/tmp", "field_potential_kernel"),
        os.path.join("/tmp", "field_potential_meeg"),
        os.path.join(temp_dir, "field_potential_proxy"),
        os.path.join(temp_dir, "field_potential_kernel"),
        os.path.join(temp_dir, "field_potential_meeg"),
    ]
    for fp_dir in fp_dirs:
        if not os.path.isdir(fp_dir):
            continue
        for root, _, files in os.walk(fp_dir):
            for name in files:
                if not name.endswith(".pkl"):
                    continue
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
    return redirect(url_for('dashboard'))

@app.route("/new_sim")
def new_sim():
    return render_template("1.2.0.new_sim.html")

@app.route("/new_sim_brunel")
def new_sim_brunel():
    return render_template("1.2.2.new_sim_brunel.html")

@app.route("/new_sim_four_area")
def new_sim_four_area():
    return render_template("1.2.3.new_sim_four_area.html")

@app.route("/new_sim_custom")
def new_sim_custom():
    return render_template("1.2.1.new_sim_custom.html")


def _simulation_computation(job_id, job_status, params):
    try:
        model_type = params["model_type"]
        form = params["form"]
        estimate_seconds = params.get("estimate_seconds", 60.0)

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if model_type == "hagen":
            example_root = os.path.join(
                repo_root, "examples", "simulation", "Hagen_model", "simulation"
            )
            network_params_content = _build_hagen_network_params(form)
            sim_defaults = HAGEN_DEFAULTS
        else:
            example_root = os.path.join(
                repo_root, "examples", "simulation", "four_area_cortical_model", "simulation"
            )
            network_params_content = _build_four_area_network_params(form)
            sim_defaults = FOUR_AREA_DEFAULTS

        simulation_params_content = _build_simulation_params(form, sim_defaults)

        run_id = str(uuid.uuid4())
        run_root = os.path.join(tempfile.gettempdir(), "simulation_runs", run_id)
        params_dir = os.path.join(run_root, "params")
        python_dir = os.path.join(run_root, "python")
        os.makedirs(params_dir, exist_ok=True)
        os.makedirs(python_dir, exist_ok=True)

        output_dir = os.path.join(tempfile.gettempdir(), "simulation_data")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(params_dir, "network_params.py"), "w", encoding="utf-8") as f:
            f.write(network_params_content)
        with open(os.path.join(params_dir, "simulation_params.py"), "w", encoding="utf-8") as f:
            f.write(simulation_params_content)

        shutil.copy(
            os.path.join(example_root, "python", "network.py"),
            os.path.join(python_dir, "network.py"),
        )
        shutil.copy(
            os.path.join(example_root, "python", "simulation.py"),
            os.path.join(python_dir, "simulation.py"),
        )

        example_script_path = os.path.join(run_root, "example_model_simulation.py")
        example_script = "\n".join([
            "import ncpi",
            "",
            "if __name__ == \"__main__\":",
            "    # Create a Simulation object",
            "    sim = ncpi.Simulation(param_folder='params', python_folder='python', output_folder=%s)"
            % repr(output_dir),
            "",
            "    # Run the network and simulation scripts (analysis is intentionally skipped)",
            "    sim.network('network.py', 'network_params.py')",
            "    sim.simulate('simulation.py', 'simulation_params.py')",
            "",
        ])
        with open(example_script_path, "w", encoding="utf-8") as f:
            f.write(example_script)

        returncode, output_text = _run_process_with_progress(
            ["python", "example_model_simulation.py"],
            run_root,
            job_status,
            job_id,
            estimate_seconds,
        )

        if returncode != 0:
            error_msg = output_text or "Unknown error"
            raise RuntimeError(error_msg)

        job_status[job_id].update({
            "status": "finished",
            "progress": 100,
            "results": output_dir,
            "error": False,
        })

    except Exception as exc:
        job_status[job_id].update({
            "status": "failed",
            "error": str(exc),
            "progress": job_status[job_id].get("progress", 0),
        })


def _simulation_computation_custom(job_id, job_status, params):
    temp_run_dir = None
    upload_root = params.get("upload_root")
    try:
        input_paths = params["input_paths"]
        estimate_seconds = params.get("estimate_seconds", 60.0)

        run_id = str(uuid.uuid4())
        temp_run_dir = os.path.join(tempfile.gettempdir(), "simulation_runs", run_id)
        params_dir = os.path.join(temp_run_dir, "params")
        python_dir = os.path.join(temp_run_dir, "python")
        os.makedirs(params_dir, exist_ok=True)
        os.makedirs(python_dir, exist_ok=True)

        shutil.copy(input_paths["network_params"], os.path.join(params_dir, "network_params.py"))
        shutil.copy(input_paths["simulation_params"], os.path.join(params_dir, "simulation_params.py"))
        shutil.copy(input_paths["network_py"], os.path.join(python_dir, "network.py"))
        shutil.copy(input_paths["simulation_py"], os.path.join(python_dir, "simulation.py"))

        output_dir = os.path.join(tempfile.gettempdir(), "simulation_data")
        os.makedirs(output_dir, exist_ok=True)

        example_script_path = os.path.join(temp_run_dir, "example_model_simulation.py")
        example_script = "\n".join([
            "import ncpi",
            "",
            "if __name__ == \"__main__\":",
            "    # Create a Simulation object",
            "    sim = ncpi.Simulation(param_folder='params', python_folder='python', output_folder=%s)"
            % repr(output_dir),
            "",
            "    # Run the network and simulation scripts (analysis is intentionally skipped)",
            "    sim.network('network.py', 'network_params.py')",
            "    sim.simulate('simulation.py', 'simulation_params.py')",
            "",
        ])
        with open(example_script_path, "w", encoding="utf-8") as f:
            f.write(example_script)

        returncode, output_text = _run_process_with_progress(
            ["python", "example_model_simulation.py"],
            temp_run_dir,
            job_status,
            job_id,
            estimate_seconds,
        )

        if returncode != 0:
            error_msg = output_text or "Unknown error"
            raise RuntimeError(error_msg)

        job_status[job_id].update({
            "status": "finished",
            "progress": 100,
            "results": output_dir,
            "error": False,
        })

    except Exception as exc:
        job_status[job_id].update({
            "status": "failed",
            "error": str(exc),
            "progress": job_status[job_id].get("progress", 0),
        })
    finally:
        if temp_run_dir and os.path.isdir(temp_run_dir):
            shutil.rmtree(temp_run_dir, ignore_errors=True)
        if upload_root and os.path.isdir(upload_root):
            shutil.rmtree(upload_root, ignore_errors=True)


@app.route("/run_trial_simulation/<model_type>", methods=["POST"])
def run_trial_simulation(model_type):
    model_type = model_type.lower()
    if model_type not in {"hagen", "four_area"}:
        return "Model type is not valid", 400

    form = request.form.to_dict()

    if model_type == "hagen":
        sim_defaults = HAGEN_DEFAULTS
    else:
        sim_defaults = FOUR_AREA_DEFAULTS
    estimated_duration = _estimate_duration_seconds(form, sim_defaults, model_type)

    job_id = str(uuid.uuid4())
    start_time = time.time()
    job_status[job_id] = {
        "status": "in_progress",
        "progress": 0,
        "start_time": start_time,
        "estimated_time_remaining": None,
        "results": None,
        "error": False,
        "progress_mode": "manual",
        "output": "",
    }

    executor.submit(
        _simulation_computation,
        job_id,
        job_status,
        {"model_type": model_type, "form": form, "estimate_seconds": estimated_duration},
    )

    return redirect(url_for("job_status_page", job_id=job_id, computation_type="simulation"))


@app.route("/run_trial_simulation_custom", methods=["POST"])
def run_trial_simulation_custom():
    required_fields = {
        "network_params_file": "network_params",
        "network_py_file": "network_py",
        "simulation_params_file": "simulation_params",
        "simulation_py_file": "simulation_py",
    }

    missing = [field for field in required_fields if field not in request.files]
    if missing:
        flash("Missing required files for custom simulation.", "error")
        return redirect(request.referrer or url_for("new_sim_custom"))

    run_id = str(uuid.uuid4())
    upload_root = os.path.join(tempfile.gettempdir(), "simulation_custom_uploads", run_id)
    os.makedirs(upload_root, exist_ok=True)

    input_paths = {}
    for field, key in required_fields.items():
        file = request.files.get(field)
        if not file or not file.filename:
            flash("All custom simulation files are required.", "error")
            return redirect(request.referrer or url_for("new_sim_custom"))
        dest_path = os.path.join(upload_root, f"{key}.py")
        file.save(dest_path)
        input_paths[key] = dest_path

    job_id = str(uuid.uuid4())
    start_time = time.time()
    estimated_duration = 60.0
    job_status[job_id] = {
        "status": "in_progress",
        "progress": 0,
        "start_time": start_time,
        "estimated_time_remaining": None,
        "results": None,
        "error": False,
        "progress_mode": "manual",
        "output": "",
    }

    executor.submit(
        _simulation_computation_custom,
        job_id,
        job_status,
        {"input_paths": input_paths, "upload_root": upload_root, "estimate_seconds": estimated_duration},
    )

    return redirect(url_for("job_status_page", job_id=job_id, computation_type="simulation"))

# Field potential configuration page
@app.route("/field_potential")
def field_potential():
    return render_template("2.field_potential.html")

@app.route("/field_potential_kernel")
def field_potential_kernel():
    mc_models_default = os.path.expandvars(
        os.path.expanduser(os.path.join("$HOME", "multicompartment_neuron_network"))
    )
    mc_outputs_default = os.path.join(
        mc_models_default, "output", "adb947bfb931a5a8d09ad078a6d256b0"
    )
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    kernel_params_default = os.path.join(
        repo_root,
        "examples",
        "simulation",
        "Hagen_model",
        "simulation",
        "params",
        "analysis_params.py",
    )
    default_dir = "/tmp/simulation_data"
    default_paths = {
        "times": os.path.join(default_dir, "times.pkl"),
        "gids": os.path.join(default_dir, "gids.pkl"),
        "vm": os.path.join(default_dir, "vm.pkl"),
        "ampa": os.path.join(default_dir, "ampa.pkl"),
        "gaba": os.path.join(default_dir, "gaba.pkl"),
        "nu_ext": os.path.join(default_dir, "nu_ext.pkl"),
        "population_sizes": os.path.join(default_dir, "population_sizes.pkl"),
    }
    default_sim = {key: os.path.exists(path) for key, path in default_paths.items()}
    kernel_output_root = "/tmp/field_potential_kernel"
    preferred_cdm = []
    for fname in ("kernel_approx_cdm.pkl", "current_dipole_moment.pkl"):
        preferred_cdm.extend(glob.glob(os.path.join(kernel_output_root, "*", fname)))
    fallback_cdm = []
    for fname in ("gauss_cylinder_potential.pkl",):
        fallback_cdm.extend(glob.glob(os.path.join(kernel_output_root, "*", fname)))
    default_cdm_path = None
    if preferred_cdm:
        default_cdm_path = max(preferred_cdm, key=os.path.getmtime)
    elif fallback_cdm:
        default_cdm_path = max(fallback_cdm, key=os.path.getmtime)
    default_meeg = {
        "cdm": default_cdm_path,
        "cdm_exists": bool(default_cdm_path and os.path.exists(default_cdm_path)),
    }
    if default_meeg["cdm"]:
        default_meeg["cdm_name"] = os.path.basename(default_meeg["cdm"])
    else:
        default_meeg["cdm_name"] = ""
    requested_tab = request.args.get("tab", "")
    allowed_tabs = {"create_kernel", "cdm_computation", "meeg"}
    initial_tab = requested_tab if requested_tab in allowed_tabs else "create_kernel"
    return render_template(
        "2.1.field_potential_kernel.html",
        mc_models_default=mc_models_default,
        mc_outputs_default=mc_outputs_default,
        kernel_params_default=kernel_params_default,
        default_sim=default_sim,
        default_sim_paths=default_paths,
        default_meeg=default_meeg,
        initial_tab=initial_tab,
    )

@app.route("/field_potential_proxy")
def field_potential_proxy():
    default_dir = "/tmp/simulation_data"
    default_paths = {
        "times": os.path.join(default_dir, "times.pkl"),
        "gids": os.path.join(default_dir, "gids.pkl"),
        "vm": os.path.join(default_dir, "vm.pkl"),
        "ampa": os.path.join(default_dir, "ampa.pkl"),
        "gaba": os.path.join(default_dir, "gaba.pkl"),
        "nu_ext": os.path.join(default_dir, "nu_ext.pkl"),
    }
    default_sim = {key: os.path.exists(path) for key, path in default_paths.items()}
    return render_template(
        "2.2.field_potential_proxy.html",
        default_sim=default_sim,
        default_sim_paths=default_paths,
    )

# Features configuration page
@app.route("/features", methods=["GET", "POST"])
def features():
    return render_template("3.features.html")

# Inference configuration page
@app.route("/inference")
def inference():
    return render_template("4.inference.html")

# Analysis configuration page
@app.route("/analysis")
def analysis():
    analysis_data_dir = os.path.join(tempfile.gettempdir(), "analysis_data")
    if os.path.isdir(analysis_data_dir):
        analysis_data_files = sorted(
            f for f in os.listdir(analysis_data_dir)
            if (f.endswith(".pkl") or f.endswith(".pickle"))
            and os.path.isfile(os.path.join(analysis_data_dir, f))
        )
    else:
        analysis_data_files = []
    return render_template(
        "5.analysis.html",
        analysis_data_files=analysis_data_files,
        has_analysis_data=bool(analysis_data_files),
    )


def _analysis_data_path():
    analysis_data_dir = os.path.join(tempfile.gettempdir(), "analysis_data")
    if not os.path.isdir(analysis_data_dir):
        return None
    candidates = []
    for name in os.listdir(analysis_data_dir):
        if not (name.endswith(".pkl") or name.endswith(".pickle")):
            continue
        path = os.path.join(analysis_data_dir, name)
        if os.path.isfile(path):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _analysis_plot_error(message, status=400, log_output=""):
    return (
        render_template(
            "analysis_plot_result.html",
            title="Analysis plot",
            subtitle="Plotting failed.",
            error=message,
            image_data=None,
            log_output=log_output,
        ),
        status,
    )


def _render_analysis_plot(title, subtitle, image_bytes, log_output=""):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return render_template(
        "analysis_plot_result.html",
        title=title,
        subtitle=subtitle,
        error=None,
        image_data=encoded,
        log_output=log_output,
    )


def _find_column(df, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        cand_lower = str(cand).lower()
        if cand_lower in lower_map:
            return lower_map[cand_lower]
    return None


def _pick_value_column(df, exclude):
    preferred = ["Predictions", "prediction", "predictions", "Y", "y", "value", "Value", "Values", "data", "Data"]
    for cand in preferred:
        if cand in df.columns and cand not in exclude:
            return cand
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if numeric_cols:
        return numeric_cols[0]
    return None


@app.route("/analysis/columns", methods=["POST"])
def analysis_columns():
    upload = request.files.get("dataframe")
    if upload is None or upload.filename == "":
        return jsonify({"error": "No dataframe file uploaded."}), 400

    filename = secure_filename(upload.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension not in {".pkl", ".pickle"}:
        return jsonify({"error": "Only .pkl/.pickle files are supported."}), 400

    analysis_data_dir = os.path.join(tempfile.gettempdir(), "analysis_data")
    os.makedirs(analysis_data_dir, exist_ok=True)
    for existing in os.listdir(analysis_data_dir):
        if not (existing.endswith(".pkl") or existing.endswith(".pickle")):
            continue
        existing_path = os.path.join(analysis_data_dir, existing)
        if os.path.isfile(existing_path):
            try:
                os.remove(existing_path)
            except OSError:
                pass
    temp_path = os.path.join(analysis_data_dir, filename)
    upload.save(temp_path)

    try:
        df = compute_utils.read_df_file(temp_path)
        columns = [str(col) for col in df.columns]
        return jsonify({"columns": columns})
    except Exception as exc:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return jsonify({"error": str(exc)}), 400


@app.route("/analysis/columns/current", methods=["GET"])
def analysis_columns_current():
    data_path = _analysis_data_path()
    if data_path is None:
        return jsonify({"error": "No analysis dataframe found."}), 404
    try:
        df = compute_utils.read_df_file(data_path)
        columns = [str(col) for col in df.columns]
        return jsonify({"columns": columns})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/analysis/column_values", methods=["GET"])
def analysis_column_values():
    data_path = _analysis_data_path()
    if data_path is None:
        return jsonify({"error": "No analysis dataframe found."}), 404
    column = request.args.get("column")
    if not column:
        return jsonify({"error": "Column is required."}), 400
    try:
        df = compute_utils.read_df_file(data_path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    if column not in df.columns:
        return jsonify({"error": f'Column "{column}" not found in the dataframe.'}), 400

    series = df[column].dropna()
    values = series.tolist()
    if not values:
        return jsonify({"values": []})

    has_complex = any(isinstance(v, (list, tuple, dict, set, np.ndarray)) for v in values)
    if has_complex:
        unique_vals = sorted({str(v) for v in values})
        return jsonify({"values": unique_vals})

    numeric = []
    non_numeric = []
    for v in values:
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            non_numeric.append(v)

    if numeric and not non_numeric:
        unique_vals = sorted(set(numeric))
        out = [str(v) for v in unique_vals]
    else:
        unique_vals = sorted({str(v) for v in values})
        out = unique_vals
    return jsonify({"values": out})


@app.route("/analysis/plot/boxplot", methods=["POST"])
def analysis_plot_boxplot():
    log_buffer = io.StringIO()
    def _log(message):
        print(message, file=log_buffer)
    def _plot_error(message, status=400):
        return _analysis_plot_error(message, status=status, log_output=log_buffer.getvalue())

    data_path = _analysis_data_path()
    if data_path is None:
        return _plot_error("No analysis dataframe found. Upload a .pkl file first.")

    group_col = request.form.get("boxplot_group_by")
    if not group_col:
        return _plot_error("Select a grouping column for the x-axis.")

    try:
        df = compute_utils.read_df_file(data_path)
    except Exception as exc:
        return _plot_error(str(exc))
    _log("Loaded dataframe.")

    if group_col not in df.columns:
        return _plot_error(f'Grouping column "{group_col}" not found in the dataframe.')

    value_col = request.form.get("boxplot_value_col")
    if not value_col:
        return _plot_error("Select a y-axis variable for the boxplot.")
    if value_col not in df.columns:
        return _plot_error(f'Value column "{value_col}" not found in the dataframe.')

    def _parse_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_color(value, default):
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        if "," in text:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) in (3, 4):
                try:
                    return tuple(float(p) for p in parts)
                except ValueError:
                    pass
        return text

    showfliers = request.form.get("boxplot_showfliers") is not None
    box_width = _parse_float(request.form.get("boxplot_width"), 0.5)
    line_width = _parse_float(request.form.get("boxplot_linewidth"), 0.5)
    median_color = _parse_color(request.form.get("boxplot_median_color"), "red")
    median_linewidth = _parse_float(request.form.get("boxplot_median_linewidth"), 0.8)
    box_edge_width = _parse_float(request.form.get("boxplot_box_edge_width"), 0.2)
    box_facecolor = _parse_color(request.form.get("boxplot_facecolor"), "none")
    colormap_name = request.form.get("boxplot_colormap", "viridis")
    colormap_alpha = _parse_float(request.form.get("boxplot_color_alpha"), 0.35)
    if colormap_alpha is None or not (0.0 <= colormap_alpha <= 1.0):
        colormap_alpha = 0.35
    show_cohend = request.form.get("boxplot_show_cohend") is not None
    control_group_raw = request.form.get("boxplot_control_group") if show_cohend else None
    control_group_raw = control_group_raw.strip() if control_group_raw else ""
    if show_cohend and not control_group_raw:
        return _plot_error("Provide a control group to compute Cohen's d.")
    _log(f"Boxplot settings: group_col={group_col}, value_col={value_col}, show_cohend={show_cohend}.")

    df_use = df[[group_col, value_col]].copy()
    groups = [g for g in df_use[group_col].dropna().unique().tolist()]
    if not groups:
        return _plot_error("No groups found for the selected grouping column.")
    def _match_control_group(items, raw_value):
        if not raw_value:
            return None
        for item in items:
            if str(item) == raw_value:
                return item
        try:
            raw_num = float(raw_value)
        except (TypeError, ValueError):
            return None
        for item in items:
            try:
                if float(item) == raw_num:
                    return item
            except (TypeError, ValueError):
                continue
        return None

    def _maybe_sort_groups(items):
        if not items:
            return items, None
        numeric_values = []
        for idx, item in enumerate(items):
            if isinstance(item, (int, float, np.integer, np.floating)):
                if pd.isna(item):
                    return items, None
                numeric_values.append((float(item), idx, item))
                continue
            try:
                value = float(str(item))
            except (TypeError, ValueError):
                return items, None
            if np.isnan(value):
                return items, None
            numeric_values.append((value, idx, item))
        numeric_values.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in numeric_values], numeric_values

    def _infer_vector_length(values):
        lengths = set()
        for v in values:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v)
                if arr.ndim == 0:
                    lengths.add(1)
                else:
                    lengths.add(arr.size)
            else:
                lengths.add(1)
        if not lengths:
            return 0, None
        if len(lengths) == 1:
            return lengths.pop(), None
        if 1 in lengths:
            return None, "mixed scalars and vectors"
        return None, "inconsistent vector lengths"

    vector_len, length_error = _infer_vector_length(df_use[value_col])
    if vector_len is None:
        return _plot_error(
            f'Value column "{value_col}" has {length_error}. Use a column with consistent lengths.'
        )
    if vector_len == 0:
        return _plot_error(f'Value column "{value_col}" contains no data to plot.')

    groups, numeric_sort = _maybe_sort_groups(groups)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return _plot_error(f"Matplotlib is required for plotting: {exc}", status=500)
    _log("Matplotlib initialized.")

    def _coerce_scalar(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim == 0:
                try:
                    return float(arr)
                except (TypeError, ValueError):
                    return None
            flat = arr.ravel()
            if flat.size != 1:
                return None
            try:
                return float(flat[0])
            except (TypeError, ValueError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_vector(value, length):
        if not isinstance(value, (list, tuple, np.ndarray)):
            return None
        arr = np.asarray(value)
        if arr.ndim == 0:
            return None
        flat = arr.ravel()
        if flat.size != length:
            return None
        try:
            return flat.astype(float)
        except (TypeError, ValueError):
            return None

    def _add_cohen_bar(ax, d_map, control_value, group_values):
        if not d_map:
            return
        labels = []
        values = []
        for g in group_values:
            if g == control_value:
                continue
            val = d_map.get(g)
            if val is None:
                continue
            try:
                if np.isnan(val):
                    continue
            except TypeError:
                pass
            labels.append(str(g))
            values.append(float(val))
        if not values:
            return
        inset = ax.inset_axes([1.12, 0.10, 0.30, 0.80], transform=ax.transAxes)
        inset.set_facecolor("white")
        inset.patch.set_alpha(0.95)
        y = np.arange(len(values))
        colors = ["#F97316" if v >= 0 else "#2563EB" for v in values]
        inset.barh(y, values, color=colors, edgecolor="#0F172A", linewidth=0.3)
        inset.axvline(0, color="#0F172A", linewidth=0.6)
        inset.set_yticks(y)
        inset.set_yticklabels(labels, fontsize=7)
        inset.tick_params(axis="x", labelsize=7)
        inset.set_title("Cohen's d", fontsize=7, pad=2)
        inset.grid(False)
        for spine in inset.spines.values():
            spine.set_visible(False)

    positions = np.arange(1, len(groups) + 1)
    boxplot_kwargs = dict(
        positions=positions,
        showfliers=showfliers,
        widths=box_width,
        patch_artist=True,
        medianprops=dict(color=median_color, linewidth=median_linewidth),
        whiskerprops=dict(color="black", linewidth=line_width),
        capprops=dict(color="black", linewidth=line_width),
        boxprops=dict(linewidth=line_width, facecolor=box_facecolor),
    )
    colormap = None
    if colormap_name:
        name = str(colormap_name).strip().lower()
        if name not in ("none", "off", "false", "no"):
            try:
                colormap = plt.colormaps[str(colormap_name).strip()]
            except KeyError:
                colormap = plt.colormaps["viridis"]

    if vector_len == 1:
        grouped_values = {g: [] for g in groups}
        for g, v in df_use.itertuples(index=False):
            if pd.isna(g) or v is None:
                continue
            val = _coerce_scalar(v)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            grouped_values[g].append(val)

        groups = [g for g in groups if grouped_values[g]]
        if not groups:
            return _plot_error("No numeric data available for the selected y-axis variable.")

        cohen_map = None
        control_group_value = None
        if show_cohend:
            control_group_value = _match_control_group(groups, control_group_raw)
            if control_group_value is None:
                return _plot_error(
                    f'Control group "{control_group_raw}" not found in the grouping column.'
                )
            try:
                import ncpi
            except Exception as exc:
                return _plot_error(f"ncpi is required for Cohen's d: {exc}", status=500)
            _log(f"Computing Cohen's d vs {control_group_value}.")
            rows = []
            for g, values in grouped_values.items():
                for val in values:
                    rows.append((g, val))
            df_cohen = pd.DataFrame(rows, columns=[group_col, value_col])
            sensor_col = "__boxplot_sensor__"
            df_cohen[sensor_col] = "__all__"
            analysis = ncpi.Analysis(df_cohen)
            try:
                results = analysis.cohend(
                    control_group=control_group_value,
                    data_col=value_col,
                    data_index=-1,
                    group_col=group_col,
                    sensor_col=sensor_col,
                    drop_zeros=False,
                )
            except Exception as exc:
                return _plot_error(f"Failed to compute Cohen's d: {exc}")
            cohen_map = {}
            for g in groups:
                if g == control_group_value:
                    continue
                key = f"{g}vs{control_group_value}"
                comp_df = results.get(key)
                if comp_df is None or comp_df.empty:
                    cohen_map[g] = np.nan
                    continue
                row = comp_df.loc[comp_df[sensor_col] == "__all__"]
                if row.empty:
                    cohen_map[g] = comp_df["d"].iloc[0]
                else:
                    cohen_map[g] = row["d"].iloc[0]

        positions = np.arange(1, len(groups) + 1)
        data_plot = [grouped_values[g] for g in groups]

        fig, ax = plt.subplots(figsize=(13.8, 6.2))
        _log("Rendering boxplot.")
        box = ax.boxplot(data_plot, **{**boxplot_kwargs, "positions": positions})
        for patch in box.get("boxes", []):
            patch.set_linewidth(box_edge_width)
        if colormap is not None:
            color_positions = np.linspace(0, 1, num=len(groups)) if len(groups) > 1 else [0.5]
            for patch, color_pos in zip(box.get("boxes", []), color_positions):
                color = colormap(color_pos)
                patch.set_facecolor((color[0], color[1], color[2], colormap_alpha))
        if show_cohend and cohen_map:
            _add_cohen_bar(ax, cohen_map, control_group_value, groups)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_xlabel(group_col)
        ax.set_ylabel(value_col)
        ax.set_title(f"{value_col} by {group_col}")
        fig.suptitle("")
        fig.tight_layout()
        fig.subplots_adjust(right=0.64)
    else:
        grouped_arrays = {g: [] for g in groups}
        for g, v in df_use.itertuples(index=False):
            if pd.isna(g) or v is None:
                continue
            arr = _coerce_vector(v, vector_len)
            if arr is None:
                return _analysis_plot_error(
                    f'Value column "{value_col}" must contain arrays/lists of length {vector_len}.'
                )
            grouped_arrays[g].append(arr)

        groups = [g for g in groups if grouped_arrays[g]]
        if not groups:
            return _plot_error("No data available for the selected y-axis variable.")

        cohen_maps = None
        control_group_value = None
        if show_cohend:
            control_group_value = _match_control_group(groups, control_group_raw)
            if control_group_value is None:
                return _plot_error(
                    f'Control group "{control_group_raw}" not found in the grouping column.'
                )
            try:
                import ncpi
            except Exception as exc:
                return _plot_error(f"ncpi is required for Cohen's d: {exc}", status=500)
            _log(f"Computing Cohen's d vs {control_group_value} for each dimension.")
            rows = []
            for g, arrs in grouped_arrays.items():
                for arr in arrs:
                    rows.append((g, arr))
            df_cohen = pd.DataFrame(rows, columns=[group_col, value_col])
            sensor_col = "__boxplot_sensor__"
            df_cohen[sensor_col] = "__all__"
            analysis = ncpi.Analysis(df_cohen)
            cohen_maps = []
            for dim in range(vector_len):
                try:
                    results = analysis.cohend(
                        control_group=control_group_value,
                        data_col=value_col,
                        data_index=dim,
                        group_col=group_col,
                        sensor_col=sensor_col,
                        drop_zeros=False,
                    )
                except Exception as exc:
                    return _plot_error(f"Failed to compute Cohen's d: {exc}")
                d_map = {}
                for g in groups:
                    if g == control_group_value:
                        continue
                    key = f"{g}vs{control_group_value}"
                    comp_df = results.get(key)
                    if comp_df is None or comp_df.empty:
                        d_map[g] = np.nan
                        continue
                    row = comp_df.loc[comp_df[sensor_col] == "__all__"]
                    if row.empty:
                        d_map[g] = comp_df["d"].iloc[0]
                    else:
                        d_map[g] = row["d"].iloc[0]
                cohen_maps.append(d_map)

        import math

        cols = 1
        rows = vector_len
        fig, axes = plt.subplots(rows, cols, figsize=(11.2, 4.6 * rows))
        _log(f"Rendering {vector_len} boxplot panels.")
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.ravel()

        for dim in range(vector_len):
            ax = axes[dim]
            data_plot = []
            has_data = False
            for g in groups:
                values = [arr[dim] for arr in grouped_arrays[g]]
                values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
                if values:
                    has_data = True
                data_plot.append(values)

            positions = np.arange(1, len(groups) + 1)
            if has_data:
                box = ax.boxplot(data_plot, **{**boxplot_kwargs, "positions": positions})
                for patch in box.get("boxes", []):
                    patch.set_linewidth(box_edge_width)
                if colormap is not None:
                    color_positions = np.linspace(0, 1, num=len(groups)) if len(groups) > 1 else [0.5]
                    for patch, color_pos in zip(box.get("boxes", []), color_positions):
                        color = colormap(color_pos)
                        patch.set_facecolor((color[0], color[1], color[2], colormap_alpha))
                if show_cohend and cohen_maps:
                    _add_cohen_bar(ax, cohen_maps[dim], control_group_value, groups)
                ax.set_xticks(positions)
                ax.set_xticklabels([str(g) for g in groups])
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_xlabel(group_col)
            ax.set_ylabel(f"{value_col}[{dim}]")
            ax.set_title(f"{value_col}[{dim}] by {group_col}")

        for extra in range(vector_len, len(axes)):
            fig.delaxes(axes[extra])
        fig.suptitle("")
        fig.tight_layout()
        fig.subplots_adjust(right=0.64)

    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=160)
    plt.close(fig)
    output.seek(0)
    return _render_analysis_plot(
        title="Boxplot result",
        subtitle=f"{value_col} grouped by {group_col}.",
        image_bytes=output.getvalue(),
        log_output=log_buffer.getvalue(),
    )


@app.route("/analysis/plot/topomap", methods=["POST"])
def analysis_plot_topomap():
    log_buffer = io.StringIO()
    def _log(message):
        print(message, file=log_buffer)
    def _plot_error(message, status=400):
        return _analysis_plot_error(message, status=status, log_output=log_buffer.getvalue())

    data_path = _analysis_data_path()
    if data_path is None:
        return _plot_error("No analysis dataframe found. Upload a .pkl file first.")

    group_col = request.form.get("topomap_group_by")
    if not group_col:
        return _plot_error("Select a grouping column for the topomap.")

    grouping_mode = request.form.get("topomap_grouping_mode", "per_sensor")
    compare_method = request.form.get("topomap_compare_method", "raw")
    control_group_raw = request.form.get("topomap_control_group") if grouping_mode == "compare_categories" else None
    control_group_raw = control_group_raw.strip() if control_group_raw else ""

    try:
        df = compute_utils.read_df_file(data_path)
    except Exception as exc:
        return _plot_error(str(exc))
    _log("Loaded dataframe.")

    if group_col not in df.columns:
        return _plot_error(f'Grouping column "{group_col}" not found in the dataframe.')

    sensor_col = _find_column(df, ["sensor", "Sensor", "channel", "Channel", "ch", "Ch", "electrode", "Electrode"])
    if sensor_col is None:
        return _plot_error("Sensor/channel column not found. Expected a column like 'sensor' or 'Sensor'.")

    value_col = request.form.get("topomap_value_col")
    if not value_col:
        return _plot_error("Select a value column for the topomap.")
    if value_col not in df.columns:
        return _plot_error(f'Value column "{value_col}" not found in the dataframe.')

    df_use = df[[group_col, sensor_col, value_col]].copy()
    df_use = df_use.dropna(subset=[sensor_col, group_col])
    if df_use.empty:
        return _plot_error("No valid rows found for the selected group and sensor columns.")
    _log(f"Topomap settings: group_col={group_col}, value_col={value_col}, mode={grouping_mode}.")

    def _infer_vector_length(values):
        lengths = set()
        for v in values:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v)
                if arr.ndim == 0:
                    lengths.add(1)
                else:
                    lengths.add(arr.size)
            else:
                lengths.add(1)
        if not lengths:
            return 0, None
        if len(lengths) == 1:
            return lengths.pop(), None
        if 1 in lengths:
            return None, "mixed scalars and vectors"
        return None, "inconsistent vector lengths"

    vector_len, length_error = _infer_vector_length(df_use[value_col])
    if vector_len is None:
        return _plot_error(
            f'Value column "{value_col}" has {length_error}. Use a column with consistent lengths.'
        )
    if vector_len == 0:
        return _plot_error(f'Value column "{value_col}" contains no data to plot.')

    groups = [g for g in df_use[group_col].dropna().unique().tolist()]
    if not groups:
        return _plot_error("No groups found for the selected grouping column.")
    if grouping_mode == "compare_categories" and not control_group_raw:
        return _plot_error("Provide a control group for category comparisons.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return _plot_error(f"Matplotlib is required for plotting: {exc}", status=500)
    _log("Matplotlib initialized.")

    try:
        import ncpi
    except Exception as exc:
        return _plot_error(f"ncpi is required for topomap plotting: {exc}", status=500)
    _log("ncpi loaded.")

    # Parse numeric inputs for plotting
    def _parse_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    head_radius = _parse_float(request.form.get("head-radius"))
    head_pos_x = _parse_float(request.form.get("head-pos-x"))
    show_colorbar = request.form.get("show-colorbar") is not None
    scale_mode = request.form.get("topomap_scale_mode", "section")
    use_diverging = grouping_mode == "compare_categories"
    compare_cmap = "bwr" if use_diverging else None

    sphere = "auto"
    if head_radius is not None:
        x = head_pos_x if head_pos_x is not None else 0.0
        sphere = (x, 0.0, 0.0, head_radius)

    analysis = ncpi.Analysis(df_use)

    def _coerce_scalar(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim == 0:
                try:
                    return float(arr)
                except (TypeError, ValueError):
                    return None
            flat = arr.ravel()
            if flat.size != 1:
                return None
            try:
                return float(flat[0])
            except (TypeError, ValueError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_vector(value, length):
        if not isinstance(value, (list, tuple, np.ndarray)):
            return None
        arr = np.asarray(value)
        if arr.ndim == 0:
            return None
        flat = arr.ravel()
        if flat.size != length:
            return None
        try:
            return flat.astype(float)
        except (TypeError, ValueError):
            return None

    def _match_control_group(items, raw_value):
        if not raw_value:
            return None
        for item in items:
            if str(item) == raw_value:
                return item
        try:
            raw_num = float(raw_value)
        except (TypeError, ValueError):
            return None
        for item in items:
            try:
                if float(item) == raw_num:
                    return item
            except (TypeError, ValueError):
                continue
        return None

    def _symmetric_limits(min_val, max_val):
        if min_val is None or max_val is None:
            return min_val, max_val
        max_abs = max(abs(min_val), abs(max_val))
        if max_abs == 0:
            return -1.0, 1.0
        return -max_abs, max_abs

    def _build_series_for_dim(dim):
        items_local = []
        if grouping_mode == "compare_categories":
            control_group = _match_control_group(groups, control_group_raw)
            if control_group is None:
                return None, f'Control group "{control_group_raw}" not found in the grouping column.'
            if compare_method == "cohen_d":
                try:
                    compare_results = analysis.cohend(
                        control_group=str(control_group),
                        data_col=value_col,
                        data_index=dim,
                        group_col=group_col,
                        sensor_col=sensor_col,
                        drop_zeros=False,
                    )
                except Exception as exc:
                    return None, f"Failed to compute group comparisons: {exc}"
                for label, comp_df in compare_results.items():
                    if comp_df.empty:
                        continue
                    series = comp_df.set_index(sensor_col)["d"]
                    items_local.append((label, series))
            else:
                control_df = df_use[df_use[group_col] == control_group]
                if dim == -1:
                    control_values = control_df[value_col].apply(_coerce_scalar)
                else:
                    control_values = control_df[value_col].apply(
                        lambda x: _coerce_vector(x, vector_len)[dim] if _coerce_vector(x, vector_len) is not None else np.nan
                    )
                control_series = (
                    pd.DataFrame({sensor_col: control_df[sensor_col], "value": control_values})
                    .groupby(sensor_col)["value"]
                    .mean()
                    .dropna()
                )
                for g in groups:
                    if g == control_group:
                        continue
                    group_df = df_use[df_use[group_col] == g]
                    if dim == -1:
                        group_values = group_df[value_col].apply(_coerce_scalar)
                    else:
                        group_values = group_df[value_col].apply(
                            lambda x: _coerce_vector(x, vector_len)[dim] if _coerce_vector(x, vector_len) is not None else np.nan
                        )
                    group_series = (
                        pd.DataFrame({sensor_col: group_df[sensor_col], "value": group_values})
                        .groupby(sensor_col)["value"]
                        .mean()
                        .dropna()
                    )
                    diff = group_series.subtract(control_series, fill_value=np.nan)
                    if not diff.empty:
                        items_local.append((f"{g} - {control_group}", diff))
            if not items_local:
                return None, "No comparison results available to plot."
        else:
            for g in groups:
                if dim == -1:
                    series_data = df_use[df_use[group_col] == g][value_col].apply(_coerce_scalar)
                else:
                    series_data = df_use[df_use[group_col] == g][value_col].apply(
                        lambda x: _coerce_vector(x, vector_len)[dim] if _coerce_vector(x, vector_len) is not None else np.nan
                    )
                series = (
                    pd.DataFrame({sensor_col: df_use.loc[df_use[group_col] == g, sensor_col], "value": series_data})
                    .groupby(sensor_col)["value"]
                    .mean()
                    .dropna()
                )
                if not series.empty:
                    items_local.append((str(g), series))
            if not items_local:
                return None, "No data available to plot for the selected grouping."
        return items_local, None

    sections = []
    if vector_len == 1:
        df_use[value_col] = df_use[value_col].apply(_coerce_scalar)
        df_use = df_use.dropna(subset=[value_col])
        if df_use.empty:
            return _plot_error(f'Value column "{value_col}" has no numeric values to plot.')
        items, err = _build_series_for_dim(-1)
        if err:
            return _plot_error(err)
        sections.append({"label": value_col, "items": items})
    else:
        for dim in range(vector_len):
            dim_items, err = _build_series_for_dim(dim)
            if err:
                return _plot_error(err)
            sections.append({"label": f"{value_col}[{dim}]", "items": dim_items})
    _log(f"Prepared {len(sections)} section(s) for plotting.")

    import math

    max_items = max((len(section["items"]) for section in sections), default=1)
    cols = min(3, max_items)
    plot_rows_total = sum(math.ceil(len(section["items"]) / cols) for section in sections) or 1
    title_rows_total = len(sections)
    total_rows = plot_rows_total + title_rows_total
    fig_height = (3.6 * plot_rows_total) + (0.2 * title_rows_total)
    fig = plt.figure(figsize=(4.2 * cols, fig_height))
    height_ratios = []
    for section in sections:
        height_ratios.append(0.10)
        rows_needed = math.ceil(len(section["items"]) / cols) if section["items"] else 1
        height_ratios.extend([1.0] * rows_needed)
    gs = fig.add_gridspec(total_rows, cols, hspace=0.28, height_ratios=height_ratios)

    row_cursor = 0
    for section in sections:
        section_items = section["items"]
        section_vmin = None
        section_vmax = None
        if scale_mode != "plot":
            section_values = []
            for _, series in section_items:
                values = series.to_numpy(dtype=float)
                if values.size:
                    section_values.append(values)
            if section_values:
                section_all = np.concatenate(section_values)
                section_vmin = np.nanmin(section_all)
                section_vmax = np.nanmax(section_all)
                if use_diverging:
                    section_vmin, section_vmax = _symmetric_limits(section_vmin, section_vmax)
        rows_needed = math.ceil(len(section_items) / cols) if section_items else 1
        section_row_start = row_cursor

        title_ax = fig.add_subplot(gs[row_cursor, :])
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.5,
            section["label"],
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
        )
        row_cursor += 1

        if not section_items:
            row_cursor += rows_needed
            continue

        for idx, (label, series) in enumerate(section_items):
            r = row_cursor + (idx // cols)
            c = idx % cols
            ax = fig.add_subplot(gs[r, c])
            try:
                plot_vmin = section_vmin
                plot_vmax = section_vmax
                if scale_mode == "plot":
                    values = series.to_numpy(dtype=float)
                    if values.size:
                        plot_vmin = np.nanmin(values)
                        plot_vmax = np.nanmax(values)
                    else:
                        plot_vmin = None
                        plot_vmax = None
                    if use_diverging:
                        plot_vmin, plot_vmax = _symmetric_limits(plot_vmin, plot_vmax)
                im, _ = analysis.eeg_topomap(
                    series,
                    axes=ax,
                    show=False,
                    vmin=plot_vmin,
                    vmax=plot_vmax,
                    cmap=compare_cmap,
                    colorbar=False,
                    sensors=True,
                    montage="standard_1020",
                    extrapolate="local",
                    sphere=sphere,
                )
            except Exception as exc:
                plt.close(fig)
                return _plot_error(f"Topomap plotting failed: {exc}")
            if show_colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(label)
        row_cursor += rows_needed

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=160)
    plt.close(fig)
    output.seek(0)
    def _trim_whitespace(png_bytes, pad=2, threshold=250):
        try:
            from PIL import Image
            import numpy as np
        except Exception:
            return png_bytes
        try:
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        except Exception:
            return png_bytes
        arr = np.asarray(img)
        mask = np.any(arr < threshold, axis=2)
        if not mask.any():
            return png_bytes
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        y0 = max(int(y0) - pad, 0)
        x0 = max(int(x0) - pad, 0)
        y1 = min(int(y1) + pad, arr.shape[0])
        x1 = min(int(x1) + pad, arr.shape[1])
        cropped = img.crop((x0, y0, x1, y1))
        out = io.BytesIO()
        cropped.save(out, format="PNG")
        return out.getvalue()

    image_bytes = _trim_whitespace(output.getvalue())
    return _render_analysis_plot(
        title="Topomap result",
        subtitle="EEG topographic plot.",
        image_bytes=image_bytes,
        log_output=log_buffer.getvalue(),
    )


@app.route("/clear_analysis_data", methods=["POST"])
def clear_analysis_data():
    analysis_data_dir = os.path.join(tempfile.gettempdir(), "analysis_data")
    if os.path.isdir(analysis_data_dir):
        for name in os.listdir(analysis_data_dir):
            if not (name.endswith(".pkl") or name.endswith(".pickle")):
                continue
            path = os.path.join(analysis_data_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    return redirect(request.referrer or url_for('dashboard'))


@app.route("/start_computation_redirect/<computation_type>", methods=["POST"])
def start_computation_redirect(computation_type):
    """Starts the background job and redirects to the status page."""
    # Allowed function names to redirect to
    allowed_functions = {'features', 'inference', 'analysis', 'field_potential_proxy', 'field_potential_kernel', 'field_potential_meeg'}

    if computation_type not in allowed_functions:
        return f"Type of computation is not valid", 400

    # Build the name of the function to compute depending on the page form this function was called from
    func_name_string = f"{computation_type}_computation"
    func = getattr(compute_utils, func_name_string) # filtered function name for security reasons

    # Get all files from the request
    files = request.files
    
    # Check how many files were uploaded with actual filenames
    uploaded_files = [f for f in files.values() if f.filename]

    # First check if ANY files were uploaded. If at least one file was uploaded
    if len(uploaded_files) == 0 and computation_type not in {'field_potential_proxy', 'field_potential_kernel', 'field_potential_meeg'}:
        # Flash an error message
        flash('No files uploaded, please try again.', 'error')
        return redirect(request.referrer)

    # File filter and checks for every computation type
    if computation_type == 'features':
        # Expect exactly 1 file
        if len(uploaded_files) != 1:
            flash('This computation requires exactly 1 file.', 'error')
            return redirect(request.referrer or url_for('features'))
        estimated_time_remaining = time.time() + 68 # 68 seconds of estimated time remaining

    if computation_type == 'inference':
        # Expect 5 files
        if len(uploaded_files) != 5 and len(uploaded_files) != 1:
            flash('This computation requires you to upload all 5 files or only the features prediction file.', 'error')
            return redirect(request.referrer or url_for('inference'))
        estimated_time_remaining = time.time() + 130 # 130 seconds of estimated time remaining

    if computation_type == 'analysis':
        estimated_time_remaining = time.time() + 10 # 15 seconds of estimated time remaining
    if computation_type == 'field_potential_proxy':
        estimated_time_remaining = time.time() + 30
    if computation_type == 'field_potential_kernel':
        estimated_time_remaining = time.time() + 60
    if computation_type == 'field_potential_meeg':
        estimated_time_remaining = time.time() + 30

    # Unique id for job
    job_id = str(uuid.uuid4())

    # If everything is OK, save the file(s)
    file_paths = {}
    for i, file_key in enumerate(request.files):
        file = request.files[file_key]
        if not file or not file.filename:
            if computation_type in {'field_potential_proxy', 'field_potential_kernel', 'field_potential_meeg'}:
                continue
        unique_filename = f"{computation_type}_{file_key}_{i}_{job_id}_{file.filename}" # E.g. features_ data_file_ 0_ 444961cc-5b72-43fc-b87e-3f4c8304ecdd_ df_inputIn_features_lfp.pkl
        file_path = os.path.join(temp_uploaded_files, unique_filename)
        # Ensure directory exists
        os.makedirs(temp_uploaded_files, exist_ok=True)
        file.save(file_path)
        # Save dictionary with file_key: file_path
        file_paths[file_key] = file_path

    data = request.form.to_dict() # Get parameters from form POST
    # Add file information to the data dictionary
    data['file_paths'] = file_paths

    # Store initial status
    job_status[job_id] = {
        "status": "in_progress",
        "progress": 0,
        "start_time": time.time(),
        "estimated_time_remaining": estimated_time_remaining,
        "results": None,
        "error": False,
        "output": "",
    }

    # Submit the long-running task according to the computation type
    executor.submit(func, job_id, job_status, data, temp_uploaded_files)

    # Redirect immediately to the loading page (PRG pattern)
    return redirect(url_for('job_status_page', job_id=job_id, computation_type=computation_type))

@app.route("/job_status/<job_id>")
def job_status_page(job_id):
    """Renders the loading page that begins polling."""
    # Get computation_type from the query parameters
    computation_type = request.args.get('computation_type') 
    # Pass the job_id to the template for use in Alpine.js
    return render_template("loading_page.html", job_id=job_id, computation_type=computation_type)

@app.route("/status/<job_id>")
def get_status(job_id):
    """AJAX endpoint for the client to poll for status updates."""
    status = job_status.get(job_id)
    if not status:
        return jsonify({
            "status": "failed", 
            "error": "Job not found"
        }), 404

    # Calculate progress based on time elapsed
    elapsed = time.time() - status["start_time"]
    total_estimated = None
    if status.get("estimated_time_remaining") is not None:
        total_estimated = status["estimated_time_remaining"] - status["start_time"]

    # Progress as percentage (0-100), capped at 99 until finished
    progress_mode = status.get("progress_mode", "time")
    if status["status"] == "in_progress":
        if progress_mode == "manual":
            progress = status.get("progress", 0)
        else:
            if not total_estimated or total_estimated <= 0:
                progress = status.get("progress", 0)
            else:
                progress = min(99, int((elapsed / total_estimated) * 100))
    elif status["status"] == "finished":
        progress = 100
    else:
        progress = status.get("progress", 0)

    # Update the progress in job_status
    status["progress"] = progress
    
    return jsonify({
        "status": status["status"],
        "progress": status["progress"],
        "elapsed_time": int(time.time() - status["start_time"]),
        "estimated_time_remaining": status["estimated_time_remaining"],
        "error": status.get("error", False),
        "output": status.get("output", ""),
    })


@app.route("/download_results/<job_id>")
def download_results(job_id):
    """Handles the download of the final Pandas DataFrame."""
    status = job_status.get(job_id)

    # Get computation_type from the query parameters
    computation_type = request.args.get('computation_type') 

    if not status or status["status"] != "finished" or status["results"] is None:
        return "Results not available or computation incomplete.", 404

    # Retrieve the stored DataFrame
    output_df_path = status["results"] 

    # Remove file after downloading it (keep proxy outputs in /tmp)
    if computation_type != 'field_potential_proxy':
        @after_this_request
        def cleanup(response):
            try:
                if os.path.exists(output_df_path):
                    os.remove(output_df_path)
                    app.logger.info(f"Cleaned up {output_df_path}")
            except Exception as e:
                app.logger.error(f"Error removing file {output_df_path}: {e}")
            return response

    if computation_type == 'analysis':
        return send_file(
            f'{temp_uploaded_files}/LFP_predictions.png',
            mimetype='image/png',
            as_attachment=True,
            download_name='LFP_predictions.png'
        )
    if computation_type == 'field_potential_proxy':
        return send_file(
            output_df_path,
            mimetype='application/python-pickle',
            as_attachment=True,
            download_name=f'{computation_type}_results_{job_id}_proxy.pkl'
        )
    if computation_type in {'field_potential_kernel', 'field_potential_meeg'}:
        return send_file(
            output_df_path,
            mimetype='application/python-pickle',
            as_attachment=True,
            download_name=f'{computation_type}_results_{job_id}.pkl'
        )

    output_df = compute_utils.read_df_file(output_df_path)

    # Create an in-memory byte stream (io.BytesIO)
    output = io.BytesIO()
    # Save the DataFrame to Pickle in the in-memory stream
    output_df.to_pickle(output)
    output.seek(0)

    # Use send_file to trigger the download
    return send_file(
        output,
        mimetype='application/python-pickle',
        as_attachment=True,
        download_name=f'{computation_type}_results_{job_id}_output.pkl'
    )
