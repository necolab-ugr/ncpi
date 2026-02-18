import tempfile
import os
import shutil
import subprocess
import ast
import threading

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
    return render_template(
        "0.dashboard.html",
        simulation_pkl_files=simulation_pkl_files,
        has_simulation_pkl=bool(simulation_pkl_files),
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
    return render_template("5.analysis.html")


@app.route("/start_computation_redirect/<computation_type>", methods=["POST"])
def start_computation_redirect(computation_type):
    """Starts the background job and redirects to the status page."""
    # Allowed function names to redirect to
    allowed_functions = {'features', 'inference', 'analysis'}

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
    if len(uploaded_files) == 0:
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

    # Unique id for job
    job_id = str(uuid.uuid4())

    # If everything is OK, save the file(s)
    file_paths = {}
    for i, file_key in enumerate(request.files):
        file = request.files[file_key]
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
        "error": False
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

    # Remove file after downloading it
    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(output_df_path):
                os.remove(output_df_path)
                app.logger.info(f"Cleaned up {output_df_path}")
        except Exception as e:
            app.logger.error(f"Error removing file {output_df_path}: {e}")
        return response

    if computation_type != 'analysis':
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

    else: # computation_type == 'analysis'
        return send_file(
            f'{temp_uploaded_files}/LFP_predictions.png',
            mimetype='image/png',
            as_attachment=True,
            download_name='LFP_predictions.png'
        )
