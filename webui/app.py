import os
import shutil
import subprocess
import signal
import ast
import json
import threading
import glob
import base64
import pickle
import sys
import inspect
import re
import traceback
from pathlib import Path
from collections import deque, defaultdict
from collections.abc import Mapping as MappingABC
from itertools import product
from tmp_paths import configure_temp_environment, tmp_subdir

# Resolve shared temp root before importing other runtime modules.
configure_temp_environment()

_MODULE_TMP_NAMES = {"simulation", "field_potential", "features", "inference", "analysis"}


def _module_tmp_subdir(module_name, *parts, create=False):
    module_key = str(module_name or "").strip().lower()
    if module_key not in _MODULE_TMP_NAMES:
        raise ValueError(f"Unknown module tmp root: {module_name}")
    clean_parts = [module_key]
    for raw in parts:
        token = str(raw or "").strip().replace("\\", "/").strip("/")
        if not token:
            continue
        clean_parts.extend(seg for seg in token.split("/") if seg and seg != ".")
    rel_path = os.path.join(*clean_parts)
    return tmp_subdir(rel_path, create=create)


SIMULATION_DATA_DIR = _module_tmp_subdir("simulation", "data")
SIMULATION_RUNS_DIR = _module_tmp_subdir("simulation", "runs")
SIMULATION_CUSTOM_UPLOADS_DIR = _module_tmp_subdir("simulation", "custom_uploads")
FIELD_POTENTIAL_PROXY_DIR = _module_tmp_subdir("field_potential", "proxy")
FIELD_POTENTIAL_KERNEL_DIR = _module_tmp_subdir("field_potential", "kernel")
FIELD_POTENTIAL_MEEG_DIR = _module_tmp_subdir("field_potential", "meeg")
FIELD_POTENTIAL_KERNEL_LOCAL_UPLOADS_DIR = _module_tmp_subdir("field_potential", "kernel", "local_folder_uploads")
FEATURES_DATA_DIR = _module_tmp_subdir("features", "data")
FEATURES_INSPECTION_DIR = _module_tmp_subdir("features", "inspection")
PREDICTIONS_DATA_DIR = _module_tmp_subdir("inference", "predictions")
INFERENCE_UPLOADS_DIR = _module_tmp_subdir("inference", "uploads")
ANALYSIS_DATA_DIR = _module_tmp_subdir("analysis", "data")


def _module_uploads_dir_for(computation_type):
    key = str(computation_type or "").strip().lower()
    if key.startswith("field_potential"):
        return _module_tmp_subdir("field_potential", "uploads")
    if key == "simulation":
        return _module_tmp_subdir("simulation", "uploads")
    if key == "features":
        return _module_tmp_subdir("features", "uploads")
    if key in {"inference", "inference_training"}:
        return _module_tmp_subdir("inference", "uploads")
    if key == "analysis":
        return _module_tmp_subdir("analysis", "uploads")
    return INFERENCE_UPLOADS_DIR

# Prefer the local repository package over any globally installed ncpi version.
_webui_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_webui_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from flask import Flask, render_template, request, jsonify, url_for, redirect, send_file, after_this_request, flash, session
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
import io
import pandas as pd
import numpy as np
import compute_utils
import ncpi
from ncpi.EphysDatasetParser import EphysDatasetParser, ParseConfig, CanonicalFields

# Main app object
app = Flask(__name__)

# Set secret key for sessions (necessary to show alert messages)
app.secret_key = '602e6444-80b2-431c-b26c-b6cda2ac9c09'
app.config['TEMPLATES_AUTO_RELOAD'] = True
# In-memory thread pool
executor = ThreadPoolExecutor(max_workers=5) 

# Dictionary to store job progress/results (job_id: status_dict)
# NOTE: This dictionary is volatile and will reset if the server restarts.
job_status = {}
job_futures = {}
# Maximum number of terminal lines kept in memory for each job.
# Set NCPI_MAX_OUTPUT_LINES <= 0 for unlimited output retention (default).
MAX_OUTPUT_LINES = max(0, int(os.environ.get("NCPI_MAX_OUTPUT_LINES", "0")))
FEATURES_PARSER_FILE_EXTENSIONS = {
    ".mat", ".json", ".npy", ".csv", ".parquet", ".pkl", ".pickle", ".xlsx", ".xls", ".feather", ".set", ".tsv"
}
FEATURES_MAX_SUBFOLDER_DEPTH = 6
FILE_EXTRACTED_VIRTUAL_FIELD = "__file_extracted_label__"
FILE_EXTRACTED_VIRTUAL_FIELD_PREFIX = "__file_extracted_chain_"
FILE_ID_METADATA_LITERAL = "file_ID"
PICKLE_EXTENSIONS = {".pkl", ".pickle"}
MAX_EMPIRICAL_UPLOAD_BYTES = int(os.environ.get("NCPI_MAX_EMPIRICAL_UPLOAD_BYTES", str(1024 * 1024 * 1024)))
MAX_SIMULATION_GRID_COMBINATIONS = 256
GRID_PREFIX = "grid="
SIMULATION_GRID_METADATA_FILE = "grid_metadata.pkl"
SIMULATION_GRID_METADATA_LEGACY_FILES = {"simulation_grid_metadata.json", "simulation_grid_metadata.pkl"}
NATIVE_PATH_PICKER_ENABLED = str(os.environ.get("NCPI_ENABLE_NATIVE_PATH_PICKER", "0")).strip().lower() in {"1", "true", "yes", "on"}
PATH_HISTORY_CLIENT_ID_SESSION_KEY = "webui_path_history_client_id"
PATH_HISTORY_STORAGE_DIR = tmp_subdir("ncpi_webui_path_history")
PATH_HISTORY_MAX_FIELDS = 256
PATH_HISTORY_MAX_VALUE_LEN = 4096


HAGEN_DEFAULTS = {
    "tstop": 12000.0,
    "dt": 2 ** -4,
    "local_num_threads": 1,
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
    "local_num_threads": 1,
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

HAGEN_GRID_KEYS = [
    "tstop",
    "dt",
    "local_num_threads",
    "X",
    "N_X",
    "C_m_X",
    "tau_m_X",
    "E_L_X",
    "model",
    "C_YX",
    "J_YX",
    "delay_YX",
    "tau_syn_YX",
    "n_ext",
    "nu_ext",
    "J_ext",
]

FOUR_AREA_GRID_KEYS = [
    "tstop",
    "dt",
    "local_num_threads",
    "areas",
    "X",
    "N_X",
    "C_m_X",
    "tau_m_X",
    "E_L_X",
    "model",
    "J_EE",
    "J_IE",
    "J_EI",
    "J_II",
    "C_YX",
    "J_YX",
    "delay_YX",
    "tau_syn_YX",
    "n_ext",
    "nu_ext",
    "J_ext",
    "inter_area_scale",
    "inter_area_p",
    "inter_area_delay",
    "inter_area.C_YX",
    "inter_area.J_YX",
    "inter_area.delay_YX",
]


def _get_form_value(form, key):
    value = form.get(key)
    if value is None:
        return None
    value = value.strip()
    return value if value != "" else None


def _normalize_path_history_candidate(raw_value):
    text = str(raw_value or "").strip()
    if not text:
        return ""
    if len(text) > PATH_HISTORY_MAX_VALUE_LEN:
        return ""
    lowered = text.lower()
    if lowered in {"upload", "server-path", "local-picker", "none", "null"}:
        return ""
    if not (
        "/" in text
        or "\\" in text
        or text.startswith(("~", ".", "$"))
        or re.match(r"^[A-Za-z]:[\\/]", text)
    ):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(text))
    return os.path.realpath(expanded)


def _get_path_history_store():
    store = {}
    history_path = _path_history_storage_path()
    if history_path and os.path.isfile(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                store = loaded
        except Exception:
            store = {}
    by_field = store.get("by_field")
    if not isinstance(by_field, dict):
        by_field = {}
    latest_path = store.get("latest_path")
    if not isinstance(latest_path, str):
        latest_path = ""
    return {
        "by_field": by_field,
        "latest_path": latest_path.strip(),
    }


def _set_path_history_store(store):
    by_field = store.get("by_field") if isinstance(store, dict) else {}
    latest_path = store.get("latest_path") if isinstance(store, dict) else ""
    if not isinstance(by_field, dict):
        by_field = {}
    if len(by_field) > PATH_HISTORY_MAX_FIELDS:
        trimmed_items = list(by_field.items())[-PATH_HISTORY_MAX_FIELDS:]
        by_field = {key: value for key, value in trimmed_items}
    if not isinstance(latest_path, str):
        latest_path = ""
    payload = {
        "by_field": by_field,
        "latest_path": latest_path.strip(),
    }
    history_path = _path_history_storage_path()
    if not history_path:
        return
    try:
        parent = os.path.dirname(history_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{history_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
        os.replace(tmp_path, history_path)
    except Exception:
        return


def _path_history_client_id():
    existing = session.get(PATH_HISTORY_CLIENT_ID_SESSION_KEY)
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    new_id = uuid.uuid4().hex
    session[PATH_HISTORY_CLIENT_ID_SESSION_KEY] = new_id
    session.modified = True
    return new_id


def _path_history_storage_path():
    try:
        client_id = _path_history_client_id()
    except Exception:
        return ""
    if not client_id:
        return ""
    safe_client_id = re.sub(r"[^a-zA-Z0-9_-]", "", client_id)[:64]
    if not safe_client_id:
        return ""
    return os.path.join(PATH_HISTORY_STORAGE_DIR, f"{safe_client_id}.json")


def _remember_path_history(field_key, value):
    normalized = _normalize_path_history_candidate(value)
    if not normalized:
        return ""
    key = str(field_key or "").strip() or "path"
    store = _get_path_history_store()
    store["by_field"][key] = normalized
    store["latest_path"] = normalized
    _set_path_history_store(store)
    return normalized


def _remember_path_history_from_form(form, route_key):
    if form is None:
        return
    try:
        keys = list(form.keys())
    except Exception:
        return
    for key in keys:
        key_text = str(key or "").strip()
        if not key_text:
            continue
        try:
            values = form.getlist(key_text)
        except Exception:
            values = [form.get(key_text)]
        for idx, raw_value in enumerate(values):
            value_text = str(raw_value or "")
            chunks = value_text.replace("\r", "\n").split("\n")
            for chunk in chunks:
                label = f"{route_key}:{key_text}"
                if len(values) > 1:
                    label = f"{label}[{idx}]"
                _remember_path_history(label, chunk)


def _path_history_namespace_key(history_key):
    raw = str(history_key or "").strip()
    if not raw:
        return ""
    safe = re.sub(r"[^a-zA-Z0-9_.:-]", "", raw)[:120]
    if not safe:
        return ""
    return f"browser:{safe}"


def _path_history_value_for_key(history_key):
    namespace_key = _path_history_namespace_key(history_key)
    if not namespace_key:
        return ""
    store = _get_path_history_store()
    by_field = store.get("by_field")
    if not isinstance(by_field, dict):
        return ""
    value = by_field.get(namespace_key)
    return str(value).strip() if isinstance(value, str) else ""


def _remember_path_history_for_key(history_key, value):
    namespace_key = _path_history_namespace_key(history_key)
    if not namespace_key:
        return ""
    return _remember_path_history(namespace_key, value)


def _existing_directory_from_candidate(path_value):
    candidate = str(path_value or "").strip()
    if not candidate:
        return ""
    if os.path.isdir(candidate):
        return candidate
    parent = os.path.dirname(candidate)
    if parent and os.path.isdir(parent):
        return parent
    return ""


def _path_history_latest(default_value=""):
    latest = _get_path_history_store().get("latest_path") or ""
    return latest if latest else default_value


def _path_history_start_directory(default_value, history_key=None):
    namespace_key = _path_history_namespace_key(history_key)
    if namespace_key:
        keyed_value = _path_history_value_for_key(history_key)
        keyed_dir = _existing_directory_from_candidate(keyed_value)
        if keyed_dir:
            return keyed_dir
        return default_value
    latest = _path_history_latest(default_value="")
    latest_dir = _existing_directory_from_candidate(latest)
    if latest_dir:
        return latest_dir
    return default_value


def _path_history_snapshot():
    store = _get_path_history_store()
    by_field = {
        str(key): str(value)
        for key, value in store.get("by_field", {}).items()
        if isinstance(key, str) and isinstance(value, str)
    }
    return {
        "latest_path": store.get("latest_path", ""),
        "by_field": by_field,
    }


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


def _parse_optional_sim_numpy_seed(form):
    use_fixed_seed = _get_form_value(form, "sim_use_numpy_seed") is not None
    if not use_fixed_seed:
        return None
    seed = _parse_int(form, "sim_numpy_seed", 0)
    if seed < 0:
        raise ValueError("NumPy seed must be a non-negative integer.")
    return int(seed)


def _format_value(value):
    return repr(value)


def _append_job_output(job_status, job_id, message):
    if job_id not in job_status:
        return
    line = str(message).strip()
    if not line:
        return
    current = str(job_status[job_id].get("output", "") or "")
    if current:
        current = f"{current}\n{line}"
    else:
        current = line
    if MAX_OUTPUT_LINES > 0:
        lines = current.splitlines()
        if len(lines) > MAX_OUTPUT_LINES:
            current = "\n".join(lines[-MAX_OUTPUT_LINES:])
    job_status[job_id]["output"] = current


def _announce_saved_output_folders(job_status, job_id, module_name, *paths):
    folders = []
    seen = set()
    for raw in paths:
        if not raw:
            continue
        candidate = os.path.realpath(str(raw))
        folder = candidate if os.path.isdir(candidate) else os.path.dirname(candidate)
        if not folder:
            continue
        folder = os.path.realpath(folder)
        if folder in seen:
            continue
        seen.add(folder)
        folders.append(folder)
    if folders:
        message = f"[{module_name}] output folder(s): {', '.join(folders)}"
        _append_job_output(job_status, job_id, message)
        print(message, flush=True)


class JobCancelledError(RuntimeError):
    """Raised when a running job is cancelled by user request."""


def _is_job_cancel_requested(job_id):
    status = job_status.get(job_id)
    return bool(isinstance(status, dict) and status.get("cancel_requested"))


def _pid_exists(pid):
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def _pkill_python_children(parent_pid, sig_name="TERM"):
    pkill_path = shutil.which("pkill")
    if not pkill_path:
        return "pkill unavailable; skipped child-process kill."

    cmd = [pkill_path, f"-{sig_name}", "-P", str(int(parent_pid)), "-f", "python"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        return f"pkill {sig_name} failed: {exc}"

    stderr_text = (proc.stderr or "").strip()
    if proc.returncode == 0:
        return f"pkill {sig_name}: sent to python child process(es)."
    if proc.returncode == 1:
        return f"pkill {sig_name}: no matching python child processes."
    return f"pkill {sig_name} returned code {proc.returncode}: {stderr_text or 'no details'}"


def _terminate_tracked_worker_process(job_id):
    status = job_status.get(job_id)
    if not isinstance(status, dict):
        return None

    worker_pid = status.get("worker_pid")
    if worker_pid is None:
        return "No tracked worker process for this job."

    try:
        worker_pid = int(worker_pid)
    except (TypeError, ValueError):
        status.pop("worker_pid", None)
        status.pop("worker_isolated_pgrp", None)
        return "Tracked worker pid is invalid."

    if worker_pid <= 0:
        status.pop("worker_pid", None)
        status.pop("worker_isolated_pgrp", None)
        return "Tracked worker pid is invalid."

    if not _pid_exists(worker_pid):
        status.pop("worker_pid", None)
        status.pop("worker_isolated_pgrp", None)
        return f"Tracked worker process {worker_pid} already exited."

    isolated_group = bool(status.get("worker_isolated_pgrp"))
    try:
        if isolated_group:
            os.killpg(worker_pid, signal.SIGTERM)
        else:
            os.kill(worker_pid, signal.SIGTERM)
    except ProcessLookupError:
        status.pop("worker_pid", None)
        status.pop("worker_isolated_pgrp", None)
        return f"Tracked worker process {worker_pid} already exited."
    except Exception as exc:
        return f"Failed to send SIGTERM to worker process {worker_pid}: {exc}"

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not _pid_exists(worker_pid):
            status.pop("worker_pid", None)
            status.pop("worker_isolated_pgrp", None)
            return f"Worker process {worker_pid} terminated."
        time.sleep(0.1)

    try:
        if isolated_group:
            os.killpg(worker_pid, signal.SIGKILL)
        else:
            os.kill(worker_pid, signal.SIGKILL)
    except ProcessLookupError:
        status.pop("worker_pid", None)
        status.pop("worker_isolated_pgrp", None)
        return f"Worker process {worker_pid} terminated."
    except Exception as exc:
        return f"Failed to send SIGKILL to worker process {worker_pid}: {exc}"

    status.pop("worker_pid", None)
    status.pop("worker_isolated_pgrp", None)
    return f"Worker process {worker_pid} force-killed."


def _cancel_job_python_processes(job_id):
    status = job_status.get(job_id)
    if not isinstance(status, dict):
        return []

    messages = []
    tracked_msg = _terminate_tracked_worker_process(job_id)
    if tracked_msg:
        messages.append(tracked_msg)

    computation_type = str(status.get("computation_type") or "").strip().lower()
    should_pkill_children = computation_type in {
        "features",
        "inference",
        "inference_training",
        "field_potential_proxy",
        "field_potential_kernel",
        "field_potential_meeg",
    }
    if should_pkill_children:
        parent_pid = os.getpid()
        messages.append(_pkill_python_children(parent_pid, "TERM"))
        time.sleep(0.2)
        messages.append(_pkill_python_children(parent_pid, "KILL"))

    return messages


def _mark_job_cancelled(job_id, reason="Computation cancelled by user."):
    status = job_status.get(job_id)
    if not isinstance(status, dict):
        return
    already_cancelled = status.get("status") == "cancelled"
    status.update({
        "status": "cancelled",
        "error": False,
        "cancel_requested": True,
        "cancelled_at": time.time(),
        "cancel_message": reason,
        "estimated_time_remaining": time.time(),
        "progress": status.get("progress", 0),
    })
    if not already_cancelled:
        _append_job_output(job_status, job_id, reason)


def _mark_job_failed(job_id, exc, *, log_traceback=True, prefix="Error"):
    status = job_status.get(job_id)
    if isinstance(status, dict) and (status.get("status") == "cancelled" or status.get("cancel_requested")):
        _mark_job_cancelled(job_id, status.get("cancel_message") or "Computation cancelled by user.")
        return

    message = str(exc).strip()
    if message:
        error_text = f"{type(exc).__name__}: {message}"
    else:
        error_text = type(exc).__name__
    _append_job_output(job_status, job_id, f"{prefix}: {error_text}")

    if log_traceback:
        tb_text = traceback.format_exc().strip()
        if tb_text and tb_text != "NoneType: None":
            for line in tb_text.splitlines():
                _append_job_output(job_status, job_id, line)

    if isinstance(status, dict):
        status.update({
            "status": "failed",
            "error": error_text,
            "progress": status.get("progress", 0),
        })


def _is_numeric_default(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_grid_candidate(value, default, key):
    if isinstance(default, str):
        if not isinstance(value, str):
            return str(value)
        return value
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid grid candidate for '{key}': {value}") from exc
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid grid candidate for '{key}': {value}") from exc
    return value


def _expand_numeric_range(spec, key):
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid grid range for '{key}'. Use 'grid=start:stop:step'."
        )
    try:
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"Invalid grid range for '{key}'. Use numeric values in 'start:stop:step'."
        ) from exc
    if step == 0:
        raise ValueError(f"Invalid grid range for '{key}': step cannot be 0.")

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

    if not values:
        raise ValueError(f"Grid range for '{key}' generated no values.")
    return values


def _parse_grid_candidates(form, key, default):
    raw = _get_form_value(form, key)
    if raw is None:
        return [default], False

    if not raw.lower().startswith(GRID_PREFIX):
        if isinstance(default, str):
            return [_parse_str({key: raw}, key, default)], False
        if isinstance(default, int) and not isinstance(default, bool):
            return [_parse_int({key: raw}, key, default)], False
        if isinstance(default, float):
            return [_parse_float({key: raw}, key, default)], False
        return [_parse_literal({key: raw}, key, default)], False

    spec = raw[len(GRID_PREFIX):].strip()
    if not spec:
        raise ValueError(
            f"Grid definition for '{key}' is empty. Use e.g. grid=[v1, v2] or grid=start:stop:step."
        )

    if _is_numeric_default(default) and ":" in spec and not any(ch in spec for ch in "[]{}(),"):
        candidates = _expand_numeric_range(spec, key)
    else:
        try:
            parsed = ast.literal_eval(spec)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Invalid grid definition for '{key}'. Use e.g. grid=[v1, v2] or grid=start:stop:step."
            ) from exc
        if isinstance(parsed, (list, tuple)):
            candidates = list(parsed)
        else:
            candidates = [parsed]

    if not candidates:
        raise ValueError(f"Grid definition for '{key}' produced no values.")
    coerced = [_coerce_grid_candidate(candidate, default, key) for candidate in candidates]
    return coerced, True


def _simulation_grid_defaults(model_type):
    if model_type == "hagen":
        defaults = {
            "tstop": HAGEN_DEFAULTS["tstop"],
            "dt": HAGEN_DEFAULTS["dt"],
            "local_num_threads": HAGEN_DEFAULTS["local_num_threads"],
        }
        defaults.update({key: HAGEN_DEFAULTS[key] for key in HAGEN_GRID_KEYS if key in HAGEN_DEFAULTS})
        return defaults

    inter_area_p = FOUR_AREA_DEFAULTS["inter_area_p"]
    inter_area_scale = FOUR_AREA_DEFAULTS["inter_area_scale"]
    inter_area_delay = FOUR_AREA_DEFAULTS["inter_area_delay"]
    j_ee = FOUR_AREA_DEFAULTS["J_EE"]
    j_ie = FOUR_AREA_DEFAULTS["J_IE"]
    j_ei = FOUR_AREA_DEFAULTS["J_EI"]
    j_ii = FOUR_AREA_DEFAULTS["J_II"]
    defaults = {
        "tstop": FOUR_AREA_DEFAULTS["tstop"],
        "dt": FOUR_AREA_DEFAULTS["dt"],
        "local_num_threads": FOUR_AREA_DEFAULTS["local_num_threads"],
        "areas": FOUR_AREA_DEFAULTS["areas"],
        "X": FOUR_AREA_DEFAULTS["X"],
        "N_X": FOUR_AREA_DEFAULTS["N_X"],
        "C_m_X": FOUR_AREA_DEFAULTS["C_m_X"],
        "tau_m_X": FOUR_AREA_DEFAULTS["tau_m_X"],
        "E_L_X": FOUR_AREA_DEFAULTS["E_L_X"],
        "model": FOUR_AREA_DEFAULTS["model"],
        "J_EE": j_ee,
        "J_IE": j_ie,
        "J_EI": j_ei,
        "J_II": j_ii,
        "C_YX": FOUR_AREA_DEFAULTS["C_YX"],
        "J_YX": [[j_ee, j_ie], [j_ei, j_ii]],
        "delay_YX": FOUR_AREA_DEFAULTS["delay_YX"],
        "tau_syn_YX": FOUR_AREA_DEFAULTS["tau_syn_YX"],
        "n_ext": FOUR_AREA_DEFAULTS["n_ext"],
        "nu_ext": FOUR_AREA_DEFAULTS["nu_ext"],
        "J_ext": FOUR_AREA_DEFAULTS["J_ext"],
        "inter_area_scale": inter_area_scale,
        "inter_area_p": inter_area_p,
        "inter_area_delay": inter_area_delay,
        "inter_area.C_YX": [[inter_area_p, inter_area_p], [0.0, 0.0]],
        "inter_area.J_YX": [
            [j_ee * inter_area_scale, j_ie * inter_area_scale],
            [0.0, 0.0],
        ],
        "inter_area.delay_YX": [[inter_area_delay, inter_area_delay], [0.0, 0.0]],
    }
    return defaults


def _to_form_value(value):
    if isinstance(value, str):
        return value
    return _format_value(value)


def _expand_simulation_forms(model_type, form):
    run_mode = (_get_form_value(form, "sim_run_mode") or "single").lower()
    if run_mode not in {"single", "grid"}:
        raise ValueError("Invalid simulation mode. Use single trial or parameter grid sweep.")

    repetitions = _parse_int(form, "sim_repetitions", 1)
    if repetitions < 1:
        raise ValueError("Repetitions per parameter configuration must be >= 1.")

    if run_mode == "single":
        expanded_forms = []
        for repeat_index in range(repetitions):
            trial_form = dict(form)
            trial_form["sim_run_mode"] = "single"
            trial_form["sim_repetitions"] = str(repetitions)
            trial_form["sim_combo_index"] = "0"
            trial_form["sim_repeat_index"] = str(repeat_index)
            expanded_forms.append(trial_form)
        return run_mode, expanded_forms

    grid_defaults = _simulation_grid_defaults(model_type)
    ordered_keys = list(grid_defaults.keys())

    candidate_lists = {}
    combinations = 1
    for key in ordered_keys:
        candidates, _ = _parse_grid_candidates(form, key, grid_defaults[key])
        candidate_lists[key] = candidates
        combinations *= len(candidates)

    total = combinations * repetitions

    if total > MAX_SIMULATION_GRID_COMBINATIONS:
        raise ValueError(
            f"Grid expands to {combinations} configurations x {repetitions} repetition(s) = {total} simulations; "
            f"maximum allowed is {MAX_SIMULATION_GRID_COMBINATIONS}."
        )

    expanded_forms = []
    for combo_index, combo in enumerate(product(*(candidate_lists[key] for key in ordered_keys))):
        for repeat_index in range(repetitions):
            trial_form = dict(form)
            trial_form["sim_run_mode"] = "grid"
            trial_form["sim_repetitions"] = str(repetitions)
            trial_form["sim_combo_index"] = str(combo_index)
            trial_form["sim_repeat_index"] = str(repeat_index)
            for key, value in zip(ordered_keys, combo):
                trial_form[key] = _to_form_value(value)
            expanded_forms.append(trial_form)

    if not expanded_forms:
        raise ValueError("No simulations generated from the selected parameter grid sweep.")
    return run_mode, expanded_forms


def _values_equal_for_grid(left, right, tol=1e-12):
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_values_equal_for_grid(lv, rv, tol=tol) for lv, rv in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_values_equal_for_grid(left[k], right[k], tol=tol) for k in left.keys())
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tol
    return left == right


def _normalize_simulation_form_values(form, defaults):
    normalized = {}
    for key, default in defaults.items():
        candidates, _ = _parse_grid_candidates(form, key, default)
        normalized[key] = candidates[0] if candidates else default
    return normalized


def _build_simulation_grid_metadata(model_type, run_mode, run_forms):
    if run_mode != "grid":
        return None
    defaults = _simulation_grid_defaults(model_type)
    ordered_keys = list(defaults.keys())
    trial_values = [_normalize_simulation_form_values(form, defaults) for form in run_forms]
    if not trial_values:
        return None

    baseline = trial_values[0]
    changed_keys = []
    for key in ordered_keys:
        if any(
            not _values_equal_for_grid(trial.get(key), baseline.get(key))
            for trial in trial_values[1:]
        ):
            changed_keys.append(key)

    repetitions = 1
    if run_forms:
        try:
            repetitions = max(1, _parse_int(run_forms[0], "sim_repetitions", 1))
        except Exception:
            repetitions = 1
    configuration_count = max(1, (len(trial_values) + repetitions - 1) // repetitions)

    trials = []
    for trial_idx, trial in enumerate(trial_values):
        try:
            combo_index = max(0, _parse_int(run_forms[trial_idx], "sim_combo_index", trial_idx // repetitions))
        except Exception:
            combo_index = trial_idx // repetitions
        try:
            repeat_index = max(0, _parse_int(run_forms[trial_idx], "sim_repeat_index", trial_idx % repetitions))
        except Exception:
            repeat_index = trial_idx % repetitions
        changed = {key: trial.get(key) for key in changed_keys}
        trials.append({
            "trial_index": int(trial_idx),
            "configuration_index": int(combo_index),
            "repeat_index": int(repeat_index),
            "changed": changed,
        })

    return {
        "version": 1,
        "model_type": model_type,
        "run_mode": run_mode,
        "trial_count": len(trial_values),
        "configuration_count": configuration_count,
        "repetitions_per_configuration": int(repetitions),
        "changed_keys": changed_keys,
        "trials": trials,
    }


def _simulation_grid_metadata_path(output_dir=None):
    root = output_dir if output_dir else SIMULATION_DATA_DIR
    return os.path.join(root, SIMULATION_GRID_METADATA_FILE)


def _clear_simulation_grid_metadata_file(output_dir=None):
    root = output_dir if output_dir else SIMULATION_DATA_DIR
    targets = {SIMULATION_GRID_METADATA_FILE, *SIMULATION_GRID_METADATA_LEGACY_FILES}
    for name in targets:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _write_simulation_grid_metadata(output_dir, metadata):
    if not metadata:
        return None
    metadata_path = _simulation_grid_metadata_path(output_dir)
    with open(metadata_path, "wb") as handle:
        pickle.dump(metadata, handle)
    return metadata_path


def _ensure_sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"'{name}' must be a list or tuple.")


def _field_potential_dirs():
    return [
        FIELD_POTENTIAL_PROXY_DIR,
        FIELD_POTENTIAL_KERNEL_DIR,
        FIELD_POTENTIAL_MEEG_DIR,
    ]


def _infer_field_potential_type_from_filename(filename):
    lower_name = os.path.basename(filename or "").lower()
    if not lower_name:
        return None

    if lower_name.startswith("proxy") or "proxy" in lower_name:
        return "proxy"
    if lower_name in {"kernel_approx_cdm.pkl", "current_dipole_moment.pkl"} or "cdm" in lower_name:
        return "cdm"
    if lower_name in {"gauss_cylinder_potential.pkl", "probe_outputs.pkl", "lfp.pkl"} or "lfp" in lower_name:
        return "lfp"
    if lower_name == "eeg.pkl" or "eeg" in lower_name:
        return "eeg"
    if lower_name == "meg.pkl" or "meg" in lower_name:
        return "meg"
    if lower_name == "meeg.pkl":
        return "meeg"
    return None


def _field_potential_output_name(fp_type, safe_name):
    # Preserve user-provided names for precomputed load-data uploads.
    return safe_name


def _features_data_dir(create=False):
    return _module_tmp_subdir("features", "data", create=create)


def _predictions_data_dir(create=False):
    return _module_tmp_subdir("inference", "predictions", create=create)


def _features_data_candidate_dirs():
    return [os.path.realpath(_features_data_dir(create=False))]


def _sync_features_data_from_candidates():
    primary = os.path.realpath(_features_data_dir(create=False))
    if not os.path.isdir(primary):
        return
    for candidate_dir in _features_data_candidate_dirs():
        if candidate_dir == primary:
            continue
        if not os.path.isdir(candidate_dir):
            continue
        try:
            names = os.listdir(candidate_dir)
        except OSError:
            continue
        for name in names:
            if not (name.endswith(".pkl") or name.endswith(".pickle")):
                continue
            src = os.path.join(candidate_dir, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(primary, name)
            should_copy = False
            if not os.path.isfile(dst):
                should_copy = True
            else:
                try:
                    should_copy = os.path.getmtime(src) > os.path.getmtime(dst)
                except OSError:
                    should_copy = False
            if should_copy:
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    continue


def _remove_dir_if_empty(path):
    target = os.path.realpath(path or "")
    if not target or not os.path.isdir(target):
        return False
    try:
        if os.listdir(target):
            return False
        os.rmdir(target)
        return True
    except OSError:
        return False


def _list_features_data_files():
    _sync_features_data_from_candidates()
    features_dir = _features_data_dir(create=False)
    if not os.path.isdir(features_dir):
        return []
    files = []
    for root, _, names in os.walk(features_dir):
        for name in names:
            lower = name.lower()
            if not (lower.endswith(".pkl") or lower.endswith(".pickle")):
                continue
            abs_path = os.path.join(root, name)
            if not os.path.isfile(abs_path):
                continue
            rel_path = os.path.relpath(abs_path, features_dir)
            files.append(rel_path)
    return sorted(files)


def _bootstrap_features_data_from_previous_steps():
    """
    Ensure inference can auto-detect features from earlier steps.
    If features_data folder is empty, try to recover the latest persisted features
    output from completed features jobs and copy it into features_data.
    """
    if _list_features_data_files():
        return False

    candidates = []
    for status in job_status.values():
        if not isinstance(status, dict):
            continue
        candidate = status.get("dashboard_features_path")
        if not candidate:
            continue
        if not os.path.isfile(candidate):
            continue
        ext = Path(candidate).suffix.lower()
        if ext not in {".pkl", ".pickle"}:
            continue
        try:
            mtime = os.path.getmtime(candidate)
        except OSError:
            continue
        candidates.append((mtime, candidate))

    if not candidates:
        return False

    _, src_path = max(candidates, key=lambda item: item[0])
    features_dir = _features_data_dir(create=True)
    dst_name = os.path.basename(src_path)
    dst_path = os.path.join(features_dir, dst_name)

    # Avoid noisy copy errors if source and destination are already the same.
    if os.path.realpath(src_path) != os.path.realpath(dst_path):
        shutil.copy2(src_path, dst_path)
    return True


def _list_predictions_data_files():
    predictions_dir = _predictions_data_dir(create=False)
    if not os.path.isdir(predictions_dir):
        return []
    files = []
    for root, _, names in os.walk(predictions_dir):
        for name in names:
            lower = name.lower()
            if not (lower.endswith(".pkl") or lower.endswith(".pickle")):
                continue
            abs_path = os.path.join(root, name)
            if not os.path.isfile(abs_path):
                continue
            rel_path = os.path.relpath(abs_path, predictions_dir)
            files.append(rel_path)
    return sorted(files)


def _list_simulation_data_files():
    simulation_dir = SIMULATION_DATA_DIR
    if not os.path.isdir(simulation_dir):
        return []
    files = []
    for name in os.listdir(simulation_dir):
        lower = name.lower()
        if not (lower.endswith(".pkl") or lower.endswith(".pickle")):
            continue
        abs_path = os.path.join(simulation_dir, name)
        if not os.path.isfile(abs_path):
            continue
        files.append(name)
    return sorted(files)


def _list_field_potential_detected_files():
    entries = []
    seen_paths = set()
    unique_roots = []
    for root in _field_potential_dirs():
        real_root = os.path.realpath(root)
        if real_root in unique_roots:
            continue
        unique_roots.append(real_root)

    for root in unique_roots:
        if not os.path.isdir(root):
            continue
        root_label = os.path.basename(root)
        if "proxy" in root_label:
            mode_label = "Field Potential Proxy"
        elif "kernel" in root_label:
            mode_label = "Field Potential Kernel"
        elif "meeg" in root_label:
            mode_label = "Field Potential M/EEG"
        else:
            mode_label = "Field Potential"

        for current_root, _, files in os.walk(root):
            for name in files:
                lower = name.lower()
                if not (lower.endswith(".pkl") or lower.endswith(".pickle")):
                    continue
                abs_path = os.path.realpath(os.path.join(current_root, name))
                if not os.path.isfile(abs_path):
                    continue
                if abs_path in seen_paths:
                    continue
                seen_paths.add(abs_path)
                rel_name = os.path.relpath(abs_path, root)
                display_name = os.path.basename(abs_path)
                entries.append({
                    "key": f"field_potential::{abs_path}",
                    "name": display_name,
                    "relative_name": rel_name,
                    "source": "field_potential",
                    "label": mode_label,
                    "path": abs_path,
                })
    entries.sort(key=lambda item: (item["label"], item["name"]))
    return entries


def _list_detected_analysis_data_files():
    entries = []

    features_root = _features_data_dir(create=False)
    for name in _list_features_data_files():
        entries.append({
            "key": f"features::{name}",
            "name": name,
            "source": "features",
            "label": "Features",
            "path": os.path.join(features_root, name),
        })

    predictions_root = _predictions_data_dir(create=False)
    for name in _list_predictions_data_files():
        entries.append({
            "key": f"predictions::{name}",
            "name": name,
            "source": "predictions",
            "label": "Predictions",
            "path": os.path.join(predictions_root, name),
        })

    simulation_root = SIMULATION_DATA_DIR
    for name in _list_simulation_data_files():
        entries.append({
            "key": f"simulation::{name}",
            "name": name,
            "source": "simulation",
            "label": "Simulation",
            "path": os.path.join(simulation_root, name),
        })

    entries.extend(_list_field_potential_detected_files())
    return entries


def _bootstrap_predictions_data_from_previous_steps():
    if _list_predictions_data_files():
        return False

    candidates = []
    for status in job_status.values():
        if not isinstance(status, dict):
            continue
        candidate = status.get("dashboard_predictions_path")
        if not candidate:
            continue
        if not os.path.isfile(candidate):
            continue
        ext = Path(candidate).suffix.lower()
        if ext not in {".pkl", ".pickle"}:
            continue
        try:
            mtime = os.path.getmtime(candidate)
        except OSError:
            continue
        candidates.append((mtime, candidate))

    if not candidates:
        return False

    _, src_path = max(candidates, key=lambda item: item[0])
    predictions_dir = _predictions_data_dir(create=True)
    dst_name = os.path.basename(src_path)
    dst_path = os.path.join(predictions_dir, dst_name)
    if os.path.realpath(src_path) != os.path.realpath(dst_path):
        shutil.copy2(src_path, dst_path)
    return True


def _collect_empirical_folder_files(folder_path):
    root = os.path.realpath((folder_path or "").strip())
    if not root:
        raise ValueError("Provide a folder path for empirical recordings.")
    if not os.path.isdir(root):
        raise ValueError(f"Empirical folder does not exist: {root}")

    matches = []
    for current_root, _, files in os.walk(root):
        for name in files:
            ext = Path(name).suffix.lower()
            if ext in FEATURES_PARSER_FILE_EXTENSIONS:
                matches.append(os.path.join(current_root, name))
    matches.sort()
    if not matches:
        raise ValueError(f"No supported empirical files found in folder: {root}")
    return matches


def _collect_simulation_folder_files(folder_path):
    root = os.path.realpath((folder_path or "").strip())
    if not root:
        raise ValueError("Provide a folder path for simulation outputs.")
    if not os.path.isdir(root):
        raise ValueError(f"Simulation outputs folder does not exist: {root}")

    matches = []
    for current_root, _, files in os.walk(root):
        for name in files:
            ext = Path(name).suffix.lower()
            if ext in FEATURES_PARSER_FILE_EXTENSIONS:
                matches.append(os.path.join(current_root, name))
    matches.sort()
    if not matches:
        raise ValueError(f"No supported simulation output files found in folder: {root}")
    return matches


def _parse_folder_paths_payload(raw_value):
    raw = str(raw_value or "").strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    if "\n" in raw:
        parts = raw.splitlines()
    elif "," in raw:
        parts = raw.split(",")
    else:
        parts = [raw]
    return [str(item).strip() for item in parts if str(item).strip()]


def _extract_folder_paths_from_form(form, singular_key, plural_key):
    candidates = []
    if plural_key:
        plural_values = form.getlist(plural_key)
        for raw in plural_values:
            candidates.extend(_parse_folder_paths_payload(raw))
        if not plural_values:
            candidates.extend(_parse_folder_paths_payload(form.get(plural_key)))
    if singular_key:
        singular_value = (form.get(singular_key) or "").strip()
        if singular_value:
            candidates.append(singular_value)

    normalized = []
    seen = set()
    for raw in candidates:
        real = os.path.realpath(str(raw or "").strip())
        if not real or real in seen:
            continue
        seen.add(real)
        normalized.append(real)
    return normalized


def _extract_text_list_from_form(form, key):
    values = []
    raw_values = form.getlist(key)
    for raw in raw_values:
        values.extend(_parse_folder_paths_payload(raw))
    if not raw_values:
        values.extend(_parse_folder_paths_payload(form.get(key)))

    normalized = []
    seen = set()
    for raw in values:
        token = str(raw or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _extract_data_file_selection_map_from_form(form, key):
    raw_value = (form.get(key) or "").strip()
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}

    selections = {}
    for raw_folder, raw_value in parsed.items():
        folder_token = str(raw_folder or "").strip()
        value_token = str(raw_value or "").replace("\\", "/").strip().lstrip("/")
        if not folder_token or not value_token:
            continue
        folder_path = os.path.realpath(folder_token)
        if folder_path:
            selections[folder_path] = value_token
        selections[folder_token] = value_token
    return selections


def _extract_subfolder_filter_map_from_form(form, key):
    """Parse {folder_path: [included_subfolder, ...]} from a form field JSON string.

    Returns {} if absent or malformed.
    A null/missing entry for a folder means 'include all subfolders' (no filter).
    An explicit list means only those subfolders are included.
    """
    raw = (form.get(key) or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    result = {}
    for raw_folder, raw_list in parsed.items():
        token = str(raw_folder or "").strip()
        if not token:
            continue
        included = [str(s).strip() for s in (raw_list or []) if str(s).strip()]
        real_path = os.path.realpath(token)
        if real_path:
            result[real_path] = included
        result[token] = included
    return result


def _to_posix_relpath(path):
    token = str(path or "").replace("\\", "/").strip().lstrip("/")
    return token or "."


def _format_extension_count_map(ext_map):
    items = []
    for ext, count in sorted((ext_map or {}).items(), key=lambda item: str(item[0])):
        items.append(f"{ext}:{int(count)}")
    return ", ".join(items) if items else "(no files)"


def _numeric_wildcard_stem(stem):
    token = str(stem or "").strip()
    if not token:
        return "", None
    wildcard = re.sub(r"\d+", "*", token)
    if "*" not in wildcard:
        return wildcard, None
    regex_src = "^" + re.escape(wildcard).replace(r"\*", r"\d+") + "$"
    return wildcard, re.compile(regex_src)


def _extract_analysis_folder_name_tokens_from_form(form, key="parser_analysis_folder_names"):
    tokens = []
    for raw in _extract_text_list_from_form(form, key):
        safe = secure_filename(str(raw or "")).strip("_")
        token = safe or "selected_folder"
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _resolve_selected_analysis_folder_paths(form, folder_summaries):
    available_paths = []
    seen = set()
    for item in folder_summaries or []:
        path = str(item.get("folder_path") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        available_paths.append(path)

    if not available_paths:
        return []

    selected_paths = _extract_folder_paths_from_form(
        form,
        singular_key=None,
        plural_key="parser_analysis_folder_paths",
    )
    if len(selected_paths) > 1:
        raise ValueError("Select only one folder for feature extraction.")

    selected_path = selected_paths[0] if selected_paths else ""
    available_set = set(available_paths)
    unknown = [selected_path] if selected_path and selected_path not in available_set else []
    if unknown:
        raise ValueError(
            "Unknown folders selected for feature extraction: "
            + ", ".join(unknown[:4])
            + (", ..." if len(unknown) > 4 else "")
        )

    if "parser_analysis_folder_paths" in form and len(available_paths) > 1 and not selected_path:
        raise ValueError("Select one folder for feature extraction.")

    if not selected_path:
        return [available_paths[0]]
    return [selected_path]


def _validate_uniform_supported_extension_per_folder(entries, label):
    ext_by_folder = defaultdict(set)
    folder_labels = {}
    for row in entries or []:
        folder_key = str(row.get("folder_path") or row.get("folder_name") or "selected_folder").strip() or "selected_folder"
        folder_labels[folder_key] = str(row.get("folder_name") or row.get("folder_path") or folder_key).strip() or folder_key
        ext = str(row.get("extension") or "").strip().lower()
        if ext:
            ext_by_folder[folder_key].add(ext)

    for folder_key, ext_set in sorted(ext_by_folder.items(), key=lambda item: str(item[0])):
        if len(ext_set) <= 1:
            continue
        folder_label = folder_labels.get(folder_key, folder_key)
        ext_list = ", ".join(sorted(ext_set))
        raise ValueError(
            f"{label} folder '{folder_label}' contains multiple supported file extensions: {ext_list}. "
            "Use one file extension per folder."
        )


def _folder_display_name(folder_path, fallback_index=1):
    base = os.path.basename(os.path.normpath(str(folder_path or "")))
    token = secure_filename(base).strip("_")
    if token:
        return token
    return f"folder_{int(fallback_index)}"


def _prefixed_file_name(file_name, folder_name=None, apply_prefix=False):
    safe_name = secure_filename(str(file_name or ""))
    if not safe_name:
        safe_name = "file"
    if not apply_prefix:
        return safe_name
    folder_token = secure_filename(str(folder_name or "")).strip("_")
    if not folder_token:
        return safe_name
    return f"{folder_token}_{safe_name}"


def _collect_supported_folder_file_entries(folder_paths, kind_label, data_file_selection_map=None):
    roots = []
    seen_roots = set()
    for idx, raw in enumerate(folder_paths or [], start=1):
        root = os.path.realpath((raw or "").strip())
        if not root:
            continue
        if root in seen_roots:
            continue
        if not os.path.isdir(root):
            raise ValueError(f"{kind_label} folder does not exist: {root}")
        seen_roots.add(root)
        roots.append(root)

    if not roots:
        raise ValueError(f"Provide at least one folder path for {kind_label.lower()} recordings.")

    entries = []
    folder_summaries = []
    extension_counts = defaultdict(int)
    folder_contexts = []
    selection_map = data_file_selection_map if isinstance(data_file_selection_map, dict) else {}

    for folder_idx, root in enumerate(roots, start=1):
        folder_name = _folder_display_name(root, fallback_index=folder_idx)
        folder_entries = []
        branch_directories = defaultdict(set)
        branch_dir_ext_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for current_root, _, files in os.walk(root):
            rel_dir = os.path.relpath(current_root, root)
            rel_dir_posix = _to_posix_relpath(rel_dir)
            if rel_dir_posix == ".":
                dir_depth = 0
            else:
                dir_depth = len([token for token in rel_dir_posix.split("/") if token])
            if dir_depth > FEATURES_MAX_SUBFOLDER_DEPTH:
                raise ValueError(
                    f"{kind_label} folder '{folder_name}' exceeds the maximum supported nested depth "
                    f"({FEATURES_MAX_SUBFOLDER_DEPTH}) at '{current_root}'."
                )
            if rel_dir_posix != ".":
                parts = [token for token in rel_dir_posix.split("/") if token]
                branch_name = parts[0]
                branch_rel_dir = "/".join(parts[1:]) if len(parts) > 1 else "."
                branch_directories[branch_name].add(branch_rel_dir or ".")

            for name in files:
                ext = Path(name).suffix.lower()
                if ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                    continue
                full_path = os.path.join(current_root, name)
                rel_path = _to_posix_relpath(os.path.relpath(full_path, root))
                rel_parts = [token for token in rel_path.split("/") if token]
                file_depth = max(0, len(rel_parts) - 1)
                if file_depth > FEATURES_MAX_SUBFOLDER_DEPTH:
                    raise ValueError(
                        f"{kind_label} folder '{folder_name}' contains a file deeper than the supported "
                        f"{FEATURES_MAX_SUBFOLDER_DEPTH} nested levels: '{full_path}'."
                    )
                row = {
                    "path": full_path,
                    "name": str(name),
                    "extension": ext,
                    "folder_path": root,
                    "folder_name": folder_name,
                    "relative_path": rel_path,
                }
                if len(rel_parts) >= 2:
                    level1_subfolder = rel_parts[0]
                    subfolder_relative_path = "/".join(rel_parts[1:])
                    subfolder_relative_dir = _to_posix_relpath(os.path.dirname(subfolder_relative_path))
                    row["level1_subfolder"] = level1_subfolder
                    row["subfolder_relative_path"] = subfolder_relative_path
                    row["subfolder_relative_dir"] = subfolder_relative_dir
                    branch_directories[level1_subfolder].add(subfolder_relative_dir or ".")
                    branch_dir_ext_files[level1_subfolder][subfolder_relative_dir][ext].append(row)
                else:
                    row["level1_subfolder"] = ""
                    row["subfolder_relative_path"] = rel_parts[0] if rel_parts else row["name"]
                    row["subfolder_relative_dir"] = "."
                folder_entries.append(row)
                extension_counts[ext] += 1

        folder_entries.sort(key=lambda item: item["path"])

        level1_dirs = sorted(
            [
                name
                for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            ]
        )
        nested_branches_with_files = sorted(
            {
                str(item.get("level1_subfolder") or "").strip()
                for item in folder_entries
                if str(item.get("level1_subfolder") or "").strip()
            }
        )
        has_nested_subfolders = len(nested_branches_with_files) > 0
        subfolder_file_counts = {}
        for _branch in nested_branches_with_files:
            subfolder_file_counts[_branch] = sum(
                len(rows)
                for dir_ext_files in branch_dir_ext_files.get(_branch, {}).values()
                for rows in dir_ext_files.values()
            )
        all_subfolders = [
            {"name": _branch, "file_count": subfolder_file_counts.get(_branch, 0)}
            for _branch in nested_branches_with_files
        ]
        selected_entries = list(folder_entries)
        selected_sample_entry = folder_entries[0] if folder_entries else None
        selected_data_file = ""
        selected_data_pattern = ""
        selected_data_extension = ""
        data_file_candidates = []
        matched_data_files = []
        structure_warnings = []

        if has_nested_subfolders:
            if not level1_dirs:
                level1_dirs = list(nested_branches_with_files)
            level1_dirs = sorted([token for token in level1_dirs if token])
            # Pick the sample branch whose internal directory structure is shared by
            # the most other branches (majority vote), so outlier folders like
            # 'derivatives/' don't hijack the reference structure.
            _dir_sigs = {
                b: frozenset(branch_directories.get(b, set()))
                for b in level1_dirs if b in nested_branches_with_files
            }
            _sig_counts: dict = {}
            for _sig in _dir_sigs.values():
                _sig_counts[_sig] = _sig_counts.get(_sig, 0) + 1
            _best_sig = max(_sig_counts, key=lambda s: _sig_counts[s]) if _sig_counts else None
            sample_branch = next(
                (b for b in level1_dirs if _dir_sigs.get(b) == _best_sig),
                nested_branches_with_files[0],
            )
            sample_dirs = set(branch_directories.get(sample_branch, set()))
            sample_dirs.add(".")
            for branch in level1_dirs:
                branch_dirs = set(branch_directories.get(branch, set()))
                branch_dirs.add(".")
                missing_dirs = sorted(sample_dirs - branch_dirs)
                extra_dirs = sorted(branch_dirs - sample_dirs)
                if missing_dirs:
                    missing_dir = missing_dirs[0]
                    sample_files = branch_dir_ext_files.get(sample_branch, {}).get(missing_dir, {})
                    sample_file_hint = ""
                    for ext_rows in sample_files.values():
                        if ext_rows:
                            sample_file_hint = ext_rows[0]["path"]
                            break
                    structure_warnings.append(
                        f"Subfolder '{branch}' is missing expected directory '{missing_dir}' "
                        f"(present in sample subfolder '{sample_branch}'). "
                        f"Consider excluding '{branch}' using the subfolder filter below."
                    )
                    break
                if extra_dirs:
                    extra_dir = extra_dirs[0]
                    culprit_file = ""
                    for rows in (branch_dir_ext_files.get(branch, {}).get(extra_dir, {}) or {}).values():
                        if rows:
                            culprit_file = str(rows[0].get("path") or "")
                            break
                    structure_warnings.append(
                        f"Subfolder '{branch}' contains unexpected directory '{extra_dir}' "
                        f"not present in sample subfolder '{sample_branch}'. "
                        f"Consider excluding '{branch}' using the subfolder filter below."
                    )
                    break

                for rel_dir in sorted(sample_dirs):
                    sample_counts = {
                        ext: len(rows)
                        for ext, rows in (branch_dir_ext_files.get(sample_branch, {}).get(rel_dir, {}) or {}).items()
                    }
                    branch_counts = {
                        ext: len(rows)
                        for ext, rows in (branch_dir_ext_files.get(branch, {}).get(rel_dir, {}) or {}).items()
                    }
                    if sample_counts == branch_counts:
                        continue

                    culprit_file = ""
                    extra_exts = sorted(set(branch_counts.keys()) - set(sample_counts.keys()))
                    missing_exts = sorted(set(sample_counts.keys()) - set(branch_counts.keys()))
                    if extra_exts:
                        first_ext = extra_exts[0]
                        culprit_rows = branch_dir_ext_files.get(branch, {}).get(rel_dir, {}).get(first_ext, [])
                        culprit_file = culprit_rows[0]["path"] if culprit_rows else ""
                    elif missing_exts:
                        first_ext = missing_exts[0]
                        culprit_rows = branch_dir_ext_files.get(sample_branch, {}).get(rel_dir, {}).get(first_ext, [])
                        culprit_file = culprit_rows[0]["path"] if culprit_rows else ""
                    else:
                        for ext in sorted(sample_counts.keys()):
                            if sample_counts.get(ext) == branch_counts.get(ext):
                                continue
                            culprit_rows = branch_dir_ext_files.get(branch, {}).get(rel_dir, {}).get(ext, [])
                            culprit_file = culprit_rows[0]["path"] if culprit_rows else ""
                            if not culprit_file:
                                culprit_rows = branch_dir_ext_files.get(sample_branch, {}).get(rel_dir, {}).get(ext, [])
                                culprit_file = culprit_rows[0]["path"] if culprit_rows else ""
                            break

                    structure_warnings.append(
                        f"Subfolder '{branch}' has inconsistent file counts at directory '{rel_dir}' "
                        f"compared to sample subfolder '{sample_branch}'. "
                        f"Consider excluding '{branch}' using the subfolder filter below."
                    )
                    break

            sample_branch_entries = sorted(
                [item for item in folder_entries if str(item.get("level1_subfolder") or "") == sample_branch],
                key=lambda item: str(item.get("subfolder_relative_path") or ""),
            )
            if not sample_branch_entries:
                structure_warnings.append(
                    f"Sample subfolder '{sample_branch}' has no supported files — inspection may be incomplete."
                )

            data_file_candidates = [
                {
                    "value": str(item.get("subfolder_relative_path") or ""),
                    "label": str(item.get("subfolder_relative_path") or ""),
                    "extension": str(item.get("extension") or ""),
                    "file_name": str(item.get("name") or ""),
                }
                for item in sample_branch_entries
            ]
            candidate_map = {row["value"]: row for row in data_file_candidates if row.get("value")}
            requested_selection = (
                selection_map.get(root)
                or selection_map.get(folder_name)
                or ""
            )
            requested_selection = _to_posix_relpath(requested_selection) if requested_selection else ""
            if requested_selection.startswith(f"{sample_branch}/"):
                requested_selection = requested_selection[len(sample_branch) + 1:]
            if requested_selection and requested_selection not in candidate_map:
                raise ValueError(
                    f"{kind_label} folder '{folder_name}' received an unknown Data file selection "
                    f"'{requested_selection}' for sample subfolder '{sample_branch}'."
                )
            selected_data_file = requested_selection or sample_branch_entries[0]["subfolder_relative_path"]
            selected_sample_entry = next(
                (
                    item
                    for item in sample_branch_entries
                    if str(item.get("subfolder_relative_path") or "") == selected_data_file
                ),
                sample_branch_entries[0],
            )
            selected_data_file = str(selected_sample_entry.get("subfolder_relative_path") or "")
            selected_data_extension = str(selected_sample_entry.get("extension") or "")
            selected_data_name = str(selected_sample_entry.get("name") or "")
            selected_dir = str(selected_sample_entry.get("subfolder_relative_dir") or ".")
            wildcard_stem, stem_regex = _numeric_wildcard_stem(Path(selected_data_name).stem)
            selected_data_pattern = f"{wildcard_stem}{selected_data_extension}" if wildcard_stem else selected_data_name

            matched_entries = []
            for branch in level1_dirs:
                branch_rows = [
                    item for item in folder_entries
                    if str(item.get("level1_subfolder") or "") == branch
                    and str(item.get("subfolder_relative_dir") or ".") == selected_dir
                    and str(item.get("extension") or "").lower() == selected_data_extension.lower()
                ]
                if not branch_rows:
                    structure_warnings.append(
                        f"Subfolder '{branch}' has no data file at '{selected_dir}' with extension "
                        f"'{selected_data_extension}'. Consider excluding it using the subfolder filter below."
                    )
                    continue
                exact_matches = [
                    item for item in branch_rows
                    if str(item.get("name") or "") == selected_data_name
                ]
                if len(exact_matches) > 1:
                    structure_warnings.append(
                        f"Subfolder '{branch}' has multiple files named '{selected_data_name}' — skipped."
                    )
                    continue
                if len(exact_matches) == 1:
                    chosen = exact_matches[0]
                else:
                    if stem_regex is None:
                        structure_warnings.append(
                            f"Subfolder '{branch}' is missing expected file '{selected_data_name}'. "
                            f"Consider excluding it using the subfolder filter below."
                        )
                        continue
                    pattern_matches = [
                        item for item in branch_rows
                        if stem_regex.match(Path(str(item.get("name") or "")).stem)
                    ]
                    if len(pattern_matches) == 0:
                        structure_warnings.append(
                            f"Subfolder '{branch}' has no file matching pattern '{selected_data_pattern}' "
                            f"at '{selected_dir}'. Consider excluding it using the subfolder filter below."
                        )
                        continue
                    if len(pattern_matches) > 1:
                        structure_warnings.append(
                            f"Subfolder '{branch}' has multiple files matching pattern '{selected_data_pattern}' "
                            f"at '{selected_dir}' — skipped."
                        )
                        continue
                    chosen = pattern_matches[0]
                matched_entries.append(chosen)

            matched_entries.sort(key=lambda item: str(item.get("path") or ""))
            selected_entries = matched_entries
            matched_data_files = [
                {
                    "level1_subfolder": str(item.get("level1_subfolder") or ""),
                    "relative_path": str(item.get("relative_path") or ""),
                    "subfolder_relative_path": str(item.get("subfolder_relative_path") or ""),
                    "name": str(item.get("name") or ""),
                    "path": str(item.get("path") or ""),
                    "extension": str(item.get("extension") or ""),
                }
                for item in matched_entries
            ]
            sample_subfolder_path = os.path.join(root, sample_branch)
        else:
            sample_branch = ""
            sample_subfolder_path = ""
            ext_groups = defaultdict(list)
            for item in folder_entries:
                ext_groups[str(item.get("extension") or "").strip().lower()].append(item)

            ext_candidates = sorted([ext for ext in ext_groups.keys() if ext])
            data_file_candidates = [
                {
                    "value": f"__ext__:{ext}",
                    "label": f"{ext} ({len(ext_groups.get(ext, []))} file(s))",
                    "extension": ext,
                    "file_name": "",
                }
                for ext in ext_candidates
            ]

            requested_selection = (
                selection_map.get(root)
                or selection_map.get(folder_name)
                or ""
            )
            requested_selection = _to_posix_relpath(requested_selection) if requested_selection else ""
            requested_extension = ""
            if requested_selection.startswith("__ext__:"):
                requested_extension = requested_selection[len("__ext__:"):].strip().lower()
            elif requested_selection:
                requested_extension = Path(requested_selection).suffix.lower() or requested_selection.lower()

            if requested_extension not in ext_groups:
                requested_extension = ext_candidates[0] if ext_candidates else ""

            selected_data_extension = requested_extension
            selected_data_file = f"__ext__:{selected_data_extension}" if selected_data_extension else ""
            selected_data_pattern = f"*{selected_data_extension}" if selected_data_extension else ""
            selected_entries = list(ext_groups.get(selected_data_extension, [])) if selected_data_extension else list(folder_entries)
            selected_entries.sort(key=lambda item: str(item.get("path") or ""))
            selected_sample_entry = selected_entries[0] if selected_entries else (folder_entries[0] if folder_entries else None)
            matched_data_files = [
                {
                    "level1_subfolder": str(item.get("level1_subfolder") or ""),
                    "relative_path": str(item.get("relative_path") or ""),
                    "subfolder_relative_path": str(item.get("subfolder_relative_path") or ""),
                    "name": str(item.get("name") or ""),
                    "path": str(item.get("path") or ""),
                    "extension": str(item.get("extension") or ""),
                }
                for item in selected_entries
            ]

        folder_summaries.append({
            "folder_path": root,
            "folder_name": folder_name,
            "file_count": len(folder_entries),
        })
        entries.extend(folder_entries)
        folder_contexts.append({
            "folder_path": root,
            "folder_name": folder_name,
            "has_nested_subfolders": bool(has_nested_subfolders),
            "sample_subfolder_name": sample_branch,
            "sample_subfolder_path": sample_subfolder_path,
            "data_file_candidates": data_file_candidates,
            "selected_data_file": selected_data_file,
            "selected_data_pattern": selected_data_pattern,
            "selected_data_extension": selected_data_extension,
            "matched_data_files": matched_data_files,
            "selected_entries": selected_entries,
            "sample_selected_entry": selected_sample_entry,
            "all_subfolders": all_subfolders,
            "structure_warnings": structure_warnings,
        })

    entries.sort(key=lambda item: item["path"])

    if not entries:
        joined = ", ".join(roots)
        raise ValueError(f"No supported {kind_label.lower()} files found in selected folder(s): {joined}")

    return entries, folder_summaries, dict(sorted(extension_counts.items())), folder_contexts


def _validate_simulation_file_path(file_path):
    candidate = os.path.realpath((file_path or "").strip())
    if not candidate:
        raise ValueError("Provide a simulation output file path.")
    if not os.path.isfile(candidate):
        raise ValueError(f"Simulation output file does not exist: {candidate}")
    ext = Path(candidate).suffix.lower()
    if ext not in FEATURES_PARSER_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported simulation output file extension: {ext}")
    return candidate


def _extract_filename_text_label(file_name):
    chains = _extract_filename_text_chains(file_name)
    return chains[-1] if chains else ""


def _extract_filename_text_chains(file_name):
    stem = Path(str(file_name or "")).stem
    if not stem:
        return []
    return [tok for tok in stem.split("_") if tok.strip()]


def _build_file_extracted_virtual_fields(file_names):
    chain_rows = [_extract_filename_text_chains(name) for name in (file_names or [])]
    max_chains = max((len(row) for row in chain_rows), default=0)
    fields = []

    for idx in range(max_chains):
        ordered_values = []
        seen = set()
        for row in chain_rows:
            if idx >= len(row):
                continue
            token = str(row[idx] or "").strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered_values.append(token)
        if not ordered_values:
            continue
        preview = "/".join(ordered_values[:6])
        if len(ordered_values) > 6:
            preview += "/..."
        fields.append({
            "key": f"{FILE_EXTRACTED_VIRTUAL_FIELD_PREFIX}{idx}",
            "label": f"file extracted: {preview}",
            "values": ordered_values,
        })

    return fields


def _attach_file_extracted_virtual_field(description, file_names):
    field_infos = _build_file_extracted_virtual_fields(file_names)
    if not field_infos:
        return description
    enriched = dict(description or {})
    candidates = [str(v) for v in (enriched.get("candidate_fields") or [])]
    virtual_labels = dict(enriched.get("virtual_field_labels") or {})
    extracted_values = dict(enriched.get("file_extracted_values") or {})
    for field_info in field_infos:
        key = str(field_info.get("key") or "").strip()
        if not key:
            continue
        if key not in candidates:
            candidates.append(key)
        virtual_labels[key] = field_info.get("label") or key
        extracted_values[key] = list(field_info.get("values") or [])
    enriched["candidate_fields"] = candidates
    enriched["virtual_field_labels"] = virtual_labels
    enriched["file_extracted_values"] = extracted_values
    return enriched


def _file_extracted_chain_index(locator):
    if not isinstance(locator, str):
        return None
    if locator == FILE_EXTRACTED_VIRTUAL_FIELD:
        # Backward compatibility with previous single extracted field.
        return -1
    if not locator.startswith(FILE_EXTRACTED_VIRTUAL_FIELD_PREFIX):
        return None
    suffix = locator[len(FILE_EXTRACTED_VIRTUAL_FIELD_PREFIX):].strip()
    if not suffix.isdigit():
        return None
    return int(suffix)


def _resolve_file_extracted_value(file_name, locator):
    index = _file_extracted_chain_index(locator)
    if index is None:
        return None
    chains = _extract_filename_text_chains(file_name)
    if not chains:
        return None
    if index < 0:
        return chains[-1]
    if index >= len(chains):
        return None
    return chains[index]


def _collect_feature_pipeline_inputs():
    source_labels = {
        "proxy": "Field Potential Proxy",
        "kernel": "Field Potential Kernel",
        "meeg": "Field Potential M/EEG",
        "field_potential_proxy": "Field Potential Proxy",
        "field_potential_kernel": "Field Potential Kernel",
        "field_potential_meeg": "Field Potential M/EEG",
    }

    def _normalize_fp_source_key(folder_name):
        key = str(folder_name or "").strip().lower()
        if key in {"field_potential_proxy", "proxy"}:
            return "proxy"
        if key in {"field_potential_kernel", "kernel"}:
            return "kernel"
        if key in {"field_potential_meeg", "meeg"}:
            return "meeg"
        return key
    discovered = {}
    for fp_dir in _field_potential_dirs():
        if not os.path.isdir(fp_dir):
            continue
        source_key = _normalize_fp_source_key(os.path.basename(fp_dir))
        source_label = source_labels.get(source_key, source_key)
        for root, _, files in os.walk(fp_dir):
            for name in files:
                path = os.path.realpath(os.path.join(root, name))
                ext = os.path.splitext(name)[1].lower()
                if ext not in {".pkl", ".pickle", ".csv", ".parquet", ".feather", ".xlsx", ".xls"}:
                    continue
                lower_name = name.lower()
                # Only expose final outputs relevant for feature extraction:
                # - proxy outputs
                # - CDM/LFP probe outputs
                # - M/EEG outputs
                # Exclude intermediate kernel objects (e.g., kernels.pkl).
                if source_key == "proxy":
                    if not lower_name.startswith("proxy"):
                        continue
                elif source_key == "kernel":
                    allowed_kernel_outputs = {
                        "kernel_approx_cdm.pkl",
                        "current_dipole_moment.pkl",
                        "gauss_cylinder_potential.pkl",
                        "probe_outputs.pkl",
                    }
                    if lower_name not in allowed_kernel_outputs:
                        continue
                elif source_key == "meeg":
                    if lower_name not in {"meeg.pkl", "eeg.pkl", "meg.pkl", "lfp.pkl"}:
                        continue
                else:
                    continue
                try:
                    modified = os.path.getmtime(path)
                except OSError:
                    continue
                rel_run = os.path.basename(root)
                record = {
                    "name": name,
                    "path": path,
                    "source": source_label,
                    "run_id": rel_run if rel_run and rel_run != source_key else "",
                    "_modified": modified,
                }
                existing = discovered.get(path)
                if existing is None or existing["_modified"] < modified:
                    discovered[path] = record
    ordered = sorted(discovered.values(), key=lambda item: item["_modified"], reverse=True)
    for item in ordered:
        item.pop("_modified", None)
    return ordered


def _allowed_feature_existing_paths():
    return {entry["path"] for entry in _collect_feature_pipeline_inputs()}


def _validate_feature_existing_path(path):
    candidate = os.path.realpath(path or "")
    if candidate not in _allowed_feature_existing_paths():
        raise ValueError("Selected pipeline file is not available anymore. Refresh the page and select again.")
    return candidate


def _extract_dataframe_fs_hint(df):
    def _to_float(value):
        if value is None:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        return val

    def _metadata_mapping(value):
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _meta_get(mapping, keys):
        if not isinstance(mapping, dict):
            return None
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    fs_hint = None
    dt_ms = None
    decimation = 1

    if "dt_ms" in df.columns:
        series = pd.to_numeric(df["dt_ms"], errors="coerce").dropna()
        if not series.empty:
            dt_ms = float(series.iloc[0])

    if "decimation_factor" in df.columns:
        series = pd.to_numeric(df["decimation_factor"], errors="coerce").dropna()
        if not series.empty:
            val = int(float(series.iloc[0]))
            if val > 0:
                decimation = val

    if "fs_hz" in df.columns:
        series = pd.to_numeric(df["fs_hz"], errors="coerce").dropna()
        if not series.empty:
            fs_hint = float(series.iloc[0])

    metadata_candidates = []
    if "metadata" in df.columns:
        meta_series = df["metadata"].dropna()
        if not meta_series.empty:
            metadata_candidates.append(meta_series.iloc[0])

    df_attrs = getattr(df, "attrs", None)
    if isinstance(df_attrs, dict):
        metadata_candidates.append(df_attrs)
        if "metadata" in df_attrs:
            metadata_candidates.append(df_attrs.get("metadata"))

    for candidate in metadata_candidates:
        meta = _metadata_mapping(candidate)
        if not meta:
            continue
        if dt_ms is None:
            dt_meta = _to_float(_meta_get(meta, ["dt_ms", "simulation_step_ms", "dt"]))
            if dt_meta is not None and dt_meta > 0:
                dt_ms = dt_meta
        if decimation == 1:
            dec_meta = _to_float(_meta_get(meta, ["decimation_factor", "decimation", "decimate_factor"]))
            if dec_meta is not None and dec_meta > 0:
                decimation = int(dec_meta)
        if fs_hint is None:
            fs_meta = _to_float(_meta_get(meta, ["fs_hz", "fs", "sampling_rate", "sampling_frequency_hz"]))
            if fs_meta is not None and fs_meta > 0:
                fs_hint = fs_meta

    if fs_hint is None and dt_ms is not None and dt_ms > 0:
        effective_dt_ms = dt_ms * max(decimation, 1)
        if effective_dt_ms > 0:
            fs_hint = 1000.0 / effective_dt_ms

    if fs_hint is not None and (not np.isfinite(fs_hint) or fs_hint <= 0):
        fs_hint = None

    note = None
    if fs_hint is not None:
        if decimation > 1:
            note = f"Detected decimation x{decimation} in previous stage. Suggested fs: {fs_hint:g} Hz."
        else:
            note = f"Suggested fs from previous stage: {fs_hint:g} Hz."
        if dt_ms is not None:
            note += f" (dt={dt_ms:g} ms)."

    return fs_hint, note


def _optional_float(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return float(_safe_eval_numeric_expression(value))


def _parse_nonnegative_int(raw_value, default=0, field_name="value"):
    if raw_value is None:
        return int(default)
    value = str(raw_value).strip()
    if value == "":
        return int(default)
    try:
        parsed = int(value)
    except Exception:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return parsed


def _safe_eval_numeric_expression(expr):
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = _eval(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv)
        ):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            return left ** right
        raise ValueError(f"Unsupported numeric expression: {expr}")

    return _eval(tree)


def _extract_locator_candidates(obj, max_depth=6, max_items=120):
    candidates = []
    seen = set()
    visited = set()

    def _is_terminal(value):
        # Scalars and strings: terminal
        if isinstance(value, (str, bytes, int, float, bool)):
            return True
        # NumPy Arrays: only if not object-type
        if isinstance(value, np.ndarray) and value.dtype != np.object_ and value.dtype.names is None:
            return True
        return False

    def _append(path):
        if path and path not in seen and len(candidates) < max_items:
            seen.add(path)
            candidates.append(path)

    def _walk(value, prefix, depth):
        if len(candidates) >= max_items or depth > max_depth:
            return
        marker = id(value)
        if marker in visited:
            return
        visited.add(marker)

        if isinstance(value, MappingABC):
            for key in value.keys():
                key_str = str(key)
                if key_str.startswith("__"):
                    continue
                path = f"{prefix}.{key_str}" if prefix else key_str
                _append(path)
                if getattr(value, "__lazy_hdf5__", False):
                    continue
                nested = value.get(key)
                if not _is_terminal(nested):
                    _walk(nested, path, depth + 1)
            return

        # NumPy Arrays object-type
        if isinstance(value, np.ndarray) and value.dtype == np.object_:
            if value.size == 0:
                return
            # Take the first element as a sample
            first_item = value.flat[0]
            # If the first item is a structure (structured array or mat_struct), explore its fields with the same prefix (without index)
            if hasattr(first_item, 'dtype') and first_item.dtype.names:
                # Structured array: explore its fields
                for field_name in first_item.dtype.names:
                    field_path = f"{prefix}.{field_name}" if prefix else field_name
                    _append(field_path)
                    field_value = first_item[field_name]
                    if not _is_terminal(field_value):
                        _walk(field_value, field_path, depth + 1)
            elif hasattr(first_item, '_fieldnames'):  # mat_struct
                for field_name in first_item._fieldnames:
                    field_path = f"{prefix}.{field_name}" if prefix else field_name
                    _append(field_path)
                    field_value = getattr(first_item, field_name)
                    if not _is_terminal(field_value):
                        _walk(field_value, field_path, depth + 1)
            else:
                _append(prefix)
            return

        # Structured arrays
        if isinstance(value, np.ndarray) and value.dtype.names:
            if value.size == 0:
                return
            # Take first element as a sample (assume homogeneous structure)
            first_elem = value.flat[0]
            for field_name in value.dtype.names:
                field_path = f"{prefix}.{field_name}" if prefix else field_name
                _append(field_path)
                field_value = first_elem[field_name]
                if not _is_terminal(field_value):
                    _walk(field_value, field_path, depth + 1)
            return

        if isinstance(value, (list, tuple)):
            if not value:
                return
            # Explore all elements, not just the first one
            for idx, item in enumerate(value):
                # Generate path with index, eg: "cfg.filtering[0]"
                indexed_path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                _append(indexed_path)
                if not _is_terminal(item):
                    _walk(item, indexed_path, depth + 1)
            return

        if hasattr(value, "__dict__"):
            for attr, nested in vars(value).items():
                if attr.startswith("_"):
                    continue
                path = f"{prefix}.{attr}" if prefix else attr
                _append(path)
                if not _is_terminal(nested):
                    _walk(nested, path, depth + 1)

    _walk(obj, "", 0)
    return candidates[:max_items]


def _pick_field_guess(candidates, patterns):
    lower_map = {str(item).lower(): item for item in candidates}
    for pattern in patterns:
        if pattern in lower_map:
            return lower_map[pattern]
    for key, value in lower_map.items():
        if any(key.endswith(f".{pattern}") for pattern in patterns):
            return value
    return ""


def _pick_field_guess_fuzzy(candidates, patterns):
    direct = _pick_field_guess(candidates, patterns)
    if direct:
        return direct

    normalized_patterns = [str(pattern or "").strip().lower() for pattern in patterns if str(pattern or "").strip()]
    if not normalized_patterns:
        return ""

    for item in candidates:
        key = str(item or "")
        lower_key = key.lower()
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", lower_key) if tok]
        for pattern in normalized_patterns:
            if not pattern:
                continue
            if pattern in tokens:
                return item
            if lower_key.endswith(f".{pattern}"):
                return item
            boundary = rf"(^|[^a-z0-9]){re.escape(pattern)}([^a-z0-9]|$)"
            if re.search(boundary, lower_key):
                return item
            if pattern in lower_key:
                return item
    return ""

def _expand_mat_struct(value, prefix=""):
    """Recursivamente extrae campos de un mat_struct o cell array."""
    fields = []
    if hasattr(value, '_fieldnames') and hasattr(value, '__dict__'):
        for field in value._fieldnames:
            full = f"{prefix}.{field}" if prefix else field
            fields.append(full)
            nested = getattr(value, field)
            if hasattr(nested, '_fieldnames'):
                fields.extend(_expand_mat_struct(nested, full))
            elif isinstance(nested, (list, tuple)) and nested and hasattr(nested[0], '_fieldnames'):
                fields.extend(_expand_mat_struct(nested[0], f"{full}[0]"))
    return fields


def _summarize_value_for_ui(value):
    summary = {
        "python_type": type(value).__name__,
        "dtype": "",
        "shape": None,
        "detail": "",
    }

    if value is None:
        summary["python_type"] = "NoneType"
        summary["detail"] = "No value"
        return summary

    if isinstance(value, pd.DataFrame):
        summary["python_type"] = "DataFrame"
        summary["shape"] = [int(value.shape[0]), int(value.shape[1])]
        summary["detail"] = f"{value.shape[0]} rows x {value.shape[1]} columns"
        return summary

    if isinstance(value, pd.Series):
        summary["python_type"] = "Series"
        summary["dtype"] = str(value.dtype)
        summary["shape"] = [int(value.shape[0])]
        sample = value.dropna()
        if not sample.empty:
            inner = _summarize_value_for_ui(sample.iloc[0])
            if inner.get("python_type"):
                summary["detail"] = f"sample: {inner['python_type']}"
                if inner.get("shape") is not None:
                    summary["detail"] += f", shape={inner['shape']}"
        else:
            summary["detail"] = "empty series"
        return summary

    if isinstance(value, np.ndarray):
        summary["python_type"] = "ndarray"
        summary["dtype"] = str(value.dtype)
        summary["shape"] = [int(dim) for dim in value.shape]
        summary["detail"] = f"{value.ndim}D array"

        if value.dtype == object and value.size > 0:
            try:
                inner_arr = np.asarray(value.flat[0])
                if inner_arr.ndim > 0:
                    full_shape = list(value.shape) + list(inner_arr.shape)
                    summary["detail"] += f" of arrays -> effective shape {full_shape}"
            except Exception:
                pass
        return summary

    if isinstance(value, dict):
        keys = [str(k) for k in value.keys()]
        summary["python_type"] = "dict"
        summary["shape"] = [len(keys)]
        preview = ", ".join(keys[:5])
        if len(keys) > 5:
            preview += ", ..."
        summary["detail"] = f"{len(keys)} keys ({preview})" if keys else "empty dict"
        return summary

    if isinstance(value, (list, tuple)):
        summary["python_type"] = type(value).__name__
        summary["shape"] = [len(value)]
        if value:
            inner = _summarize_value_for_ui(value[0])
            summary["detail"] = f"length {len(value)}, first item: {inner['python_type']}"
            if inner.get("shape") is not None:
                summary["detail"] += f" shape={inner['shape']}"
        else:
            summary["detail"] = "empty sequence"
        return summary

    if isinstance(value, (str, bytes)):
        summary["shape"] = [len(value)]
        summary["detail"] = f"length {len(value)}"
        return summary

    if hasattr(value, '_fieldnames') and hasattr(value, '__dict__'):
        subfields = _expand_mat_struct(value)
        summary["python_type"] = "mat_struct"
        summary["shape"] = [len(subfields)]
        if subfields:
            preview = ", ".join(subfields[:10])
            if len(subfields) > 10:
                preview += ", ..."
            summary["detail"] = f"mat_struct with fields: {preview}"
        else:
            summary["detail"] = "mat_struct (no fields)"
        return summary

    try:
        arr = np.asarray(value)
    except Exception:
        arr = None
    if arr is not None and arr.ndim > 0 and arr.size > 1:
        summary["shape"] = [int(dim) for dim in arr.shape]
        summary["dtype"] = str(arr.dtype)
        summary["detail"] = f"array-like {arr.ndim}D"
        return summary

    return summary


def _describe_source_fields_for_ui(source_obj, candidate_fields, source_type):
    details = []
    lazy_mapping = bool(getattr(source_obj, "__lazy_hdf5__", False))
    for field in candidate_fields or []:
        locator = str(field or "").strip()
        if not locator:
            continue
        origin = "dataframe" if source_type == "dataframe" else "field"
        if lazy_mapping and "." not in locator and "[" not in locator and locator != "__self__":
            details.append({
                "field": locator,
                "origin": origin,
                "python_type": "HDF5 key",
                "dtype": "",
                "shape": None,
                "detail": "Lazy-loaded MATLAB v7.3 key (resolved when selected).",
            })
            continue
        if source_type == "dataframe" and isinstance(source_obj, pd.DataFrame) and locator in source_obj.columns:
            value = source_obj[locator]
        elif locator == "__self__":
            value = source_obj
        else:
            value = _resolve_locator_value(source_obj, locator)

        if value is None and locator != "__self__":
            details.append({
                "field": locator,
                "origin": origin,
                "python_type": "Unknown",
                "dtype": "",
                "shape": None,
                "detail": "Could not resolve field in the inspected sample.",
            })
            continue

        summarized = _summarize_value_for_ui(value)
        details.append({
            "field": locator,
            "origin": origin,
            "python_type": summarized.get("python_type") or "",
            "dtype": summarized.get("dtype") or "",
            "shape": summarized.get("shape"),
            "detail": summarized.get("detail") or "",
        })
    return details


def _manual_field_details_for_ui():
    return [
        {
            "field": "__manual_data__",
            "label": "Data locator selection",
            "origin": "manual",
            "python_type": "str",
            "detail": "Manually select which field/column contains the signal data.",
        },
        {
            "field": "__numeric_fs__",
            "label": "Numeric sampling frequency",
            "origin": "manual",
            "python_type": "float",
            "detail": "Provide fs directly in Hz when no field contains it.",
        },
        {
            "field": "__manual_ch_names__",
            "label": "Manual channel names list",
            "origin": "manual",
            "python_type": "list[str]",
            "detail": "Provide comma-separated channel labels.",
        },
        {
            "field": "__autocomplete_ch_names__",
            "label": "Autocomplete channel names",
            "origin": "manual",
            "python_type": "callable",
            "detail": "Generate ch0..chN from detected data dimensions and axis.",
        },
        {
            "field": "__recording_type_value__",
            "label": "Recording type value",
            "origin": "manual",
            "python_type": "str",
            "detail": "Set recording type directly (LFP, EEG, MEG, ...).",
        },
    ]


def _build_virtual_field_details_for_ui(file_names):
    normalized_names = []
    seen_names = set()
    for raw in file_names or []:
        token = str(raw or "").strip()
        if not token or token in seen_names:
            continue
        seen_names.add(token)
        normalized_names.append(token)

    file_id_examples = normalized_names[:6]
    file_id_usage = [
        "Use `file_ID` as Subject ID locator to assign one subject per file.",
    ]
    if file_id_examples:
        file_id_usage.append(
            "Example subject IDs from current selection: "
            + ", ".join(file_id_examples[:4])
            + ("..." if len(file_id_examples) > 4 else "")
        )

    details = [{
        "field": "__file_id__",
        "label": "file_ID",
        "origin": "virtual",
        "python_type": "str",
        "detail": "Full file name-based identifier generated for each file.",
        "example_values": file_id_examples,
        "usage_examples": file_id_usage,
    }]
    for item in _build_file_extracted_virtual_fields(file_names):
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        values = list(item.get("values") or [])
        preview = ", ".join(values[:5])
        if len(values) > 5:
            preview += ", ..."
        position = _file_extracted_chain_index(key)
        usage_examples = []
        if position is not None and position >= 0:
            usage_examples.append(
                f"Represents token position {position + 1} extracted from file names."
            )
        usage_examples.append(
            "Use this virtual field as Group/Condition locator when file names encode categories."
        )
        if values:
            usage_examples.append(
                "Example options in current selection: "
                + ", ".join(values[:6])
                + ("..." if len(values) > 6 else "")
            )
            if len(values) >= 2:
                usage_examples.append(
                    f"Example usage: map `{key}` to `group` (e.g., {values[0]} vs {values[1]})."
                )
        details.append({
            "field": key,
            "label": str(item.get("label") or key),
            "origin": "virtual",
            "python_type": "str",
            "detail": f"Tokens extracted from file names: {preview}" if preview else "Tokens extracted from file names.",
            "example_values": values[:8],
            "usage_examples": usage_examples,
        })
    return details


def _summarize_listed_file_names(listed_file_names):
    cleaned = [str(name or "").replace("\\", "/").strip() for name in (listed_file_names or []) if str(name or "").strip()]
    if not cleaned:
        return {
            "folder_summaries": [],
            "extension_summaries": [],
            "total_files": 0,
        }

    folder_counts = defaultdict(int)
    ext_counts = defaultdict(int)
    for raw in cleaned:
        parts = [part for part in raw.split("/") if part not in {"", "."}]
        folder_name = parts[0] if len(parts) > 1 else "selected_folder"
        folder_counts[folder_name] += 1
        ext = Path(parts[-1]).suffix.lower() if parts else Path(raw).suffix.lower()
        ext_counts[ext or "(no extension)"] += 1

    folder_summaries = [{"folder_name": name, "file_count": int(count)} for name, count in sorted(folder_counts.items())]
    extension_summaries = [{"extension": ext, "count": int(count)} for ext, count in sorted(ext_counts.items())]
    return {
        "folder_summaries": folder_summaries,
        "extension_summaries": extension_summaries,
        "total_files": int(len(cleaned)),
    }


def _validate_listed_file_names_by_folder_extension(listed_file_names, label="Selected"):
    entries = []
    for raw in listed_file_names or []:
        token = str(raw or "").replace("\\", "/").strip()
        if not token:
            continue
        parts = [part for part in token.split("/") if part not in {"", "."}]
        if not parts or any(part == ".." for part in parts):
            continue
        if len(parts) > 2:
            raise ValueError(
                f"{label} local upload contains nested subfolders, which are not supported in local mode. "
                f"Nested path detected: {token}"
            )
        folder_name = parts[0] if len(parts) > 1 else "selected_folder"
        extension = Path(parts[-1]).suffix.lower()
        if extension not in FEATURES_PARSER_FILE_EXTENSIONS:
            continue
        entries.append({
            "folder_name": folder_name,
            "folder_path": folder_name,
            "extension": extension,
        })
    # Mixed supported extensions are allowed. The parser flow resolves the
    # selected data source via per-folder data-file selections.


def _build_folder_inspection_profiles(folder_entries, folder_summaries):
    entries_by_folder = defaultdict(list)
    for row in folder_entries or []:
        folder_path = str(row.get("folder_path") or "").strip()
        if not folder_path:
            continue
        entries_by_folder[folder_path].append(row)

    folder_profiles = []
    combined_candidates = []
    seen_candidates = set()
    candidate_field_folders = defaultdict(list)
    extension_counts_global = defaultdict(int)
    combined_logical_names = []

    for folder_summary in folder_summaries or []:
        folder_path = str(folder_summary.get("folder_path") or "").strip()
        folder_name = str(folder_summary.get("folder_name") or folder_path or "folder").strip()
        rows = sorted(entries_by_folder.get(folder_path, []), key=lambda item: str(item.get("path") or ""))
        if not rows:
            continue

        folder_file_names = [str(item.get("logical_name") or item.get("name") or "") for item in rows if str(item.get("logical_name") or item.get("name") or "").strip()]
        combined_logical_names.extend(folder_file_names)

        by_ext = defaultdict(list)
        for row in rows:
            ext = str(row.get("extension") or "").strip().lower() or "(no extension)"
            by_ext[ext].append(row)
            extension_counts_global[ext] += 1

        extension_summaries = [
            {"extension": ext, "count": len(ext_rows)}
            for ext, ext_rows in sorted(by_ext.items())
        ]

        extension_profiles = []
        for ext, ext_rows in sorted(by_ext.items()):
            sample = ext_rows[0]
            ext_names = [
                str(item.get("logical_name") or item.get("name") or "").strip()
                for item in ext_rows
                if str(item.get("logical_name") or item.get("name") or "").strip()
            ]
            ext_description = _describe_parser_source(sample["path"])
            ext_description = _attach_file_extracted_virtual_field(ext_description, ext_names)
            ext_description["virtual_field_details"] = _build_virtual_field_details_for_ui(ext_names)
            ext_description["manual_field_details"] = _manual_field_details_for_ui()
            ext_description["extension"] = ext
            ext_description["sample_file"] = str(sample.get("logical_name") or sample.get("name") or "")
            ext_description["sample_file_path"] = sample["path"]
            ext_description["file_count"] = len(ext_rows)
            ext_description["folder_name"] = folder_name
            ext_description["folder_path"] = folder_path
            extension_profiles.append(ext_description)

            for candidate in ext_description.get("candidate_fields") or []:
                token = str(candidate or "").strip()
                if not token:
                    continue
                if token not in seen_candidates:
                    seen_candidates.add(token)
                    combined_candidates.append(token)
                if folder_name not in candidate_field_folders[token]:
                    candidate_field_folders[token].append(folder_name)

        folder_profiles.append({
            "folder_name": folder_name,
            "folder_path": folder_path,
            "file_count": len(rows),
            "extension_summaries": extension_summaries,
            "extension_profiles": extension_profiles,
            "virtual_field_details": _build_virtual_field_details_for_ui(folder_file_names),
        })

    extension_summaries_global = [
        {"extension": ext, "count": int(count)}
        for ext, count in sorted(extension_counts_global.items())
    ]
    return (
        folder_profiles,
        extension_summaries_global,
        combined_candidates,
        dict(candidate_field_folders),
        combined_logical_names,
    )


def _aggregate_candidate_metadata_from_file_entries(folder_entries):
    combined_candidates = []
    seen_candidates = set()
    candidate_field_folders = defaultdict(list)
    candidate_field_origins = defaultdict(set)
    field_detail_map = {}
    dataframe_fields = set()
    inspected_count = 0

    for row in sorted(folder_entries or [], key=lambda item: str(item.get("path") or "")):
        source_path = str(row.get("path") or "").strip()
        if not source_path:
            continue
        folder_name = str(row.get("folder_name") or row.get("folder_path") or "").strip() or "selected_folder"
        try:
            description = _describe_parser_source(source_path)
        except Exception as exc:
            raise ValueError(
                f"Failed to inspect metadata in file '{source_path}': {exc}"
            )
        inspected_count += 1

        for token in (description.get("candidate_fields") or []):
            field_name = str(token or "").strip()
            if not field_name:
                continue
            if field_name not in seen_candidates:
                seen_candidates.add(field_name)
                combined_candidates.append(field_name)
            if folder_name not in candidate_field_folders[field_name]:
                candidate_field_folders[field_name].append(folder_name)

        for detail in (description.get("field_details") or []):
            if not isinstance(detail, dict):
                continue
            field_name = str(detail.get("field") or "").strip()
            if not field_name:
                continue
            if field_name not in field_detail_map:
                field_detail_map[field_name] = detail
            origin = str(detail.get("origin") or "").strip().lower()
            if origin:
                candidate_field_origins[field_name].add(origin)

        for token in _collect_dataframe_candidate_fields_from_description(description):
            field_name = str(token or "").strip()
            if field_name:
                dataframe_fields.add(field_name)

    return {
        "candidate_fields": combined_candidates,
        "field_details": [field_detail_map[key] for key in combined_candidates if key in field_detail_map],
        "candidate_field_folders": {
            key: value
            for key, value in candidate_field_folders.items()
        },
        "candidate_field_origins": {
            key: sorted(values)
            for key, values in candidate_field_origins.items()
        },
        "dataframe_candidate_fields": [field for field in combined_candidates if field in dataframe_fields],
        "inspected_file_count": inspected_count,
    }


def _parse_sensor_names(raw_value):
    if raw_value is None:
        return None
    names = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return names or None


def _parse_aggregate_over(raw_value):
    if raw_value is None:
        return None
    parts = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return tuple(parts) if parts else None


def _parse_aggregate_labels(raw_value):
    value = (raw_value or "").strip()
    if not value:
        return None
    try:
        parsed = ast.literal_eval(value)
    except Exception as exc:
        raise ValueError(f"Invalid aggregate labels mapping: {exc}")
    if not isinstance(parsed, dict):
        raise ValueError("Aggregate labels must be a dict, e.g. {'sensor': 'all'}.")
    return {str(k): str(v) for k, v in parsed.items()}


def _parse_metadata_field_source(form, source_key, value_key):
    source = (form.get(source_key) or "").strip()
    if source == "__none__":
        return None
    if source == "__file_id__":
        return FILE_ID_METADATA_LITERAL
    if source == "__value__":
        value = (form.get(value_key) or "").strip()
        return value or None
    if source:
        return source
    value = (form.get(value_key) or "").strip()
    return value or None


def _collect_dataframe_candidate_fields_from_description(description):
    if not isinstance(description, dict):
        return []
    candidates = [
        str(item or "").strip()
        for item in (description.get("candidate_fields") or [])
        if str(item or "").strip()
    ]
    details = description.get("field_details") or []
    dataframe_fields = set()
    for item in details:
        if not isinstance(item, dict):
            continue
        origin = str(item.get("origin") or "").strip().lower()
        if origin != "dataframe":
            continue
        token = str(item.get("field") or "").strip()
        if token:
            dataframe_fields.add(token)
    if not dataframe_fields and str(description.get("source_type") or "").strip().lower() == "dataframe":
        dataframe_fields = set(candidates)
    return [field for field in candidates if field in dataframe_fields]


def _validate_data_locator_against_dataframe_source(data_locator, source_obj, source_label="selected source"):
    locator = str(data_locator or "").strip()
    if not locator:
        raise ValueError("Select a field for Data locator.")
    if isinstance(source_obj, pd.DataFrame):
        columns = [str(col) for col in source_obj.columns]
        if locator not in columns:
            preview = ", ".join(columns[:12])
            if len(columns) > 12:
                preview += ", ..."
            raise ValueError(
                f"Data locator '{locator}' is not a dataframe column in {source_label}. "
                f"Available columns: {preview or '(none)'}."
            )
        return
    if locator == "__self__":
        return
    if getattr(source_obj, "__lazy_hdf5__", False):
        # Fast-path for large MATLAB v7.3 lazy mappings: validate only root token
        # to avoid expensive dataset materialization in the request thread.
        root = re.split(r"[.\[]", locator, maxsplit=1)[0].strip()
        if not root:
            raise ValueError("Select a valid field for Data locator.")
        try:
            has_root = root in source_obj
        except Exception:
            has_root = False
        if not has_root:
            available = []
            try:
                available = [str(k) for k in list(source_obj.keys())[:12]]
            except Exception:
                available = []
            preview = ", ".join(available) if available else "(unavailable)"
            raise ValueError(
                f"Data locator '{locator}' root key '{root}' was not found in {source_label}. "
                f"Top-level keys: {preview}."
            )
        return
    resolved = _resolve_locator_value(source_obj, locator)
    if resolved is None:
        raise ValueError(
            f"Data locator '{locator}' could not be resolved in {source_label}."
        )


def _validate_data_locator_against_source_path(data_locator, source_path, source_label="selected source"):
    source_obj = _load_features_source(source_path)
    sample_name = os.path.basename(str(source_path or "").strip()) or "sample"
    label = f"{source_label} '{sample_name}'"
    _validate_data_locator_against_dataframe_source(data_locator, source_obj, label)


def _estimate_sensor_count(value, axis=None):
    if isinstance(value, dict):
        keys = [k for k in value.keys() if not str(k).startswith("__")]
        if not keys:
            return None
        if len(keys) == 1:
            nested = _estimate_sensor_count(value.get(keys[0]))
            if nested is not None and nested > 0:
                return int(nested)
        return int(len(keys))

    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if axis is not None:
        try:
            axis_idx = int(axis)
        except Exception:
            axis_idx = None
        if axis_idx is not None and arr.ndim > 0 and 0 <= axis_idx < arr.ndim:
            axis_size = int(arr.shape[axis_idx])
            if axis_size > 0:
                return axis_size

    if arr.ndim == 0:
        return 1
    if arr.ndim == 1:
        return 1
    if arr.ndim == 2:
        d0, d1 = int(arr.shape[0]), int(arr.shape[1])
        if d0 <= 0 and d1 <= 0:
            return 1
        if d0 == 1 and d1 > 1:
            return d1 if d1 <= 512 else 1
        if d1 == 1 and d0 > 1:
            return d0 if d0 <= 512 else 1
        if d0 > 0 and d1 > 0:
            small, large = (d0, d1) if d0 <= d1 else (d1, d0)
            # In typical ephys arrays channels are much fewer than samples.
            if large >= 8 * small and small <= 512:
                return small
            if d0 <= 512 < d1:
                return d0
            if d1 <= 512 < d0:
                return d1
        return d0 if d0 > 0 else 1
    if arr.ndim >= 3:
        dims = [int(v) for v in arr.shape if int(v) > 0]
        if not dims:
            return 1
        small_dims = [d for d in dims if d <= 512]
        if small_dims:
            return int(min(small_dims))
        return int(min(dims))
    return None


def _auto_channel_names(value, axis=None):
    count = _estimate_sensor_count(value, axis=axis)
    if count is None or count < 1:
        count = 1
    return [f"ch{i}" for i in range(int(count))]


def _resolve_locator_value(obj, locator):
    if not locator:
        return None
    parser = EphysDatasetParser(ParseConfig())
    try:
        return parser._resolve(obj, locator)
    except Exception:
        return None


def _load_parser_source(path):
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pkl", ".pickle"}:
        try:
            return pd.read_pickle(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)
    parser = EphysDatasetParser(ParseConfig())
    source_obj, _ = parser._load_source(path)  # Internal helper is acceptable for UI inspection.
    return source_obj


def _coerce_trial_dataframe_sequence(source_obj):
    """
    Field-potential pipeline files with multiple trials are commonly stored as
    a list/tuple of per-trial DataFrames. Flatten them into one DataFrame so
    parser inspection can expose dataframe fields in the UI.
    """
    if not isinstance(source_obj, (list, tuple)):
        return source_obj
    if len(source_obj) == 0:
        return source_obj
    if not all(isinstance(item, pd.DataFrame) for item in source_obj):
        return source_obj

    frames = []
    for idx, frame in enumerate(source_obj):
        frame_use = frame.copy()
        if "trial_index" not in frame_use.columns:
            frame_use["trial_index"] = int(idx)
        frames.append(frame_use)
    return pd.concat(frames, ignore_index=True)


def _load_features_source(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext in {".xlsx", ".xls", ".feather"}:
        loaded = compute_utils.read_df_file(path)
    else:
        loaded = _load_parser_source(path)
    return _coerce_trial_dataframe_sequence(loaded)


def _load_uploaded_source_in_memory(upload):
    safe_name = secure_filename(upload.filename or "")
    if not safe_name:
        raise ValueError("Invalid uploaded file name.")
    ext = Path(safe_name).suffix.lower()

    raw = upload.read()
    if not raw:
        raise ValueError(f"Uploaded file '{safe_name}' is empty.")

    if ext in {".pkl", ".pickle"}:
        bio = io.BytesIO(raw)
        try:
            return pd.read_pickle(bio)
        except Exception:
            bio.seek(0)
            return pickle.load(bio)

    if ext == ".json":
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON file '{safe_name}': {exc}")

    if ext == ".npy":
        bio = io.BytesIO(raw)
        return np.load(bio, allow_pickle=True)

    if ext == ".csv":
        return pd.read_csv(io.BytesIO(raw))
    if ext == ".tsv":
        return pd.read_csv(io.BytesIO(raw), sep="\t")

    if ext == ".parquet":
        return pd.read_parquet(io.BytesIO(raw))

    if ext == ".feather":
        return pd.read_feather(io.BytesIO(raw))

    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw))

    if ext == ".mat":
        return compute_utils._load_mat_with_fallback(raw, in_memory=True, source_name=safe_name)

    raise ValueError(f"Unsupported file type for in-memory parsing: {ext}")


def _copy_parse_config(base_cfg, *, fields):
    return ParseConfig(
        fields=fields,
        epoch_length_s=base_cfg.epoch_length_s,
        epoch_step_s=base_cfg.epoch_step_s,
        preload=base_cfg.preload,
        pick_types=base_cfg.pick_types,
        max_seconds=base_cfg.max_seconds,
        drop_bads=base_cfg.drop_bads,
        zscore=base_cfg.zscore,
        aggregate_over=base_cfg.aggregate_over,
        aggregate_method=base_cfg.aggregate_method,
        aggregate_labels=base_cfg.aggregate_labels,
        warn_unimplemented=base_cfg.warn_unimplemented,
    )


def _copy_canonical_fields(base_fields, **overrides):
    payload = {
        "data": base_fields.data,
        "fs": base_fields.fs,
        "ch_names": base_fields.ch_names,
        "time": base_fields.time,
        "data_domain": base_fields.data_domain,
        "freqs": base_fields.freqs,
        "spectral_kind": base_fields.spectral_kind,
        "epoch": base_fields.epoch,
        "metadata": dict(base_fields.metadata or {}),
        "table_layout": base_fields.table_layout,
        "channel_columns": base_fields.channel_columns,
        "long_channel_col": base_fields.long_channel_col,
        "long_value_col": base_fields.long_value_col,
        "long_time_col": base_fields.long_time_col,
    }
    payload.update(overrides)
    return CanonicalFields(**payload)


def _resolve_data_candidate_for_channels(source_obj, data_locator):
    if isinstance(data_locator, str):
        if data_locator == "__self__":
            return source_obj
        return _resolve_locator_value(source_obj, data_locator)
    if callable(data_locator):
        try:
            return data_locator(source_obj)
        except Exception:
            return None
    return None


def _extract_dataframe_data_column(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Data locator '{column_name}' is not a dataframe column.")

    series = df[column_name]
    if series.empty:
        raise ValueError(f"Data column '{column_name}' is empty.")

    if len(series) == 1:
        return series.iloc[0]

    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy()

    arrays = []
    for value in series.tolist():
        arr = np.asarray(value).squeeze()
        if arr.ndim == 0:
            return series.iloc[0]
        if arr.ndim != 1:
            return series.iloc[0]
        arrays.append(arr)

    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) == 1:
        return np.vstack(arrays)
    return series.iloc[0]


def _extract_dataframe_scalar_column(df, column_name):
    if not column_name or column_name not in df.columns:
        return None
    series = df[column_name].dropna()
    if series.empty:
        return None
    value = series.iloc[0]
    arr = np.asarray(value)
    if arr.ndim == 0:
        try:
            return float(arr.item())
        except Exception:
            return None
    return None


def _extract_dataframe_channel_names(df, column_name):
    if not column_name or column_name not in df.columns:
        return None
    series = df[column_name].dropna()
    if series.empty:
        return None
    if len(series) == 1:
        value = series.iloc[0]
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                return [str(v) for v in value.tolist()]
            return None
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]

    out = []
    for value in series.tolist():
        arr = np.asarray(value)
        if arr.ndim != 0:
            return None
        out.append(str(value))
    return out or None


def _build_mapping_source_from_dataframe(df, parse_cfg):
    fields = parse_cfg.fields
    if not isinstance(fields.data, str):
        raise ValueError("For dataframe auto mapping, parser data locator must be a column name.")

    data_value = _extract_dataframe_data_column(df, fields.data)
    mapped = {"data": data_value}

    fs_value = None
    if isinstance(fields.fs, str):
        fs_value = _extract_dataframe_scalar_column(df, fields.fs)
    elif isinstance(fields.fs, (int, float, np.integer, np.floating)):
        fs_value = float(fields.fs)
    if fs_value is not None:
        mapped["fs"] = fs_value

    ch_value = None
    if isinstance(fields.ch_names, str):
        ch_value = _extract_dataframe_channel_names(df, fields.ch_names)
    elif callable(fields.ch_names):
        try:
            ch_value = fields.ch_names(df)
        except Exception:
            ch_value = None
        if isinstance(ch_value, np.ndarray):
            if ch_value.ndim == 1:
                ch_value = [str(v) for v in ch_value.tolist()]
            else:
                ch_value = None
        elif isinstance(ch_value, (list, tuple)):
            ch_value = [str(v) for v in ch_value]
    elif isinstance(fields.ch_names, np.ndarray):
        if fields.ch_names.ndim == 1:
            ch_value = [str(v) for v in fields.ch_names.tolist()]
    elif isinstance(fields.ch_names, (list, tuple)):
        ch_value = [str(v) for v in fields.ch_names]
    if ch_value is None:
        ch_value = _auto_channel_names(data_value)
    mapped["ch_names"] = ch_value

    metadata = {}
    for key, locator in (fields.metadata or {}).items():
        if isinstance(locator, str) and locator in df.columns:
            series = df[locator].dropna()
            metadata[key] = series.iloc[0] if not series.empty else None
        else:
            metadata[key] = locator

    mapped_fields = CanonicalFields(
        data="data",
        fs="fs" if "fs" in mapped else None,
        ch_names="ch_names",
        metadata=metadata,
    )
    mapped_cfg = _copy_parse_config(parse_cfg, fields=mapped_fields)
    return mapped, mapped_cfg


def _describe_parser_source(path):
    source_obj = _load_features_source(path)
    fs_hint_hz = None
    fs_hint_note = None

    def _build_defaults(candidates):
        return {
            "data": _pick_field_guess_fuzzy(candidates, ["data", "signal", "proxy", "lfp", "eeg", "meg", "ecog", "cdm"]),
            "fs": _pick_field_guess_fuzzy(
                candidates,
                ["fs", "sfreq", "freq", "frequency", "sampling_rate", "sampling_frequency", "sampling_frequency_hz"],
            ),
            "ch_names": _pick_field_guess_fuzzy(
                candidates,
                ["ch_names", "channels", "channel_names", "channel", "sensors", "sensor", "ch"],
            ),
            "recording_type": _pick_field_guess(candidates, ["recording_type", "modality", "recording", "type"]),
            "subject_id": _pick_field_guess(candidates, ["subject_id", "subject", "subj", "participant"]),
            "group": _pick_field_guess(candidates, ["group", "cohort", "class"]),
            "species": _pick_field_guess(candidates, ["species", "animal", "organism"]),
            "condition": _pick_field_guess(candidates, ["condition", "state", "task"]),
        }

    if isinstance(source_obj, pd.DataFrame):
        fs_hint_hz, fs_hint_note = _extract_dataframe_fs_hint(source_obj)
        columns = [str(col) for col in source_obj.columns]
        defaults = _build_defaults(columns)
        if fs_hint_hz is None and defaults["fs"] and defaults["fs"] in source_obj.columns:
            fs_series = pd.to_numeric(source_obj[defaults["fs"]], errors="coerce").dropna()
            if not fs_series.empty:
                val = float(fs_series.iloc[0])
                if np.isfinite(val) and val > 0:
                    fs_hint_hz = val
                    fs_hint_note = f"Detected fs column in data source. Suggested fs: {val:g} Hz."
        sensor_count = None
        data_guess = defaults.get("data")
        if data_guess:
            try:
                data_value = _extract_dataframe_data_column(source_obj, data_guess)
                sensor_count = _estimate_sensor_count(data_value)
            except Exception:
                sensor_count = None
        if sensor_count is None:
            sensor_col = next((c for c in source_obj.columns if str(c).lower() == "sensor"), None)
            if sensor_col is not None:
                try:
                    sensor_count = int(source_obj[sensor_col].nunique(dropna=True))
                except Exception:
                    sensor_count = None
        if sensor_count is None:
            exclude = {"time", "t", "fs", "sfreq"}
            numeric_cols = [c for c in source_obj.columns if np.issubdtype(source_obj[c].dtype, np.number)]
            sensor_count = len([c for c in numeric_cols if str(c).lower() not in exclude])

        payload = {
            "source_type": "dataframe",
            "candidate_fields": columns,
            "defaults": defaults,
            "summary": f"DataFrame with {source_obj.shape[0]} rows and {source_obj.shape[1]} columns.",
            "sensor_count_estimate": sensor_count,
            "multi_sensor_detected": bool(sensor_count and sensor_count > 1),
            "fs_hint_hz": fs_hint_hz,
            "fs_hint_note": fs_hint_note,
        }
        payload["field_details"] = _describe_source_fields_for_ui(source_obj, columns, "dataframe")
        payload["manual_field_details"] = _manual_field_details_for_ui()
        return payload

    if isinstance(source_obj, np.ndarray):
        sensor_count = _estimate_sensor_count(source_obj)
        payload = {
            "source_type": "ndarray",
            "candidate_fields": ["__self__"],
            "defaults": {
                "data": "__self__",
                "fs": "",
                "ch_names": "",
                "recording_type": "",
                "subject_id": "",
                "group": "",
                "species": "",
                "condition": "",
            },
            "summary": f"NumPy array with shape {list(source_obj.shape)}.",
            "sensor_count_estimate": sensor_count,
            "multi_sensor_detected": bool(sensor_count and sensor_count > 1),
            "fs_hint_hz": fs_hint_hz,
            "fs_hint_note": fs_hint_note,
        }
        payload["field_details"] = _describe_source_fields_for_ui(source_obj, ["__self__"], "field")
        payload["manual_field_details"] = _manual_field_details_for_ui()
        return payload

    # MNE Raw / Epochs objects (duck-typed to avoid hard MNE import)
    if hasattr(source_obj, "get_data") and hasattr(source_obj, "ch_names") and hasattr(source_obj, "info"):
        try:
            n_ch = len(source_obj.ch_names)
        except Exception:
            n_ch = "?"
        try:
            sfreq = float(getattr(source_obj.info, "sfreq", 0))
        except Exception:
            sfreq = None
        mne_candidates = [
            "get_data",
            "ch_names",
            "info.sfreq",
            "info.subject_info",
            "info.description",
        ]
        mne_defaults = {
            "data": "get_data",
            "fs": "info.sfreq",
            "ch_names": "ch_names",
            "recording_type": "",
            "subject_id": "",
            "group": "",
            "species": "",
            "condition": "",
        }
        summary_parts = [f"MNE {type(source_obj).__name__}", f"{n_ch} channels"]
        if sfreq:
            summary_parts.append(f"{sfreq:g} Hz")
        payload = {
            "source_type": "mne_raw",
            "candidate_fields": mne_candidates,
            "defaults": mne_defaults,
            "summary": " · ".join(summary_parts) + ".",
            "sensor_count_estimate": n_ch if isinstance(n_ch, int) else None,
            "multi_sensor_detected": isinstance(n_ch, int) and n_ch > 1,
            "fs_hint_hz": sfreq,
            "fs_hint_note": f"Sampling rate from MNE info: {sfreq:g} Hz." if sfreq else None,
        }
        payload["field_details"] = _describe_source_fields_for_ui(source_obj, mne_candidates, "field")
        payload["manual_field_details"] = _manual_field_details_for_ui()
        return payload

    candidate_fields = _extract_locator_candidates(source_obj)
    defaults = _build_defaults(candidate_fields)

    if isinstance(source_obj, MappingABC):
        top_keys = [str(k) for k in source_obj.keys() if not str(k).startswith("__")]
        data_guess = defaults.get("data")
        resolved_data = _resolve_locator_value(source_obj, data_guess)
        sensor_count = _estimate_sensor_count(resolved_data) if resolved_data is not None else None
        payload = {
            "source_type": "mapping",
            "candidate_fields": candidate_fields,
            "top_keys": top_keys,
            "defaults": defaults,
            "summary": f"Mapping with {len(top_keys)} top-level keys.",
            "sensor_count_estimate": sensor_count,
            "multi_sensor_detected": bool(sensor_count and sensor_count > 1),
            "fs_hint_hz": fs_hint_hz,
            "fs_hint_note": fs_hint_note,
        }
        payload["field_details"] = _describe_source_fields_for_ui(source_obj, candidate_fields, "field")
        payload["manual_field_details"] = _manual_field_details_for_ui()
        return payload

    data_guess = defaults.get("data")
    resolved_data = _resolve_locator_value(source_obj, data_guess)
    sensor_count = _estimate_sensor_count(resolved_data) if resolved_data is not None else None
    payload = {
        "source_type": "object",
        "candidate_fields": candidate_fields,
        "defaults": defaults,
        "summary": f"Object of type {type(source_obj).__name__}.",
        "sensor_count_estimate": sensor_count,
        "multi_sensor_detected": bool(sensor_count and sensor_count > 1),
        "fs_hint_hz": fs_hint_hz,
        "fs_hint_note": fs_hint_note,
    }
    payload["field_details"] = _describe_source_fields_for_ui(source_obj, candidate_fields, "field")
    payload["manual_field_details"] = _manual_field_details_for_ui()
    return payload


def _build_parse_config_from_form(form):
    data_locator = (form.get("parser_data_locator") or "").strip()
    if not data_locator:
        raise ValueError("Select a locator for the parser data field.")

    data_source_kind = (form.get("data_source_kind") or "").strip()
    zscore = str(form.get("parser_zscore", "")).lower() in {"1", "true", "on", "yes"}
    epoching_enabled = str(form.get("parser_enable_epoching", "")).lower() in {"1", "true", "on", "yes"}
    aggregate_enabled = str(form.get("parser_enable_aggregate", "")).lower() in {"1", "true", "on", "yes"}
    epoch_length_s = _optional_float(form.get("parser_epoch_length_s")) if epoching_enabled else None
    epoch_step_s = _optional_float(form.get("parser_epoch_step_s")) if epoching_enabled else None
    if epoching_enabled and (epoch_length_s is None or epoch_step_s is None):
        raise ValueError("Epoching is enabled. Provide both epoch length and epoch step in seconds.")

    aggregate_over = None
    aggregate_method = "sum"
    aggregate_labels = None
    if aggregate_enabled:
        aggregate_over = _parse_aggregate_over(form.get("parser_aggregate_over"))
        if not aggregate_over:
            raise ValueError("Aggregation is enabled. Provide at least one dimension in 'aggregate over'.")
        aggregate_method = (form.get("parser_aggregate_method") or "sum").strip().lower()
        if aggregate_method not in {"sum", "mean", "median"}:
            raise ValueError("Aggregate method must be one of: sum, mean, median.")
        aggregate_labels = _parse_aggregate_labels(form.get("parser_aggregate_labels"))

    fs_locator = (form.get("parser_fs_locator") or "").strip() or None
    fs_source = (form.get("parser_fs_source") or "").strip()
    fs_manual = _optional_float(form.get("parser_fs_manual"))
    if fs_source == "__numeric__":
        if fs_manual is None:
            # Defensive fallback for duplicated/hidden form controls:
            # if a locator is present, prefer it; for empirical sources allow missing fs.
            if fs_locator:
                fs_value = fs_locator
            elif data_source_kind == "new-empirical":
                fs_value = None
            else:
                raise ValueError("Sampling frequency numeric value is required when source is set to numeric.")
        else:
            fs_value = float(fs_manual)
    elif fs_source == "__none__":
        fs_value = None
    elif fs_source:
        fs_value = fs_source
    else:
        # Backward-compatible fallback for old forms.
        if fs_locator and fs_manual is not None:
            raise ValueError("Sampling frequency locator and numeric value are mutually exclusive.")
        if fs_locator:
            fs_value = fs_locator
        elif fs_manual is not None:
            fs_value = float(fs_manual)
        else:
            raise ValueError("Sampling frequency is required (field, numeric value, or None).")

    recording_type_source = (form.get("parser_recording_type_source") or "").strip()
    recording_type_locator = (form.get("parser_recording_type_locator") or "").strip() or None
    recording_type = (form.get("parser_recording_type") or "").strip()
    subject_id_locator = (form.get("parser_subject_id_locator") or "").strip() or None
    group_locator = (form.get("parser_group_locator") or "").strip() or None
    species_locator = (form.get("parser_species_locator") or "").strip() or None
    condition_locator = (form.get("parser_condition_locator") or "").strip() or None
    if recording_type_source == "__value__":
        recording_type_value = recording_type or "LFP"
    elif recording_type_source == "__none__":
        recording_type_value = None
    elif recording_type_source:
        recording_type_value = recording_type_source
    else:
        # Backward-compatible fallback for old forms.
        if recording_type_locator and recording_type:
            raise ValueError("Recording type locator and recording type value are mutually exclusive.")
        recording_type = recording_type or "LFP"
        recording_type_value = recording_type_locator if recording_type_locator else recording_type

    # Array axis mapping
    array_axes_enabled = str(form.get("parser_array_axes_enabled", "")).lower() in {"1", "on", "true", "yes"}
    array_axes = None
    if array_axes_enabled:
        try:
            axis_channels = int(form.get("parser_axis_channels") or 0)
            axis_samples = int(form.get("parser_axis_samples") or 1)
            axis_epochs_raw = form.get("parser_axis_epochs") or "-1"
            axis_epochs = int(axis_epochs_raw)
            axis_ids_raw = form.get("parser_axis_ids") or "-1"
            axis_ids = int(axis_ids_raw)
        except (ValueError, TypeError):
            raise ValueError("Invalid array axis mapping values. Axes must be integers.")
        all_axes = [axis_channels, axis_samples]
        if axis_epochs >= 0:
            all_axes.append(axis_epochs)
        if axis_ids >= 0:
            all_axes.append(axis_ids)
        if len(set(all_axes)) != len(all_axes):
            raise ValueError("Array axis mapping: each dimension can only be assigned to one role.")
        array_axes = {"channels": axis_channels, "samples": axis_samples}
        if axis_epochs >= 0:
            array_axes["epochs"] = axis_epochs
        if axis_ids >= 0:
            array_axes["ids"] = axis_ids
            # Data is already pre-epoched: disable parser-level epoching
        if axis_epochs >= 0:
            epoching_enabled = False
            epoch_length_s = None
            epoch_step_s = None

    sensor_names = _parse_sensor_names(form.get("parser_sensor_names"))
    ch_names_source = (form.get("parser_ch_names_source") or "").strip()
    ch_names_locator = (form.get("parser_ch_names_locator") or "").strip() or None
    ch_names_autocomplete_axis = _parse_nonnegative_int(
        form.get("parser_ch_names_autocomplete_axis"),
        default=0,
        field_name="Autocomplete channel axis",
    )
    if ch_names_source == "__manual__":
        if sensor_names is None:
            raise ValueError("Provide manual channel names when channel names source is set to manual.")
        ch_names_value = sensor_names
    elif ch_names_source == "__autocomplete__":
        def _auto_ch_names_locator(
            obj,
            locator=data_locator,
            axis=ch_names_autocomplete_axis,
        ):
            data_candidate = obj if locator == "__self__" else _resolve_locator_value(obj, locator)
            if data_candidate is None:
                return None
            return _auto_channel_names(data_candidate, axis=axis)

        ch_names_value = _auto_ch_names_locator
    elif ch_names_source == "__none__":
        ch_names_value = None
    elif ch_names_source:
        ch_names_value = ch_names_source
    else:
        # Backward-compatible fallback for old forms.
        ch_names_value = sensor_names if sensor_names is not None else ch_names_locator

    if data_source_kind in {"pipeline", "new-simulation"}:
        metadata_mode = (form.get("parser_metadata_mode") or "empirical").strip().lower()
        if metadata_mode not in {"simulated", "empirical"}:
            metadata_mode = "empirical"

        metadata = {}
        if recording_type_value is not None:
            metadata["recording_type"] = recording_type_value

        if metadata_mode == "simulated":
            metadata.setdefault("subject_id", 0)
            metadata.setdefault("group", "simulation")
            metadata.setdefault("species", "simulated")
            metadata.setdefault("condition", "simulation_pipeline")
        else:
            subject_id_value = _parse_metadata_field_source(
                form,
                "parser_metadata_subject_id_source",
                "parser_metadata_subject_id",
            )
            group_value = _parse_metadata_field_source(
                form,
                "parser_metadata_group_source",
                "parser_metadata_group",
            )
            species_value = _parse_metadata_field_source(
                form,
                "parser_metadata_species_source",
                "parser_metadata_species",
            )
            condition_value = _parse_metadata_field_source(
                form,
                "parser_metadata_condition_source",
                "parser_metadata_condition",
            )
            if subject_id_value:
                metadata["subject_id"] = subject_id_value
            if group_value:
                metadata["group"] = group_value
            if species_value:
                metadata["species"] = species_value
            if condition_value:
                metadata["condition"] = condition_value

        fields = CanonicalFields(
            data=data_locator,
            fs=fs_value,
            ch_names=ch_names_value,
            metadata=metadata,
            array_axes=array_axes,
        )
        return ParseConfig(
            fields=fields,
            epoch_length_s=epoch_length_s,
            epoch_step_s=epoch_step_s,
            zscore=zscore,
            aggregate_over=aggregate_over,
            aggregate_method=aggregate_method,
            aggregate_labels=aggregate_labels if aggregate_labels is not None else {"sensor": "aggregate"},
        )

    table_layout = (form.get("parser_table_layout") or "").strip().lower() or None
    if table_layout not in {None, "wide", "long"}:
        raise ValueError("Invalid table layout. Use 'wide' or 'long'.")

    metadata = {"recording_type": recording_type_value}
    if data_source_kind == "new-empirical":
        if subject_id_locator:
            metadata["subject_id"] = FILE_ID_METADATA_LITERAL if subject_id_locator == "__file_id__" else subject_id_locator
        if group_locator:
            metadata["group"] = FILE_ID_METADATA_LITERAL if group_locator == "__file_id__" else group_locator
        if species_locator:
            metadata["species"] = FILE_ID_METADATA_LITERAL if species_locator == "__file_id__" else species_locator
        if condition_locator:
            metadata["condition"] = FILE_ID_METADATA_LITERAL if condition_locator == "__file_id__" else condition_locator

    fields = CanonicalFields(
        data=data_locator,
        fs=fs_value,
        ch_names=ch_names_value,
        metadata=metadata,
        table_layout=table_layout,
        long_time_col=(form.get("parser_long_time_col") or "").strip() or None,
        long_channel_col=(form.get("parser_long_channel_col") or "").strip() or None,
        long_value_col=(form.get("parser_long_value_col") or "").strip() or None,
        array_axes=array_axes,
    )
    return ParseConfig(
        fields=fields,
        epoch_length_s=epoch_length_s,
        epoch_step_s=epoch_step_s,
        zscore=zscore,
        aggregate_over=aggregate_over,
        aggregate_method=aggregate_method,
        aggregate_labels=aggregate_labels if aggregate_labels is not None else {"sensor": "aggregate"},
    )


def _normalize_features_input_path(source_path, form, job_id, upload_dir=None):
    source_obj = _load_features_source(source_path)
    data_source_kind = (form.get("data_source_kind") or "").strip()

    if not (form.get("parser_data_locator") or "").strip():
        raise ValueError(
            "Configure EphysDatasetParser (at least the data locator)."
        )

    parse_cfg = _build_parse_config_from_form(form)
    _validate_data_locator_against_dataframe_source(
        parse_cfg.fields.data,
        source_obj,
        source_label=f"source file '{os.path.basename(str(source_path or '').strip()) or 'sample'}'",
    )

    parser_source = source_obj
    parser_cfg = parse_cfg

    if isinstance(source_obj, pd.DataFrame):
        if data_source_kind == "pipeline":
            parser_source, parser_cfg = _build_mapping_source_from_dataframe(source_obj, parse_cfg)
        else:
            layout = parse_cfg.fields.table_layout
            if layout not in {"wide", "long"}:
                parser_source, parser_cfg = _build_mapping_source_from_dataframe(source_obj, parse_cfg)

    # If channel names are still missing, generate ch0..chN from detected sensor count.
    if parser_cfg.fields.ch_names is None or parser_cfg.fields.ch_names == "":
        data_candidate = _resolve_data_candidate_for_channels(parser_source, parser_cfg.fields.data)
        sensor_count = _estimate_sensor_count(data_candidate)
        if sensor_count is not None and sensor_count >= 1:
            auto_names = [f"ch{i}" for i in range(int(sensor_count))]
            auto_fields = _copy_canonical_fields(parser_cfg.fields, ch_names=auto_names)
            parser_cfg = _copy_parse_config(parser_cfg, fields=auto_fields)

    parser = EphysDatasetParser(parser_cfg)
    parsed_df = parser.parse(parser_source)
    metadata_map = dict(parser_cfg.fields.metadata or {})
    file_id_fields = [
        str(key)
        for key, locator in metadata_map.items()
        if isinstance(locator, str) and locator == FILE_ID_METADATA_LITERAL
    ]
    file_extracted_fields = {
        str(key): str(locator)
        for key, locator in metadata_map.items()
        if _file_extracted_chain_index(locator) is not None
    }
    source_label = os.path.basename(str(source_path or ""))
    if file_id_fields:
        for col_name in file_id_fields:
            parsed_df[col_name] = source_label if source_label else None
    if file_extracted_fields:
        for col_name, locator in file_extracted_fields.items():
            parsed_df[col_name] = _resolve_file_extracted_value(source_label, locator)
    target_upload_dir = upload_dir or _module_uploads_dir_for("features")
    os.makedirs(target_upload_dir, exist_ok=True)
    normalized_path = os.path.join(target_upload_dir, f"features_data_file_0_{job_id}_parsed.pkl")
    parsed_df.to_pickle(normalized_path)
    return normalized_path


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


def _run_process_with_progress(
    cmd,
    cwd,
    job_status,
    job_id,
    estimate_seconds,
    progress_callback=None,
    line_callback=None,
    preserve_existing_output=False,
):
    output_lines = []
    progress_seen = threading.Event()
    output_buffer = deque(maxlen=MAX_OUTPUT_LINES) if MAX_OUTPUT_LINES > 0 else None
    output_text_holder = [""]
    cancelled = False

    if job_id in job_status:
        job_status[job_id].setdefault("output", "")
        if preserve_existing_output:
            existing_output = job_status[job_id].get("output", "")
            if existing_output:
                if output_buffer is not None:
                    for existing_line in existing_output.splitlines(keepends=True):
                        output_buffer.append(existing_line)
                    output_text_holder[0] = "".join(output_buffer)
                else:
                    output_text_holder[0] = str(existing_output)
        job_status[job_id]["output"] = output_text_holder[0]

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    if job_id in job_status:
        job_status[job_id]["worker_pid"] = process.pid
        job_status[job_id]["worker_isolated_pgrp"] = True

    start = time.time()

    def _maybe_update_progress(line):
        value = None
        if "PROGRESS:" in line:
            marker_index = line.find("PROGRESS:")
            if marker_index != -1:
                value = line[marker_index + len("PROGRESS:"):].strip()
        else:
            segment_match = re.search(r"SIM_SEGMENT\s+(\d+)\s*/\s*(\d+)", line)
            if segment_match:
                done = int(segment_match.group(1))
                total = int(segment_match.group(2))
                if total > 0:
                    value = str((100.0 * done) / float(total))
        if value is None:
            return
        try:
            pct = int(float(value))
        except ValueError:
            return
        pct = max(0, min(99, pct))
        progress_seen.set()
        if progress_callback is not None:
            progress_callback(pct)
        else:
            job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), pct)
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
            if output_buffer is not None:
                output_buffer.append(line)
                output_text_holder[0] = "".join(output_buffer)
            else:
                output_text_holder[0] = f"{output_text_holder[0]}{line}"
            if job_id in job_status:
                job_status[job_id]["output"] = output_text_holder[0]
            if line_callback is not None:
                line_callback(line)
        process.stdout.close()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    initial_estimate = estimate_seconds
    while True:
        if _is_job_cancel_requested(job_id):
            cancelled = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass
            break
        if process.poll() is not None:
            break
        if not progress_seen.is_set() and job_id in job_status:
            elapsed = time.time() - start
            estimate = max(1.0, float(initial_estimate or estimate_seconds or 60.0))
            pct = int(min(95, max(0.0, (elapsed / estimate) * 95.0)))
            if progress_callback is not None:
                progress_callback(pct)
            else:
                job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), pct)
                job_status[job_id]["estimated_time_remaining"] = start + estimate
        time.sleep(1)

    reader_thread.join(timeout=2)
    if job_id in job_status:
        job_status[job_id].pop("worker_pid", None)
        job_status[job_id].pop("worker_isolated_pgrp", None)
    if cancelled:
        raise JobCancelledError("Computation cancelled by user.")
    output_text = "".join(output_lines).strip()
    return process.returncode, output_text


def _build_simulation_params(form, defaults):
    tstop = _parse_float(form, "tstop", defaults["tstop"])
    dt = _parse_float(form, "dt", defaults["dt"])
    local_num_threads = _parse_int(form, "local_num_threads", defaults["local_num_threads"])
    numpy_seed = _parse_optional_sim_numpy_seed(form)
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
        "# Optional fixed NumPy seed used to derive deterministic randomization in simulation scripts",
        f"numpy_seed = {numpy_seed if numpy_seed is not None else 'None'}",
        "",
    ])


def _enforce_simulation_chunk_seconds(simulation_py_path, chunk_ms=1000.0):
    """Force _simulate_with_progress to use fixed chunk duration in milliseconds."""
    try:
        with open(simulation_py_path, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError:
        return

    # Replace legacy adaptive chunking if present.
    updated = source
    updated = re.sub(
        r"step\s*=\s*max\(\s*dt\s*,\s*tstop\s*/\s*50(?:\.0)?\s*\)",
        f"step = max(dt, {float(chunk_ms):.1f})",
        updated,
    )

    # Ensure fixed chunking is present in _simulate_with_progress.
    if "_simulate_with_progress" in updated:
        updated = re.sub(
            r"step\s*=\s*max\(\s*dt\s*,\s*[0-9]+(?:\.[0-9]+)?\s*\)",
            f"step = max(dt, {float(chunk_ms):.1f})",
            updated,
        )

    if updated == source:
        return

    try:
        with open(simulation_py_path, "w", encoding="utf-8") as f:
            f.write(updated)
    except OSError:
        return


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
    simulation_data_dir = SIMULATION_DATA_DIR
    if os.path.isdir(simulation_data_dir):
        simulation_pkl_files = sorted(
            f for f in os.listdir(simulation_data_dir)
            if f.endswith(".pkl") and os.path.isfile(os.path.join(simulation_data_dir, f))
        )
    else:
        simulation_pkl_files = []
    field_potential_files = []
    for fp_dir in _field_potential_dirs():
        if not os.path.isdir(fp_dir):
            continue
        for root, _, files in os.walk(fp_dir):
            for name in files:
                if name.endswith(".pkl"):
                    field_potential_files.append(name)
    analysis_data_dir = ANALYSIS_DATA_DIR
    if os.path.isdir(analysis_data_dir):
        analysis_data_files = sorted(
            f for f in os.listdir(analysis_data_dir)
            if (f.endswith(".pkl") or f.endswith(".pickle"))
            and os.path.isfile(os.path.join(analysis_data_dir, f))
        )
    else:
        analysis_data_files = []
    features_data_files = _list_features_data_files()
    _bootstrap_predictions_data_from_previous_steps()
    predictions_data_files = _list_predictions_data_files()
    return render_template(
        "0.dashboard.html",
        simulation_pkl_files=simulation_pkl_files,
        has_simulation_pkl=bool(simulation_pkl_files),
        field_potential_files=sorted(set(field_potential_files)),
        has_field_potential_data=bool(field_potential_files),
        features_data_files=features_data_files,
        has_features_data=bool(features_data_files),
        predictions_data_files=predictions_data_files,
        has_predictions_data=bool(predictions_data_files),
        analysis_data_files=analysis_data_files,
        has_analysis_data=bool(analysis_data_files),
    )

# Simulation configuration page
@app.route("/simulation")
def simulation():
    options = [
        {
            "url": url_for('upload_sim'),
            "title": "Load data",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">upload_file</span>',
        },
        {
            "url": url_for('new_sim'),
            "title": "New simulation",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">add_circle</span>',
        },
    ]

    return render_template("1.simulation.html", options=options)

@app.route("/simulation/upload_sim")
def upload_sim():
    simulation_data_dir = SIMULATION_DATA_DIR
    if os.path.isdir(simulation_data_dir):
        simulation_pkl_files = sorted(
            f for f in os.listdir(simulation_data_dir)
            if Path(f).suffix.lower() in PICKLE_EXTENSIONS and os.path.isfile(os.path.join(simulation_data_dir, f))
        )
    else:
        simulation_pkl_files = []
    return render_template(
        "1.1.0.upload_sim.html",
        simulation_pkl_files=simulation_pkl_files,
        has_simulation_pkl=bool(simulation_pkl_files),
    )

@app.route("/upload_sim_files", methods=["POST"])
def upload_sim_files():
    _remember_path_history_from_form(request.form, "upload_sim_files")
    source_mode = (request.form.get("simulation_source_mode") or "upload").strip().lower()
    simulation_data_dir = SIMULATION_DATA_DIR
    os.makedirs(simulation_data_dir, exist_ok=True)

    if source_mode == "server-path":
        server_file_paths = _extract_server_file_paths(request.form, "simulation_server_file_path")
        legacy_folder_path = (request.form.get("simulation_server_folder_path") or "").strip()
        if server_file_paths:
            copied = 0
            for file_path in server_file_paths:
                try:
                    source_path = _validate_existing_pickle_file_path(file_path, "Simulation server file")
                except Exception as exc:
                    flash(str(exc), "error")
                    return redirect(request.referrer or url_for('upload_sim'))

                filename = secure_filename(os.path.basename(source_path))
                if not filename:
                    continue
                shutil.copy2(source_path, os.path.join(simulation_data_dir, filename))
                copied += 1
            if copied == 0:
                flash("No valid simulation server files selected.", "error")
                return redirect(request.referrer or url_for('upload_sim'))
        elif legacy_folder_path:
            # Backward compatibility with older clients sending a folder path.
            try:
                source_files = _collect_pickle_files_from_folder(legacy_folder_path, recursive=True)
            except Exception as exc:
                flash(f"Invalid simulation server folder path: {exc}", "error")
                return redirect(request.referrer or url_for('upload_sim'))

            copied = 0
            for source_path in source_files:
                filename = secure_filename(os.path.basename(source_path))
                if not filename:
                    continue
                shutil.copy2(source_path, os.path.join(simulation_data_dir, filename))
                copied += 1
            if copied == 0:
                flash("No valid simulation files were copied from server folder.", "error")
                return redirect(request.referrer or url_for('upload_sim'))
        else:
            flash("No simulation server file selected.", "error")
            return redirect(request.referrer or url_for('upload_sim'))
    else:
        files = request.files.getlist("simulation_files")
        uploaded_files = [f for f in files if f and f.filename]
        if len(uploaded_files) == 0:
            flash('No files uploaded, please try again.', 'error')
            return redirect(request.referrer or url_for('upload_sim'))
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            if not filename:
                continue
            file.save(os.path.join(simulation_data_dir, filename))

    return redirect(request.referrer or url_for('upload_sim'))


@app.route("/simulation/remove_file", methods=["POST"])
def remove_simulation_file():
    filename = (request.form.get("filename") or "").strip()
    simulation_root = SIMULATION_DATA_DIR
    try:
        target_path = _validate_relative_pickle_path(filename, simulation_root, "Simulation file")
        os.remove(target_path)
    except Exception as exc:
        flash(str(exc), "error")
    return redirect(url_for("upload_sim"))

@app.route("/clear_simulation_data", methods=["POST"])
def clear_simulation_data():
    clear_scope = (request.form.get("clear_scope") or "all").strip().lower()
    _clear_simulation_data_files()
    if clear_scope != "only":
        _clear_field_potential_data_files()
        _clear_features_data_files()
        _clear_predictions_data_files()
        _clear_analysis_state()
    return redirect(url_for('dashboard'))

@app.route("/clear_field_potential_data", methods=["POST"])
def clear_field_potential_data():
    clear_scope = (request.form.get("clear_scope") or "all").strip().lower()
    _clear_field_potential_data_files()
    if clear_scope != "only":
        _clear_features_data_files()
        _clear_predictions_data_files()
        _clear_analysis_state()
    return redirect(url_for('dashboard'))


@app.route("/clear_features_data", methods=["POST"])
def clear_features_data():
    clear_scope = (request.form.get("clear_scope") or "all").strip().lower()
    _clear_features_data_files()
    if clear_scope != "only":
        _clear_predictions_data_files()
        _clear_analysis_state()
    return redirect(url_for('dashboard'))


@app.route("/clear_predictions_data", methods=["POST"])
def clear_predictions_data():
    clear_scope = (request.form.get("clear_scope") or "all").strip().lower()
    _clear_predictions_data_files()
    if clear_scope != "only":
        _clear_analysis_state()
    return redirect(url_for('dashboard'))


def _clear_simulation_data_files():
    simulation_data_dir = SIMULATION_DATA_DIR
    if not os.path.isdir(simulation_data_dir):
        return
    for name in os.listdir(simulation_data_dir):
        lower = name.lower()
        if not (
            lower.endswith(".pkl")
            or lower.endswith(".pickle")
            or name == SIMULATION_GRID_METADATA_FILE
            or name in SIMULATION_GRID_METADATA_LEGACY_FILES
        ):
            continue
        path = os.path.join(simulation_data_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _clear_simulation_output_folder_all_files():
    simulation_data_dir = SIMULATION_DATA_DIR
    if not os.path.isdir(simulation_data_dir):
        return
    for name in os.listdir(simulation_data_dir):
        path = os.path.join(simulation_data_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


def _clear_field_potential_data_files():
    for fp_dir in _field_potential_dirs():
        if not os.path.isdir(fp_dir):
            continue
        for root, _, files in os.walk(fp_dir):
            for name in files:
                lower = name.lower()
                if not (lower.endswith(".pkl") or lower.endswith(".pickle")):
                    continue
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


def _clear_features_data_files():
    features_data_dir = _features_data_dir(create=False)
    if not os.path.isdir(features_data_dir):
        return
    for name in os.listdir(features_data_dir):
        if not (name.endswith(".pkl") or name.endswith(".pickle")):
            continue
        path = os.path.join(features_data_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _clear_predictions_data_files():
    predictions_data_dir = _predictions_data_dir(create=False)
    if not os.path.isdir(predictions_data_dir):
        return
    for name in os.listdir(predictions_data_dir):
        if not (name.endswith(".pkl") or name.endswith(".pickle")):
            continue
        path = os.path.join(predictions_data_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _clear_analysis_state():
    _clear_analysis_data_files()
    _clear_analysis_selection_mode()
    _clear_analysis_selected_simulation_keys()


def _clear_downstream_data_for_module(module_key):
    """Clear data produced by downstream modules in the processing pipeline."""
    key = (module_key or "").strip().lower()
    cleared = []
    if key == "simulation":
        _clear_field_potential_data_files()
        _clear_features_data_files()
        _clear_predictions_data_files()
        _clear_analysis_state()
        cleared = ["field_potential", "features", "inference", "analysis"]
    elif key == "field_potential":
        _clear_features_data_files()
        _clear_predictions_data_files()
        _clear_analysis_state()
        cleared = ["features", "inference", "analysis"]
    elif key == "features":
        _clear_predictions_data_files()
        _clear_analysis_state()
        cleared = ["inference", "analysis"]
    elif key == "inference":
        _clear_analysis_state()
        cleared = ["analysis"]
    return cleared


def _run_job_with_post_success_cleanup(job_id, module_key, func, *func_args):
    """Run a background job and clear downstream data only on successful completion."""
    try:
        if _is_job_cancel_requested(job_id):
            _mark_job_cancelled(job_id, "Computation cancelled by user.")
            return
        func(job_id, job_status, *func_args)
    except Exception as exc:
        app.logger.exception("[compute %s] unhandled worker exception in %s", job_id, getattr(func, "__name__", "worker"))
        status = job_status.get(job_id)
        if isinstance(status, dict):
            if status.get("cancel_requested") or status.get("status") == "cancelled":
                _mark_job_cancelled(job_id, status.get("cancel_message") or "Computation cancelled by user.")
            elif status.get("status") not in {"finished", "failed"}:
                _mark_job_failed(job_id, exc, prefix="Unhandled worker exception")
        return
    finally:
        job_futures.pop(job_id, None)

    status = job_status.get(job_id, {})
    if status.get("cancel_requested") or status.get("status") == "cancelled":
        _mark_job_cancelled(job_id, status.get("cancel_message") or "Computation cancelled by user.")
        return
    if status.get("status") != "finished" or bool(status.get("error")):
        return

    cleared = _clear_downstream_data_for_module(module_key)
    if cleared:
        app.logger.warning(
            "[compute %s] upstream '%s' finished; cleared downstream data: %s",
            job_id,
            module_key,
            ", ".join(cleared),
        )

@app.route("/simulation/new_sim")
def new_sim():
    return render_template("1.2.0.new_sim.html")

@app.route("/simulation/new_sim/brunel")
def new_sim_brunel():
    return render_template("1.2.2.new_sim_brunel.html")

@app.route("/simulation/new_sim/four_area")
def new_sim_four_area():
    return render_template("1.2.3.new_sim_four_area.html")

@app.route("/simulation/new_sim/custom")
def new_sim_custom():
    return render_template("1.2.1.new_sim_custom.html")


def _simulation_computation(job_id, job_status, params):
    simulation_runs_root = SIMULATION_RUNS_DIR
    try:
        model_type = params["model_type"]
        run_mode = params.get("run_mode", "single")
        run_forms = params.get("run_forms") or [params["form"]]
        estimate_seconds = float(params.get("estimate_seconds", 60.0))
        per_run_estimate = max(1.0, estimate_seconds / max(1, len(run_forms)))

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if model_type == "hagen":
            example_root = os.path.join(
                repo_root, "examples", "simulation", "Hagen_model", "simulation"
            )
            sim_defaults = HAGEN_DEFAULTS
        else:
            example_root = os.path.join(
                repo_root, "examples", "simulation", "four_area_cortical_model", "simulation"
            )
            sim_defaults = FOUR_AREA_DEFAULTS

        output_dir = SIMULATION_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        _clear_simulation_grid_metadata_file(output_dir)
        for name in os.listdir(output_dir):
            if name.endswith(".pkl"):
                path = os.path.join(output_dir, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        total_runs = len(run_forms)
        is_grid = run_mode == "grid"
        grid_metadata = _build_simulation_grid_metadata(model_type, run_mode, run_forms)
        if is_grid:
            repetitions = int((grid_metadata or {}).get("repetitions_per_configuration", 1))
            configuration_count = int((grid_metadata or {}).get("configuration_count", total_runs))
            _append_job_output(
                job_status,
                job_id,
                f"Parameter grid sweep mode enabled with {configuration_count} configuration(s), "
                f"{repetitions} repetition(s) each ({total_runs} simulation(s) total).",
            )
        else:
            if total_runs > 1:
                _append_job_output(
                    job_status,
                    job_id,
                    f"Single trial mode enabled with {total_runs} repetition(s).",
                )
            else:
                _append_job_output(job_status, job_id, "Single trial mode enabled.")

        job_status[job_id]["simulation_total"] = total_runs
        job_status[job_id]["simulation_completed"] = 0
        job_status[job_id]["estimated_time_remaining"] = time.time() + estimate_seconds

        aggregated_outputs = {}
        generated_files = None

        for run_idx, form in enumerate(run_forms, start=1):
            if _is_job_cancel_requested(job_id):
                raise JobCancelledError("Computation cancelled by user.")
            if model_type == "hagen":
                network_params_content = _build_hagen_network_params(form)
            else:
                network_params_content = _build_four_area_network_params(form)
            simulation_params_content = _build_simulation_params(form, sim_defaults)

            run_id = str(uuid.uuid4())
            run_root = os.path.join(simulation_runs_root, run_id)
            params_dir = os.path.join(run_root, "params")
            python_dir = os.path.join(run_root, "python")
            trial_output_dir = os.path.join(run_root, "output")
            os.makedirs(params_dir, exist_ok=True)
            os.makedirs(python_dir, exist_ok=True)
            os.makedirs(trial_output_dir, exist_ok=True)

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
            _enforce_simulation_chunk_seconds(os.path.join(python_dir, "simulation.py"), chunk_ms=1000.0)

            example_script_path = os.path.join(run_root, "example_model_simulation.py")
            example_script = "\n".join([
                "import os",
                "import sys",
                f"sys.path.insert(0, {repr(repo_root)})",
                "import ncpi",
                "",
                "if __name__ == \"__main__\":",
                "    # Create a Simulation object",
                "    sim = ncpi.Simulation(param_folder='params', python_folder='python', output_folder=%s)"
                % repr(trial_output_dir),
                "",
                "    # Run the network and simulation scripts (analysis is intentionally skipped)",
                "    sim.network('network.py', 'network_params.py')",
                "    sim.simulate('simulation.py', 'simulation_params.py')",
                "",
            ])
            with open(example_script_path, "w", encoding="utf-8") as f:
                f.write(example_script)

            if total_runs > 1:
                _append_job_output(job_status, job_id, f"Starting simulation {run_idx}/{total_runs}...")

            def _grid_progress(pct, current_run=run_idx):
                if total_runs <= 1:
                    return
                completed = current_run - 1
                overall = int(((completed + (pct / 100.0)) / total_runs) * 99)
                overall = max(0, min(99, overall))
                job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), overall)

            returncode, output_text = _run_process_with_progress(
                ["python", "example_model_simulation.py"],
                run_root,
                job_status,
                job_id,
                per_run_estimate,
                progress_callback=_grid_progress if total_runs > 1 else None,
                preserve_existing_output=total_runs > 1,
            )

            if returncode != 0:
                error_msg = output_text or "Unknown error"
                raise RuntimeError(f"Simulation {run_idx}/{total_runs} failed.\n{error_msg}")

            trial_files = sorted(
                name for name in os.listdir(trial_output_dir)
                if name.endswith(".pkl") and os.path.isfile(os.path.join(trial_output_dir, name))
            )
            if not trial_files:
                raise RuntimeError(f"No .pkl simulation outputs produced for run {run_idx}/{total_runs}.")

            if generated_files is None:
                generated_files = trial_files
            else:
                missing = [name for name in generated_files if name not in trial_files]
                if missing:
                    raise RuntimeError(
                        f"Simulation {run_idx}/{total_runs} did not produce expected files: {missing}"
                    )

            if total_runs == 1:
                for name in trial_files:
                    shutil.copy2(os.path.join(trial_output_dir, name), os.path.join(output_dir, name))
            else:
                for name in generated_files:
                    src = os.path.join(trial_output_dir, name)
                    with open(src, "rb") as handle:
                        payload = pickle.load(handle)
                    aggregated_outputs.setdefault(name, []).append(payload)

            job_status[job_id]["simulation_completed"] = run_idx
            if total_runs > 1:
                coarse = int((run_idx / total_runs) * 99)
                job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), coarse)
                _append_job_output(job_status, job_id, f"Completed simulation {run_idx}/{total_runs}.")
                remaining = total_runs - run_idx
                job_status[job_id]["estimated_time_remaining"] = time.time() + (remaining * per_run_estimate)
            else:
                job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 99)

            shutil.rmtree(run_root, ignore_errors=True)

        if total_runs > 1:
            for name, payload_list in aggregated_outputs.items():
                dst = os.path.join(output_dir, name)
                with open(dst, "wb") as handle:
                    pickle.dump(payload_list, handle)
            if is_grid:
                _append_job_output(
                    job_status,
                    job_id,
                    f"Stored grid outputs in {output_dir} ({total_runs} entries per file).",
                )
            else:
                _append_job_output(
                    job_status,
                    job_id,
                    f"Stored repeated single-trial outputs in {output_dir} ({total_runs} entries per file).",
                )
        if grid_metadata:
            try:
                metadata_path = _write_simulation_grid_metadata(output_dir, grid_metadata)
                if metadata_path:
                    _append_job_output(job_status, job_id, f"Stored grid metadata in {metadata_path}.")
            except Exception as meta_exc:
                _append_job_output(job_status, job_id, f"Warning: failed to store grid metadata: {meta_exc}")

        job_status[job_id].update({
            "status": "finished",
            "progress": 100,
            "results": output_dir,
            "error": False,
        })
        _announce_saved_output_folders(job_status, job_id, "simulation", output_dir)

    except Exception as exc:
        _clear_simulation_output_folder_all_files()
        if isinstance(exc, JobCancelledError):
            _mark_job_cancelled(job_id, str(exc))
        else:
            _mark_job_failed(job_id, exc)
    finally:
        _remove_dir_if_empty(simulation_runs_root)


def _simulation_computation_custom(job_id, job_status, params):
    temp_run_dir = None
    upload_root = params.get("upload_root")
    simulation_runs_root = SIMULATION_RUNS_DIR
    try:
        input_paths = params["input_paths"]
        estimate_seconds = params.get("estimate_seconds", 60.0)

        run_id = str(uuid.uuid4())
        temp_run_dir = os.path.join(simulation_runs_root, run_id)
        params_dir = os.path.join(temp_run_dir, "params")
        python_dir = os.path.join(temp_run_dir, "python")
        os.makedirs(params_dir, exist_ok=True)
        os.makedirs(python_dir, exist_ok=True)

        shutil.copy(input_paths["network_params"], os.path.join(params_dir, "network_params.py"))
        shutil.copy(input_paths["simulation_params"], os.path.join(params_dir, "simulation_params.py"))
        shutil.copy(input_paths["network_py"], os.path.join(python_dir, "network.py"))
        shutil.copy(input_paths["simulation_py"], os.path.join(python_dir, "simulation.py"))
        _enforce_simulation_chunk_seconds(os.path.join(python_dir, "simulation.py"), chunk_ms=1000.0)

        output_dir = SIMULATION_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        _clear_simulation_grid_metadata_file(output_dir)

        example_script_path = os.path.join(temp_run_dir, "example_model_simulation.py")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        example_script = "\n".join([
            "import os",
            "import sys",
            f"sys.path.insert(0, {repr(repo_root)})",
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
        _announce_saved_output_folders(job_status, job_id, "simulation", output_dir)

    except Exception as exc:
        _clear_simulation_output_folder_all_files()
        if isinstance(exc, JobCancelledError):
            _mark_job_cancelled(job_id, str(exc))
        else:
            _mark_job_failed(job_id, exc)
    finally:
        if temp_run_dir and os.path.isdir(temp_run_dir):
            shutil.rmtree(temp_run_dir, ignore_errors=True)
        if upload_root and os.path.isdir(upload_root):
            shutil.rmtree(upload_root, ignore_errors=True)
        _remove_dir_if_empty(simulation_runs_root)
        _remove_dir_if_empty(SIMULATION_CUSTOM_UPLOADS_DIR)


@app.route("/run_trial_simulation/<model_type>", methods=["POST"])
def run_trial_simulation(model_type):
    model_type = model_type.lower()
    if model_type not in {"hagen", "four_area"}:
        return "Model type is not valid", 400

    form = request.form.to_dict()
    ref_page = "new_sim_brunel" if model_type == "hagen" else "new_sim_four_area"

    try:
        run_mode, run_forms = _expand_simulation_forms(model_type, form)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(request.referrer or url_for(ref_page))

    if model_type == "hagen":
        sim_defaults = HAGEN_DEFAULTS
    else:
        sim_defaults = FOUR_AREA_DEFAULTS

    try:
        estimated_duration = sum(
            _estimate_duration_seconds(run_form, sim_defaults, model_type)
            for run_form in run_forms
        )
    except Exception as exc:
        flash(f"Invalid simulation parameters: {exc}", "error")
        return redirect(request.referrer or url_for(ref_page))
    estimated_duration = max(60.0, min(24 * 3600.0, float(estimated_duration)))

    job_id = str(uuid.uuid4())
    start_time = time.time()
    job_status[job_id] = {
        "status": "in_progress",
        "progress": 0,
        "start_time": start_time,
        "estimated_time_remaining": None,
        "results": None,
        "error": False,
        "cancel_requested": False,
        "computation_type": "simulation",
        "progress_mode": "manual",
        "output": "",
        "simulation_total": len(run_forms),
        "simulation_completed": 0,
        "simulation_mode": run_mode,
    }

    future = executor.submit(
        _run_job_with_post_success_cleanup,
        job_id,
        "simulation",
        _simulation_computation,
        {
            "model_type": model_type,
            "form": form,
            "run_forms": run_forms,
            "run_mode": run_mode,
            "estimate_seconds": estimated_duration,
        },
    )
    job_futures[job_id] = future

    return redirect(url_for("job_status_page", job_id=job_id, computation_type="simulation"))


@app.route("/run_trial_simulation_custom", methods=["POST"])
def run_trial_simulation_custom():
    _remember_path_history_from_form(request.form, "run_trial_simulation_custom")
    required_fields = {
        "network_params_file": "network_params",
        "network_py_file": "network_py",
        "simulation_params_file": "simulation_params",
        "simulation_py_file": "simulation_py",
    }

    run_id = str(uuid.uuid4())
    upload_root = os.path.join(SIMULATION_CUSTOM_UPLOADS_DIR, run_id)
    os.makedirs(upload_root, exist_ok=True)

    input_paths = {}
    for field, key in required_fields.items():
        source_mode = (request.form.get(f"{field}_source_mode") or "upload").strip().lower()
        dest_path = os.path.join(upload_root, f"{key}.py")

        if source_mode == "server-path":
            server_path_raw = (request.form.get(f"{field}_server_path") or "").strip()
            server_path = os.path.realpath(os.path.expanduser(server_path_raw))
            if not server_path:
                flash("All custom simulation files are required.", "error")
                return redirect(request.referrer or url_for("new_sim_custom"))
            if not os.path.isfile(server_path):
                flash(f"Server file not found: {server_path}", "error")
                return redirect(request.referrer or url_for("new_sim_custom"))
            if Path(server_path).suffix.lower() != ".py":
                flash(f"Server file must be a .py file: {server_path}", "error")
                return redirect(request.referrer or url_for("new_sim_custom"))
            shutil.copy2(server_path, dest_path)
        else:
            file = request.files.get(field)
            if not file or not file.filename:
                flash("All custom simulation files are required.", "error")
                return redirect(request.referrer or url_for("new_sim_custom"))
            filename = secure_filename(file.filename)
            if not filename:
                flash("Invalid uploaded file name for custom simulation.", "error")
                return redirect(request.referrer or url_for("new_sim_custom"))
            if Path(filename).suffix.lower() != ".py":
                flash("Custom simulation uploads must be Python files (.py).", "error")
                return redirect(request.referrer or url_for("new_sim_custom"))
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
        "cancel_requested": False,
        "computation_type": "simulation",
        "progress_mode": "manual",
        "output": "",
    }

    future = executor.submit(
        _run_job_with_post_success_cleanup,
        job_id,
        "simulation",
        _simulation_computation_custom,
        {"input_paths": input_paths, "upload_root": upload_root, "estimate_seconds": estimated_duration},
    )
    job_futures[job_id] = future

    return redirect(url_for("job_status_page", job_id=job_id, computation_type="simulation"))

# Field potential configuration page
@app.route("/field_potential")
def field_potential():
    options = [
        {
            "url": url_for('field_potential_load'),
            "title": "Load data",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">upload_file</span>',
        },
        {
            "url": url_for('field_potential_kernel'),
            "title": "Kernel",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">grain</span>',
        },
        {
            "url": url_for('field_potential_proxy'),
            "title": "Proxy",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">analytics</span>',
        },
    ]
    return render_template("2.field_potential.html", options=options)


@app.route("/field_potential/load")
def field_potential_load():
    field_potential_entries = _list_field_potential_detected_files()
    return render_template(
        "2.1.0.field_potential_load.html",
        field_potential_entries=field_potential_entries,
        has_field_potential_data=bool(field_potential_entries),
    )


@app.route("/field_potential/remove_file", methods=["POST"])
def remove_field_potential_file():
    file_key = (request.form.get("file_key") or "").strip()
    entries = _list_field_potential_detected_files()
    target = next((entry for entry in entries if entry.get("key") == file_key), None)
    if not target:
        flash("Field potential file not found.", "error")
        return redirect(url_for("field_potential_load"))

    target_path = os.path.realpath(target.get("path") or "")
    if not target_path or not os.path.isfile(target_path):
        flash("Field potential file does not exist.", "error")
        return redirect(url_for("field_potential_load"))
    allowed_roots = [os.path.realpath(root) for root in _field_potential_dirs()]
    if not any(_is_path_within_root(target_path, root) for root in allowed_roots):
        flash("Field potential file is outside allowed directories.", "error")
        return redirect(url_for("field_potential_load"))
    try:
        os.remove(target_path)
    except OSError as exc:
        flash(f"Failed to remove field potential file: {exc}", "error")
    return redirect(url_for("field_potential_load"))


@app.route("/field_potential/load_precomputed", methods=["POST"])
def field_potential_load_precomputed():
    _remember_path_history_from_form(request.form, "field_potential_load_precomputed")
    source_mode = (request.form.get("precomputed_fp_source_mode") or "upload").strip().lower()
    uploads = [file for file in request.files.getlist("precomputed_fp_file") if file and file.filename]
    server_file_paths = _extract_server_file_paths(request.form, "precomputed_fp_server_file_path")
    fp_type = (request.form.get("precomputed_fp_type") or "").strip().lower()

    destination_roots = {
        "proxy": FIELD_POTENTIAL_PROXY_DIR,
        "cdm": FIELD_POTENTIAL_KERNEL_DIR,
        "lfp": FIELD_POTENTIAL_KERNEL_DIR,
        "eeg": FIELD_POTENTIAL_MEEG_DIR,
        "meg": FIELD_POTENTIAL_MEEG_DIR,
        "meeg": FIELD_POTENTIAL_MEEG_DIR,
    }

    inputs = []
    if source_mode == "server-path":
        if not server_file_paths:
            flash("Select at least one server field potential file (.pkl/.pickle).", "error")
            return redirect(request.referrer or url_for("field_potential_load"))
        for server_path in server_file_paths:
            try:
                source_path = _validate_existing_pickle_file_path(server_path, "Field potential server file")
                safe_name = secure_filename(os.path.basename(source_path))
            except Exception as exc:
                flash(str(exc), "error")
                return redirect(request.referrer or url_for("field_potential_load"))
            if not safe_name:
                continue
            inputs.append({"safe_name": safe_name, "source_path": source_path, "upload": None})
    else:
        if len(uploads) == 0:
            flash("Upload at least one precomputed field potential file (.pkl/.pickle).", "error")
            return redirect(request.referrer or url_for("field_potential_load"))
        for upload in uploads:
            safe_name = secure_filename(upload.filename)
            if not safe_name:
                continue
            ext = Path(safe_name).suffix.lower()
            if ext not in PICKLE_EXTENSIONS:
                flash("Precomputed field potential must be a pickle file (.pkl/.pickle).", "error")
                return redirect(request.referrer or url_for("field_potential_load"))
            inputs.append({"safe_name": safe_name, "source_path": None, "upload": upload})

    if not inputs:
        flash("No valid precomputed field potential files were provided.", "error")
        return redirect(request.referrer or url_for("field_potential_load"))

    for item in inputs:
        safe_name = item["safe_name"]
        file_fp_type = fp_type or (_infer_field_potential_type_from_filename(safe_name) or "proxy")
        if file_fp_type not in destination_roots:
            file_fp_type = _infer_field_potential_type_from_filename(safe_name) or "proxy"

        output_root = destination_roots[file_fp_type]
        output_name = _field_potential_output_name(file_fp_type, safe_name)
        run_dir = os.path.join(output_root, f"loaded_{uuid.uuid4().hex[:12]}")
        os.makedirs(run_dir, exist_ok=True)
        output_path = os.path.join(run_dir, output_name)
        if source_mode == "server-path":
            shutil.copy2(item["source_path"], output_path)
        else:
            item["upload"].save(output_path)

    return redirect(request.referrer or url_for("field_potential_load"))


@app.route("/field_potential/kernel")
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

    remembered_mc_folder = _path_history_value_for_key("field_potential_kernel:mc_folder_browser")
    if remembered_mc_folder:
        mc_models_default = remembered_mc_folder

    remembered_output_sim = _path_history_value_for_key("field_potential_kernel:output_sim_path_browser")
    if remembered_output_sim:
        mc_outputs_default = remembered_output_sim

    remembered_kernel_params = _path_history_value_for_key("field_potential_kernel:kernel_params_module_browser")
    if remembered_kernel_params:
        kernel_params_default = remembered_kernel_params

    remembered_mc_folder_local = _path_history_value_for_key("field_potential_kernel:mc_folder_local_staged")
    if remembered_mc_folder_local and not os.path.isdir(remembered_mc_folder_local):
        remembered_mc_folder_local = ""

    remembered_output_sim_local = _path_history_value_for_key("field_potential_kernel:output_sim_path_local_staged")
    if remembered_output_sim_local and not os.path.isdir(remembered_output_sim_local):
        remembered_output_sim_local = ""

    default_dir = SIMULATION_DATA_DIR
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
    kernel_output_root = FIELD_POTENTIAL_KERNEL_DIR
    preferred_cdm = []
    for fname in ("kernel_approx_cdm.pkl", "current_dipole_moment.pkl"):
        direct_path = os.path.join(kernel_output_root, fname)
        if os.path.isfile(direct_path):
            preferred_cdm.append(direct_path)
        preferred_cdm.extend(glob.glob(os.path.join(kernel_output_root, "*", fname)))
    fallback_cdm = []
    for fname in ("gauss_cylinder_potential.pkl",):
        direct_path = os.path.join(kernel_output_root, fname)
        if os.path.isfile(direct_path):
            fallback_cdm.append(direct_path)
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
    biophys_options = []
    try:
        neuron_utils_mod = getattr(ncpi, "neuron_utils", None)
        if neuron_utils_mod is not None:
            excluded_biophys_methods = {"compute_nu_X"}
            for name, obj in inspect.getmembers(neuron_utils_mod):
                if name.startswith("_") or name in excluded_biophys_methods:
                    continue
                if not callable(obj):
                    continue
                try:
                    signature = inspect.signature(obj)
                    params = list(signature.parameters.values())
                except Exception:
                    continue
                if params and params[0].name == "cell":
                    biophys_options.append(name)
    except Exception:
        biophys_options = []
    biophys_options = sorted(set(biophys_options))
    if not biophys_options:
        biophys_options = ["set_Ih_linearized_hay2011", "make_cell_uniform"]
    return render_template(
        "2.2.0.field_potential_kernel.html",
        mc_models_default=mc_models_default,
        mc_outputs_default=mc_outputs_default,
        kernel_params_default=kernel_params_default,
        default_sim=default_sim,
        default_sim_paths=default_paths,
        default_meeg=default_meeg,
        initial_tab=initial_tab,
        biophys_options=biophys_options,
        mc_models_local_staged_default=remembered_mc_folder_local,
        mc_outputs_local_staged_default=remembered_output_sim_local,
    )

@app.route("/field_potential/proxy")
def field_potential_proxy():
    default_dir = SIMULATION_DATA_DIR
    default_paths = {
        "times": os.path.join(default_dir, "times.pkl"),
        "gids": os.path.join(default_dir, "gids.pkl"),
        "vm": os.path.join(default_dir, "vm.pkl"),
        "ampa": os.path.join(default_dir, "ampa.pkl"),
        "gaba": os.path.join(default_dir, "gaba.pkl"),
        "nu_ext": os.path.join(default_dir, "nu_ext.pkl"),
    }
    default_sim = {key: os.path.exists(path) for key, path in default_paths.items()}
    webui_runtime = _detect_webui_runtime_context(request)
    return render_template(
        "2.3.0.field_potential_proxy.html",
        default_sim=default_sim,
        default_sim_paths=default_paths,
        webui_runtime=webui_runtime,
    )


@app.route("/field_potential/proxy/infer_trials", methods=["POST"])
def field_potential_proxy_infer_trials():
    file_key = (request.form.get("file_key") or "").strip()
    default_names = {
        "times_file": "times.pkl",
        "gids_file": "gids.pkl",
        "vm_file": "vm.pkl",
        "ampa_file": "ampa.pkl",
        "gaba_file": "gaba.pkl",
    }
    if file_key not in default_names:
        return jsonify({"error": "Invalid simulation file key for trial detection."}), 400

    source_path = None
    temp_uploaded_path = None
    source_kind = "unknown"
    try:
        local_file = request.files.get("local_file")
        server_path_raw = (request.form.get("server_path") or "").strip()
        use_default = (request.form.get("use_default") or "").strip().lower() in {"1", "true", "yes", "on"}

        if local_file and local_file.filename:
            safe_name = secure_filename(local_file.filename)
            if not safe_name:
                return jsonify({"error": "Invalid uploaded file name."}), 400
            ext = Path(safe_name).suffix.lower()
            if ext not in PICKLE_EXTENSIONS:
                return jsonify({"error": "Trial detection expects a .pkl/.pickle file."}), 400
            uploads_dir = _module_uploads_dir_for("field_potential_proxy")
            os.makedirs(uploads_dir, exist_ok=True)
            temp_name = f"trial_detect_{uuid.uuid4()}_{safe_name}"
            temp_uploaded_path = os.path.join(uploads_dir, temp_name)
            local_file.save(temp_uploaded_path)
            source_path = temp_uploaded_path
            source_kind = "local-upload"
        elif server_path_raw:
            source_path = _validate_existing_pickle_file_path(server_path_raw, "Server simulation file")
            source_kind = "server-path"
        elif use_default:
            default_path = os.path.join(SIMULATION_DATA_DIR, default_names[file_key])
            if not os.path.isfile(default_path):
                return jsonify({"error": f"Default file not found: {default_path}"}), 404
            source_path = default_path
            source_kind = "default-simulation"
        else:
            return jsonify({"error": "No file source provided for trial detection."}), 400

        payload = compute_utils.read_file(source_path)
        trial_count = int(compute_utils._infer_trial_count_from_values(payload))
        trial_count = max(1, trial_count)
        return jsonify({
            "trial_count": trial_count,
            "source_kind": source_kind,
            "file_key": file_key,
        })
    except Exception as exc:
        return jsonify({"error": f"Failed to detect trials: {exc}"}), 400
    finally:
        if temp_uploaded_path and os.path.isfile(temp_uploaded_path):
            try:
                os.remove(temp_uploaded_path)
            except OSError:
                pass


def _is_loopback_identifier(value):
    candidate = (value or "").strip().lower()
    if not candidate:
        return False
    if "," in candidate:
        candidate = candidate.split(",", 1)[0].strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if ":" in candidate and candidate.count(":") == 1 and "." in candidate:
        host_part, _ = candidate.rsplit(":", 1)
        if host_part:
            candidate = host_part.strip().lower()
    return candidate in {"localhost", "::1", "0:0:0:0:0:0:0:1"} or candidate.startswith("127.")


def _detect_webui_runtime_context(req):
    host_header = (req.host or "").strip()
    host_only = host_header.split(":", 1)[0].strip().lower() if host_header else ""
    forwarded_for = (req.headers.get("X-Forwarded-For") or "").strip()
    forwarded_host = (req.headers.get("X-Forwarded-Host") or "").strip()
    remote_addr = (req.remote_addr or "").strip()

    client_addr = forwarded_for.split(",", 1)[0].strip() if forwarded_for else remote_addr
    effective_host = forwarded_host.split(",", 1)[0].strip().split(":", 1)[0].lower() if forwarded_host else host_only

    forced_mode = (os.environ.get("NCPI_WEBUI_RUNTIME_MODE") or "").strip().lower()
    has_display = bool((os.environ.get("DISPLAY") or "").strip())
    if forced_mode in {"server", "remote", "cluster"}:
        is_server_runtime = True
    elif forced_mode in {"local", "desktop"}:
        is_server_runtime = False
    else:
        loopback_request = _is_loopback_identifier(effective_host) and _is_loopback_identifier(client_addr)
        ssh_session = any(os.environ.get(key) for key in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))
        # When Flask runs on a remote machine and browser traffic arrives via SSH tunneling
        # (-L localhost:...:localhost:...), both host and client appear loopback.
        # In that case, treat runtime as server-side. Also default loopback to server-side
        # when no local GUI session is detected, which is typical for remote/headless runs.
        if loopback_request and ssh_session:
            is_server_runtime = True
        elif loopback_request:
            is_server_runtime = not has_display
        else:
            is_server_runtime = True

    path_history = _path_history_snapshot()
    server_real_home_dir = os.path.realpath(os.path.expanduser("~"))
    server_default_browse_dir = _path_history_start_directory(default_value=server_real_home_dir)
    return {
        "is_server_runtime": is_server_runtime,
        "native_picker_available": bool(
            NATIVE_PATH_PICKER_ENABLED and shutil.which("zenity") and os.environ.get("DISPLAY")
        ),
        "default_empirical_source_mode": "server-path" if is_server_runtime else "upload",
        "default_simulation_source_mode": "server-path" if is_server_runtime else "upload",
        "default_analysis_source_mode": "server-path" if is_server_runtime else "upload",
        "default_path_picker_source_mode": "server-path" if is_server_runtime else "local-picker",
        "server_home_dir": server_default_browse_dir,
        "server_real_home_dir": server_real_home_dir,
        "path_history_latest": path_history.get("latest_path", ""),
        "path_history_by_field": path_history.get("by_field", {}),
    }


def _validate_existing_file_path(path_value, label, allowed_extensions=None):
    resolved = os.path.realpath(os.path.expanduser((path_value or "").strip()))
    if not resolved:
        raise ValueError(f"{label} path is required.")
    if not os.path.isfile(resolved):
        raise ValueError(f"{label} file does not exist: {resolved}")
    if allowed_extensions:
        normalized_allowed = set()
        for raw_ext in allowed_extensions:
            ext = str(raw_ext or "").strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            normalized_allowed.add(ext)
        if normalized_allowed:
            ext = Path(resolved).suffix.lower()
            if ext not in normalized_allowed:
                allowed_display = ", ".join(sorted(normalized_allowed))
                raise ValueError(f"{label} must have one of these extensions ({allowed_display}): {resolved}")
    return resolved


def _is_path_within_root(path_value, root_path):
    path_real = os.path.realpath(path_value)
    root_real = os.path.realpath(root_path)
    try:
        return os.path.commonpath([path_real, root_real]) == root_real
    except ValueError:
        return False


def _validate_relative_pickle_path(path_value, root_path, label):
    rel = str(path_value or "").strip()
    if not rel:
        raise ValueError(f"{label} is required.")
    root_real = os.path.realpath(root_path)
    candidate = os.path.realpath(os.path.join(root_real, rel))
    if not _is_path_within_root(candidate, root_real):
        raise ValueError(f"{label} is outside the allowed directory.")
    if not os.path.isfile(candidate):
        raise ValueError(f"{label} does not exist: {candidate}")
    ext = Path(candidate).suffix.lower()
    if ext not in PICKLE_EXTENSIONS:
        raise ValueError(f"{label} must be a .pkl/.pickle file: {candidate}")
    return candidate


def _validate_existing_pickle_file_path(path_value, label):
    return _validate_existing_file_path(path_value, label, allowed_extensions=PICKLE_EXTENSIONS)


def _collect_pickle_files_from_folder(folder_path, recursive=True):
    root = os.path.realpath(os.path.expanduser((folder_path or "").strip()))
    if not root:
        raise ValueError("Server folder path is required.")
    if not os.path.isdir(root):
        raise ValueError(f"Server folder does not exist: {root}")

    found = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for filename in sorted(filenames):
                ext = Path(filename).suffix.lower()
                if ext not in PICKLE_EXTENSIONS:
                    continue
                abs_path = os.path.realpath(os.path.join(dirpath, filename))
                if os.path.isfile(abs_path):
                    found.append(abs_path)
    else:
        with os.scandir(root) as entries:
            for entry in sorted(entries, key=lambda item: item.name.lower()):
                if not entry.is_file(follow_symlinks=False):
                    continue
                ext = Path(entry.name).suffix.lower()
                if ext not in PICKLE_EXTENSIONS:
                    continue
                found.append(os.path.realpath(entry.path))

    if not found:
        raise ValueError(f"No .pkl/.pickle files found in server folder: {root}")
    return found


def _extract_server_file_paths(form, field_name):
    raw_values = []
    try:
        raw_values.extend(form.getlist(field_name))
    except Exception:
        pass
    if not raw_values:
        fallback = form.get(field_name)
        if fallback is not None:
            raw_values.append(fallback)

    paths = []
    seen = set()
    for raw in raw_values:
        text = str(raw or "").replace("\r", "\n")
        for chunk in text.split("\n"):
            candidate = chunk.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            paths.append(candidate)
    return paths


def _infer_inference_asset_role(file_name):
    base_name = os.path.basename(str(file_name or "")).strip().lower()
    if not base_name:
        return None
    stem = Path(base_name).stem.lower()
    normalized_stem = stem.replace("-", "_")
    if normalized_stem.startswith("model") or normalized_stem.startswith("posterior"):
        return "model_file"
    if normalized_stem.startswith("scaler"):
        return "scaler_file"
    if "density_estimator" in normalized_stem or normalized_stem.startswith("density"):
        return "density_estimator_file"
    return None


def _collect_inference_assets_folder_files(folder_path):
    root = os.path.realpath(os.path.expanduser((folder_path or "").strip()))
    if not root:
        raise ValueError("Inference assets server folder path is required.")
    if not os.path.isdir(root):
        raise ValueError(f"Inference assets server folder does not exist: {root}")

    matched = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            key = _infer_inference_asset_role(filename)
            if key and key not in matched:
                abs_path = os.path.realpath(os.path.join(dirpath, filename))
                if os.path.isfile(abs_path):
                    matched[key] = abs_path
    return matched


@app.context_processor
def inject_webui_runtime_context():
    return {"webui_runtime": _detect_webui_runtime_context(request)}


# Features main selection menu page
@app.route("/features")
def features():
    options = [
        {
            "url": url_for('features_load_data'),
            "title": "Load data",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">upload_file</span>',
        },
        {
            "url": url_for('features_methods', entry='compute'),
            "title": "Compute new features",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">add_circle</span>',
        },
    ]
    return render_template("3.features.html", options=options)


# Features configuration page
@app.route("/features/methods", methods=["GET", "POST"])
def features_methods():
    requested_entry = (request.args.get("entry") or "").strip().lower()
    requested_tab = (request.args.get("tab") or "").strip().lower()
    if requested_entry == "load" or requested_tab == "load-precomputed":
        return redirect(url_for("features_load_data"))

    pipeline_files = _collect_feature_pipeline_inputs()
    features_data_files = _list_features_data_files()
    runtime_context = _detect_webui_runtime_context(request)
    return render_template(
        "3.1.features_methods.html",
        pipeline_files=pipeline_files,
        has_pipeline_files=bool(pipeline_files),
        features_data_files=features_data_files,
        has_features_data=bool(features_data_files),
        features_runtime=runtime_context,
    )


@app.route("/features/load_data")
def features_load_data():
    features_data_files = _list_features_data_files()
    return render_template(
        "3.0.load_precomputed_features.html",
        features_data_files=features_data_files,
        has_features_data=bool(features_data_files),
    )


@app.route("/features/browse_dirs", methods=["GET"])
def features_browse_dirs():
    requested = (request.args.get("path") or "").strip()
    requested_history_key = (request.args.get("history_key") or "").strip()
    if requested:
        current = os.path.realpath(os.path.expanduser(requested))
    else:
        current = _path_history_start_directory(
            default_value=os.path.realpath(os.path.expanduser("~")),
            history_key=requested_history_key,
        )

    if not os.path.isdir(current):
        return jsonify({"error": f"Not a directory: {current}"}), 400

    include_files = (request.args.get("include_files") or "").strip().lower() in {"1", "true", "yes", "on"}
    requested_exts = (request.args.get("extensions") or "").strip()
    allowed_exts = set()
    if requested_exts:
        for raw in requested_exts.split(","):
            ext = raw.strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            allowed_exts.add(ext)

    try:
        dirs = []
        files = []
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    dirs.append({
                        "name": entry.name,
                        "path": os.path.realpath(entry.path),
                    })
                    continue
                if include_files and entry.is_file(follow_symlinks=False):
                    ext = Path(entry.name).suffix.lower()
                    if allowed_exts and ext not in allowed_exts:
                        continue
                    files.append({
                        "name": entry.name,
                        "path": os.path.realpath(entry.path),
                    })
    except PermissionError:
        return jsonify({"error": f"Permission denied: {current}"}), 403
    except OSError as exc:
        return jsonify({"error": f"Failed to list directory: {exc}"}), 400

    dirs.sort(key=lambda item: item["name"].lower())
    if include_files:
        files.sort(key=lambda item: item["name"].lower())
    parent = os.path.dirname(current)
    if parent == current:
        parent = ""
    payload = {
        "path": current,
        "parent": parent,
        "dirs": dirs[:1000],
    }
    if include_files:
        payload["files"] = files[:1000]
    if requested_history_key:
        _remember_path_history_for_key(requested_history_key, current)
    _remember_path_history("features_browse_dirs:path", current)
    return jsonify(payload)


@app.route("/inference/inspect_assets_folder", methods=["POST"])
def inference_inspect_assets_folder():
    _remember_path_history_from_form(request.form, "inference_inspect_assets_folder")
    folder_path = (request.form.get("folder_path") or "").strip()
    root = os.path.realpath(os.path.expanduser(folder_path))
    try:
        matched = _collect_inference_assets_folder_files(folder_path)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    files_preview = []
    files_total = 0
    preview_limit = 120
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            abs_path = os.path.realpath(os.path.join(dirpath, filename))
            if not os.path.isfile(abs_path):
                continue
            files_total += 1
            if len(files_preview) < preview_limit:
                files_preview.append(os.path.relpath(abs_path, root))

    return jsonify({
        "ok": True,
        "folder_path": root,
        "assets": {
            "model_file": matched.get("model_file") or "",
            "scaler_file": matched.get("scaler_file") or "",
            "density_estimator_file": matched.get("density_estimator_file") or "",
        },
        "files_preview": files_preview,
        "files_total": files_total,
        "files_more": max(0, files_total - len(files_preview)),
    })


@app.route("/features/select_folder", methods=["POST"])
def features_select_folder():
    _remember_path_history_from_form(request.form, "features_select_folder")
    # Opens a native path picker on the machine running the Flask server.
    # This is intended for local desktop usage.
    mode = (request.form.get("mode") or "folder").strip().lower()
    allow_multiple = (request.form.get("allow_multiple") or "").strip().lower() in {"1", "true", "yes", "on"}
    requested_exts = (request.form.get("extensions") or "").strip()
    requested_start_path = (request.form.get("start_path") or "").strip()
    requested_history_key = (request.form.get("history_key") or "").strip()
    allowed_exts = set()
    if requested_exts:
        for raw in requested_exts.split(","):
            ext = raw.strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            allowed_exts.add(ext)

    start_path = ""
    if requested_start_path:
        candidate = os.path.realpath(os.path.expanduser(requested_start_path))
        if os.path.isdir(candidate):
            start_path = candidate
    if not start_path:
        start_path = _path_history_start_directory(default_value="", history_key=requested_history_key)

    def _normalize_and_validate_paths(picked_items):
        if not picked_items:
            return None, jsonify({"error": "No path selected."}), 400
        if mode == "folder":
            picked = os.path.realpath(os.path.expanduser(str(picked_items[0]).strip()))
            if not os.path.isdir(picked):
                return None, jsonify({"error": f"Selected path is not a directory: {picked}"}), 400
            if requested_history_key:
                _remember_path_history_for_key(requested_history_key, picked)
            _remember_path_history("features_select_folder:selected_folder", picked)
            return {"path": picked}, None, None

        normalized_paths = []
        seen = set()
        for raw_path in picked_items:
            picked = os.path.realpath(os.path.expanduser(str(raw_path).strip()))
            if not picked or picked in seen:
                continue
            seen.add(picked)
            if not os.path.isfile(picked):
                return None, jsonify({"error": f"Selected path is not a file: {picked}"}), 400
            ext = Path(picked).suffix.lower()
            if allowed_exts:
                if ext not in allowed_exts:
                    return None, jsonify({"error": f"Unsupported selected file type: {ext}"}), 400
            elif ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                return None, jsonify({"error": f"Unsupported selected file type: {ext}"}), 400
            normalized_paths.append(picked)
        if not normalized_paths:
            return None, jsonify({"error": "No valid files selected."}), 400
        for idx, normalized_path in enumerate(normalized_paths):
            if requested_history_key:
                _remember_path_history_for_key(requested_history_key, normalized_path)
            _remember_path_history(f"features_select_folder:selected_file[{idx}]", normalized_path)
        return {"path": normalized_paths[0], "paths": normalized_paths}, None, None

    if not NATIVE_PATH_PICKER_ENABLED:
        return jsonify({
            "error": (
                "Native file/folder picker is unavailable in this runtime. "
                "Use Server folders or type the path manually."
            ),
            "details": "Native picker disabled by configuration (NCPI_ENABLE_NATIVE_PATH_PICKER=0).",
        }), 400

    picker_errors = []

    if shutil.which("zenity"):
        try:
            zenity_title = "NCPI - Select data folder" if mode == "folder" else "NCPI - Select data file"
            cmd = ["zenity", "--file-selection", "--modal", f"--title={zenity_title}"]

            # Best effort: attach the picker to the currently active X11 window so it is raised.
            if shutil.which("xprop"):
                try:
                    active_proc = subprocess.run(
                        ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    active_out = (active_proc.stdout or "").strip().lower()
                    marker = "window id # "
                    if marker in active_out:
                        active_id = active_out.split(marker, 1)[1].strip().split()[0]
                        if active_id and active_id != "0x0":
                            cmd.append(f"--attach={active_id}")
                except Exception:
                    pass

            if start_path:
                zenity_start = start_path
                if not zenity_start.endswith(os.sep):
                    zenity_start = f"{zenity_start}{os.sep}"
                cmd.append(f"--filename={zenity_start}")
            if mode == "folder":
                cmd.append("--directory")
            else:
                if allow_multiple:
                    cmd.extend(["--multiple", "--separator=\n"])

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )

            focus_stop = threading.Event()
            focus_thread = None

            def _focus_zenity_window():
                # Some window managers only put Zenity in the taskbar/sidebar;
                # repeatedly request activation during the initial popup period.
                deadline = time.time() + 15.0
                has_xdotool = bool(shutil.which("xdotool"))
                has_wmctrl = bool(shutil.which("wmctrl"))
                while not focus_stop.is_set() and proc.poll() is None and time.time() < deadline:
                    focused = False
                    if has_xdotool:
                        try:
                            search = subprocess.run(
                                ["xdotool", "search", "--name", zenity_title],
                                capture_output=True,
                                text=True,
                                timeout=1,
                            )
                            if search.returncode == 0:
                                window_ids = [line.strip() for line in (search.stdout or "").splitlines() if line.strip()]
                                if window_ids:
                                    window_id = window_ids[-1]
                                    subprocess.run(
                                        ["xdotool", "windowactivate", "--sync", window_id],
                                        capture_output=True,
                                        text=True,
                                        timeout=1,
                                    )
                                    subprocess.run(
                                        ["xdotool", "windowraise", window_id],
                                        capture_output=True,
                                        text=True,
                                        timeout=1,
                                    )
                                    focused = True
                        except Exception:
                            pass
                    if not focused and has_wmctrl:
                        try:
                            subprocess.run(
                                ["wmctrl", "-a", zenity_title],
                                capture_output=True,
                                text=True,
                                timeout=1,
                            )
                        except Exception:
                            pass
                    if focused:
                        break
                    time.sleep(0.25)

            if shutil.which("xdotool") or shutil.which("wmctrl"):
                focus_thread = threading.Thread(target=_focus_zenity_window, daemon=True)
                focus_thread.start()

            try:
                stdout_text, stderr_text = proc.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.communicate(timeout=2)
                except Exception:
                    pass
                raise
            finally:
                focus_stop.set()
                if focus_thread is not None and focus_thread.is_alive():
                    focus_thread.join(timeout=1)

            if proc.returncode == 0:
                raw_output = (stdout_text or "").replace("\r", "\n")
                picked_items = [chunk.strip() for chunk in raw_output.split("\n") if chunk.strip()]
                payload, error_response, error_code = _normalize_and_validate_paths(picked_items)
                if error_response is not None:
                    return error_response, error_code
                return jsonify(payload)
            stderr_text = (stderr_text or "").strip()
            if proc.returncode == 1 and not stderr_text:
                return jsonify({"error": "Path selection cancelled."}), 400
            picker_errors.append(f"zenity failed (code {proc.returncode}): {stderr_text or 'no error details'}")
        except subprocess.TimeoutExpired:
            picker_errors.append("zenity timed out while waiting for a selection.")
        except Exception as exc:
            picker_errors.append(f"zenity error: {exc}")
    else:
        picker_errors.append("zenity not found")

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.update()

            dialog_kwargs = {}
            if start_path:
                dialog_kwargs["initialdir"] = start_path

            if mode == "folder":
                selected = filedialog.askdirectory(title="Select data folder", **dialog_kwargs)
                picked_items = [selected] if selected else []
            else:
                filetypes = []
                if allowed_exts:
                    patterns = " ".join(f"*{ext}" for ext in sorted(allowed_exts))
                    filetypes.append(("Allowed files", patterns))
                else:
                    patterns = " ".join(f"*{ext}" for ext in sorted(FEATURES_PARSER_FILE_EXTENSIONS))
                    filetypes.append(("Supported files", patterns))
                filetypes.append(("All files", "*.*"))
                if allow_multiple:
                    selected = filedialog.askopenfilenames(title="Select data file(s)", filetypes=filetypes, **dialog_kwargs)
                    picked_items = list(selected) if selected else []
                else:
                    selected = filedialog.askopenfilename(title="Select data file", filetypes=filetypes, **dialog_kwargs)
                    picked_items = [selected] if selected else []
        finally:
            if root is not None:
                root.destroy()

        if not picked_items:
            return jsonify({"error": "Path selection cancelled."}), 400
        payload, error_response, error_code = _normalize_and_validate_paths(picked_items)
        if error_response is not None:
            return error_response, error_code
        return jsonify(payload)
    except Exception as exc:
        picker_errors.append(f"tkinter failed: {exc}")

    details = "; ".join(picker_errors)
    error_message = "Native file/folder picker is unavailable in this runtime. Use Server folders or type the path manually."
    if details:
        error_message = f"{error_message} Details: {details}"
    return jsonify({"error": error_message, "details": details}), 400


@app.route("/path_history/remember", methods=["POST"])
def path_history_remember():
    history_key = (request.form.get("history_key") or "").strip()
    path_value = (request.form.get("path") or "").strip()
    if not history_key:
        return jsonify({"ok": False, "error": "Missing history key."}), 400
    if not path_value:
        return jsonify({"ok": False, "error": "Missing path value."}), 400
    remembered = _remember_path_history_for_key(history_key, path_value)
    if not remembered:
        return jsonify({"ok": False, "error": "Invalid path value for history."}), 400
    return jsonify({"ok": True, "path": remembered})


@app.route("/field_potential/kernel/stage_local_folder", methods=["POST"])
def stage_kernel_local_folder():
    uploads = [item for item in request.files.getlist("folder_files") if item and item.filename]
    if not uploads:
        return jsonify({"error": "No local folder files were uploaded."}), 400

    target_field = (request.form.get("target_field") or "").strip().lower()
    if target_field not in {"mc_folder", "output_sim_path"}:
        target_field = "folder"

    base_root = FIELD_POTENTIAL_KERNEL_LOCAL_UPLOADS_DIR
    session_id = uuid.uuid4().hex
    staging_root = os.path.realpath(os.path.join(base_root, f"{target_field}_{session_id}"))
    os.makedirs(staging_root, exist_ok=True)

    top_level_name = ""
    saved_count = 0

    for upload in uploads:
        raw_name = str(upload.filename or "").strip().replace("\\", "/")
        parts = [part for part in raw_name.split("/") if part not in {"", "."}]
        if not parts:
            continue
        if any(part == ".." for part in parts):
            return jsonify({"error": f"Invalid relative path in upload: {raw_name}"}), 400

        if not top_level_name and parts:
            top_level_name = parts[0]

        relative_path = os.path.join(*parts)
        destination = os.path.realpath(os.path.join(staging_root, relative_path))
        if not _is_path_within_root(destination, staging_root):
            return jsonify({"error": f"Invalid destination path for upload: {raw_name}"}), 400
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        upload.save(destination)
        saved_count += 1

    if saved_count == 0:
        return jsonify({"error": "No valid files were uploaded from the selected folder."}), 400

    staged_path = staging_root
    if top_level_name:
        candidate = os.path.realpath(os.path.join(staging_root, top_level_name))
        if _is_path_within_root(candidate, staging_root) and os.path.isdir(candidate):
            staged_path = candidate

    return jsonify({
        "path": staged_path,
        "files_saved": saved_count,
    })


@app.route("/features/parser/inspect", methods=["POST"])
def features_parser_inspect():
    _remember_path_history_from_form(request.form, "features_parser_inspect")
    existing_data_path = (request.form.get("existing_data_path") or "").strip()
    empirical_folder_paths = _extract_folder_paths_from_form(
        request.form,
        singular_key="empirical_folder_path",
        plural_key="empirical_folder_paths",
    )
    simulation_file_path = (request.form.get("simulation_file_path") or "").strip()
    simulation_folder_paths = _extract_folder_paths_from_form(
        request.form,
        singular_key="simulation_folder_path",
        plural_key="simulation_folder_paths",
    )
    simulation_data_file_selections = _extract_data_file_selection_map_from_form(
        request.form,
        "simulation_data_file_selections",
    )
    empirical_data_file_selections = _extract_data_file_selection_map_from_form(
        request.form,
        "empirical_data_file_selections",
    )
    listed_file_names = [str(item).strip() for item in request.form.getlist("listed_file_names") if str(item).strip()]
    uploads = [f for f in request.files.getlist("file") if f and f.filename]
    _metadata_server_paths_raw = [p.strip() for p in request.form.getlist("metadata_server_path") if p.strip()]
    inspect_path = None
    cleanup_paths = []
    name_candidates = list(listed_file_names)
    file_name = ""
    folder_entries = []
    folder_summaries = []
    extension_summaries = []
    extension_profiles = []
    folder_profiles = []
    folder_contexts = []
    candidate_field_folders = {}
    folder_file_count = 0

    try:
        if listed_file_names:
            _validate_listed_file_names_by_folder_extension(listed_file_names, label="Selected")

        if existing_data_path:
            inspect_path = _validate_feature_existing_path(existing_data_path)
            file_name = os.path.basename(inspect_path)
            if not name_candidates:
                name_candidates.append(file_name)
        elif empirical_folder_paths:
            folder_entries, folder_summaries, extension_counts, folder_contexts = _collect_supported_folder_file_entries(
                empirical_folder_paths,
                "Empirical",
                data_file_selection_map=empirical_data_file_selections,
            )
            use_prefix = len(folder_summaries) > 1
            for row in folder_entries:
                row["logical_name"] = _prefixed_file_name(row["name"], row["folder_name"], apply_prefix=use_prefix)
            inspect_entry = next(
                (
                    context.get("sample_selected_entry")
                    for context in folder_contexts
                    if isinstance(context.get("sample_selected_entry"), dict)
                ),
                folder_entries[0] if folder_entries else None,
            )
            if inspect_entry is None:
                raise ValueError("No supported empirical files were found for inspection.")
            inspect_path = inspect_entry["path"]
            file_name = inspect_entry.get("logical_name") or inspect_entry["name"]
            folder_file_count = len(folder_entries)
            (
                folder_profiles,
                extension_summaries,
                _combined_candidates_from_profiles,
                _candidate_field_folders_from_profiles,
                combined_logical_names,
            ) = _build_folder_inspection_profiles(folder_entries, folder_summaries)
            if not name_candidates:
                name_candidates = combined_logical_names
            extension_profiles = [
                profile
                for folder_profile in folder_profiles
                for profile in (folder_profile.get("extension_profiles") or [])
            ]
        elif simulation_file_path:
            inspect_path = _validate_simulation_file_path(simulation_file_path)
            file_name = os.path.basename(inspect_path)
            if not name_candidates:
                name_candidates.append(file_name)
        elif simulation_folder_paths:
            folder_entries, folder_summaries, extension_counts, folder_contexts = _collect_supported_folder_file_entries(
                simulation_folder_paths,
                "Simulation outputs",
                data_file_selection_map=simulation_data_file_selections,
            )
            use_prefix = len(folder_summaries) > 1
            for row in folder_entries:
                row["logical_name"] = _prefixed_file_name(row["name"], row["folder_name"], apply_prefix=use_prefix)
            inspect_entry = next(
                (
                    context.get("sample_selected_entry")
                    for context in folder_contexts
                    if isinstance(context.get("sample_selected_entry"), dict)
                ),
                folder_entries[0] if folder_entries else None,
            )
            if inspect_entry is None:
                raise ValueError("No supported simulation files were found for inspection.")
            inspect_path = inspect_entry["path"]
            file_name = inspect_entry.get("logical_name") or inspect_entry["name"]
            folder_file_count = len(folder_entries)
            (
                folder_profiles,
                extension_summaries,
                _combined_candidates_from_profiles,
                _candidate_field_folders_from_profiles,
                combined_logical_names,
            ) = _build_folder_inspection_profiles(folder_entries, folder_summaries)
            if not name_candidates:
                name_candidates = combined_logical_names
            extension_profiles = [
                profile
                for folder_profile in folder_profiles
                for profile in (folder_profile.get("extension_profiles") or [])
            ]
        elif uploads or _metadata_server_paths_raw:
            inspect_root = FEATURES_INSPECTION_DIR
            os.makedirs(inspect_root, exist_ok=True)
            metadata_server_paths = _metadata_server_paths_raw
            upload_entries = []
            for upl in uploads:
                upl_name = secure_filename(upl.filename)
                if not upl_name:
                    return jsonify({"error": "Invalid file name."}), 400
                ext = Path(upl_name).suffix.lower()
                if ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                    return jsonify({"error": f"Unsupported file type for inspection: {ext}"}), 400
                temp_name = f"{uuid.uuid4()}_{upl_name}"
                temp_path = os.path.join(inspect_root, temp_name)
                upl.save(temp_path)
                cleanup_paths.append(temp_path)
                name_candidates.append(upl_name)
                upload_entries.append({"path": temp_path, "name": upl_name, "folder_name": upl_name})
            for srv_path in metadata_server_paths:
                real_path = os.path.realpath(srv_path)
                if not os.path.isfile(real_path):
                    return jsonify({"error": f"Server file not found: {srv_path}"}), 400
                srv_name = os.path.basename(real_path)
                ext = Path(srv_name).suffix.lower()
                if ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                    return jsonify({"error": f"Unsupported file type for inspection: {ext}"}), 400
                name_candidates.append(srv_name)
                upload_entries.append({"path": real_path, "name": srv_name, "folder_name": srv_name})
            if not upload_entries:
                return jsonify({"error": "No valid files provided for inspection."}), 400
            inspect_path = upload_entries[0]["path"]
            file_name = upload_entries[0]["name"]
            if len(upload_entries) > 1:
                folder_entries = upload_entries
                folder_file_count = len(upload_entries)
                file_name = f"{len(upload_entries)} files"

        else:
            return jsonify({"error": "Provide an existing pipeline file, a simulation folder path, an empirical folder path, or upload a file to inspect."}), 400

        if folder_entries:
            selected_description = _describe_parser_source(inspect_path)
            selected_description["manual_field_details"] = _manual_field_details_for_ui()
            selected_description["dataframe_candidate_fields"] = _collect_dataframe_candidate_fields_from_description(selected_description)

            selected_data_candidates = [
                str(item or "").strip()
                for item in (selected_description.get("candidate_fields") or [])
                if str(item or "").strip()
            ]
            selected_data_field_details = [
                item for item in (selected_description.get("field_details") or [])
                if isinstance(item, dict)
            ]
            selected_data_dataframe_fields = _collect_dataframe_candidate_fields_from_description(selected_description)

            metadata_summary = _aggregate_candidate_metadata_from_file_entries(folder_entries)
            combined_candidates = metadata_summary.get("candidate_fields") or list(selected_data_candidates)
            combined_field_details = metadata_summary.get("field_details") or list(selected_data_field_details)
            candidate_field_folders = metadata_summary.get("candidate_field_folders") or {}
            candidate_field_origins = metadata_summary.get("candidate_field_origins") or {}
            dataframe_candidate_fields = metadata_summary.get("dataframe_candidate_fields") or []

            combined_defaults = selected_description.get("defaults") or {
                "data": _pick_field_guess_fuzzy(selected_data_candidates, ["data", "signal", "lfp", "eeg", "meg", "ecog", "cdm"]),
                "fs": _pick_field_guess_fuzzy(
                    selected_data_candidates,
                    ["fs", "sfreq", "freq", "frequency", "sampling_rate", "sampling_frequency", "sampling_frequency_hz"],
                ),
                "ch_names": _pick_field_guess_fuzzy(
                    selected_data_candidates,
                    ["ch_names", "channels", "channel_names", "channel", "sensors", "sensor", "ch"],
                ),
                "recording_type": _pick_field_guess(selected_data_candidates, ["recording_type", "modality", "recording", "type"]),
                "subject_id": _pick_field_guess(selected_data_candidates, ["subject_id", "subject", "subj", "participant"]),
                "group": _pick_field_guess(selected_data_candidates, ["group", "cohort", "class"]),
                "species": _pick_field_guess(selected_data_candidates, ["species", "animal", "organism"]),
                "condition": _pick_field_guess(selected_data_candidates, ["condition", "state", "task"]),
            }

            context_by_path = {
                str(item.get("folder_path") or "").strip(): item
                for item in (folder_contexts or [])
                if isinstance(item, dict) and str(item.get("folder_path") or "").strip()
            }
            folder_structure_profiles = []
            for context in folder_contexts or []:
                folder_path = str(context.get("folder_path") or "").strip()
                if not folder_path:
                    continue
                folder_structure_profiles.append({
                    "folder_path": folder_path,
                    "folder_name": str(context.get("folder_name") or folder_path),
                    "has_nested_subfolders": bool(context.get("has_nested_subfolders")),
                    "sample_subfolder_name": str(context.get("sample_subfolder_name") or ""),
                    "sample_subfolder_path": str(context.get("sample_subfolder_path") or ""),
                    "selected_data_file": str(context.get("selected_data_file") or ""),
                    "selected_data_pattern": str(context.get("selected_data_pattern") or ""),
                    "selected_data_extension": str(context.get("selected_data_extension") or ""),
                    "data_file_candidates": list(context.get("data_file_candidates") or []),
                    "matched_data_files": list(context.get("matched_data_files") or []),
                    "selected_data_file_count": int(len(context.get("matched_data_files") or [])),
                    "all_subfolders": list(context.get("all_subfolders") or []),
                    "structure_warnings": list(context.get("structure_warnings") or []),
                })

            for folder_profile in folder_profiles:
                folder_path = str(folder_profile.get("folder_path") or "").strip()
                context = context_by_path.get(folder_path)
                if not context:
                    continue
                folder_profile["has_nested_subfolders"] = bool(context.get("has_nested_subfolders"))
                folder_profile["sample_subfolder_name"] = str(context.get("sample_subfolder_name") or "")
                folder_profile["sample_subfolder_path"] = str(context.get("sample_subfolder_path") or "")
                folder_profile["selected_data_file"] = str(context.get("selected_data_file") or "")
                folder_profile["selected_data_pattern"] = str(context.get("selected_data_pattern") or "")
                folder_profile["selected_data_extension"] = str(context.get("selected_data_extension") or "")
                folder_profile["data_file_candidates"] = list(context.get("data_file_candidates") or [])
                folder_profile["matched_data_files"] = list(context.get("matched_data_files") or [])
                folder_profile["all_subfolders"] = list(context.get("all_subfolders") or [])
                folder_profile["structure_warnings"] = list(context.get("structure_warnings") or [])
                if not context.get("has_nested_subfolders"):
                    continue

                sample_entry = context.get("sample_selected_entry")
                if isinstance(sample_entry, dict) and sample_entry.get("path"):
                    sample_path = str(sample_entry.get("path") or "")
                    sample_names = [
                        str(item.get("name") or "").strip()
                        for item in (context.get("matched_data_files") or [])
                        if str(item.get("name") or "").strip()
                    ]
                    sample_profile = _describe_parser_source(sample_path)
                    sample_profile = _attach_file_extracted_virtual_field(sample_profile, sample_names or [str(sample_entry.get("name") or "")])
                    sample_profile["virtual_field_details"] = _build_virtual_field_details_for_ui(sample_names or [str(sample_entry.get("name") or "")])
                    sample_profile["manual_field_details"] = _manual_field_details_for_ui()
                    sample_profile["extension"] = str(sample_entry.get("extension") or "")
                    sample_profile["sample_file"] = str(sample_entry.get("subfolder_relative_path") or sample_entry.get("relative_path") or sample_entry.get("name") or "")
                    sample_profile["sample_file_path"] = sample_path
                    sample_profile["file_count"] = int(len(context.get("matched_data_files") or []))
                    sample_profile["folder_name"] = str(folder_profile.get("folder_name") or "")
                    sample_profile["folder_path"] = folder_path
                    folder_profile["extension_profiles"] = [sample_profile]

            fs_hint_hz = selected_description.get("fs_hint_hz")
            fs_hint_note = selected_description.get("fs_hint_note")

            description = {
                "source_type": "multi_folder",
                "candidate_fields": combined_candidates,
                "defaults": combined_defaults,
                "summary": (
                    f"Selected {len(folder_summaries)} folder(s), {folder_file_count} file(s). "
                    f"Metadata inspected in {int(metadata_summary.get('inspected_file_count') or 0)} file(s)."
                ),
                "field_details": combined_field_details,
                "manual_field_details": _manual_field_details_for_ui(),
                "virtual_field_details": _build_virtual_field_details_for_ui(name_candidates),
                "extension_profiles": extension_profiles,
                "folder_profiles": folder_profiles,
                "folder_summaries": folder_summaries,
                "extension_summaries": extension_summaries,
                "folder_structure_profiles": folder_structure_profiles,
                "data_file_selection_map": {
                    str(item.get("folder_path") or ""): str(item.get("selected_data_file") or "")
                    for item in folder_structure_profiles
                    if str(item.get("selected_data_file") or "").strip()
                },
                "candidate_field_folders": candidate_field_folders,
                "candidate_field_origins": candidate_field_origins,
                "dataframe_candidate_fields": dataframe_candidate_fields,
                "data_file_candidate_fields": selected_data_candidates,
                "data_file_field_details": selected_data_field_details,
                "data_file_dataframe_candidate_fields": selected_data_dataframe_fields,
                "data_file_source_type": str(selected_description.get("source_type") or ""),
                "folder_file_count": folder_file_count,
                "fs_hint_hz": fs_hint_hz,
                "fs_hint_note": fs_hint_note,
            }
            if candidate_field_folders and len(folder_summaries) > 1:
                field_labels = {}
                for field_name in combined_candidates:
                    folder_names = [str(name).strip() for name in (candidate_field_folders.get(field_name) or []) if str(name).strip()]
                    if not folder_names:
                        continue
                    suffix = ", ".join(folder_names[:4])
                    if len(folder_names) > 4:
                        suffix += ", ..."
                    field_labels[field_name] = f"{field_name} [folders: {suffix}]"
                if field_labels:
                    description["virtual_field_labels"] = field_labels
            
            # Override summary for the multi-upload metadata case
            if not folder_summaries and folder_entries:
                description["summary"] = (
                    f"{len(folder_entries)} file(s) inspected. "
                    f"Combined fields detected: {len(combined_candidates)}."
                )
                description["source_type"] = "uploaded_files"

        else:
            description = _describe_parser_source(inspect_path)
            description["manual_field_details"] = _manual_field_details_for_ui()
            description["dataframe_candidate_fields"] = _collect_dataframe_candidate_fields_from_description(description)
            description["data_file_candidate_fields"] = [
                str(item or "").strip()
                for item in (description.get("candidate_fields") or [])
                if str(item or "").strip()
            ]
            description["data_file_field_details"] = [
                item for item in (description.get("field_details") or [])
                if isinstance(item, dict)
            ]
            description["data_file_dataframe_candidate_fields"] = list(description.get("dataframe_candidate_fields") or [])
            description["data_file_source_type"] = str(description.get("source_type") or "")
            if "candidate_field_origins" not in description:
                candidate_field_origins = defaultdict(set)
                for item in description.get("field_details") or []:
                    field_name = str(item.get("field") or "").strip()
                    origin_name = str(item.get("origin") or "").strip().lower()
                    if field_name and origin_name:
                        candidate_field_origins[field_name].add(origin_name)
                description["candidate_field_origins"] = {
                    key: sorted(values)
                    for key, values in candidate_field_origins.items()
                }
            
            # Build extension_profiles so the response renders in "Folder and File Inspection"
            _ext_key = Path(str(file_name)).suffix.lower()
            _ext_profile = {
                "extension": _ext_key,
                "source_type": description.get("source_type", ""),
                "candidate_fields": list(description.get("candidate_fields") or []),
                "field_details": [d for d in (description.get("field_details") or []) if isinstance(d, dict)],
                "summary": description.get("summary", ""),
                "sample_file": file_name,
                "file_count": 1,
                "folder_name": file_name,
                "folder_path": "",
                "virtual_field_details": list(description.get("virtual_field_details") or []),
            }
            description["extension_profiles"] = [_ext_profile]
            description["extension_summaries"] = [{"extension": _ext_key, "count": 1}]
            if not description.get("folder_summaries"):
                description["folder_summaries"] = [{"folder_name": file_name, "folder_path": "", "file_count": 1}]

            
        description = _attach_file_extracted_virtual_field(description, name_candidates)
        if "dataframe_candidate_fields" not in description:
            description["dataframe_candidate_fields"] = _collect_dataframe_candidate_fields_from_description(description)
        if "data_file_candidate_fields" not in description:
            description["data_file_candidate_fields"] = [
                str(item or "").strip()
                for item in (description.get("candidate_fields") or [])
                if str(item or "").strip()
            ]
        if "data_file_field_details" not in description:
            description["data_file_field_details"] = [
                item for item in (description.get("field_details") or [])
                if isinstance(item, dict)
            ]
        if "data_file_dataframe_candidate_fields" not in description:
            description["data_file_dataframe_candidate_fields"] = _collect_dataframe_candidate_fields_from_description(description)
        if "data_file_source_type" not in description:
            description["data_file_source_type"] = str(description.get("source_type") or "")
        if "virtual_field_details" not in description:
            description["virtual_field_details"] = _build_virtual_field_details_for_ui(name_candidates)
        description["source_name"] = file_name
        if folder_summaries:
            description["folder_path"] = folder_summaries[0]["folder_path"]
        elif empirical_folder_paths or simulation_folder_paths:
            selected_folder = (empirical_folder_paths or simulation_folder_paths)[0]
            description["folder_path"] = os.path.realpath(selected_folder)
        if simulation_file_path:
            description["selected_file_path"] = os.path.realpath(simulation_file_path)

        listed_summary = _summarize_listed_file_names(listed_file_names)
        if listed_summary["total_files"] > 0 and not description.get("folder_summaries"):
            description["folder_summaries"] = listed_summary["folder_summaries"]
            description["extension_summaries"] = listed_summary["extension_summaries"]
            description["folder_file_count"] = listed_summary["total_files"]
            if not description.get("summary"):
                description["summary"] = f"Selected {listed_summary['total_files']} file(s)."
        return jsonify(description)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        for _cpath in cleanup_paths:
            if _cpath and os.path.exists(_cpath):
                try:
                    os.remove(_cpath)
                except OSError:
                    pass


@app.route("/features/load_precomputed", methods=["POST"])
def features_load_precomputed():
    def _redirect_to_features_load_tab():
        return redirect(url_for("features_load_data"))

    _remember_path_history_from_form(request.form, "features_load_precomputed")
    source_mode = (request.form.get("precomputed_features_source_mode") or "upload").strip().lower()
    uploads = [file for file in request.files.getlist("precomputed_features_file") if file and file.filename]
    server_file_paths = _extract_server_file_paths(request.form, "precomputed_features_server_file_path")

    inputs = []
    if source_mode == "server-path":
        if not server_file_paths:
            flash("Select at least one server features file (.pkl/.pickle).", "error")
            return _redirect_to_features_load_tab()
        for server_path in server_file_paths:
            try:
                source_path = _validate_existing_pickle_file_path(server_path, "Features server file")
                safe_name = secure_filename(os.path.basename(source_path))
            except Exception as exc:
                flash(str(exc), "error")
                return _redirect_to_features_load_tab()
            if not safe_name:
                continue
            inputs.append({"safe_name": safe_name, "source_path": source_path, "upload": None})
    else:
        if len(uploads) == 0:
            flash("Upload at least one precomputed features dataframe (.pkl/.pickle).", "error")
            return _redirect_to_features_load_tab()
        for upload in uploads:
            safe_name = secure_filename(upload.filename)
            if not safe_name:
                continue
            ext = Path(safe_name).suffix.lower()
            if ext not in PICKLE_EXTENSIONS:
                flash("Precomputed features must be a pickle file (.pkl/.pickle).", "error")
                return _redirect_to_features_load_tab()
            inputs.append({"safe_name": safe_name, "source_path": None, "upload": upload})

    if not inputs:
        flash("No valid precomputed features files were provided.", "error")
        return _redirect_to_features_load_tab()

    features_dir = _features_data_dir(create=True)
    loaded_count = 0
    errors = []
    for item in inputs:
        safe_name = item["safe_name"]
        temp_path = os.path.join(features_dir, f"tmp_{uuid.uuid4().hex}_{safe_name}")
        try:
            if source_mode == "server-path":
                shutil.copy2(item["source_path"], temp_path)
            else:
                item["upload"].save(temp_path)
            df = compute_utils.read_df_file(temp_path)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Uploaded object is not a pandas dataframe.")
            if "Features" not in df.columns:
                raise ValueError("Uploaded dataframe does not contain a 'Features' column.")
            output_name = safe_name
            output_path = os.path.join(features_dir, output_name)
            df.to_pickle(output_path)
            loaded_count += 1
        except Exception as exc:
            errors.append(f"{safe_name}: {exc}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    if loaded_count == 0:
        detail = errors[0] if errors else "Unknown error."
        flash(f"Failed to load precomputed features: {detail}", "error")
        return _redirect_to_features_load_tab()

    if errors:
        shown = "; ".join(errors[:3])
        if len(errors) > 3:
            shown += f"; ... ({len(errors) - 3} more)"
        flash(f"Loaded {loaded_count} features file(s), but some failed: {shown}", "error")

    return _redirect_to_features_load_tab()


@app.route("/features/remove_file", methods=["POST"])
def remove_features_file():
    filename = (request.form.get("filename") or "").strip()
    features_root = _features_data_dir(create=False)
    try:
        target_path = _validate_relative_pickle_path(filename, features_root, "Features file")
        os.remove(target_path)
    except Exception as exc:
        flash(str(exc), "error")
    return redirect(request.referrer or url_for("features_load_data"))

# Inference configuration page
@app.route("/inference")
def inference():
    options = [
        {
            "url": url_for('inference_load_data'),
            "title": "Load data",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">upload_file</span>',
        },
        {
            "url": url_for('new_training'),
            "title": "New training",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">add_circle</span>',
        },
        {
            "url": url_for('compute_predictions'),
            "title": "Compute predictions",
            "icon": '<span class="material-symbols-outlined text-4xl text-slate-600 dark:text-slate-300 group-hover:text-primary">query_stats</span>',
        },
    ]
    return render_template("4.inference.html", options=options)

# Load precomputed predictions for inference module
@app.route("/inference/load_data")
def inference_load_data():
    predictions_data_files = _list_predictions_data_files()
    return render_template(
        "4.1.0.load_precomputed_predictions.html",
        predictions_data_files=predictions_data_files,
        has_predictions_data=bool(predictions_data_files),
    )


@app.route("/inference/load_precomputed", methods=["POST"])
def inference_load_precomputed():
    def _redirect_to_inference_load_tab():
        return redirect(url_for("inference_load_data"))

    _remember_path_history_from_form(request.form, "inference_load_precomputed")
    source_mode = (request.form.get("precomputed_predictions_source_mode") or "upload").strip().lower()
    uploads = [file for file in request.files.getlist("precomputed_predictions_file") if file and file.filename]
    server_file_paths = _extract_server_file_paths(request.form, "precomputed_predictions_server_file_path")

    inputs = []
    if source_mode == "server-path":
        if not server_file_paths:
            flash("Select at least one server predictions file (.pkl/.pickle).", "error")
            return _redirect_to_inference_load_tab()
        for server_path in server_file_paths:
            try:
                source_path = _validate_existing_pickle_file_path(server_path, "Predictions server file")
                safe_name = secure_filename(os.path.basename(source_path))
            except Exception as exc:
                flash(str(exc), "error")
                return _redirect_to_inference_load_tab()
            if not safe_name:
                continue
            inputs.append({"safe_name": safe_name, "source_path": source_path, "upload": None})
    else:
        if len(uploads) == 0:
            flash("Upload at least one precomputed predictions dataframe (.pkl/.pickle).", "error")
            return _redirect_to_inference_load_tab()
        for upload in uploads:
            safe_name = secure_filename(upload.filename)
            if not safe_name:
                continue
            ext = Path(safe_name).suffix.lower()
            if ext not in PICKLE_EXTENSIONS:
                flash("Precomputed predictions must be a pickle file (.pkl/.pickle).", "error")
                return _redirect_to_inference_load_tab()
            inputs.append({"safe_name": safe_name, "source_path": None, "upload": upload})

    if not inputs:
        flash("No valid precomputed predictions files were provided.", "error")
        return _redirect_to_inference_load_tab()

    predictions_dir = _predictions_data_dir(create=True)
    loaded_count = 0
    errors = []
    for item in inputs:
        safe_name = item["safe_name"]
        temp_path = os.path.join(predictions_dir, f"tmp_{uuid.uuid4().hex}_{safe_name}")
        try:
            if source_mode == "server-path":
                shutil.copy2(item["source_path"], temp_path)
            else:
                item["upload"].save(temp_path)
            df = compute_utils.read_df_file(temp_path)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Uploaded object is not a pandas dataframe.")
            if "Predictions" not in df.columns:
                raise ValueError("Uploaded dataframe does not contain a 'Predictions' column.")
            output_name = safe_name
            output_path = os.path.join(predictions_dir, output_name)
            df.to_pickle(output_path)
            loaded_count += 1
        except Exception as exc:
            errors.append(f"{safe_name}: {exc}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    if loaded_count == 0:
        detail = errors[0] if errors else "Unknown error."
        flash(f"Failed to load precomputed predictions: {detail}", "error")
        return _redirect_to_inference_load_tab()

    if errors:
        shown = "; ".join(errors[:3])
        if len(errors) > 3:
            shown += f"; ... ({len(errors) - 3} more)"
        flash(f"Loaded {loaded_count} predictions file(s), but some failed: {shown}", "error")
    return _redirect_to_inference_load_tab()


@app.route("/inference/remove_file", methods=["POST"])
def remove_inference_file():
    filename = (request.form.get("filename") or "").strip()
    predictions_root = _predictions_data_dir(create=False)
    try:
        target_path = _validate_relative_pickle_path(filename, predictions_root, "Predictions file")
        os.remove(target_path)
    except Exception as exc:
        flash(str(exc), "error")
    return redirect(url_for("inference_load_data"))

# New training for inference configuration page
@app.route("/inference/new_training")
def new_training():
    return render_template("4.2.0.new_training.html")

# Compute predictions for inference configuration page
@app.route("/inference/compute_predictions")
def compute_predictions():
    _bootstrap_features_data_from_previous_steps()
    feature_data_files = _list_features_data_files()
    # Hard fallback: directly scan canonical folder in case helper discovery is bypassed.
    if not feature_data_files:
        hard_root = _features_data_dir(create=False)
        if os.path.isdir(hard_root):
            recovered = []
            for root, _, names in os.walk(hard_root):
                for name in names:
                    lower = name.lower()
                    if not (lower.endswith(".pkl") or lower.endswith(".pickle")):
                        continue
                    abs_path = os.path.join(root, name)
                    if not os.path.isfile(abs_path):
                        continue
                    recovered.append(os.path.relpath(abs_path, hard_root))
            if recovered:
                feature_data_files = sorted(recovered)
    app.logger.warning(
        "[inference] features auto-detect dir=%s files=%s",
        _features_data_dir(create=False),
        feature_data_files,
    )
    print(
        f"[inference] features auto-detect dir={_features_data_dir(create=False)} files={feature_data_files}",
        flush=True,
    )
    default_feature_file = ""
    if feature_data_files:
        features_root = _features_data_dir(create=False)
        try:
            default_feature_file = max(
                feature_data_files,
                key=lambda name: os.path.getmtime(os.path.join(features_root, name))
            )
        except Exception:
            default_feature_file = feature_data_files[-1]
    return render_template(
        "4.3.0.compute_predictions.html",
        feature_data_files=feature_data_files,
        has_feature_data=bool(feature_data_files),
        default_feature_file=default_feature_file,
    )


@app.route("/inference/detect_model_backend", methods=["POST"])
def inference_detect_model_backend():
    upload = request.files.get("model_file")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "error": "Upload a model file first."}), 400

    safe_name = secure_filename(upload.filename)
    if not safe_name:
        return jsonify({"ok": False, "error": "Invalid uploaded file name."}), 400

    inference_upload_dir = _module_uploads_dir_for("inference")
    os.makedirs(inference_upload_dir, exist_ok=True)
    temp_path = os.path.join(inference_upload_dir, f"detect_model_{uuid.uuid4().hex}_{safe_name}")
    upload.save(temp_path)

    try:
        model_obj = compute_utils._load_pickle_file(temp_path)
        backend, inferred_type = compute_utils._infer_artifact_backend(model_obj)
        if backend not in {"sbi", "sklearn"}:
            return jsonify({
                "ok": False,
                "error": f"Could not infer backend from uploaded model artifact type '{inferred_type}'.",
                "backend": "",
                "inferred_type": inferred_type,
            }), 200
        return jsonify({
            "ok": True,
            "backend": backend,
            "inferred_type": inferred_type,
            "is_sbi": backend == "sbi",
        }), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 200
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

# Analysis configuration page
def _analysis_data_dir(create=False):
    path = ANALYSIS_DATA_DIR
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _analysis_mode_path(create=False):
    return os.path.join(_analysis_data_dir(create=create), ".selection_mode")


def _analysis_selected_keys_path(create=False):
    return os.path.join(_analysis_data_dir(create=create), ".selected_simulation_keys.json")


def _set_analysis_selected_simulation_keys(keys):
    selected = [
        str(key).strip()
        for key in (keys or [])
        if str(key).strip()
    ]
    try:
        with open(_analysis_selected_keys_path(create=True), "w", encoding="utf-8") as f:
            json.dump(selected, f)
    except OSError:
        pass


def _get_analysis_selected_simulation_keys():
    path = _analysis_selected_keys_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def _clear_analysis_selected_simulation_keys():
    path = _analysis_selected_keys_path()
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    _remove_dir_if_empty(_analysis_data_dir(create=False))


def _set_analysis_selection_mode(mode):
    mode_value = (mode or "").strip().lower()
    if not mode_value:
        _clear_analysis_selection_mode()
        return
    try:
        with open(_analysis_mode_path(create=True), "w", encoding="utf-8") as f:
            f.write(mode_value)
    except OSError:
        pass


def _get_analysis_selection_mode():
    path = _analysis_mode_path()
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (f.read() or "").strip().lower()
    except OSError:
        return ""


def _clear_analysis_selection_mode():
    path = _analysis_mode_path()
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    _remove_dir_if_empty(_analysis_data_dir(create=False))


def _list_analysis_data_files():
    analysis_data_dir = _analysis_data_dir(create=False)
    if not os.path.isdir(analysis_data_dir):
        return []
    return sorted(
        f for f in os.listdir(analysis_data_dir)
        if (f.endswith(".pkl") or f.endswith(".pickle"))
        and os.path.isfile(os.path.join(analysis_data_dir, f))
    )


def _clear_analysis_data_files():
    removed_files = []
    analysis_data_dir = _analysis_data_dir(create=False)
    if not os.path.isdir(analysis_data_dir):
        return removed_files
    for name in os.listdir(analysis_data_dir):
        if not (name.endswith(".pkl") or name.endswith(".pickle")):
            continue
        path = os.path.join(analysis_data_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed_files.append(name)
            except OSError:
                pass
    _remove_dir_if_empty(analysis_data_dir)
    return removed_files


@app.route("/analysis")
def analysis():
    analysis_data_files = _list_analysis_data_files()
    selection_mode = _get_analysis_selection_mode()
    preselected_simulation_keys = _get_analysis_selected_simulation_keys() if selection_mode == "simulation" else []
    if selection_mode == "simulation":
        has_analysis_dataframe = False
    elif selection_mode == "dataframe":
        has_analysis_dataframe = bool(analysis_data_files)
    else:
        # Backward compatibility with existing persisted files from older sessions.
        has_analysis_dataframe = bool(analysis_data_files)
    feature_data_files = _list_features_data_files()
    predictions_data_files = _list_predictions_data_files()
    detected_data_files = [
        {k: v for k, v in entry.items() if k != "path"}
        for entry in _list_detected_analysis_data_files()
    ]
    simulation_available = False
    simulation_trials = 0
    simulation_model = ""
    simulation_welch_defaults = None
    try:
        sim_data = _load_simulation_outputs()
        simulation_available = True
        simulation_trials = _simulation_trial_count(sim_data)
        simulation_model = _simulation_model_type(sim_data)
        simulation_welch_defaults = _simulation_default_welch_preview(sim_data)
    except Exception:
        simulation_available = False
        simulation_trials = 0
        simulation_model = ""
        simulation_welch_defaults = None
    return render_template(
        "5.1.0.analysis.html",
        analysis_data_files=analysis_data_files,
        has_analysis_data=bool(analysis_data_files),
        has_analysis_dataframe=has_analysis_dataframe,
        feature_data_files=feature_data_files,
        has_feature_data=bool(feature_data_files),
        predictions_data_files=predictions_data_files,
        has_predictions_data=bool(predictions_data_files),
        detected_data_files=detected_data_files,
        preselected_simulation_keys=preselected_simulation_keys,
        simulation_available=simulation_available,
        simulation_trials=simulation_trials,
        simulation_model=simulation_model,
        simulation_welch_defaults=simulation_welch_defaults,
    )


def _analysis_data_path():
    if _get_analysis_selection_mode() == "simulation":
        return None
    analysis_data_dir = _analysis_data_dir(create=False)
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


def _simulation_output_root():
    return SIMULATION_DATA_DIR


def _simulation_output_file(name):
    path = os.path.join(_simulation_output_root(), name)
    return path if os.path.isfile(path) else None


def _load_simulation_outputs():
    required = ["times.pkl", "gids.pkl", "dt.pkl", "tstop.pkl", "network.pkl"]
    paths = {name: _simulation_output_file(name) for name in required}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"Missing simulation output files in {_simulation_output_root()}: {', '.join(missing)}"
        )

    payload = {}
    for name, path in paths.items():
        with open(path, "rb") as f:
            payload[name] = pickle.load(f)

    metadata = None
    metadata_path = _simulation_grid_metadata_path(_simulation_output_root())
    if metadata_path and os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "rb") as handle:
                maybe_meta = pickle.load(handle)
            if isinstance(maybe_meta, dict):
                metadata = maybe_meta
        except Exception:
            metadata = None
    if metadata is None:
        legacy_candidates = ["simulation_grid_metadata.pkl", "simulation_grid_metadata.json"]
        for legacy_name in legacy_candidates:
            legacy_path = os.path.join(_simulation_output_root(), legacy_name)
            if not os.path.isfile(legacy_path):
                continue
            try:
                if legacy_name.endswith(".json"):
                    with open(legacy_path, "r", encoding="utf-8") as handle:
                        maybe_meta = json.load(handle)
                else:
                    with open(legacy_path, "rb") as handle:
                        maybe_meta = pickle.load(handle)
                if isinstance(maybe_meta, dict):
                    metadata = maybe_meta
                    break
            except Exception:
                continue

    return {
        "times": payload["times.pkl"],
        "gids": payload["gids.pkl"],
        "dt": payload["dt.pkl"],
        "tstop": payload["tstop.pkl"],
        "network": payload["network.pkl"],
        "grid_metadata": metadata,
    }


def _simulation_trial_count(sim_data):
    times = sim_data["times"]
    return len(times) if isinstance(times, list) else 1


def _simulation_model_type(sim_data):
    net = sim_data.get("network", {}) or {}
    if isinstance(net, list):
        net = net[0] if net else {}
    if isinstance(net, dict) and "areas" in net:
        return "four_area"
    return "hagen"


def _simulation_area_names(sim_data, trial_idx=0):
    try:
        trial = _simulation_trial_data(sim_data, trial_idx)
        trial_network = trial.get("network", {})
        if isinstance(trial_network, dict):
            areas = trial_network.get("areas")
            if isinstance(areas, (list, tuple)):
                return list(areas)
    except Exception:
        pass

    net = sim_data.get("network", {}) or {}
    if isinstance(net, dict):
        areas = net.get("areas")
        if isinstance(areas, (list, tuple)):
            return list(areas)
    if isinstance(net, list) and net:
        first = net[0]
        if isinstance(first, dict):
            areas = first.get("areas")
            if isinstance(areas, (list, tuple)):
                return list(areas)
    return []


def _simulation_trial_data(sim_data, trial_idx):
    times_all = sim_data["times"]
    gids_all = sim_data["gids"]
    dt_all = sim_data["dt"]
    tstop_all = sim_data["tstop"]
    trial_count = _simulation_trial_count(sim_data)
    if trial_idx < 0 or trial_idx >= trial_count:
        raise IndexError(f"Trial index {trial_idx} out of bounds for {trial_count} trial(s).")

    def _pick(obj):
        if isinstance(obj, list):
            return obj[trial_idx]
        return obj

    return {
        "times": _pick(times_all),
        "gids": _pick(gids_all),
        "dt": float(_pick(dt_all)),
        "tstop": float(_pick(tstop_all)),
        "network": _pick(sim_data["network"]),
    }


def _resolve_default_welch_params(signal_len, fs_hz):
    signal_len = int(max(1, signal_len))
    fs_hz = float(max(1.0, fs_hz))
    one_second_samples = int(max(1, round(fs_hz)))
    if signal_len > one_second_samples:
        nperseg = one_second_samples
    else:
        nperseg = max(1, signal_len // 2)
    return {
        "nperseg": int(nperseg),
        "noverlap": None,
        "nfft": None,
    }


def _simulation_default_welch_preview(sim_data):
    try:
        trial = _simulation_trial_data(sim_data, 0)
        dt = float(trial.get("dt", 0.0))
        tstop = float(trial.get("tstop", 0.0))
    except Exception:
        return None
    if not np.isfinite(dt) or not np.isfinite(tstop) or dt <= 0.0 or tstop <= 0.0:
        return None
    fs_hz = 1000.0 / dt
    signal_len = int(max(1, round(tstop / dt)))
    params = _resolve_default_welch_params(signal_len, fs_hz)
    params["fs_hz"] = float(fs_hz)
    params["signal_len"] = int(signal_len)
    params["dt_ms"] = float(dt)
    return params


def _format_trial_param_value(value, max_len=32):
    if isinstance(value, float):
        text = f"{value:.6g}"
    elif isinstance(value, (int, bool)):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            text = repr(value)
    text = str(text)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _simulation_trial_changed_params_text(sim_data, trial_idx, max_params=3):
    meta = sim_data.get("grid_metadata")
    if not isinstance(meta, dict):
        return ""
    changed_keys = meta.get("changed_keys")
    trials = meta.get("trials")
    if not isinstance(changed_keys, list) or not changed_keys:
        return ""
    if not isinstance(trials, list) or trial_idx < 0 or trial_idx >= len(trials):
        return ""
    trial_entry = trials[trial_idx]
    if not isinstance(trial_entry, dict):
        return ""
    changed = trial_entry.get("changed")
    if not isinstance(changed, dict):
        return ""
    parts = []
    for key in changed_keys:
        if key not in changed:
            continue
        parts.append(f"{key}={_format_trial_param_value(changed[key])}")
    if not parts:
        return ""
    if len(parts) > max_params:
        remaining = len(parts) - max_params
        parts = parts[:max_params] + [f"+{remaining} more"]
    return ", ".join(parts)


def _simulation_trial_plot_title(sim_data, trial_idx, suffix):
    base = f"Trial {trial_idx} {suffix}".strip()
    changed = _simulation_trial_changed_params_text(sim_data, trial_idx)
    if changed:
        return f"{base}\n{changed}"
    return base


def _simulation_trial_legend_label(sim_data, trial_idx):
    changed = _simulation_trial_changed_params_text(sim_data, trial_idx)
    if changed:
        return f"trial {trial_idx} | {changed}"
    return f"trial {trial_idx}"


def _simulation_trial_grid_entry(sim_data, trial_idx):
    meta = sim_data.get("grid_metadata")
    if not isinstance(meta, dict):
        return None
    trials = meta.get("trials")
    if not isinstance(trials, list) or trial_idx < 0 or trial_idx >= len(trials):
        return None
    entry = trials[trial_idx]
    return entry if isinstance(entry, dict) else None


def _simulation_trial_repeat_index(sim_data, trial_idx):
    entry = _simulation_trial_grid_entry(sim_data, trial_idx)
    if isinstance(entry, dict):
        try:
            value = int(entry.get("repeat_index"))
            if value >= 0:
                return value
        except (TypeError, ValueError):
            pass

    meta = sim_data.get("grid_metadata")
    if not isinstance(meta, dict):
        return None
    try:
        repetitions = int(meta.get("repetitions_per_configuration"))
    except (TypeError, ValueError):
        repetitions = 0
    if repetitions > 1:
        return int(trial_idx % repetitions)
    return None


def _simulation_trial_configuration_index(sim_data, trial_idx):
    entry = _simulation_trial_grid_entry(sim_data, trial_idx)
    if isinstance(entry, dict):
        try:
            return int(entry.get("configuration_index"))
        except (TypeError, ValueError):
            pass

    meta = sim_data.get("grid_metadata")
    if isinstance(meta, dict):
        try:
            repetitions = int(meta.get("repetitions_per_configuration"))
        except (TypeError, ValueError):
            repetitions = 0
        if repetitions > 0:
            return int(trial_idx // repetitions)
    return int(trial_idx)


def _simulation_available_repetition_indices(sim_data, trial_indices):
    repeats = []
    for trial_idx in trial_indices:
        rep_idx = _simulation_trial_repeat_index(sim_data, trial_idx)
        if rep_idx is None:
            continue
        repeats.append(int(rep_idx))
    return sorted(set(repeats))


def _simulation_group_trials_by_configuration(sim_data, trial_indices):
    grouped = {}
    for trial_idx in trial_indices:
        cfg_idx = _simulation_trial_configuration_index(sim_data, trial_idx)
        grouped.setdefault(int(cfg_idx), []).append(int(trial_idx))
    return grouped


def _simulation_configuration_legend_label(sim_data, configuration_index, trial_indices):
    changed = ""
    for trial_idx in trial_indices:
        changed = _simulation_trial_changed_params_text(sim_data, trial_idx)
        if changed:
            break
    base = f"config {configuration_index}"
    if changed:
        return f"{base} | {changed}"
    return base


def _population_color(name, idx):
    if str(name).upper().startswith("E"):
        return "C0"
    if str(name).upper().startswith("I"):
        return "C1"
    return f"C{idx % 10}"


def _spike_rate(times, dt, tstop):
    bins = np.arange(0.0, tstop + dt, dt)
    hist, _ = np.histogram(np.asarray(times), bins=bins)
    return bins[:-1], hist.astype(float)


def _compute_trial_cdm_proxy(trial):
    dt = float(trial["dt"])
    tstop = float(trial["tstop"])
    times = trial["times"]
    model = _simulation_model_type({"network": trial["network"]})
    bins = np.arange(0.0, tstop + dt, dt)
    centers = bins[:-1]

    if model == "hagen":
        cdm = np.zeros_like(centers, dtype=float)
        for pop_name, pop_times in times.items():
            hist, _ = np.histogram(np.asarray(pop_times), bins=bins)
            sign = 1.0 if str(pop_name).upper().startswith("E") else -1.0
            cdm += sign * hist.astype(float)
        return centers, cdm

    # four_area
    cdm = np.zeros_like(centers, dtype=float)
    for area_data in times.values():
        for pop_name, pop_times in area_data.items():
            hist, _ = np.histogram(np.asarray(pop_times), bins=bins)
            sign = 1.0 if str(pop_name).upper().startswith("E") else -1.0
            cdm += sign * hist.astype(float)
    return centers, cdm


def _parse_selected_analysis_file_keys(selected_keys):
    simulation_files = set()
    field_potential_paths = []
    for raw_key in selected_keys:
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        if not key:
            continue
        if key.startswith("simulation::"):
            try:
                _, name = key.split("::", 1)
            except ValueError:
                continue
            name = (name or "").strip()
            if name:
                simulation_files.add(name)
            continue
        if key.startswith("field_potential::"):
            try:
                _, path = key.split("::", 1)
            except ValueError:
                continue
            path = (path or "").strip()
            if path and os.path.isfile(path):
                field_potential_paths.append(path)
    return simulation_files, field_potential_paths


def _pick_trial_value(value, trial_idx):
    if isinstance(value, list):
        if not value:
            return None
        if 0 <= trial_idx < len(value):
            return value[trial_idx]
        return value[-1]
    return value


def _to_1d_signal(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == 3:
            return arr[2]
        if arr.shape[1] == 3:
            return arr[:, 2]
        if arr.shape[0] <= arr.shape[1]:
            return np.sum(arr, axis=0)
        return np.sum(arr, axis=1)
    return np.ravel(arr)


def _sum_signals(base, extra):
    if base is None:
        return np.array(extra, dtype=float, copy=True)
    n = min(base.size, extra.size)
    if n <= 0:
        return base
    base[:n] += extra[:n]
    return base


def _extract_area_from_signal_key(key, areas):
    key_str = str(key)
    # Support keys like "E(frontal):I(parietal)" by extracting parenthesized tokens first.
    paren_tokens = re.findall(r"\(([^()]+)\)", key_str)
    candidates = [key_str]
    candidates.extend(paren_tokens)
    for sep in (":", "/", "|", "-", "_"):
        split_tokens = []
        for item in candidates:
            split_tokens.extend(item.split(sep))
        candidates.extend(split_tokens)
    lowered = {str(area).lower(): str(area) for area in areas}
    for token in candidates:
        area = lowered.get(str(token).strip().lower())
        if area:
            return area
    return None


def _field_potential_payload_to_area_series(payload, model, areas, trial_idx):
    obj = _pick_trial_value(payload, trial_idx)
    if obj is None:
        return {}, None

    dt_ms = None
    raw_signals = None
    data_signal = None

    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return {}, None
        row = obj.iloc[0]
        if "dt_ms" in row:
            try:
                dt_ms = float(row["dt_ms"])
            except Exception:
                dt_ms = None
        raw_signals = row.get("raw_signals") if "raw_signals" in row else None
        data_signal = row.get("data") if "data" in row else None
    elif isinstance(obj, dict):
        raw_signals = obj.get("raw_signals")
        data_signal = obj.get("data")
        if obj.get("dt_ms") is not None:
            try:
                dt_ms = float(obj.get("dt_ms"))
            except Exception:
                dt_ms = None
    else:
        data_signal = obj

    series_by_area = {}
    if isinstance(raw_signals, dict) and raw_signals:
        for signal_key, signal_val in raw_signals.items():
            signal = _to_1d_signal(signal_val)
            if signal.size == 0:
                continue
            if model == "four_area":
                area = _extract_area_from_signal_key(signal_key, areas)
                if area is None:
                    continue
            else:
                area = "global"
            series_by_area[area] = _sum_signals(series_by_area.get(area), signal)

    if not series_by_area and data_signal is not None:
        arr = np.asarray(data_signal, dtype=float)
        if model == "four_area":
            if arr.ndim == 2 and arr.shape[0] == len(areas):
                for i, area in enumerate(areas):
                    series_by_area[area] = _to_1d_signal(arr[i])
            elif arr.ndim == 2 and arr.shape[1] == len(areas):
                for i, area in enumerate(areas):
                    series_by_area[area] = _to_1d_signal(arr[:, i])
            elif arr.ndim == 1:
                # Proxy outputs can be global 1D signals; broadcast to all areas for plotting.
                for area in areas:
                    series_by_area[area] = _to_1d_signal(arr)
        else:
            series_by_area["global"] = _to_1d_signal(arr)

    if model == "four_area" and areas:
        # If only a global signal is available, broadcast it across all areas.
        if "global" in series_by_area:
            global_sig = _to_1d_signal(series_by_area["global"])
            for area in areas:
                series_by_area.setdefault(area, np.array(global_sig, copy=True))
            series_by_area.pop("global", None)

        # If no explicit area keys were found but there is exactly one signal, reuse it for all areas.
        if not all(area in series_by_area for area in areas) and len(series_by_area) == 1:
            only_sig = _to_1d_signal(next(iter(series_by_area.values())))
            series_by_area = {area: np.array(only_sig, copy=True) for area in areas}

    return series_by_area, dt_ms


@app.route("/analysis/select_features_file", methods=["POST"])
def analysis_select_features_file():
    selector = (request.form.get("data_file_key") or request.form.get("features_file") or "").strip()
    if not selector:
        return jsonify({"error": "Select a detected file."}), 400

    detected_entries = _list_detected_analysis_data_files()
    by_key = {entry["key"]: entry for entry in detected_entries}

    selected = by_key.get(selector)
    if selected is None:
        # Backward compatibility for old keys/plain filenames.
        if "::" in selector:
            source, filename = selector.split("::", 1)
            fallback_key = f"{source.strip().lower()}::{filename.strip()}"
        else:
            fallback_key = f"features::{selector}"
        selected = by_key.get(fallback_key)

    if selected is None:
        return jsonify({"error": "Selected file is not available. Refresh the page and try again."}), 400

    source = selected.get("source", "unknown")
    filename = selected.get("name", "")
    src_path = selected.get("path")
    if not src_path or not os.path.isfile(src_path):
        return jsonify({"error": "Selected file was not found on disk."}), 404

    _clear_analysis_data_files()
    analysis_data_dir = _analysis_data_dir(create=True)

    dst_name = secure_filename(filename)
    dst_path = os.path.join(analysis_data_dir, dst_name)
    try:
        shutil.copy2(src_path, dst_path)
    except OSError as exc:
        return jsonify({"error": f"Failed to stage selected file for analysis: {exc}"}), 500

    if source in {"simulation", "field_potential"}:
        _set_analysis_selection_mode("simulation")
        _set_analysis_selected_simulation_keys([selected["key"]])
        return jsonify({
            "columns": [],
            "filename": dst_name,
            "source": source,
            "is_dataframe": False,
        })

    try:
        df = compute_utils.read_df_file(dst_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Selected file is not a pandas dataframe.")
        columns = [str(col) for col in df.columns]
        _set_analysis_selection_mode("dataframe")
        _clear_analysis_selected_simulation_keys()
        return jsonify({
            "columns": columns,
            "filename": dst_name,
            "source": source,
            "is_dataframe": True,
        })
    except Exception as exc:
        try:
            if os.path.exists(dst_path):
                os.remove(dst_path)
        except OSError:
            pass
        return jsonify({"error": str(exc)}), 400


@app.route("/analysis/sync_selected_simulation_files", methods=["POST"])
def analysis_sync_selected_simulation_files():
    selected_keys = [
        (key or "").strip()
        for key in request.form.getlist("selected_keys")
        if (key or "").strip()
    ]

    detected_entries = _list_detected_analysis_data_files()
    by_key = {entry["key"]: entry for entry in detected_entries}
    selected_entries = []
    for key in selected_keys:
        entry = by_key.get(key)
        if entry is None:
            continue
        if entry.get("source") not in {"simulation", "field_potential"}:
            continue
        src_path = entry.get("path")
        if not src_path or not os.path.isfile(src_path):
            continue
        selected_entries.append(entry)

    _clear_analysis_data_files()
    analysis_data_dir = _analysis_data_dir(create=True)

    copied_files = []
    used_names = set()
    for idx, entry in enumerate(selected_entries):
        src_path = entry["path"]
        src_name = os.path.basename(entry.get("name") or src_path)
        safe_name = secure_filename(src_name) or f"analysis_selected_{idx + 1}.pkl"
        stem, ext = os.path.splitext(safe_name)
        ext = ext or ".pkl"
        candidate = safe_name
        suffix = 1
        while candidate in used_names:
            candidate = f"{stem}_{suffix}{ext}"
            suffix += 1
        used_names.add(candidate)
        dst_path = os.path.join(analysis_data_dir, candidate)
        try:
            shutil.copy2(src_path, dst_path)
        except OSError:
            continue
        copied_files.append(candidate)

    if copied_files:
        _set_analysis_selection_mode("simulation")
        _set_analysis_selected_simulation_keys(selected_keys)
    else:
        _clear_analysis_selection_mode()
        _clear_analysis_selected_simulation_keys()

    return jsonify({"ok": True, "copied_files": copied_files})


@app.route("/analysis/upload_simulation_files", methods=["POST"])
def analysis_upload_simulation_files():
    uploads = [f for f in request.files.getlist("simulation_files") if f and f.filename]
    server_file_paths = _extract_server_file_paths(request.form, "simulation_server_file_path")
    if not uploads and not server_file_paths:
        return jsonify({"error": "No simulation files were provided."}), 400

    inputs = []
    for server_path in server_file_paths:
        try:
            source_path = _validate_existing_pickle_file_path(server_path, "Simulation server file")
            safe_name = secure_filename(os.path.basename(source_path))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        if not safe_name:
            continue
        inputs.append({"safe_name": safe_name, "source_path": source_path, "upload": None})

    for upload in uploads:
        raw_name = os.path.basename(upload.filename or "")
        safe_name = secure_filename(raw_name)
        if not safe_name:
            continue
        ext = Path(safe_name).suffix.lower()
        if ext not in PICKLE_EXTENSIONS:
            continue
        inputs.append({"safe_name": safe_name, "source_path": None, "upload": upload})

    if not inputs:
        return jsonify({"error": "No supported simulation .pkl/.pickle files were provided."}), 400

    simulation_root = _simulation_output_root()
    os.makedirs(simulation_root, exist_ok=True)

    # Replace previous simulation pickle outputs to avoid mixing datasets.
    for name in os.listdir(simulation_root):
        lower = name.lower()
        if not (lower.endswith(".pkl") or lower.endswith(".pickle")):
            continue
        path = os.path.join(simulation_root, name)
        if not os.path.isfile(path):
            continue
        try:
            os.remove(path)
        except OSError:
            pass

    required_name_by_stem = {
        "times": "times.pkl",
        "gids": "gids.pkl",
        "dt": "dt.pkl",
        "tstop": "tstop.pkl",
        "network": "network.pkl",
        "grid_metadata": SIMULATION_GRID_METADATA_FILE,
    }
    copied_files = []
    used_names = set()
    for item in inputs:
        safe_name = item["safe_name"]

        stem = Path(safe_name).stem.lower()
        preferred_name = required_name_by_stem.get(stem, safe_name)
        preferred_name = secure_filename(preferred_name)
        if not preferred_name:
            continue
        preferred_stem, preferred_ext = os.path.splitext(preferred_name)
        preferred_ext = preferred_ext or ".pkl"
        candidate = preferred_name
        suffix = 1
        while candidate in used_names:
            candidate = f"{preferred_stem}_{suffix}{preferred_ext}"
            suffix += 1
        dst_path = os.path.join(simulation_root, candidate)
        try:
            if item["source_path"]:
                shutil.copy2(item["source_path"], dst_path)
            else:
                item["upload"].save(dst_path)
        except OSError:
            continue
        if not os.path.isfile(dst_path) or os.path.getsize(dst_path) <= 0:
            try:
                if os.path.exists(dst_path):
                    os.remove(dst_path)
            except OSError:
                pass
            continue
        used_names.add(candidate)
        copied_files.append(candidate)

    if not copied_files:
        return jsonify({"error": "No supported simulation .pkl/.pickle files were uploaded."}), 400

    priority = {
        "times.pkl": 0,
        "gids.pkl": 1,
        "dt.pkl": 2,
        "tstop.pkl": 3,
        "network.pkl": 4,
        SIMULATION_GRID_METADATA_FILE: 5,
    }
    copied_files = sorted(copied_files, key=lambda name: (priority.get(name, 1000), name))
    selected_keys = [f"simulation::{name}" for name in copied_files]
    _set_analysis_selection_mode("simulation")
    _set_analysis_selected_simulation_keys(selected_keys)
    _clear_analysis_data_files()

    required = {"times.pkl", "gids.pkl", "dt.pkl", "tstop.pkl", "network.pkl"}
    missing_required = sorted(name for name in required if name not in set(copied_files))

    simulation_available = False
    simulation_trials = 0
    simulation_model = ""
    simulation_welch_defaults = None
    try:
        sim_data = _load_simulation_outputs()
        simulation_available = True
        simulation_trials = _simulation_trial_count(sim_data)
        simulation_model = _simulation_model_type(sim_data)
        simulation_welch_defaults = _simulation_default_welch_preview(sim_data)
    except Exception:
        simulation_available = False
        simulation_trials = 0
        simulation_model = ""
        simulation_welch_defaults = None

    return jsonify({
        "ok": True,
        "copied_files": copied_files,
        "selected_keys": selected_keys,
        "missing_required": missing_required,
        "simulation_available": simulation_available,
        "simulation_trials": simulation_trials,
        "simulation_model": simulation_model,
        "simulation_welch_defaults": simulation_welch_defaults,
    })


def _analysis_plot_error(message, status=400, log_output=""):
    return (
        render_template(
            "5.2.0.analysis_plot_results.html",
            title="Analysis plot",
            subtitle="Plotting failed.",
            error=message,
            image_data=None,
            log_output=log_output,
        ),
        status,
    )


def _analysis_default_plot_filename(title):
    base_name = secure_filename(str(title or "").strip())
    if not base_name:
        base_name = "analysis_plot"
    if not base_name.lower().endswith(".png"):
        base_name = f"{base_name}.png"
    return base_name


def _render_analysis_plot(title, subtitle, image_bytes, log_output="", **extra_context):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    context = {
        "title": title,
        "subtitle": subtitle,
        "error": None,
        "image_data": encoded,
        "log_output": log_output,
        "default_filename": _analysis_default_plot_filename(title),
    }
    if extra_context:
        context.update(extra_context)
    return render_template("5.2.0.analysis_plot_results.html", **context)


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
    _remember_path_history_from_form(request.form, "analysis_columns")
    dataframe_path = (request.form.get("dataframe_path") or "").strip()
    upload = request.files.get("dataframe")
    
    _clear_analysis_data_files()
    analysis_data_dir = _analysis_data_dir(create=True)

    if dataframe_path:
        source_path = os.path.realpath(os.path.expanduser(dataframe_path))
        if not os.path.isfile(source_path):
            return jsonify({"error": f"Selected path is not a file: {source_path}"}), 400
        file_extension = os.path.splitext(source_path)[1].lower()
        if file_extension not in {".pkl", ".pickle"}:
            return jsonify({"error": "Only .pkl/.pickle files are supported."}), 400
        filename = secure_filename(os.path.basename(source_path))
        if not filename:
            return jsonify({"error": "Invalid filename."}), 400
        temp_path = os.path.join(analysis_data_dir, filename)
        try:
            shutil.copy2(source_path, temp_path)
        except OSError as exc:
            return jsonify({"error": f"Failed to copy selected file: {exc}"}), 400
    else:
        if upload is None or upload.filename == "":
            return jsonify({"error": "No dataframe file uploaded."}), 400
        filename = secure_filename(upload.filename)
        if not filename:
            return jsonify({"error": "Invalid filename."}), 400
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in {".pkl", ".pickle"}:
            return jsonify({"error": "Only .pkl/.pickle files are supported."}), 400
        temp_path = os.path.join(analysis_data_dir, filename)
        try:
            upload.save(temp_path)
        except OSError as exc:
            return jsonify({"error": f"Failed to save uploaded file: {exc}"}), 400

    try:
        df = compute_utils.read_df_file(temp_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "Uploaded file does not contain a pandas dataframe. "
                "Select 'Simulation files' mode for simulation outputs."
            )
        columns = [str(col) for col in df.columns]
        _set_analysis_selection_mode("dataframe")
        _clear_analysis_selected_simulation_keys()
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
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "Current analysis file is not a pandas dataframe. "
                "Select 'Simulation files' mode for simulation outputs."
            )
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
    if not isinstance(df, pd.DataFrame):
        return jsonify({
            "error": (
                "Current analysis file is not a pandas dataframe. "
                "Select 'Simulation files' mode for simulation outputs."
            )
        }), 400
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


@app.route("/analysis/plot/simulation", methods=["POST"])
def analysis_plot_simulation():
    try:
        sim_data = _load_simulation_outputs()
    except Exception as exc:
        return _analysis_plot_error(str(exc), status=400)

    total_trials = _simulation_trial_count(sim_data)
    model = _simulation_model_type(sim_data)

    def _parse_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    range_start = _parse_int(request.form.get("sim_trial_start"), 0)
    range_end = _parse_int(request.form.get("sim_trial_end"), total_trials - 1)
    range_start = max(0, range_start)
    range_end = min(total_trials - 1, range_end)
    if range_start > range_end:
        return _analysis_plot_error("Trial range is invalid: start must be <= end.", status=400)

    trial_indices_full_range = list(range(range_start, range_end + 1))
    if not trial_indices_full_range:
        return _analysis_plot_error("No trials selected to plot.", status=400)
    trial_indices = list(trial_indices_full_range)

    plot_type = (request.form.get("sim_plot_type") or "raster").strip().lower()
    available_repeat_indices = _simulation_available_repetition_indices(sim_data, trial_indices_full_range)
    sim_repeat_raw = (request.form.get("sim_repeat_index") or "").strip().lower()
    selected_repeat_index = None
    if sim_repeat_raw and sim_repeat_raw not in {"all", "*"}:
        try:
            selected_repeat_index = int(sim_repeat_raw)
        except ValueError:
            return _analysis_plot_error("Repetition selection must be an integer or 'all'.", status=400)
        if selected_repeat_index < 0:
            return _analysis_plot_error("Repetition selection must be >= 0.", status=400)

    if selected_repeat_index is not None:
        if not available_repeat_indices:
            return _analysis_plot_error("Repetition selection is not available for these simulation outputs.", status=400)
        if selected_repeat_index not in set(available_repeat_indices):
            return _analysis_plot_error(
                f"Selected repetition {selected_repeat_index + 1} is outside the available range.",
                status=400,
            )

    if plot_type in {"raster", "firing_rates", "cdm"} and selected_repeat_index is not None:
        trial_indices = [
            idx for idx in trial_indices_full_range
            if _simulation_trial_repeat_index(sim_data, idx) == selected_repeat_index
        ]
        if not trial_indices:
            return _analysis_plot_error(
                f"No trials found for repetition {selected_repeat_index + 1} in the selected trial range.",
                status=400,
            )

    selected_keys = [s for s in request.form.getlist("sim_selected_file_keys") if isinstance(s, str) and s.strip()]
    selected_sim_files, selected_fp_paths = _parse_selected_analysis_file_keys(selected_keys)
    def _parse_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    time_start = _parse_float(request.form.get("sim_time_start"), 4000.0)
    time_end = _parse_float(request.form.get("sim_time_end"), 4100.0)
    if time_start > time_end:
        time_start, time_end = time_end, time_start

    freq_min = _parse_float(request.form.get("sim_freq_min"), 20.0)
    freq_max = _parse_float(request.form.get("sim_freq_max"), 200.0)
    if freq_min > freq_max:
        freq_min, freq_max = freq_max, freq_min

    def _parse_optional_int(field_name, label, min_value=0):
        raw = request.form.get(field_name)
        if raw is None:
            return None, None
        text = str(raw).strip()
        if text == "":
            return None, None
        try:
            value = int(text)
        except ValueError:
            return None, f"{label} must be an integer."
        if value < min_value:
            return None, f"{label} must be >= {min_value}."
        return value, None

    welch_nperseg, err = _parse_optional_int("sim_welch_nperseg", "Welch nperseg", min_value=1)
    if err:
        return _analysis_plot_error(err, status=400)
    welch_noverlap, err = _parse_optional_int("sim_welch_noverlap", "Welch noverlap", min_value=0)
    if err:
        return _analysis_plot_error(err, status=400)
    welch_nfft, err = _parse_optional_int("sim_welch_nfft", "Welch nfft", min_value=1)
    if err:
        return _analysis_plot_error(err, status=400)
    if welch_nperseg is not None and welch_noverlap is not None and welch_noverlap >= welch_nperseg:
        return _analysis_plot_error("Welch noverlap must be smaller than nperseg.", status=400)
    if welch_nperseg is not None and welch_nfft is not None and welch_nfft < welch_nperseg:
        return _analysis_plot_error("Welch nfft must be >= nperseg.", status=400)

    def _resolve_welch_kwargs(signal_len, fs_hz):
        signal_len = int(max(1, signal_len))
        fs_hz = float(max(1.0, fs_hz))
        one_second_samples = int(max(1, round(fs_hz)))
        if signal_len > one_second_samples:
            default_nperseg = one_second_samples
        else:
            default_nperseg = max(1, signal_len // 2)
        nperseg = int(welch_nperseg) if welch_nperseg is not None else default_nperseg
        nperseg = max(1, min(signal_len, nperseg))

        noverlap = int(welch_noverlap) if welch_noverlap is not None else None
        if noverlap is not None:
            noverlap = max(0, min(noverlap, nperseg - 1))

        nfft = int(welch_nfft) if welch_nfft is not None else None
        if nfft is not None:
            nfft = max(nperseg, nfft)
        return {
            "nperseg": nperseg,
            "noverlap": noverlap,
            "nfft": nfft,
        }

    def _fallback_welch_text():
        if not trial_indices:
            return None
        try:
            trial = _simulation_trial_data(sim_data, trial_indices[0])
            dt_local = float(trial["dt"])
            fs = 1000.0 / max(dt_local, 1e-9)
            signal_len = int(max(1, round(float(trial["tstop"]) / max(dt_local, 1e-9))))
            return _resolve_welch_kwargs(signal_len, fs)
        except Exception:
            return None

    allowed_types = {"raster", "firing_rates", "cdm", "cdm_psd"}
    if plot_type not in allowed_types:
        return _analysis_plot_error(f"Unsupported simulation plot type: {plot_type}", status=400)

    required_by_plot = {
        "raster": {"times.pkl", "gids.pkl", "dt.pkl", "tstop.pkl", "network.pkl"},
        "firing_rates": {"times.pkl", "gids.pkl", "dt.pkl", "tstop.pkl", "network.pkl"},
        "cdm": set(),
        "cdm_psd": set(),
    }
    required_files = required_by_plot[plot_type]
    if plot_type in {"raster", "firing_rates"}:
        if not selected_sim_files:
            return _analysis_plot_error(
                "No simulation files selected. Select the required simulation output files first.",
                status=400,
            )
        missing_selected = sorted(required_files - selected_sim_files)
        if missing_selected:
            return _analysis_plot_error(
                "Selected files are incomplete for this plot type. Missing: "
                + ", ".join(missing_selected),
                status=400,
            )

    cdm_payload = None
    cdm_payload_name = ""
    if plot_type in {"cdm", "cdm_psd"}:
        if not selected_fp_paths:
            return _analysis_plot_error(
                "Field-potential plots require a computed field-potential output file. Select a field-potential file first.",
                status=400,
            )
        loaded_candidates = []
        for path in selected_fp_paths:
            try:
                with open(path, "rb") as f:
                    loaded_candidates.append((path, pickle.load(f)))
            except Exception:
                continue
        if not loaded_candidates:
            return _analysis_plot_error(
                "Unable to load selected field-potential files. Select a valid field-potential output file.",
                status=400,
            )

        preferred = None
        for path, payload in loaded_candidates:
            name = os.path.basename(path).lower()
            if "cdm" in name or "dipole" in name:
                preferred = (path, payload)
                break
        if preferred is None:
            preferred = loaded_candidates[0]

        cdm_payload_name = os.path.basename(preferred[0])
        cdm_payload = preferred[1]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scipy.signal as ss
    except Exception as exc:
        return _analysis_plot_error(f"Plot dependencies are unavailable: {exc}", status=500)

    import math
    simulation_repeat_controls = None
    repeat_display_note = ""
    if plot_type in {"raster", "firing_rates", "cdm"} and available_repeat_indices:
        repeat_display_note = (
            f" Repetition shown: {selected_repeat_index + 1}."
            if selected_repeat_index is not None
            else " Repetitions shown: all."
        )
        preserve_fields = [
            "sim_plot_type",
            "sim_trial_start",
            "sim_trial_end",
            "sim_time_start",
            "sim_time_end",
            "sim_freq_min",
            "sim_freq_max",
            "sim_welch_nperseg",
            "sim_welch_noverlap",
            "sim_welch_nfft",
        ]
        hidden_fields = []
        for field_name in preserve_fields:
            value = request.form.get(field_name)
            if value is None:
                continue
            value_text = str(value).strip()
            if value_text == "":
                continue
            hidden_fields.append((field_name, value_text))
        if not any(name == "sim_plot_type" for name, _ in hidden_fields):
            hidden_fields.append(("sim_plot_type", plot_type))
        simulation_repeat_controls = {
            "action_url": url_for("analysis_plot_simulation"),
            "hidden_fields": hidden_fields,
            "selected_file_keys": selected_keys,
            "selected_value": str(selected_repeat_index) if selected_repeat_index is not None else "all",
            "options": [
                {"value": "all", "label": "All repetitions"}
            ] + [
                {"value": str(rep), "label": f"Repetition {rep + 1}"}
                for rep in available_repeat_indices
            ],
            "hint": "Select which repetition to display for this plot type.",
        }

    def _align_psd_to_reference(reference_freqs, freqs, psd_values):
        ref = np.asarray(reference_freqs, dtype=float).ravel()
        x = np.asarray(freqs, dtype=float).ravel()
        y = np.asarray(psd_values, dtype=float).ravel()
        if ref.size == 0 or x.size == 0 or y.size == 0:
            return None
        if x.size != y.size:
            return None
        if ref.size == x.size and np.allclose(ref, x, rtol=1e-6, atol=1e-9):
            return y
        return np.interp(ref, x, y, left=np.nan, right=np.nan)

    if plot_type == "cdm_psd":
        welch_used = None
        config_groups = _simulation_group_trials_by_configuration(sim_data, trial_indices)
        config_count = len(config_groups)
        areas = _simulation_area_names(sim_data, trial_indices[0]) if model == "four_area" else []
        if model == "four_area":
            cols = min(2, max(1, len(areas)))
            rows = int(math.ceil(len(areas) / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.2 * rows), dpi=160, squeeze=False)
            flat_ax = axs.ravel()
            for area_idx, area_name in enumerate(areas):
                ax = flat_ax[area_idx]
                plotted = 0
                for config_idx, config_trials in config_groups.items():
                    ref_freqs = None
                    spectra = []
                    for trial_idx in config_trials:
                        trial = _simulation_trial_data(sim_data, trial_idx)
                        series_by_area, dt_fp = _field_potential_payload_to_area_series(
                            cdm_payload,
                            model,
                            areas,
                            trial_idx,
                        )
                        if area_name not in series_by_area:
                            continue
                        cdm = np.asarray(series_by_area[area_name], dtype=float)
                        if cdm.size < 8:
                            continue
                        dt_local = float(dt_fp) if dt_fp is not None else float(trial["dt"])
                        fs = 1000.0 / max(dt_local, 1e-9)
                        effective_welch = _resolve_welch_kwargs(cdm.size, fs)
                        if welch_used is None:
                            welch_used = dict(effective_welch)
                        freqs, psd = ss.welch(cdm, fs=fs, **effective_welch)
                        if psd.size == 0:
                            continue
                        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
                        if ref_freqs is None:
                            ref_freqs = np.asarray(freqs, dtype=float)
                            aligned = np.asarray(psd_norm, dtype=float)
                        else:
                            aligned = _align_psd_to_reference(ref_freqs, freqs, psd_norm)
                            if aligned is None:
                                continue
                        spectra.append(aligned)
                    if ref_freqs is None or not spectra:
                        continue
                    spectra_matrix = np.vstack(spectra)
                    if not np.isfinite(spectra_matrix).any():
                        continue
                    mean_psd = np.nanmean(spectra_matrix, axis=0)
                    mask = (ref_freqs >= freq_min) & (ref_freqs <= freq_max) & np.isfinite(mean_psd)
                    if not np.any(mask):
                        continue
                    ax.semilogy(
                        ref_freqs[mask],
                        mean_psd[mask],
                        linewidth=1.0,
                        label=_simulation_configuration_legend_label(sim_data, config_idx, config_trials),
                    )
                    plotted += 1
                ax.set_title(f"{area_name} PSD")
                ax.set_xlabel("f (Hz)")
                ax.set_ylabel("PSD (a.u./Hz)")
                ax.grid(alpha=0.2)
                if plotted and config_count <= 20:
                    ax.legend(fontsize=7, loc="best")
            for extra_idx in range(len(areas), len(flat_ax)):
                flat_ax[extra_idx].axis("off")
            fig.tight_layout()
            welch_text = welch_used or _fallback_welch_text() or {
                "nperseg": "auto",
                "noverlap": "auto",
                "nfft": "auto",
            }
            subtitle = (
                f"Field-potential power spectra by area (repetition-averaged per configuration). "
                f"Model: {model}. Trials {range_start} to {range_end} | configurations={config_count} "
                f"({freq_min:g}-{freq_max:g} Hz). Source: {cdm_payload_name}. "
                f"Welch: nperseg={welch_text.get('nperseg')}, "
                f"noverlap={welch_text.get('noverlap')}, "
                f"nfft={welch_text.get('nfft')}."
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=160)
            for config_idx, config_trials in config_groups.items():
                ref_freqs = None
                spectra = []
                for trial_idx in config_trials:
                    trial = _simulation_trial_data(sim_data, trial_idx)
                    series_by_area, dt_fp = _field_potential_payload_to_area_series(
                        cdm_payload,
                        model,
                        [],
                        trial_idx,
                    )
                    cdm = np.asarray(series_by_area.get("global", []), dtype=float)
                    if cdm.size < 8:
                        continue
                    dt_local = float(dt_fp) if dt_fp is not None else float(trial["dt"])
                    fs = 1000.0 / max(dt_local, 1e-9)
                    effective_welch = _resolve_welch_kwargs(cdm.size, fs)
                    if welch_used is None:
                        welch_used = dict(effective_welch)
                    freqs, psd = ss.welch(cdm, fs=fs, **effective_welch)
                    if psd.size == 0:
                        continue
                    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
                    if ref_freqs is None:
                        ref_freqs = np.asarray(freqs, dtype=float)
                        aligned = np.asarray(psd_norm, dtype=float)
                    else:
                        aligned = _align_psd_to_reference(ref_freqs, freqs, psd_norm)
                        if aligned is None:
                            continue
                    spectra.append(aligned)
                if ref_freqs is None or not spectra:
                    continue
                spectra_matrix = np.vstack(spectra)
                if not np.isfinite(spectra_matrix).any():
                    continue
                mean_psd = np.nanmean(spectra_matrix, axis=0)
                mask = (ref_freqs >= freq_min) & (ref_freqs <= freq_max) & np.isfinite(mean_psd)
                if not np.any(mask):
                    continue
                ax.semilogy(
                    ref_freqs[mask],
                    mean_psd[mask],
                    linewidth=1.2,
                    label=_simulation_configuration_legend_label(sim_data, config_idx, config_trials),
                )
            ax.set_xlabel("f (Hz)")
            ax.set_ylabel("PSD (a.u./Hz)")
            ax.set_title("Field-potential power spectra (repetition-averaged by configuration)")
            ax.grid(alpha=0.2)
            if config_count <= 20:
                ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            welch_text = welch_used or _fallback_welch_text() or {
                "nperseg": "auto",
                "noverlap": "auto",
                "nfft": "auto",
            }
            subtitle = (
                f"Model: {model}. Trials {range_start} to {range_end} "
                f"(repetition-averaged spectra, configurations={config_count}, {freq_min:g}-{freq_max:g} Hz). "
                f"Source: {cdm_payload_name}. "
                f"Welch: nperseg={welch_text.get('nperseg')}, "
                f"noverlap={welch_text.get('noverlap')}, "
                f"nfft={welch_text.get('nfft')}."
            )
    else:
        if model == "four_area":
            areas = _simulation_area_names(sim_data, trial_indices[0])
            if not areas:
                return _analysis_plot_error("Four-area model detected but no areas were found in network parameters.", status=400)
            area_cols = min(2, max(1, len(areas)))
            area_rows = int(math.ceil(len(areas) / area_cols))
            total_rows = len(trial_indices) * area_rows
            fig, axs = plt.subplots(total_rows, area_cols, figsize=(5.0 * area_cols, 3.3 * total_rows), dpi=160, squeeze=False)
            for trial_pos, trial_idx in enumerate(trial_indices):
                trial = _simulation_trial_data(sim_data, trial_idx)
                times = trial["times"]
                gids = trial["gids"]
                dt = trial["dt"]
                tstop = trial["tstop"]

                area_series = None
                dt_fp = None
                if plot_type == "cdm":
                    area_series, dt_fp = _field_potential_payload_to_area_series(
                        cdm_payload,
                        model,
                        areas,
                        trial_idx,
                    )
                    missing = [area for area in areas if area not in area_series]
                    if missing:
                        return _analysis_plot_error(
                            "Selected field-potential file does not contain area-resolved signals for all four-area regions. "
                            f"Missing: {', '.join(missing)}",
                            status=400,
                        )

                for area_idx, area_name in enumerate(areas):
                    grid_row = trial_pos * area_rows + (area_idx // area_cols)
                    grid_col = area_idx % area_cols
                    ax = axs[grid_row][grid_col]
                    if plot_type == "raster":
                        area_times = times.get(area_name, {})
                        area_gids = gids.get(area_name, {})
                        for pop_i, pop_name in enumerate(sorted(area_times.keys())):
                            t = np.asarray(area_times[pop_name])
                            g = np.asarray(area_gids.get(pop_name, []))
                            mask_t = (t >= time_start) & (t <= time_end)
                            ax.plot(
                                t[mask_t],
                                g[mask_t],
                                ".",
                                ms=0.9,
                                color=_population_color(pop_name, pop_i),
                                alpha=0.7,
                                label=str(pop_name),
                            )
                        ax.set_ylabel("gid")
                        ax.set_xlabel("t (ms)")
                        ax.set_xlim(time_start, time_end)
                        ax.set_title(_simulation_trial_plot_title(sim_data, trial_idx, f"| {area_name} raster"))
                        if trial_pos == 0 and area_idx == 0:
                            ax.legend(fontsize=7, loc="best")

                    elif plot_type == "firing_rates":
                        area_times = times.get(area_name, {})
                        for pop_i, pop_name in enumerate(sorted(area_times.keys())):
                            tb, rate = _spike_rate(area_times[pop_name], dt, tstop)
                            mask_t = (tb >= time_start) & (tb <= time_end)
                            ax.plot(
                                tb[mask_t],
                                rate[mask_t],
                                linewidth=1.0,
                                color=_population_color(pop_name, pop_i),
                                label=str(pop_name),
                            )
                        ax.set_ylabel(r"$\nu_X$ (spikes/$\Delta t$)")
                        ax.set_xlabel("t (ms)")
                        ax.set_xlim(time_start, time_end)
                        ax.set_title(_simulation_trial_plot_title(sim_data, trial_idx, f"| {area_name} rates"))
                        if trial_pos == 0 and area_idx == 0:
                            ax.legend(fontsize=7, loc="best")

                    elif plot_type == "cdm":
                        cdm = np.asarray(area_series[area_name], dtype=float)
                        dt_local = float(dt_fp) if dt_fp is not None else float(dt)
                        t_cdm = np.arange(cdm.size, dtype=float) * dt_local
                        mask_t = (t_cdm >= time_start) & (t_cdm <= time_end)
                        ax.plot(t_cdm[mask_t], cdm[mask_t], color="C0", linewidth=1.0)
                        ax.set_ylabel("Field potential (a.u.)")
                        ax.set_xlabel("t (ms)")
                        ax.set_xlim(time_start, time_end)
                        ax.set_title(_simulation_trial_plot_title(sim_data, trial_idx, f"| {area_name} field potential"))
                # Hide unused cells in this trial's area grid block.
                for extra_idx in range(len(areas), area_rows * area_cols):
                    grid_row = trial_pos * area_rows + (extra_idx // area_cols)
                    grid_col = extra_idx % area_cols
                    axs[grid_row][grid_col].axis("off")
        else:
            n = len(trial_indices)
            cols = min(3, n)
            rows = int(math.ceil(n / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), dpi=160, squeeze=False)
            flat_ax = axs.ravel()
            for pos, trial_idx in enumerate(trial_indices):
                ax = flat_ax[pos]
                trial = _simulation_trial_data(sim_data, trial_idx)
                times = trial["times"]
                gids = trial["gids"]
                dt = trial["dt"]
                tstop = trial["tstop"]

                if plot_type == "raster":
                    for pop_i, pop_name in enumerate(sorted(times.keys())):
                        t = np.asarray(times[pop_name])
                        g = np.asarray(gids[pop_name])
                        mask_t = (t >= time_start) & (t <= time_end)
                        ax.plot(t[mask_t], g[mask_t], ".", ms=1.0, color=_population_color(pop_name, pop_i), alpha=0.7, label=str(pop_name))
                    ax.set_ylabel("gid")
                    ax.set_xlabel("t (ms)")
                    ax.set_title(_simulation_trial_plot_title(sim_data, trial_idx, "raster"))
                    if pos == 0:
                        ax.legend(fontsize=7, loc="best")
                    ax.set_xlim(time_start, time_end)

                elif plot_type == "firing_rates":
                    for pop_i, pop_name in enumerate(sorted(times.keys())):
                        tb, rate = _spike_rate(times[pop_name], dt, tstop)
                        mask_t = (tb >= time_start) & (tb <= time_end)
                        ax.plot(tb[mask_t], rate[mask_t], linewidth=1.0, color=_population_color(pop_name, pop_i), label=str(pop_name))
                    ax.set_ylabel(r"$\nu_X$ (spikes/$\Delta t$)")
                    ax.set_xlabel("t (ms)")
                    ax.set_title(_simulation_trial_plot_title(sim_data, trial_idx, "firing rates"))
                    if pos == 0:
                        ax.legend(fontsize=7, loc="best")
                    ax.set_xlim(time_start, time_end)

                elif plot_type == "cdm":
                    series_by_area, dt_fp = _field_potential_payload_to_area_series(
                        cdm_payload,
                        model,
                        [],
                        trial_idx,
                    )
                    cdm = np.asarray(series_by_area.get("global", []), dtype=float)
                    if cdm.size == 0:
                        continue
                    dt_local = float(dt_fp) if dt_fp is not None else float(dt)
                    t_cdm = np.arange(cdm.size, dtype=float) * dt_local
                    mask_t = (t_cdm >= time_start) & (t_cdm <= time_end)
                    ax.plot(t_cdm[mask_t], cdm[mask_t], color="C0", linewidth=1.0)
                    ax.set_ylabel("Field potential (a.u.)")
                    ax.set_xlabel("t (ms)")
                    ax.set_title(_simulation_trial_plot_title(sim_data, trial_idx, "field potential"))
                    ax.set_xlim(time_start, time_end)

            for pos in range(len(trial_indices), len(flat_ax)):
                flat_ax[pos].axis("off")

        fig.tight_layout()
        pretty_name = {
            "raster": "Raster plots",
            "firing_rates": "Firing rates",
            "cdm": "Field potential",
        }.get(plot_type, plot_type)
        selected_info = f" Selected files: {len(selected_keys)}." if selected_keys else ""
        source_info = f" Source: {cdm_payload_name}." if plot_type == "cdm" and cdm_payload_name else ""
        subtitle = (
            f"{pretty_name}. Model: {model}. Trials {range_start} to {range_end} "
            f"({time_start:g}-{time_end:g} ms).{repeat_display_note}{selected_info}{source_info}"
        )

    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=160)
    plt.close(fig)
    output.seek(0)
    return _render_analysis_plot(
        title="Simulation outputs",
        subtitle=subtitle,
        image_bytes=output.getvalue(),
        log_output="",
        simulation_repeat_controls=simulation_repeat_controls,
    )


@app.route("/clear_analysis_data", methods=["POST"])
def clear_analysis_data():
    removed_files = _clear_analysis_data_files()
    _clear_analysis_selection_mode()
    _clear_analysis_selected_simulation_keys()
    accept_header = (request.headers.get("Accept") or "").lower()
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    if is_ajax or "application/json" in accept_header:
        return jsonify({"ok": True, "removed_files": removed_files})
    return redirect(request.referrer or url_for('dashboard'))


@app.route("/start_computation_redirect/<computation_type>", methods=["POST"])
def start_computation_redirect(computation_type):
    """Starts the background job and redirects to the status page."""
    # Allowed function names to redirect to
    allowed_functions = {
        'features',
        'inference',
        'inference_training',
        'analysis',
        'field_potential_proxy',
        'field_potential_kernel',
        'field_potential_meeg',
    }

    if computation_type not in allowed_functions:
        return f"Type of computation is not valid", 400
    _remember_path_history_from_form(request.form, f"start_computation_redirect:{computation_type}")

    # Allocate job id early so we can correlate upload/validation logs even before redirect.
    job_id = str(uuid.uuid4())
    route_started = time.perf_counter()
    app.logger.warning(
        "[compute %s] start route=%s content_length=%s",
        job_id,
        computation_type,
        request.content_length,
    )
    if (
        computation_type == "features"
        and request.content_length is not None
        and request.content_length > MAX_EMPIRICAL_UPLOAD_BYTES
    ):
        app.logger.warning(
            "[compute %s] rejecting oversized features request before form parsing: %.2f GB",
            job_id,
            request.content_length / float(1024 ** 3),
        )
        max_gb = MAX_EMPIRICAL_UPLOAD_BYTES / float(1024 ** 3)
        flash(
            f"Request payload is too large ({request.content_length / float(1024 ** 3):.2f} GB). "
            f"Use at most {max_gb:.2f} GB for browser upload or use server folder path mode.",
            "error",
        )
        return redirect(request.referrer or url_for("features_methods"))

    # Build the name of the function to compute depending on the page form this function was called from
    func_name_string = f"{computation_type}_computation"
    func = getattr(compute_utils, func_name_string) # filtered function name for security reasons

    files = None
    uploaded_files = []
    inference_features_source_mode = "upload"
    inference_model_assets_source = "upload"
    inference_features_server_path = ""
    inference_assets_server_files = {}
    inference_training_features_source_mode = "upload"
    inference_training_parameters_source_mode = "upload"
    inference_training_features_server_path = ""
    inference_training_parameters_server_path = ""

    def _ensure_files_parsed():
        nonlocal files, uploaded_files
        if files is None:
            files_parse_started = time.perf_counter()
            files = request.files
            files_parse_ms = (time.perf_counter() - files_parse_started) * 1000.0
            app.logger.warning(
                "[compute %s] request.files parsed in %.1f ms (keys=%d)",
                job_id,
                files_parse_ms,
                len(list(files.keys())),
            )
            uploaded_files = [f for f in files.values() if f.filename]
        return files

    # File filter and checks for every computation type
    if computation_type == 'features':
        try:
            app_sig = inspect.signature(ncpi.Features.compute_features)
        except Exception:
            app_sig = "unavailable"
        try:
            worker_sig = inspect.signature(compute_utils.ncpi.Features.compute_features)
        except Exception:
            worker_sig = "unavailable"
        app.logger.warning(
            "[compute %s] ncpi app=%s sig=%s | worker=%s sig=%s",
            job_id,
            getattr(ncpi, "__file__", "unknown"),
            app_sig,
            getattr(compute_utils.ncpi, "__file__", "unknown"),
            worker_sig,
        )

        selected_method = (request.form.get("select-method") or "").strip()
        app.logger.warning("[compute %s] features selected_method=%s", job_id, selected_method)
        valid_feature_methods = {"catch22", "specparam", "dfa", "fEI", "hctsa", "power_spectrum_parameterization", "fei"}
        if selected_method not in valid_feature_methods:
            flash('Select a valid features method before computing.', 'error')
            return redirect(request.referrer or url_for('features_methods'))
        if selected_method == "hctsa":
            hctsa_folder = (request.form.get("hctsa_folder") or "").strip()
            if not hctsa_folder:
                flash('hctsa_folder is required for hctsa.', 'error')
                return redirect(request.referrer or url_for('features_methods'))

        data_source_kind = (request.form.get("data_source_kind") or "new-simulation").strip()
        app.logger.warning("[compute %s] features data_source_kind=%s", job_id, data_source_kind)
        empirical_source_mode = (request.form.get("empirical_source_mode") or "upload").strip()
        existing_data_path = (request.form.get("existing_data_path") or "").strip()
        if not (request.form.get("parser_data_locator") or "").strip():
            flash('Select the data locator for EphysDatasetParser.', 'error')
            return redirect(request.referrer or url_for('features_methods'))
        epoching_enabled = str(request.form.get("parser_enable_epoching", "")).lower() in {"1", "true", "on", "yes"}
        if epoching_enabled:
            if _optional_float(request.form.get("parser_epoch_length_s")) is None:
                flash('Set an epoch length in seconds.', 'error')
                return redirect(request.referrer or url_for('features_methods'))
            if _optional_float(request.form.get("parser_epoch_step_s")) is None:
                flash('Set an epoch step in seconds.', 'error')
                return redirect(request.referrer or url_for('features_methods'))

        if data_source_kind == "pipeline":
            if uploaded_files:
                flash('Do not upload files when using "Continue simulation pipeline".', 'error')
                return redirect(request.referrer or url_for('features_methods'))
            if not existing_data_path:
                flash('Select a detected simulation pipeline file to continue.', 'error')
                return redirect(request.referrer or url_for('features_methods'))
            fs_source = (request.form.get("parser_fs_source") or "").strip()
            fs_manual = _optional_float(request.form.get("parser_fs_manual"))
            if fs_source == "__numeric__":
                if fs_manual is None:
                    flash('Provide sampling frequency numeric value.', 'error')
                    return redirect(request.referrer or url_for('features_methods'))
            elif fs_source == "__none__":
                pass
            elif fs_source:
                pass
            else:
                fs_locator = (request.form.get("parser_fs_locator") or "").strip()
                if fs_locator and fs_manual is not None:
                    flash('Sampling frequency locator and sampling frequency value are mutually exclusive.', 'error')
                    return redirect(request.referrer or url_for('features_methods'))
                if not fs_locator and fs_manual is None:
                    flash('Provide sampling frequency source (field, numeric value, or None).', 'error')
                    return redirect(request.referrer or url_for('features_methods'))

            recording_type_source = (request.form.get("parser_recording_type_source") or "").strip()
            recording_type_value = (request.form.get("parser_recording_type") or "").strip()
            if recording_type_source == "__value__":
                if not recording_type_value:
                    flash('Select a recording type value.', 'error')
                    return redirect(request.referrer or url_for('features_methods'))
            elif recording_type_source == "__none__":
                pass
            elif recording_type_source:
                pass
            else:
                recording_type_locator = (request.form.get("parser_recording_type_locator") or "").strip()
                if recording_type_locator and recording_type_value:
                    flash('Recording type locator and recording type value are mutually exclusive.', 'error')
                    return redirect(request.referrer or url_for('features_methods'))

            ch_names_source = (request.form.get("parser_ch_names_source") or "").strip()
            manual_sensor_names = _parse_sensor_names(request.form.get("parser_sensor_names"))
            if ch_names_source == "__manual__":
                if not manual_sensor_names:
                    flash('Provide manual channel names (comma-separated).', 'error')
                    return redirect(request.referrer or url_for('features_methods'))
            elif ch_names_source == "__autocomplete__":
                try:
                    _parse_nonnegative_int(
                        request.form.get("parser_ch_names_autocomplete_axis"),
                        default=0,
                        field_name="Autocomplete channel axis",
                    )
                except Exception as exc:
                    flash(str(exc), 'error')
                    return redirect(request.referrer or url_for('features_methods'))
            elif ch_names_source:
                pass
            else:
                ch_names_locator = (request.form.get("parser_ch_names_locator") or "").strip()
                if ch_names_locator and manual_sensor_names:
                    flash('Channel names locator and manual channel names are mutually exclusive.', 'error')
                    return redirect(request.referrer or url_for('features_methods'))
            try:
                _validate_feature_existing_path(existing_data_path)
            except Exception as exc:
                flash(str(exc), 'error')
                return redirect(request.referrer or url_for('features_methods'))
        elif data_source_kind == "new-simulation":
            simulation_folder_paths = _extract_folder_paths_from_form(
                request.form,
                singular_key="simulation_folder_path",
                plural_key="simulation_folder_paths",
            )
            simulation_uploads = [f for f in _ensure_files_parsed().getlist("data_file") if f and f.filename]
            has_upload = len(simulation_uploads) > 0
            if has_upload and simulation_folder_paths:
                flash('Use either a local simulation folder upload or a server simulation folder path, not both.', 'error')
                return redirect(request.referrer or url_for('features_methods'))
            if simulation_folder_paths:
                try:
                    simulation_data_file_selections = _extract_data_file_selection_map_from_form(
                        request.form,
                        "simulation_data_file_selections",
                    )
                    simulation_subfolder_filter_map = _extract_subfolder_filter_map_from_form(
                        request.form,
                        "simulation_subfolder_selections",
                    )
                    simulation_entries, folder_summaries, _, folder_contexts = _collect_supported_folder_file_entries(
                        simulation_folder_paths,
                        "Simulation outputs",
                        data_file_selection_map=simulation_data_file_selections,
                    )
                    selected_analysis_paths = _resolve_selected_analysis_folder_paths(request.form, folder_summaries)
                    selected_set = set(selected_analysis_paths)
                    selected_entries = [
                        entry for entry in simulation_entries
                        if str(entry.get("folder_path") or "") in selected_set
                    ]
                    selected_data_entries = []
                    for context in folder_contexts:
                        folder_path = str(context.get("folder_path") or "").strip()
                        if folder_path not in selected_set:
                            continue
                        selected_data_entries.extend(list(context.get("selected_entries") or []))
                    if simulation_subfolder_filter_map and selected_data_entries:
                        _filtered = []
                        for _entry in selected_data_entries:
                            _fp = str(_entry.get("folder_path") or "").strip()
                            _included = simulation_subfolder_filter_map.get(_fp, [])
                            if not _included:
                                _filtered.append(_entry)
                                continue
                            _sub = str(_entry.get("level1_subfolder") or "").strip()
                            if not _sub or _sub in _included:
                                _filtered.append(_entry)
                        selected_data_entries = _filtered
                    if selected_data_entries:
                        selected_entries = selected_data_entries
                    if not selected_entries:
                        raise ValueError("No files available in the selected analysis folder(s).")
                    app.logger.warning(
                        "[compute %s] simulation folder mode folders=%d selected_folders=%d files=%d selected_files=%d",
                        job_id,
                        len(folder_summaries),
                        len(selected_analysis_paths),
                        len(simulation_entries),
                        len(selected_entries),
                    )
                except Exception as exc:
                    flash(f"Invalid simulation outputs folder path: {exc}", 'error')
                    return redirect(request.referrer or url_for('features_methods'))
            elif has_upload:
                supported_uploads = []
                for upload in simulation_uploads:
                    raw_name = str(upload.filename or "").replace("\\", "/").strip()
                    parts = [part for part in raw_name.split("/") if part not in {"", "."}]
                    if len(parts) > 2:
                        flash(
                            f"Local simulation upload supports only flat folders. Nested path detected: {raw_name}",
                            'error',
                        )
                        return redirect(request.referrer or url_for('features_methods'))
                    ext = Path(str(upload.filename or "")).suffix.lower()
                    if ext in FEATURES_PARSER_FILE_EXTENSIONS:
                        supported_uploads.append(upload)
                app.logger.warning(
                    "[compute %s] simulation local upload count=%d supported=%d",
                    job_id,
                    len(simulation_uploads),
                    len(supported_uploads),
                )
                if not supported_uploads:
                    flash('No supported simulation output files were found in the selected local folder.', 'error')
                    return redirect(request.referrer or url_for('features_methods'))
            else:
                flash('Provide a simulation outputs folder path or upload a local simulation outputs folder.', 'error')
                return redirect(request.referrer or url_for('features_methods'))
        elif data_source_kind == "new-empirical":
            if empirical_source_mode == "server-path":
                empirical_folder_paths = _extract_folder_paths_from_form(
                    request.form,
                    singular_key="empirical_folder_path",
                    plural_key="empirical_folder_paths",
                )
                try:
                    empirical_data_file_selections = _extract_data_file_selection_map_from_form(
                        request.form,
                        "empirical_data_file_selections",
                    )
                    empirical_subfolder_filter_map_pre = _extract_subfolder_filter_map_from_form(
                        request.form,
                        "empirical_subfolder_selections",
                    )
                    empirical_entries, folder_summaries, _, folder_contexts = _collect_supported_folder_file_entries(
                        empirical_folder_paths,
                        "Empirical",
                        data_file_selection_map=empirical_data_file_selections,
                    )
                    selected_analysis_paths = _resolve_selected_analysis_folder_paths(request.form, folder_summaries)
                    selected_set = set(selected_analysis_paths)
                    selected_entries = [
                        entry for entry in empirical_entries
                        if str(entry.get("folder_path") or "") in selected_set
                    ]
                    selected_data_entries = []
                    for context in folder_contexts:
                        folder_path = str(context.get("folder_path") or "").strip()
                        if folder_path not in selected_set:
                            continue
                        selected_data_entries.extend(list(context.get("selected_entries") or []))
                    if empirical_subfolder_filter_map_pre and selected_data_entries:
                        _filtered = []
                        for _entry in selected_data_entries:
                            _fp = str(_entry.get("folder_path") or "").strip()
                            _included = empirical_subfolder_filter_map_pre.get(_fp, [])
                            if not _included:
                                _filtered.append(_entry)
                                continue
                            _sub = str(_entry.get("level1_subfolder") or "").strip()
                            if not _sub or _sub in _included:
                                _filtered.append(_entry)
                        selected_data_entries = _filtered
                    if selected_data_entries:
                        selected_entries = selected_data_entries
                    if not selected_entries:
                        raise ValueError("No files available in the selected analysis folder(s).")
                    app.logger.warning(
                        "[compute %s] empirical folder mode folders=%d selected_folders=%d files=%d selected_files=%d",
                        job_id,
                        len(folder_summaries),
                        len(selected_analysis_paths),
                        len(empirical_entries),
                        len(selected_entries),
                    )
                except Exception as exc:
                    flash(f"Invalid empirical folder path: {exc}", 'error')
                    return redirect(request.referrer or url_for('features_methods'))
            else:
                if request.content_length and request.content_length > MAX_EMPIRICAL_UPLOAD_BYTES:
                    max_gb = MAX_EMPIRICAL_UPLOAD_BYTES / float(1024 ** 3)
                    flash(
                        f"Empirical upload too large ({request.content_length / float(1024 ** 3):.2f} GB). "
                        f"Use at most {max_gb:.2f} GB or switch to server folder path mode.",
                        'error',
                    )
                    return redirect(request.referrer or url_for('features_methods'))
                empirical_uploads = [f for f in _ensure_files_parsed().getlist("empirical_files") if f and f.filename]
                for upload in empirical_uploads:
                    raw_name = str(upload.filename or "").replace("\\", "/").strip()
                    parts = [part for part in raw_name.split("/") if part not in {"", "."}]
                    if len(parts) > 2:
                        flash(
                            f"Local empirical upload supports only flat folders. Nested path detected: {raw_name}",
                            'error',
                        )
                        return redirect(request.referrer or url_for('features_methods'))
                app.logger.warning("[compute %s] empirical upload count=%d", job_id, len(empirical_uploads))
                if not empirical_uploads:
                    flash('Upload at least one empirical recording file (folder upload is supported).', 'error')
                    return redirect(request.referrer or url_for('features_methods'))
        else:
            flash('Unknown data source for features computation.', 'error')
            return redirect(request.referrer or url_for('features_methods'))
        estimated_time_remaining = None

    if computation_type != 'features':
        _ensure_files_parsed()
    if computation_type != 'features' and len(uploaded_files) == 0 and computation_type not in {'inference', 'inference_training', 'field_potential_proxy', 'field_potential_kernel', 'field_potential_meeg'}:
        flash('No files uploaded, please try again.', 'error')
        return redirect(request.referrer)

    if computation_type == 'inference_training':
        files_obj = _ensure_files_parsed()

        def _has_upload(*keys):
            for key in keys:
                upload = files_obj.get(key)
                if upload is not None and bool(upload.filename):
                    return True
            return False

        inference_training_features_source_mode = (request.form.get("training_features_source_mode") or "upload").strip().lower()
        inference_training_parameters_source_mode = (request.form.get("training_parameters_source_mode") or "upload").strip().lower()
        training_features_server_file_path = (request.form.get("training_features_server_file_path") or "").strip()
        training_parameters_server_file_path = (request.form.get("training_parameters_server_file_path") or "").strip()

        has_features = _has_upload("training_features_file", "file-upload-x", "features_train_file")
        has_parameters = _has_upload("training_parameters_file", "file-upload-y", "parameters_train_file")
        has_training_features_server_path = bool(training_features_server_file_path)
        has_training_parameters_server_path = bool(training_parameters_server_file_path)

        if inference_training_features_source_mode not in {"upload", "server-path"}:
            inference_training_features_source_mode = "server-path" if has_training_features_server_path else "upload"
        if inference_training_parameters_source_mode not in {"upload", "server-path"}:
            inference_training_parameters_source_mode = "server-path" if has_training_parameters_server_path else "upload"

        # Fallback: infer source mode from provided fields when UI mode sync fails.
        if inference_training_features_source_mode == "upload" and not has_features and has_training_features_server_path:
            inference_training_features_source_mode = "server-path"
        if inference_training_features_source_mode == "server-path" and not has_training_features_server_path and has_features:
            inference_training_features_source_mode = "upload"

        if inference_training_parameters_source_mode == "upload" and not has_parameters and has_training_parameters_server_path:
            inference_training_parameters_source_mode = "server-path"
        if inference_training_parameters_source_mode == "server-path" and not has_training_parameters_server_path and has_parameters:
            inference_training_parameters_source_mode = "upload"

        if inference_training_features_source_mode == "server-path":
            try:
                inference_training_features_server_path = _validate_existing_file_path(
                    training_features_server_file_path,
                    "Training features server file",
                )
            except Exception as exc:
                flash(str(exc), 'error')
                return redirect(request.referrer or url_for('new_training'))
        elif not has_features:
            flash('Provide Features Data for training (local upload or server file).', 'error')
            return redirect(request.referrer or url_for('new_training'))

        if inference_training_parameters_source_mode == "server-path":
            try:
                inference_training_parameters_server_path = _validate_existing_file_path(
                    training_parameters_server_file_path,
                    "Training parameters server file",
                )
            except Exception as exc:
                flash(str(exc), 'error')
                return redirect(request.referrer or url_for('new_training'))
        elif not has_parameters:
            flash('Provide Parameters Data for training (local upload or server file).', 'error')
            return redirect(request.referrer or url_for('new_training'))

        model_name = (request.form.get("training_model_name") or "").strip()
        if model_name == "__custom__":
            model_name = (request.form.get("training_model_name_custom") or "").strip()
        if not model_name:
            flash('Select a training model.', 'error')
            return redirect(request.referrer or url_for('new_training'))

        estimated_time_remaining = time.time() + 300

    if computation_type == 'inference':
        existing_features_file = (request.form.get("existing_features_file") or "").strip()
        has_existing_features = existing_features_file in set(_list_features_data_files()) if existing_features_file else False

        files_obj = _ensure_files_parsed()

        def _has_upload(*keys):
            for key in keys:
                upload = files_obj.get(key)
                if upload is not None and bool(upload.filename):
                    return True
            return False

        inference_features_source_mode = (request.form.get("features_source_mode") or "upload").strip().lower()
        inference_model_assets_source = (request.form.get("model_assets_source") or "upload").strip().lower()
        features_server_file_path = (request.form.get("features_server_file_path") or "").strip()
        inference_assets_server_folder_path = (request.form.get("inference_assets_server_folder_path") or "").strip()
        inference_model_server_file_path = (request.form.get("inference_model_server_file_path") or "").strip()
        inference_scaler_server_file_path = (request.form.get("inference_scaler_server_file_path") or "").strip()
        inference_density_server_file_path = (request.form.get("inference_density_server_file_path") or "").strip()

        has_uploaded_features = _has_upload("features_predict_file", "features_predict", "file-upload-features")
        has_server_features_path = bool(features_server_file_path)
        if inference_features_source_mode not in {"upload", "server-path"}:
            inference_features_source_mode = "server-path" if has_server_features_path else "upload"
        # Fallback: infer source mode from provided fields when UI mode sync fails.
        if inference_features_source_mode == "upload" and not has_uploaded_features and has_server_features_path:
            inference_features_source_mode = "server-path"
        if inference_features_source_mode == "server-path" and not has_server_features_path and has_uploaded_features:
            inference_features_source_mode = "upload"

        if inference_features_source_mode == "server-path":
            try:
                inference_features_server_path = _validate_existing_pickle_file_path(
                    features_server_file_path,
                    "Inference features server file",
                )
            except Exception as exc:
                flash(str(exc), 'error')
                return redirect(request.referrer or url_for('inference'))
        else:
            if not has_uploaded_features and not has_existing_features:
                flash('Upload features data or use an auto-loaded features file.', 'error')
                return redirect(request.referrer or url_for('inference'))

        has_uploaded_model = _has_upload("model_file", "model-file", "file-upload-model")
        has_uploaded_scaler = _has_upload("scaler_file", "scaler-file", "file-upload-scaler")
        has_uploaded_density = _has_upload("density_estimator_file", "density-estimator-file", "file-upload-density-estimator")
        folder_uploads = [f for f in files_obj.getlist("inference_assets_folder") if f and f.filename]
        has_folder_uploads = bool(folder_uploads)
        has_individual_assets = has_uploaded_model or has_uploaded_scaler or has_uploaded_density
        has_server_assets_folder_path = bool(inference_assets_server_folder_path)
        has_server_asset_file_paths = bool(
            inference_model_server_file_path
            or inference_scaler_server_file_path
            or inference_density_server_file_path
        )
        assets_upload_mode = (request.form.get("assets_upload_mode") or "").strip().lower()
        if assets_upload_mode not in {"individual", "folder"}:
            assets_upload_mode = "individual"
        inference_assets_server_files = {}

        if inference_model_assets_source not in {"upload", "server-path"}:
            inference_model_assets_source = "server-path" if (has_server_assets_folder_path or has_server_asset_file_paths) else "upload"
        if (
            inference_model_assets_source == "upload"
            and not has_folder_uploads
            and not has_individual_assets
            and (has_server_assets_folder_path or has_server_asset_file_paths)
        ):
            inference_model_assets_source = "server-path"
        if (
            inference_model_assets_source == "server-path"
            and not has_server_assets_folder_path
            and not has_server_asset_file_paths
            and (has_folder_uploads or has_individual_assets)
        ):
            inference_model_assets_source = "upload"

        if inference_model_assets_source == "server-path":
            if assets_upload_mode == "individual" and has_server_assets_folder_path and not has_server_asset_file_paths:
                assets_upload_mode = "folder"
            if assets_upload_mode == "folder" and not has_server_assets_folder_path and has_server_asset_file_paths:
                assets_upload_mode = "individual"

        if inference_model_assets_source == "server-path":
            if assets_upload_mode == "folder":
                try:
                    inference_assets_server_files = _collect_inference_assets_folder_files(
                        inference_assets_server_folder_path
                    )
                except Exception as exc:
                    flash(str(exc), 'error')
                    return redirect(request.referrer or url_for('inference'))
                if "model_file" not in inference_assets_server_files:
                    flash(
                        'Server assets folder must contain a model/posterior file (e.g. model.* or posterior.*).',
                        'error',
                    )
                    return redirect(request.referrer or url_for('inference'))
            else:
                try:
                    inference_assets_server_files["model_file"] = _validate_existing_file_path(
                        inference_model_server_file_path,
                        "Inference server model file",
                    )
                except Exception as exc:
                    flash(str(exc), 'error')
                    return redirect(request.referrer or url_for('inference'))
                if inference_scaler_server_file_path:
                    try:
                        inference_assets_server_files["scaler_file"] = _validate_existing_file_path(
                            inference_scaler_server_file_path,
                            "Inference server scaler file",
                        )
                    except Exception as exc:
                        flash(str(exc), 'error')
                        return redirect(request.referrer or url_for('inference'))
                if inference_density_server_file_path:
                    try:
                        inference_assets_server_files["density_estimator_file"] = _validate_existing_file_path(
                            inference_density_server_file_path,
                            "Inference server density estimator file",
                        )
                    except Exception as exc:
                        flash(str(exc), 'error')
                        return redirect(request.referrer or url_for('inference'))
        else:
            if has_folder_uploads and has_individual_assets:
                flash('Use only one assets upload mode: either individual files or one folder.', 'error')
                return redirect(request.referrer or url_for('inference'))
            if assets_upload_mode == "folder" and has_individual_assets:
                flash('Assets upload mode is set to folder, so individual asset files are not allowed.', 'error')
                return redirect(request.referrer or url_for('inference'))
            if assets_upload_mode == "individual" and has_folder_uploads:
                flash('Assets upload mode is set to individual files, so folder upload is not allowed.', 'error')
                return redirect(request.referrer or url_for('inference'))

            folder_has_model = False
            for folder_upload in folder_uploads:
                if _infer_inference_asset_role(folder_upload.filename or "") == "model_file":
                    folder_has_model = True
                    break
            if not has_uploaded_model and not folder_has_model:
                flash('Upload a model file, or upload a folder that contains a model/posterior file (e.g. model.* or posterior.*).', 'error')
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

    module_upload_dir = _module_uploads_dir_for(computation_type)
    os.makedirs(module_upload_dir, exist_ok=True)

    # If everything is OK, save/prepare the file(s)
    file_paths = {}
    kernel_params_module_override = None
    prepared_features_df = None
    empirical_upload_paths = None
    parser_config_obj = None
    if computation_type == "features":
        data_source_kind = (request.form.get("data_source_kind") or "new-simulation").strip()
        empirical_source_mode = (request.form.get("empirical_source_mode") or "upload").strip()

        if data_source_kind == "pipeline":
            existing_path = _validate_feature_existing_path(request.form.get("existing_data_path"))
            copied_name = f"features_data_file_0_{job_id}_{os.path.basename(existing_path)}"
            copied_path = os.path.join(module_upload_dir, copied_name)
            shutil.copy2(existing_path, copied_path)
            try:
                normalized_path = _normalize_features_input_path(
                    copied_path,
                    request.form,
                    job_id,
                    upload_dir=module_upload_dir,
                )
                if normalized_path != copied_path and os.path.exists(copied_path):
                    os.remove(copied_path)
                file_paths["data_file"] = normalized_path
            except Exception as exc:
                flash(f"Failed to prepare selected pipeline file: {exc}", "error")
                return redirect(request.referrer or url_for('features_methods'))

        elif data_source_kind == "new-simulation":
            try:
                cfg_started = time.perf_counter()
                parse_cfg = _build_parse_config_from_form(request.form)
                app.logger.warning(
                    "[compute %s] simulation parser config built in %.1f ms",
                    job_id,
                    (time.perf_counter() - cfg_started) * 1000.0,
                )
                parser_config_obj = parse_cfg
                empirical_upload_paths = []

                simulation_folder_paths = _extract_folder_paths_from_form(
                    request.form,
                    singular_key="simulation_folder_path",
                    plural_key="simulation_folder_paths",
                )
                if simulation_folder_paths:
                    simulation_data_file_selections = _extract_data_file_selection_map_from_form(
                        request.form,
                        "simulation_data_file_selections",
                    )
                    source_entries, folder_summaries, _, folder_contexts = _collect_supported_folder_file_entries(
                        simulation_folder_paths,
                        "Simulation outputs",
                        data_file_selection_map=simulation_data_file_selections,
                    )
                    selected_analysis_paths = _resolve_selected_analysis_folder_paths(request.form, folder_summaries)
                    selected_path_set = set(selected_analysis_paths)
                    selected_entries = [
                        entry for entry in source_entries
                        if str(entry.get("folder_path") or "") in selected_path_set
                    ]
                    selected_data_entries = []
                    for context in folder_contexts:
                        folder_path = str(context.get("folder_path") or "").strip()
                        if folder_path not in selected_path_set:
                            continue
                        selected_data_entries.extend(list(context.get("selected_entries") or []))
                    if selected_data_entries:
                        selected_entries = selected_data_entries
                    if not selected_entries:
                        raise ValueError("No files available in the selected analysis folder(s).")
                    use_prefix = len(folder_summaries) > 1
                    for entry in selected_entries:
                        logical_name = _prefixed_file_name(
                            entry["name"],
                            entry.get("folder_name"),
                            apply_prefix=use_prefix,
                        )
                        empirical_upload_paths.append({
                            "name": logical_name,
                            "ext": entry["extension"],
                            "path": entry["path"],
                            "folder_name": entry.get("folder_name"),
                            "folder_path": entry.get("folder_path"),
                        })
                    app.logger.warning(
                        "[compute %s] using simulation server-path folders=%d files=%d",
                        job_id,
                        len(folder_summaries),
                        len(empirical_upload_paths),
                    )
                else:
                    uploads = [f for f in _ensure_files_parsed().getlist("data_file") if f and f.filename]
                    upload_items = []
                    local_folder_tokens = []
                    local_folder_token_set = set()
                    for upload in uploads:
                        raw_name = str(upload.filename or "").strip().replace("\\", "/")
                        parts = [part for part in raw_name.split("/") if part not in {"", "."}]
                        if not parts or any(part == ".." for part in parts):
                            continue
                        if len(parts) > 2:
                            raise ValueError(
                                f"Local simulation upload supports only flat folders. "
                                f"Nested path detected: {raw_name}"
                            )
                        top_folder = secure_filename(parts[0]).strip("_") if len(parts) > 1 else ""
                        folder_token = top_folder or "selected_folder"
                        safe_name = secure_filename(parts[-1])
                        if not safe_name:
                            continue
                        upload_items.append((upload, safe_name, top_folder, folder_token))
                        if folder_token not in local_folder_token_set:
                            local_folder_token_set.add(folder_token)
                            local_folder_tokens.append(folder_token)

                    supported_upload_items = [
                        row for row in upload_items
                        if Path(str(row[1] or "")).suffix.lower() in FEATURES_PARSER_FILE_EXTENSIONS
                    ]
                    selected_analysis_name_tokens = _extract_analysis_folder_name_tokens_from_form(request.form)
                    if len(selected_analysis_name_tokens) > 1:
                        raise ValueError("Select only one folder for feature extraction.")
                    if "parser_analysis_folder_names" in request.form and len(local_folder_tokens) > 1 and not selected_analysis_name_tokens:
                        raise ValueError("Select one folder for feature extraction.")
                    selected_token = selected_analysis_name_tokens[0] if selected_analysis_name_tokens else (
                        local_folder_tokens[0] if local_folder_tokens else None
                    )
                    if selected_token and selected_token not in local_folder_token_set:
                        raise ValueError(
                            f"Unknown local folder selected for feature extraction: {selected_token}"
                        )
                    if selected_token:
                        upload_items = [row for row in upload_items if row[3] == selected_token]
                    if not upload_items:
                        raise ValueError("No files available in the selected analysis folder(s).")
                    use_prefix = len(local_folder_tokens) > 1

                    save_started = time.perf_counter()
                    saved_bytes = 0
                    for idx, (upload, safe_name, top_folder, folder_token) in enumerate(upload_items):
                        ext = Path(safe_name).suffix.lower()
                        if ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                            continue
                        logical_name = _prefixed_file_name(
                            safe_name,
                            top_folder,
                            apply_prefix=use_prefix and bool(top_folder),
                        )
                        unique_filename = f"features_simulation_file_{idx}_{job_id}_{safe_name}"
                        file_path = os.path.join(module_upload_dir, unique_filename)
                        save_one_started = time.perf_counter()
                        upload.save(file_path)
                        save_one_ms = (time.perf_counter() - save_one_started) * 1000.0
                        if not os.path.exists(file_path) or os.path.getsize(file_path) <= 0:
                            continue
                        file_size = os.path.getsize(file_path)
                        saved_bytes += file_size
                        file_paths[f"simulation_file_{idx}"] = file_path
                        empirical_upload_paths.append({
                            "name": logical_name,
                            "ext": ext,
                            "path": file_path,
                            "folder_name": folder_token,
                        })
                        if idx == 0 or (idx + 1) % 25 == 0 or (idx + 1) == len(upload_items):
                            app.logger.warning(
                                "[compute %s] saved simulation file %d/%d (%.2f MB, %.1f ms, cumulative %.2f MB)",
                                job_id,
                                idx + 1,
                                len(upload_items),
                                file_size / (1024 * 1024),
                                save_one_ms,
                                saved_bytes / (1024 * 1024),
                            )
                    app.logger.warning(
                        "[compute %s] simulation staging finished: files=%d total=%.2f MB in %.1f s",
                        job_id,
                        len(empirical_upload_paths),
                        saved_bytes / (1024 * 1024),
                        time.perf_counter() - save_started,
                    )

                if not empirical_upload_paths:
                    raise ValueError("No supported simulation output files were provided.")
                _validate_data_locator_against_source_path(
                    parse_cfg.fields.data,
                    empirical_upload_paths[0]["path"],
                    source_label="selected analysis sample",
                )
            except Exception as exc:
                app.logger.exception("[compute %s] simulation staging failed", job_id)
                flash(f"Simulation parser configuration failed: {exc}", "error")
                return redirect(request.referrer or url_for('features_methods'))

        elif data_source_kind == "new-empirical":
            try:
                cfg_started = time.perf_counter()
                parse_cfg = _build_parse_config_from_form(request.form)
                app.logger.warning(
                    "[compute %s] parser config built in %.1f ms",
                    job_id,
                    (time.perf_counter() - cfg_started) * 1000.0,
                )
                parser_config_obj = parse_cfg
                empirical_upload_paths = []
                if empirical_source_mode == "server-path":
                    empirical_folder_paths = _extract_folder_paths_from_form(
                        request.form,
                        singular_key="empirical_folder_path",
                        plural_key="empirical_folder_paths",
                    )
                    empirical_data_file_selections = _extract_data_file_selection_map_from_form(
                        request.form,
                        "empirical_data_file_selections",
                    )
                    empirical_subfolder_filter_map = _extract_subfolder_filter_map_from_form(
                        request.form,
                        "empirical_subfolder_selections",
                    )
                    source_entries, folder_summaries, _, folder_contexts = _collect_supported_folder_file_entries(
                        empirical_folder_paths,
                        "Empirical",
                        data_file_selection_map=empirical_data_file_selections,
                    )
                    selected_analysis_paths = _resolve_selected_analysis_folder_paths(request.form, folder_summaries)
                    selected_path_set = set(selected_analysis_paths)
                    selected_entries = [
                        entry for entry in source_entries
                        if str(entry.get("folder_path") or "") in selected_path_set
                    ]
                    selected_data_entries = []
                    for context in folder_contexts:
                        folder_path = str(context.get("folder_path") or "").strip()
                        if folder_path not in selected_path_set:
                            continue
                        selected_data_entries.extend(list(context.get("selected_entries") or []))
                    if empirical_subfolder_filter_map and selected_data_entries:
                        _filtered = []
                        for _entry in selected_data_entries:
                            _fp = str(_entry.get("folder_path") or "").strip()
                            _included = empirical_subfolder_filter_map.get(_fp, [])
                            if not _included:
                                _filtered.append(_entry)
                                continue
                            _sub = str(_entry.get("level1_subfolder") or "").strip()
                            if not _sub or _sub in _included:
                                _filtered.append(_entry)
                        selected_data_entries = _filtered
                    if selected_data_entries:
                        selected_entries = selected_data_entries
                    if not selected_entries:
                        raise ValueError("No files available in the selected analysis folder(s).")
                    use_prefix = len(folder_summaries) > 1
                    for entry in selected_entries:
                        logical_name = _prefixed_file_name(
                            entry["name"],
                            entry.get("folder_name"),
                            apply_prefix=use_prefix,
                        )
                        empirical_upload_paths.append({
                            "name": logical_name,
                            "ext": entry["extension"],
                            "path": entry["path"],
                            "folder_name": entry.get("folder_name"),
                            "folder_path": entry.get("folder_path"),
                        })
                    app.logger.warning(
                        "[compute %s] using empirical server-path folders=%d files=%d",
                        job_id,
                        len(folder_summaries),
                        len(empirical_upload_paths),
                    )
                else:
                    empirical_uploads = [f for f in _ensure_files_parsed().getlist("empirical_files") if f and f.filename]
                    upload_items = []
                    local_folder_tokens = []
                    local_folder_token_set = set()
                    for upload in empirical_uploads:
                        raw_name = str(upload.filename or "").strip().replace("\\", "/")
                        parts = [part for part in raw_name.split("/") if part not in {"", "."}]
                        if not parts or any(part == ".." for part in parts):
                            continue
                        if len(parts) > 2:
                            raise ValueError(
                                f"Local empirical upload supports only flat folders. "
                                f"Nested path detected: {raw_name}"
                            )
                        top_folder = secure_filename(parts[0]).strip("_") if len(parts) > 1 else ""
                        folder_token = top_folder or "selected_folder"
                        safe_name = secure_filename(parts[-1])
                        if not safe_name:
                            continue
                        upload_items.append((upload, safe_name, top_folder, folder_token))
                        if folder_token not in local_folder_token_set:
                            local_folder_token_set.add(folder_token)
                            local_folder_tokens.append(folder_token)

                    supported_upload_items = [
                        row for row in upload_items
                        if Path(str(row[1] or "")).suffix.lower() in FEATURES_PARSER_FILE_EXTENSIONS
                    ]
                    selected_analysis_name_tokens = _extract_analysis_folder_name_tokens_from_form(request.form)
                    if len(selected_analysis_name_tokens) > 1:
                        raise ValueError("Select only one folder for feature extraction.")
                    if "parser_analysis_folder_names" in request.form and len(local_folder_tokens) > 1 and not selected_analysis_name_tokens:
                        raise ValueError("Select one folder for feature extraction.")
                    selected_token = selected_analysis_name_tokens[0] if selected_analysis_name_tokens else (
                        local_folder_tokens[0] if local_folder_tokens else None
                    )
                    if selected_token and selected_token not in local_folder_token_set:
                        raise ValueError(
                            f"Unknown local folder selected for feature extraction: {selected_token}"
                        )

                    # Before folder filtering, try to resolve companion channel-names file
                    # from any uploaded subfolder (e.g. channels/channels.mat alongside data/).
                    if (
                        getattr(parse_cfg, "fields", None) is not None
                        and isinstance(getattr(parse_cfg.fields, "ch_names", None), str)
                        and parse_cfg.fields.ch_names not in {"__self__", ""}
                    ):
                        _ch_loc = parse_cfg.fields.ch_names
                        from compute_utils import _load_uploaded_source_path, _flatten_matlab_ch_names
                        for _up, _sname, _tf, _ftok in upload_items:
                            if os.path.splitext(_sname)[0].lower() != _ch_loc.lower():
                                continue
                            _ext = Path(_sname).suffix.lower()
                            if _ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                                continue
                            _tmp = os.path.join(module_upload_dir, f"companion_{job_id}_{secure_filename(_sname)}")
                            try:
                                _up.seek(0)
                                _up.save(_tmp)
                                _obj = _load_uploaded_source_path(_tmp, _sname, _ext)
                                if isinstance(_obj, dict):
                                    _key = _ch_loc if _ch_loc in _obj else next(
                                        (k for k in _obj if isinstance(k, str) and not k.startswith("_") and k.lower() == _ch_loc.lower()),
                                        None,
                                    )
                                    if _key:
                                        _names = _flatten_matlab_ch_names(_obj[_key])
                                        if _names:
                                            parse_cfg = _copy_parse_config(
                                                parse_cfg,
                                                fields=_copy_canonical_fields(parse_cfg.fields, ch_names=_names),
                                            )
                                            parser_config_obj = parse_cfg
                            except Exception:
                                pass
                            finally:
                                try:
                                    if os.path.exists(_tmp):
                                        os.remove(_tmp)
                                except Exception:
                                    pass
                            break

                    if selected_token:
                        upload_items = [row for row in upload_items if row[3] == selected_token]
                    if not upload_items:
                        raise ValueError("No files available in the selected analysis folder(s).")
                    use_prefix = len(local_folder_tokens) > 1

                    save_started = time.perf_counter()
                    saved_bytes = 0
                    for idx, (upload, safe_name, top_folder, folder_token) in enumerate(upload_items):
                        ext = Path(safe_name).suffix.lower()
                        if ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                            continue
                        logical_name = _prefixed_file_name(
                            safe_name,
                            top_folder,
                            apply_prefix=use_prefix and bool(top_folder),
                        )
                        unique_filename = f"features_empirical_file_{idx}_{job_id}_{safe_name}"
                        file_path = os.path.join(module_upload_dir, unique_filename)
                        save_one_started = time.perf_counter()
                        upload.save(file_path)
                        save_one_ms = (time.perf_counter() - save_one_started) * 1000.0
                        if not os.path.exists(file_path) or os.path.getsize(file_path) <= 0:
                            continue
                        file_size = os.path.getsize(file_path)
                        saved_bytes += file_size
                        file_paths[f"empirical_file_{idx}"] = file_path
                        empirical_upload_paths.append({
                            "name": logical_name,
                            "ext": ext,
                            "path": file_path,
                            "folder_name": folder_token,
                        })
                        if idx == 0 or (idx + 1) % 25 == 0 or (idx + 1) == len(upload_items):
                            app.logger.warning(
                                "[compute %s] saved empirical file %d/%d (%.2f MB, %.1f ms, cumulative %.2f MB)",
                                job_id,
                                idx + 1,
                                len(upload_items),
                                file_size / (1024 * 1024),
                                save_one_ms,
                                saved_bytes / (1024 * 1024),
                            )
                    app.logger.warning(
                        "[compute %s] empirical staging finished: files=%d total=%.2f MB in %.1f s",
                        job_id,
                        len(empirical_upload_paths),
                        saved_bytes / (1024 * 1024),
                        time.perf_counter() - save_started,
                    )

                if not empirical_upload_paths:
                    raise ValueError("No supported empirical files were uploaded.")
                _validate_data_locator_against_source_path(
                    parse_cfg.fields.data,
                    empirical_upload_paths[0]["path"],
                    source_label="selected analysis sample",
                )
            except Exception as exc:
                app.logger.exception("[compute %s] empirical staging failed", job_id)
                flash(f"Empirical parser configuration failed: {exc}", "error")
                return redirect(request.referrer or url_for('features_methods'))
    else:
        files_obj = _ensure_files_parsed()
        inference_file_key_map = {
            "features_predict_file": "features_predict",
            "file-upload-features": "features_predict",
            "features_predict": "features_predict",
            "model_file": "model_file",
            "model-file": "model_file",
            "file-upload-model": "model_file",
            "scaler_file": "scaler_file",
            "scaler-file": "scaler_file",
            "file-upload-scaler": "scaler_file",
            "density_estimator_file": "density_estimator_file",
            "density-estimator-file": "density_estimator_file",
            "file-upload-density-estimator": "density_estimator_file",
            # Backward-compatibility with older input keys.
            "features_sim": "features_sim",
            "parameters": "parameters",
        }

        for i, file_key in enumerate(files_obj):
            if computation_type == "inference" and file_key == "inference_assets_folder":
                # Folder uploads are handled below via getlist.
                continue
            file = files_obj[file_key]
            if not file or not file.filename:
                if computation_type in {'field_potential_proxy', 'field_potential_kernel', 'field_potential_meeg'}:
                    continue
                continue
            unique_filename = f"{computation_type}_{file_key}_{i}_{job_id}_{file.filename}" # E.g. features_ data_file_ 0_ 444961cc-5b72-43fc-b87e-3f4c8304ecdd_ df_inputIn_features_lfp.pkl
            file_path = os.path.join(module_upload_dir, unique_filename)
            file.save(file_path)
            # Save dictionary with file_key: file_path
            normalized_key = file_key
            if computation_type == "inference":
                normalized_key = inference_file_key_map.get(file_key, file_key)
            file_paths[normalized_key] = file_path
        if computation_type in {"field_potential_proxy", "field_potential_kernel", "field_potential_meeg"}:
            server_file_keys = []
            redirect_target = "field_potential_kernel"
            if computation_type == "field_potential_proxy":
                redirect_target = "field_potential_proxy"
                server_file_keys = [
                    "times_file",
                    "gids_file",
                    "vm_file",
                    "ampa_file",
                    "gaba_file",
                ]
            elif computation_type == "field_potential_kernel":
                server_file_keys = [
                    "kernel_spike_times_file",
                    "kernel_population_sizes_file",
                    "electrode_parameters_file",
                ]
            elif computation_type == "field_potential_meeg":
                redirect_target = "field_potential_meeg"
                server_file_keys = [
                    "meeg_cdm_file",
                    "meeg_dipole_file",
                    "meeg_sensor_file",
                ]
            for file_key in server_file_keys:
                source_mode = (request.form.get(f"{file_key}_source_mode") or "upload").strip().lower()
                if source_mode != "server-path":
                    continue
                server_path_raw = (request.form.get(f"{file_key}_server_path") or "").strip()
                if not server_path_raw:
                    continue
                try:
                    source_path = _validate_existing_pickle_file_path(server_path_raw, f"{file_key} server file")
                except Exception as exc:
                    flash(str(exc), "error")
                    return redirect(request.referrer or url_for(redirect_target))
                copied_name = f"{computation_type}_{file_key}_{job_id}_{os.path.basename(source_path)}"
                copied_path = os.path.join(module_upload_dir, copied_name)
                shutil.copy2(source_path, copied_path)
                file_paths[file_key] = copied_path
        if computation_type == "field_potential_kernel":
            kernel_params_source_mode = (request.form.get("kernel_params_module_source_mode") or "server-path").strip().lower()
            if kernel_params_source_mode not in {"upload", "server-path"}:
                kernel_params_source_mode = "server-path"

            if kernel_params_source_mode == "upload":
                uploaded_kernel_params = file_paths.get("kernel_params_file")
                if uploaded_kernel_params:
                    if Path(uploaded_kernel_params).suffix.lower() != ".py":
                        flash("Kernel parameters local upload must be a Python file (.py).", "error")
                        return redirect(request.referrer or url_for("field_potential_kernel"))
                    kernel_params_module_override = uploaded_kernel_params

            if kernel_params_module_override is None:
                kernel_params_path_raw = (request.form.get("kernel_params_module") or "").strip()
                if kernel_params_path_raw:
                    try:
                        source_path = _validate_existing_file_path(
                            kernel_params_path_raw,
                            "Kernel parameters module",
                            allowed_extensions={".py"},
                        )
                    except Exception as exc:
                        flash(str(exc), "error")
                        return redirect(request.referrer or url_for("field_potential_kernel"))
                    copied_name = f"{computation_type}_kernel_params_module_{job_id}_{os.path.basename(source_path)}"
                    copied_path = os.path.join(module_upload_dir, copied_name)
                    shutil.copy2(source_path, copied_path)
                    file_paths["kernel_params_module_file"] = copied_path
                    kernel_params_module_override = copied_path
        if computation_type == "inference":
            if inference_model_assets_source != "server-path":
                folder_uploads = [f for f in files_obj.getlist("inference_assets_folder") if f and f.filename]
                for idx, upload in enumerate(folder_uploads):
                    raw_name = upload.filename or ""
                    base_name = os.path.basename(raw_name)
                    safe_name = secure_filename(base_name)
                    if not safe_name:
                        continue
                    unique_filename = f"{computation_type}_inference_assets_folder_{idx}_{job_id}_{safe_name}"
                    file_path = os.path.join(module_upload_dir, unique_filename)
                    upload.save(file_path)

                    normalized_key = _infer_inference_asset_role(safe_name)

                    # Do not override explicit single-file uploads from dedicated controls.
                    if normalized_key and normalized_key not in file_paths:
                        file_paths[normalized_key] = file_path
                    else:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except OSError:
                            pass

            if inference_features_source_mode == "server-path" and inference_features_server_path:
                copied_name = f"inference_features_predict_0_{job_id}_{os.path.basename(inference_features_server_path)}"
                copied_path = os.path.join(module_upload_dir, copied_name)
                shutil.copy2(inference_features_server_path, copied_path)
                file_paths["features_predict"] = copied_path
            else:
                existing_features_file = (request.form.get("existing_features_file") or "").strip()
                if "features_predict" not in file_paths and existing_features_file:
                    available = set(_list_features_data_files())
                    if existing_features_file in available:
                        src_path = os.path.join(_features_data_dir(create=False), existing_features_file)
                        if os.path.isfile(src_path):
                            copied_name = f"inference_features_predict_0_{job_id}_{os.path.basename(src_path)}"
                            copied_path = os.path.join(module_upload_dir, copied_name)
                            shutil.copy2(src_path, copied_path)
                            file_paths["features_predict"] = copied_path

            if inference_model_assets_source == "server-path":
                for asset_key in ("model_file", "scaler_file", "density_estimator_file"):
                    source_path = inference_assets_server_files.get(asset_key)
                    if not source_path:
                        continue
                    copied_name = f"inference_{asset_key}_{job_id}_{os.path.basename(source_path)}"
                    copied_path = os.path.join(module_upload_dir, copied_name)
                    shutil.copy2(source_path, copied_path)
                    file_paths[asset_key] = copied_path
        if computation_type == "inference_training":
            if inference_training_features_source_mode == "server-path" and inference_training_features_server_path:
                copied_name = f"inference_training_features_{job_id}_{os.path.basename(inference_training_features_server_path)}"
                copied_path = os.path.join(module_upload_dir, copied_name)
                shutil.copy2(inference_training_features_server_path, copied_path)
                file_paths["training_features_file"] = copied_path
            if inference_training_parameters_source_mode == "server-path" and inference_training_parameters_server_path:
                copied_name = f"inference_training_parameters_{job_id}_{os.path.basename(inference_training_parameters_server_path)}"
                copied_path = os.path.join(module_upload_dir, copied_name)
                shutil.copy2(inference_training_parameters_server_path, copied_path)
                file_paths["training_parameters_file"] = copied_path

    data = request.form.to_dict() # Get parameters from form POST
    if computation_type == "inference":
        data["features_source_mode"] = inference_features_source_mode
        data["model_assets_source"] = inference_model_assets_source
        data["features_server_file_path"] = (
            inference_features_server_path or (request.form.get("features_server_file_path") or "").strip()
        )
        data["inference_assets_server_folder_path"] = inference_assets_server_folder_path
        data["inference_model_server_file_path"] = inference_model_server_file_path
        data["inference_scaler_server_file_path"] = inference_scaler_server_file_path
        data["inference_density_server_file_path"] = inference_density_server_file_path
        data["assets_upload_mode"] = assets_upload_mode
    if computation_type == "inference_training":
        data["training_features_source_mode"] = inference_training_features_source_mode
        data["training_parameters_source_mode"] = inference_training_parameters_source_mode
        data["training_features_server_file_path"] = (
            inference_training_features_server_path or (request.form.get("training_features_server_file_path") or "").strip()
        )
        data["training_parameters_server_file_path"] = (
            inference_training_parameters_server_path or (request.form.get("training_parameters_server_file_path") or "").strip()
        )
    if kernel_params_module_override is not None:
        data["kernel_params_module"] = kernel_params_module_override
    # Add file information to the data dictionary
    data['file_paths'] = file_paths
    if prepared_features_df is not None:
        data["prepared_features_df"] = prepared_features_df
    if empirical_upload_paths is not None:
        data["empirical_upload_paths"] = empirical_upload_paths
    if parser_config_obj is not None:
        data["parser_config_obj"] = parser_config_obj

    # Handle additional metadata files for parser enrichments:
    # - subject-level cross-referencing (when link field is provided)
    # - channel names resolution from additional tabular columns
    if computation_type == "features":
        _add_meta_uploads = [f for f in request.files.getlist("additional_metadata_file") if f and f.filename]
        _add_meta_server_paths = [p.strip() for p in request.form.getlist("additional_metadata_server_path") if p.strip()]
        _add_file_link_field = (request.form.get("additional_file_link_field") or "").strip()
        if _add_meta_uploads or _add_meta_server_paths:
            _additional_metadata_paths = []
            for _upl in _add_meta_uploads:
                _upl_name = secure_filename(_upl.filename)
                _ext_check = Path(_upl_name).suffix.lower()
                if _ext_check in FEATURES_PARSER_FILE_EXTENSIONS:
                    _tmp_path = os.path.join(module_upload_dir, f"{job_id}_addmeta_{_upl_name}")
                    _upl.save(_tmp_path)
                    _additional_metadata_paths.append({"path": _tmp_path, "name": _upl_name})
            for _srv in _add_meta_server_paths:
                _real = os.path.realpath(_srv)
                if os.path.isfile(_real):
                    _additional_metadata_paths.append({"path": _real, "name": os.path.basename(_real)})
            if _additional_metadata_paths:
                data["additional_metadata_paths"] = _additional_metadata_paths
            if _add_file_link_field:
                data["additional_file_link_field"] = _add_file_link_field

    # Store initial status
    job_status[job_id] = {
        "status": "in_progress",
        "progress": 0,
        "start_time": time.time(),
        "estimated_time_remaining": estimated_time_remaining,
        "results": None,
        "error": False,
        "cancel_requested": False,
        "computation_type": computation_type,
        "output": "",
        "progress_mode": "manual" if computation_type in {"features", "inference", "inference_training", "field_potential_meeg"} else "time",
    }

    # Submit the long-running task according to the computation type.
    upstream_module_by_type = {
        "field_potential_proxy": "field_potential",
        "field_potential_kernel": "field_potential",
        "field_potential_meeg": "field_potential",
        "features": "features",
        "inference": "inference",
    }
    upstream_module = upstream_module_by_type.get(computation_type)
    future = executor.submit(
        _run_job_with_post_success_cleanup,
        job_id,
        upstream_module,
        func,
        data,
        module_upload_dir,
    )
    job_futures[job_id] = future
    app.logger.warning("[compute %s] job submitted to executor", job_id)

    # Redirect immediately to the loading page (PRG pattern)
    app.logger.warning(
        "[compute %s] redirecting to status page (total route %.1f s)",
        job_id,
        time.perf_counter() - route_started,
    )
    return redirect(url_for('job_status_page', job_id=job_id, computation_type=computation_type))

@app.route("/job_status/<job_id>")
def job_status_page(job_id):
    """Renders the loading page that begins polling."""
    # Get computation_type from the query parameters
    computation_type = request.args.get('computation_type') 
    # Pass the job_id to the template for use in Alpine.js
    return render_template("loading_page.html", job_id=job_id, computation_type=computation_type, pending=False)

@app.route("/job_status_pending")
def job_status_pending():
    """Renders the loading page before a job id is assigned (used for uploads)."""
    computation_type = request.args.get('computation_type')
    return render_template("loading_page.html", job_id="", computation_type=computation_type, pending=True)


@app.route("/cancel_job/<job_id>", methods=["POST"])
def cancel_job(job_id):
    status = job_status.get(job_id)
    if not isinstance(status, dict):
        return jsonify({"ok": False, "error": "Job not found."}), 404

    if status.get("status") in {"finished", "failed", "cancelled"}:
        return jsonify({
            "ok": False,
            "error": f'Job is already in terminal state: {status.get("status")}.',
            "status": status.get("status"),
        }), 400

    status["cancel_requested"] = True
    _append_job_output(job_status, job_id, "Cancellation requested by user.")

    future = job_futures.get(job_id)
    cancelled_before_start = bool(future and future.cancel())
    if cancelled_before_start:
        job_futures.pop(job_id, None)
        _mark_job_cancelled(job_id, "Computation cancelled before execution.")
    else:
        for cancel_msg in _cancel_job_python_processes(job_id):
            _append_job_output(job_status, job_id, cancel_msg)
        _mark_job_cancelled(job_id, "Computation cancelled by user.")

    return jsonify({
        "ok": True,
        "status": job_status.get(job_id, {}).get("status", "cancelled"),
        "cancel_requested": True,
    })

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
        "cancel_requested": bool(status.get("cancel_requested")),
        "output": status.get("output", ""),
        "simulation_total": status.get("simulation_total"),
        "simulation_completed": status.get("simulation_completed"),
        "simulation_mode": status.get("simulation_mode"),
        "meeg_trial_total": status.get("meeg_trial_total"),
        "meeg_trial_index": status.get("meeg_trial_index"),
        "meeg_sensors_total": status.get("meeg_sensors_total"),
        "meeg_sensors_completed": status.get("meeg_sensors_completed"),
        "can_download": bool(status.get("status") == "finished" and status.get("results")),
    })


@app.route("/preview_results/<job_id>")
def preview_results(job_id):
    """Returns the first rows of the result DataFrame as JSON for in-UI preview."""
    status = job_status.get(job_id)
    if not status or status.get("status") != "finished" or not status.get("results"):
        return jsonify({"error": "Results not available"}), 404
    try:
        path = status["results"]
        ext = os.path.splitext(str(path))[1].lower()
        if ext in {".pkl", ".pickle"}:
            df = pd.read_pickle(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_pickle(path)
        if not isinstance(df, pd.DataFrame):
            return jsonify({"error": "Result is not a DataFrame"}), 400
        # Drop array-valued columns that can't be serialized
        array_cols = [c for c in df.columns if df[c].dtype == object and len(df) > 0
                      and hasattr(df[c].iloc[0], '__len__') and not isinstance(df[c].iloc[0], str)]
        preview = df.drop(columns=array_cols, errors="ignore").head(20)
        return jsonify({
            "columns": list(preview.columns),
            "rows": preview.astype(str).values.tolist(),
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "dropped_cols": array_cols,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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
    download_tmp_dir = _module_uploads_dir_for(computation_type)

    # Remove file after downloading it, except canonical pipeline outputs in the shared temp root.
    if computation_type not in {'field_potential_proxy', 'field_potential_kernel', 'field_potential_meeg', 'simulation'}:
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
            output_df_path,
            mimetype='image/png',
            as_attachment=True,
            download_name='LFP_predictions.png'
        )
    if computation_type == 'simulation':
        if not os.path.isdir(output_df_path):
            return "Simulation outputs directory is not available.", 404

        os.makedirs(download_tmp_dir, exist_ok=True)
        archive_base = os.path.join(download_tmp_dir, f"simulation_results_{job_id}")
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=output_df_path)

        @after_this_request
        def cleanup_sim_zip(response):
            try:
                if os.path.exists(archive_path):
                    os.remove(archive_path)
            except Exception as exc:
                app.logger.error(f"Error removing simulation archive {archive_path}: {exc}")
            return response

        return send_file(
            archive_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'simulation_results_{job_id}.zip'
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
    if computation_type == 'inference_training':
        return send_file(
            output_df_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name='inference_training_artifacts.zip'
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

if __name__ == '__main__':
    app.run(debug=True)
