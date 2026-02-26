import tempfile
import os
import shutil
import subprocess
import ast
import json
import threading
import glob
import base64
import pickle
import sys
import inspect
from pathlib import Path
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

# Temporary folder for uploaded files of forms.
# Keep this outside the repository (and OneDrive-synced paths) to avoid long blocking writes.
temp_uploaded_files = os.path.join(_tempdir, "ncpi_temp_uploaded_files")

# Prefer the local repository package over any globally installed ncpi version.
_webui_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_webui_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from flask import Flask, render_template, request, jsonify, url_for, redirect, send_file, after_this_request, flash
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

# In-memory thread pool
executor = ThreadPoolExecutor(max_workers=5) 

# Dictionary to store job progress/results (job_id: status_dict)
# NOTE: This dictionary is volatile and will reset if the server restarts.
job_status = {}
MAX_OUTPUT_LINES = 200
FEATURES_PARSER_FILE_EXTENSIONS = {
    ".mat", ".json", ".npy", ".csv", ".parquet", ".pkl", ".pickle", ".xlsx", ".xls", ".feather"
}
PICKLE_EXTENSIONS = {".pkl", ".pickle"}
MAX_EMPIRICAL_UPLOAD_BYTES = int(os.environ.get("NCPI_MAX_EMPIRICAL_UPLOAD_BYTES", str(1024 * 1024 * 1024)))


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


def _field_potential_dirs():
    temp_dir = tempfile.gettempdir()
    return [
        os.path.join("/tmp", "field_potential_proxy"),
        os.path.join("/tmp", "field_potential_kernel"),
        os.path.join("/tmp", "field_potential_meeg"),
        os.path.join(temp_dir, "field_potential_proxy"),
        os.path.join(temp_dir, "field_potential_kernel"),
        os.path.join(temp_dir, "field_potential_meeg"),
    ]


def _features_data_dir(create=False):
    path = os.path.realpath("/tmp/features_data")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _predictions_data_dir(create=False):
    path = os.path.realpath("/tmp/predictions_data")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _features_data_candidate_dirs():
    raw_paths = [
        "/tmp/features_data",
        os.path.join(tempfile.gettempdir(), "features_data"),
        os.path.join(_tempdir, "features_data"),
        "/home/necolab/tmp/features_data",
    ]
    seen = set()
    candidates = []
    for raw_path in raw_paths:
        if not raw_path:
            continue
        path = os.path.realpath(os.path.expanduser(raw_path))
        if path in seen:
            continue
        seen.add(path)
        candidates.append(path)
    return candidates


def _sync_features_data_from_candidates():
    primary = os.path.realpath(_features_data_dir(create=True))
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


def _collect_feature_pipeline_inputs():
    source_labels = {
        "field_potential_proxy": "Field Potential Proxy",
        "field_potential_kernel": "Field Potential Kernel",
        "field_potential_meeg": "Field Potential M/EEG",
    }
    discovered = {}
    for fp_dir in _field_potential_dirs():
        if not os.path.isdir(fp_dir):
            continue
        source_key = os.path.basename(fp_dir)
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
                if source_key == "field_potential_proxy":
                    if not lower_name.startswith("proxy"):
                        continue
                elif source_key == "field_potential_kernel":
                    allowed_kernel_outputs = {
                        "kernel_approx_cdm.pkl",
                        "current_dipole_moment.pkl",
                        "gauss_cylinder_potential.pkl",
                        "probe_outputs.pkl",
                    }
                    if lower_name not in allowed_kernel_outputs:
                        continue
                elif source_key == "field_potential_meeg":
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


def _extract_locator_candidates(obj, max_depth=3, max_items=120):
    candidates = []
    seen = set()
    visited = set()

    def _is_terminal(value):
        return isinstance(value, (str, bytes, int, float, bool, np.ndarray))

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

        if isinstance(value, dict):
            for key, nested in value.items():
                key_str = str(key)
                if key_str.startswith("__"):
                    continue
                path = f"{prefix}.{key_str}" if prefix else key_str
                _append(path)
                if not _is_terminal(nested):
                    _walk(nested, path, depth + 1)
            return

        if isinstance(value, (list, tuple)):
            if not value:
                return
            sample = value[0]
            if not _is_terminal(sample):
                _walk(sample, prefix, depth + 1)
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


def _estimate_sensor_count(value):
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


def _auto_channel_names(value):
    count = _estimate_sensor_count(value)
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


def _load_features_source(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext in {".xlsx", ".xls", ".feather"}:
        return compute_utils.read_df_file(path)
    return _load_parser_source(path)


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

    if ext == ".parquet":
        return pd.read_parquet(io.BytesIO(raw))

    if ext == ".feather":
        return pd.read_feather(io.BytesIO(raw))

    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw))

    if ext == ".mat":
        try:
            import scipy.io as sio
        except Exception as exc:
            raise ValueError(f"scipy is required to parse .mat files: {exc}")
        return sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)

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

    if isinstance(source_obj, pd.DataFrame):
        fs_hint_hz, fs_hint_note = _extract_dataframe_fs_hint(source_obj)
        columns = [str(col) for col in source_obj.columns]
        defaults = {
            "data": _pick_field_guess(columns, ["data", "signal", "proxy", "cdm"]),
            "fs": _pick_field_guess(columns, ["fs", "sfreq", "sampling_rate"]),
            "ch_names": _pick_field_guess(columns, ["ch_names", "channels", "channel", "sensor"]),
            "recording_type": _pick_field_guess(columns, ["recording_type", "modality", "recording", "type"]),
            "subject_id": _pick_field_guess(columns, ["subject_id", "subject", "subj", "participant"]),
            "group": _pick_field_guess(columns, ["group", "cohort", "class"]),
        }
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

        return {
            "source_type": "dataframe",
            "candidate_fields": columns,
            "defaults": defaults,
            "summary": f"DataFrame with {source_obj.shape[0]} rows and {source_obj.shape[1]} columns.",
            "sensor_count_estimate": sensor_count,
            "multi_sensor_detected": bool(sensor_count and sensor_count > 1),
            "fs_hint_hz": fs_hint_hz,
            "fs_hint_note": fs_hint_note,
        }

    if isinstance(source_obj, np.ndarray):
        sensor_count = _estimate_sensor_count(source_obj)
        return {
            "source_type": "ndarray",
            "candidate_fields": ["__self__"],
            "defaults": {
                "data": "__self__",
                "fs": "",
                "ch_names": "",
                "recording_type": "",
                "subject_id": "",
                "group": "",
            },
            "summary": f"NumPy array with shape {list(source_obj.shape)}.",
            "sensor_count_estimate": sensor_count,
            "multi_sensor_detected": bool(sensor_count and sensor_count > 1),
            "fs_hint_hz": fs_hint_hz,
            "fs_hint_note": fs_hint_note,
        }

    candidate_fields = _extract_locator_candidates(source_obj)
    defaults = {
        "data": _pick_field_guess(candidate_fields, ["data", "signal", "lfp", "cdm"]),
        "fs": _pick_field_guess(candidate_fields, ["fs", "sfreq", "sampling_rate"]),
        "ch_names": _pick_field_guess(candidate_fields, ["ch_names", "channels", "channel_names"]),
        "recording_type": _pick_field_guess(candidate_fields, ["recording_type", "modality", "recording", "type"]),
        "subject_id": _pick_field_guess(candidate_fields, ["subject_id", "subject", "subj", "participant"]),
        "group": _pick_field_guess(candidate_fields, ["group", "cohort", "class"]),
    }

    if isinstance(source_obj, dict):
        top_keys = [str(k) for k in source_obj.keys() if not str(k).startswith("__")]
        data_guess = defaults.get("data")
        resolved_data = _resolve_locator_value(source_obj, data_guess)
        sensor_count = _estimate_sensor_count(resolved_data) if resolved_data is not None else None
        return {
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

    data_guess = defaults.get("data")
    resolved_data = _resolve_locator_value(source_obj, data_guess)
    sensor_count = _estimate_sensor_count(resolved_data) if resolved_data is not None else None
    return {
        "source_type": "object",
        "candidate_fields": candidate_fields,
        "defaults": defaults,
        "summary": f"Object of type {type(source_obj).__name__}.",
        "sensor_count_estimate": sensor_count,
        "multi_sensor_detected": bool(sensor_count and sensor_count > 1),
        "fs_hint_hz": fs_hint_hz,
        "fs_hint_note": fs_hint_note,
    }


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

    sensor_names = _parse_sensor_names(form.get("parser_sensor_names"))
    ch_names_source = (form.get("parser_ch_names_source") or "").strip()
    ch_names_locator = (form.get("parser_ch_names_locator") or "").strip() or None
    if ch_names_source == "__manual__":
        if sensor_names is None:
            raise ValueError("Provide manual channel names when channel names source is set to manual.")
        ch_names_value = sensor_names
    elif ch_names_source == "__autocomplete__":
        ch_names_value = None
    elif ch_names_source == "__none__":
        ch_names_value = None
    elif ch_names_source:
        ch_names_value = ch_names_source
    else:
        # Backward-compatible fallback for old forms.
        ch_names_value = sensor_names if sensor_names is not None else ch_names_locator

    if data_source_kind in {"pipeline", "new-simulation"}:
        metadata = {
            "subject_id": 0,
            "group": "simulation",
            "species": "simulated",
            "condition": "simulation_pipeline",
            "recording_type": recording_type_value,
        }
        fields = CanonicalFields(
            data=data_locator,
            fs=fs_value,
            ch_names=ch_names_value,
            metadata=metadata,
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
            metadata["subject_id"] = "file_ID" if subject_id_locator == "__file_id__" else subject_id_locator
        if group_locator:
            metadata["group"] = group_locator

    fields = CanonicalFields(
        data=data_locator,
        fs=fs_value,
        ch_names=ch_names_value,
        metadata=metadata,
        table_layout=table_layout,
        long_time_col=(form.get("parser_long_time_col") or "").strip() or None,
        long_channel_col=(form.get("parser_long_channel_col") or "").strip() or None,
        long_value_col=(form.get("parser_long_value_col") or "").strip() or None,
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


def _normalize_features_input_path(source_path, form, job_id):
    source_obj = _load_features_source(source_path)
    data_source_kind = (form.get("data_source_kind") or "").strip()

    if not (form.get("parser_data_locator") or "").strip():
        raise ValueError(
            "Configure EphysDatasetParser (at least the data locator)."
        )

    parse_cfg = _build_parse_config_from_form(form)

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
    normalized_path = os.path.join(temp_uploaded_files, f"features_data_file_0_{job_id}_parsed.pkl")
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
    field_potential_files = []
    for fp_dir in _field_potential_dirs():
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
    return render_template("1.simulation.html")

@app.route("/simulation/upload_sim")
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
    for fp_dir in _field_potential_dirs():
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


@app.route("/clear_features_data", methods=["POST"])
def clear_features_data():
    features_data_dir = _features_data_dir(create=False)
    if os.path.isdir(features_data_dir):
        for name in os.listdir(features_data_dir):
            if not (name.endswith(".pkl") or name.endswith(".pickle")):
                continue
            path = os.path.join(features_data_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    return redirect(url_for('dashboard'))


@app.route("/clear_predictions_data", methods=["POST"])
def clear_predictions_data():
    predictions_data_dir = _predictions_data_dir(create=False)
    if os.path.isdir(predictions_data_dir):
        for name in os.listdir(predictions_data_dir):
            if not (name.endswith(".pkl") or name.endswith(".pickle")):
                continue
            path = os.path.join(predictions_data_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    return redirect(url_for('dashboard'))

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


@app.route("/field_potential/load")
def field_potential_load():
    return render_template("2.3.field_potential_load.html")


@app.route("/field_potential/load_precomputed", methods=["POST"])
def field_potential_load_precomputed():
    upload = request.files.get("precomputed_fp_file")
    fp_type = (request.form.get("precomputed_fp_type") or "proxy").strip().lower()

    if upload is None or not upload.filename:
        flash("Upload a precomputed field potential file (.pkl/.pickle).", "error")
        return redirect(request.referrer or url_for("field_potential_load"))

    safe_name = secure_filename(upload.filename)
    if not safe_name:
        flash("Invalid uploaded file name.", "error")
        return redirect(request.referrer or url_for("field_potential_load"))

    ext = Path(safe_name).suffix.lower()
    if ext not in PICKLE_EXTENSIONS:
        flash("Precomputed field potential must be a pickle file (.pkl/.pickle).", "error")
        return redirect(request.referrer or url_for("field_potential_load"))

    destination_map = {
        "proxy": (os.path.join(tempfile.gettempdir(), "field_potential_proxy"), "proxy_loaded.pkl"),
        "cdm": (os.path.join(tempfile.gettempdir(), "field_potential_kernel"), "kernel_approx_cdm.pkl"),
        "lfp": (os.path.join(tempfile.gettempdir(), "field_potential_kernel"), "gauss_cylinder_potential.pkl"),
        "eeg": (os.path.join(tempfile.gettempdir(), "field_potential_meeg"), "eeg.pkl"),
        "meg": (os.path.join(tempfile.gettempdir(), "field_potential_meeg"), "meg.pkl"),
    }

    if fp_type not in destination_map:
        flash("Unknown precomputed field potential type.", "error")
        return redirect(request.referrer or url_for("field_potential_load"))

    output_root, output_name = destination_map[fp_type]
    run_dir = os.path.join(output_root, f"loaded_{uuid.uuid4().hex[:12]}")
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, output_name)
    upload.save(output_path)

    return redirect(url_for("field_potential"))


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

@app.route("/field_potential/proxy")
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
    pipeline_files = _collect_feature_pipeline_inputs()
    features_data_files = _list_features_data_files()
    return render_template(
        "3.features.html",
        pipeline_files=pipeline_files,
        has_pipeline_files=bool(pipeline_files),
        features_data_files=features_data_files,
        has_features_data=bool(features_data_files),
    )


@app.route("/features/browse_dirs", methods=["GET"])
def features_browse_dirs():
    requested = (request.args.get("path") or "").strip()
    if requested:
        current = os.path.realpath(os.path.expanduser(requested))
    else:
        current = os.path.realpath(os.path.expanduser("~"))

    if not os.path.isdir(current):
        return jsonify({"error": f"Not a directory: {current}"}), 400

    try:
        dirs = []
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    dirs.append({
                        "name": entry.name,
                        "path": os.path.realpath(entry.path),
                    })
    except PermissionError:
        return jsonify({"error": f"Permission denied: {current}"}), 403
    except OSError as exc:
        return jsonify({"error": f"Failed to list directory: {exc}"}), 400

    dirs.sort(key=lambda item: item["name"].lower())
    parent = os.path.dirname(current)
    if parent == current:
        parent = ""
    return jsonify({
        "path": current,
        "parent": parent,
        "dirs": dirs[:1000],
    })


@app.route("/features/select_folder", methods=["POST"])
def features_select_folder():
    # Opens a native path picker on the machine running the Flask server.
    # This is intended for local desktop usage.
    mode = (request.form.get("mode") or "folder").strip().lower()
    if shutil.which("zenity"):
        try:
            cmd = ["zenity", "--file-selection"]
            if mode == "folder":
                cmd.extend(["--directory", "--title=Select data folder"])
            else:
                cmd.extend(["--title=Select data file"])
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=os.environ.copy(),
            )
            if proc.returncode != 0:
                return jsonify({"error": "Path selection cancelled."}), 400
            picked = (proc.stdout or "").strip()
            if not picked:
                return jsonify({"error": "No path selected."}), 400
            picked = os.path.realpath(os.path.expanduser(picked))
            if mode == "folder":
                if not os.path.isdir(picked):
                    return jsonify({"error": f"Selected path is not a directory: {picked}"}), 400
            else:
                if not os.path.isfile(picked):
                    return jsonify({"error": f"Selected path is not a file: {picked}"}), 400
                ext = Path(picked).suffix.lower()
                if ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                    return jsonify({"error": f"Unsupported selected file type: {ext}"}), 400
            return jsonify({"path": picked})
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Folder selector timed out."}), 408
        except Exception as exc:
            return jsonify({"error": f"Failed to open native folder selector: {exc}"}), 500

    return jsonify({"error": "Native folder picker is unavailable (zenity not found). Set the folder path manually."}), 400


@app.route("/features/parser/inspect", methods=["POST"])
def features_parser_inspect():
    existing_data_path = (request.form.get("existing_data_path") or "").strip()
    empirical_folder_path = (request.form.get("empirical_folder_path") or "").strip()
    simulation_file_path = (request.form.get("simulation_file_path") or "").strip()
    simulation_folder_path = (request.form.get("simulation_folder_path") or "").strip()
    upload = request.files.get("file")
    inspect_path = None
    cleanup_path = None

    try:
        if existing_data_path:
            inspect_path = _validate_feature_existing_path(existing_data_path)
            file_name = os.path.basename(inspect_path)
        elif empirical_folder_path:
            folder_files = _collect_empirical_folder_files(empirical_folder_path)
            inspect_path = folder_files[0]
            file_name = os.path.basename(inspect_path)
        elif simulation_file_path:
            inspect_path = _validate_simulation_file_path(simulation_file_path)
            file_name = os.path.basename(inspect_path)
        elif simulation_folder_path:
            # Backward-compatible support for older form payloads.
            folder_files = _collect_simulation_folder_files(simulation_folder_path)
            inspect_path = folder_files[0]
            file_name = os.path.basename(inspect_path)
        elif upload and upload.filename:
            file_name = secure_filename(upload.filename)
            if not file_name:
                return jsonify({"error": "Invalid file name."}), 400
            ext = Path(file_name).suffix.lower()
            if ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                return jsonify({"error": f"Unsupported file type for inspection: {ext}"}), 400
            inspect_root = os.path.join(tempfile.gettempdir(), "features_inspection")
            os.makedirs(inspect_root, exist_ok=True)
            temp_name = f"{uuid.uuid4()}_{file_name}"
            inspect_path = os.path.join(inspect_root, temp_name)
            upload.save(inspect_path)
            cleanup_path = inspect_path
        else:
            return jsonify({"error": "Provide an existing pipeline file, a simulation file path, an empirical folder path, or upload a file to inspect."}), 400

        description = _describe_parser_source(inspect_path)
        description["source_name"] = file_name
        if empirical_folder_path or simulation_folder_path:
            description["folder_file_count"] = len(folder_files)
            selected_folder = empirical_folder_path if empirical_folder_path else simulation_folder_path
            description["folder_path"] = os.path.realpath(selected_folder)
        if simulation_file_path:
            description["selected_file_path"] = os.path.realpath(simulation_file_path)
        return jsonify(description)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except OSError:
                pass


@app.route("/features/load_precomputed", methods=["POST"])
def features_load_precomputed():
    upload = request.files.get("precomputed_features_file")
    if upload is None or not upload.filename:
        flash("Upload a precomputed features dataframe (.pkl/.pickle).", "error")
        return redirect(request.referrer or url_for("features"))

    safe_name = secure_filename(upload.filename)
    if not safe_name:
        flash("Invalid uploaded file name.", "error")
        return redirect(request.referrer or url_for("features"))

    ext = Path(safe_name).suffix.lower()
    if ext not in PICKLE_EXTENSIONS:
        flash("Precomputed features must be a pickle file (.pkl/.pickle).", "error")
        return redirect(request.referrer or url_for("features"))

    features_dir = _features_data_dir(create=True)
    temp_path = os.path.join(features_dir, f"tmp_{uuid.uuid4().hex}_{safe_name}")
    upload.save(temp_path)
    try:
        df = compute_utils.read_df_file(temp_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Uploaded object is not a pandas dataframe.")
        if "Features" not in df.columns:
            raise ValueError("Uploaded dataframe does not contain a 'Features' column.")
        output_name = f"features_loaded_{uuid.uuid4().hex[:8]}_{safe_name}"
        output_path = os.path.join(features_dir, output_name)
        df.to_pickle(output_path)
    except Exception as exc:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        flash(f"Failed to load precomputed features: {exc}", "error")
        return redirect(request.referrer or url_for("features"))

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return redirect(url_for("features"))

# Inference configuration page
@app.route("/inference")
def inference():
    return render_template("4.inference.html")

# Load precomputed predictions for inference module
@app.route("/inference/load_data")
def inference_load_data():
    predictions_data_files = _list_predictions_data_files()
    return render_template(
        "4.0.load_data.html",
        predictions_data_files=predictions_data_files,
        has_predictions_data=bool(predictions_data_files),
    )


@app.route("/inference/load_precomputed", methods=["POST"])
def inference_load_precomputed():
    upload = request.files.get("precomputed_predictions_file")
    if upload is None or not upload.filename:
        flash("Upload a precomputed predictions dataframe (.pkl/.pickle).", "error")
        return redirect(request.referrer or url_for("inference_load_data"))

    safe_name = secure_filename(upload.filename)
    if not safe_name:
        flash("Invalid uploaded file name.", "error")
        return redirect(request.referrer or url_for("inference_load_data"))

    ext = Path(safe_name).suffix.lower()
    if ext not in PICKLE_EXTENSIONS:
        flash("Precomputed predictions must be a pickle file (.pkl/.pickle).", "error")
        return redirect(request.referrer or url_for("inference_load_data"))

    predictions_dir = _predictions_data_dir(create=True)
    temp_path = os.path.join(predictions_dir, f"tmp_{uuid.uuid4().hex}_{safe_name}")
    upload.save(temp_path)
    try:
        df = compute_utils.read_df_file(temp_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Uploaded object is not a pandas dataframe.")
        if "Predictions" not in df.columns:
            raise ValueError("Uploaded dataframe does not contain a 'Predictions' column.")
        output_name = "predictions.pkl"
        output_path = os.path.join(predictions_dir, output_name)
        df.to_pickle(output_path)
    except Exception as exc:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        flash(f"Failed to load precomputed predictions: {exc}", "error")
        return redirect(request.referrer or url_for("inference_load_data"))

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    flash(f"Predictions file loaded: {output_name}", "success")
    return redirect(url_for("inference_load_data"))

# New training for inference configuration page
@app.route("/inference/new_training")
def new_training():
    return render_template("4.1.new_training.html")

# Compute predictions for inference configuration page
@app.route("/inference/compute_predictions")
def compute_predictions():
    _bootstrap_features_data_from_previous_steps()
    feature_data_files = _list_features_data_files()
    # Hard fallback: directly scan canonical folder in case helper discovery is bypassed.
    if not feature_data_files:
        hard_root = os.path.realpath("/tmp/features_data")
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
        "4.2.compute_predictions.html",
        feature_data_files=feature_data_files,
        has_feature_data=bool(feature_data_files),
        default_feature_file=default_feature_file,
    )

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
    feature_data_files = _list_features_data_files()
    predictions_data_files = _list_predictions_data_files()
    detected_data_files = []
    for name in feature_data_files:
        detected_data_files.append({
            "key": f"features::{name}",
            "name": name,
            "source": "features",
            "label": "Features",
        })
    for name in predictions_data_files:
        detected_data_files.append({
            "key": f"predictions::{name}",
            "name": name,
            "source": "predictions",
            "label": "Predictions",
        })
    return render_template(
        "5.analysis.html",
        analysis_data_files=analysis_data_files,
        has_analysis_data=bool(analysis_data_files),
        feature_data_files=feature_data_files,
        has_feature_data=bool(feature_data_files),
        predictions_data_files=predictions_data_files,
        has_predictions_data=bool(predictions_data_files),
        detected_data_files=detected_data_files,
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


@app.route("/analysis/select_features_file", methods=["POST"])
def analysis_select_features_file():
    selector = (request.form.get("data_file_key") or request.form.get("features_file") or "").strip()
    if not selector:
        return jsonify({"error": "Select a detected file."}), 400

    if "::" in selector:
        source, filename = selector.split("::", 1)
        source = source.strip().lower()
        filename = filename.strip()
    else:
        # Backward compatibility: treat plain value as features file.
        source = "features"
        filename = selector

    source_dirs = {
        "features": _features_data_dir(create=False),
        "predictions": _predictions_data_dir(create=False),
    }
    source_list_fn = {
        "features": _list_features_data_files,
        "predictions": _list_predictions_data_files,
    }
    if source not in source_dirs:
        return jsonify({"error": f"Unsupported data source '{source}'."}), 400

    available = set(source_list_fn[source]())
    if filename not in available:
        return jsonify({"error": "Selected file is not available. Refresh the page and try again."}), 400

    src_path = os.path.join(source_dirs[source], filename)
    if not os.path.isfile(src_path):
        return jsonify({"error": "Selected file was not found on disk."}), 404

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

    dst_name = secure_filename(filename)
    dst_path = os.path.join(analysis_data_dir, dst_name)
    shutil.copy2(src_path, dst_path)

    try:
        df = compute_utils.read_df_file(dst_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Selected file is not a pandas dataframe.")
        columns = [str(col) for col in df.columns]
        return jsonify({"columns": columns, "filename": dst_name, "source": source})
    except Exception as exc:
        try:
            if os.path.exists(dst_path):
                os.remove(dst_path)
        except OSError:
            pass
        return jsonify({"error": str(exc)}), 400


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
        return redirect(request.referrer or url_for("features"))

    # Build the name of the function to compute depending on the page form this function was called from
    func_name_string = f"{computation_type}_computation"
    func = getattr(compute_utils, func_name_string) # filtered function name for security reasons

    files = None
    uploaded_files = []

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
            return redirect(request.referrer or url_for('features'))
        if selected_method == "hctsa":
            hctsa_folder = (request.form.get("hctsa_folder") or "").strip()
            if not hctsa_folder:
                flash('hctsa_folder is required for hctsa.', 'error')
                return redirect(request.referrer or url_for('features'))

        data_source_kind = (request.form.get("data_source_kind") or "new-simulation").strip()
        app.logger.warning("[compute %s] features data_source_kind=%s", job_id, data_source_kind)
        empirical_source_mode = (request.form.get("empirical_source_mode") or "upload").strip()
        existing_data_path = (request.form.get("existing_data_path") or "").strip()
        if not (request.form.get("parser_data_locator") or "").strip():
            flash('Select the data locator for EphysDatasetParser.', 'error')
            return redirect(request.referrer or url_for('features'))
        epoching_enabled = str(request.form.get("parser_enable_epoching", "")).lower() in {"1", "true", "on", "yes"}
        if epoching_enabled:
            if _optional_float(request.form.get("parser_epoch_length_s")) is None:
                flash('Set an epoch length in seconds.', 'error')
                return redirect(request.referrer or url_for('features'))
            if _optional_float(request.form.get("parser_epoch_step_s")) is None:
                flash('Set an epoch step in seconds.', 'error')
                return redirect(request.referrer or url_for('features'))

        if data_source_kind == "pipeline":
            if uploaded_files:
                flash('Do not upload files when using "Continue simulation pipeline".', 'error')
                return redirect(request.referrer or url_for('features'))
            if not existing_data_path:
                flash('Select a detected simulation pipeline file to continue.', 'error')
                return redirect(request.referrer or url_for('features'))
            fs_source = (request.form.get("parser_fs_source") or "").strip()
            fs_manual = _optional_float(request.form.get("parser_fs_manual"))
            if fs_source == "__numeric__":
                if fs_manual is None:
                    flash('Provide sampling frequency numeric value.', 'error')
                    return redirect(request.referrer or url_for('features'))
            elif fs_source == "__none__":
                pass
            elif fs_source:
                pass
            else:
                fs_locator = (request.form.get("parser_fs_locator") or "").strip()
                if fs_locator and fs_manual is not None:
                    flash('Sampling frequency locator and sampling frequency value are mutually exclusive.', 'error')
                    return redirect(request.referrer or url_for('features'))
                if not fs_locator and fs_manual is None:
                    flash('Provide sampling frequency source (field, numeric value, or None).', 'error')
                    return redirect(request.referrer or url_for('features'))

            recording_type_source = (request.form.get("parser_recording_type_source") or "").strip()
            recording_type_value = (request.form.get("parser_recording_type") or "").strip()
            if recording_type_source == "__value__":
                if not recording_type_value:
                    flash('Select a recording type value.', 'error')
                    return redirect(request.referrer or url_for('features'))
            elif recording_type_source == "__none__":
                pass
            elif recording_type_source:
                pass
            else:
                recording_type_locator = (request.form.get("parser_recording_type_locator") or "").strip()
                if recording_type_locator and recording_type_value:
                    flash('Recording type locator and recording type value are mutually exclusive.', 'error')
                    return redirect(request.referrer or url_for('features'))

            ch_names_source = (request.form.get("parser_ch_names_source") or "").strip()
            manual_sensor_names = _parse_sensor_names(request.form.get("parser_sensor_names"))
            if ch_names_source == "__manual__":
                if not manual_sensor_names:
                    flash('Provide manual channel names (comma-separated).', 'error')
                    return redirect(request.referrer or url_for('features'))
            elif ch_names_source == "__autocomplete__":
                pass
            elif ch_names_source:
                pass
            else:
                ch_names_locator = (request.form.get("parser_ch_names_locator") or "").strip()
                if ch_names_locator and manual_sensor_names:
                    flash('Channel names locator and manual channel names are mutually exclusive.', 'error')
                    return redirect(request.referrer or url_for('features'))
            try:
                _validate_feature_existing_path(existing_data_path)
            except Exception as exc:
                flash(str(exc), 'error')
                return redirect(request.referrer or url_for('features'))
        elif data_source_kind == "new-simulation":
            simulation_file_path = (request.form.get("simulation_file_path") or "").strip()
            data_upload = _ensure_files_parsed().get("data_file")
            has_upload = data_upload is not None and bool(data_upload.filename)
            if not has_upload:
                if not simulation_file_path:
                    flash('Provide a simulation output file path or upload a simulation output dataframe.', 'error')
                    return redirect(request.referrer or url_for('features'))
                try:
                    simulation_path = _validate_simulation_file_path(simulation_file_path)
                    app.logger.warning(
                        "[compute %s] simulation file mode path=%s",
                        job_id,
                        simulation_path,
                    )
                except Exception as exc:
                    flash(f"Invalid simulation output file path: {exc}", 'error')
                    return redirect(request.referrer or url_for('features'))
        elif data_source_kind == "new-empirical":
            if empirical_source_mode == "server-path":
                empirical_folder_path = (request.form.get("empirical_folder_path") or "").strip()
                try:
                    empirical_paths = _collect_empirical_folder_files(empirical_folder_path)
                    app.logger.warning(
                        "[compute %s] empirical folder mode path=%s files=%d",
                        job_id,
                        os.path.realpath(empirical_folder_path) if empirical_folder_path else "",
                        len(empirical_paths),
                    )
                except Exception as exc:
                    flash(f"Invalid empirical folder path: {exc}", 'error')
                    return redirect(request.referrer or url_for('features'))
            else:
                if request.content_length and request.content_length > MAX_EMPIRICAL_UPLOAD_BYTES:
                    max_gb = MAX_EMPIRICAL_UPLOAD_BYTES / float(1024 ** 3)
                    flash(
                        f"Empirical upload too large ({request.content_length / float(1024 ** 3):.2f} GB). "
                        f"Use at most {max_gb:.2f} GB or switch to server folder path mode.",
                        'error',
                    )
                    return redirect(request.referrer or url_for('features'))
                empirical_uploads = [f for f in _ensure_files_parsed().getlist("empirical_files") if f and f.filename]
                app.logger.warning("[compute %s] empirical upload count=%d", job_id, len(empirical_uploads))
                if not empirical_uploads:
                    flash('Upload at least one empirical recording file (folder upload is supported).', 'error')
                    return redirect(request.referrer or url_for('features'))
        else:
            flash('Unknown data source for features computation.', 'error')
            return redirect(request.referrer or url_for('features'))
        estimated_time_remaining = None

    if computation_type != 'features':
        _ensure_files_parsed()
    if computation_type != 'features' and len(uploaded_files) == 0 and computation_type not in {'inference', 'field_potential_proxy', 'field_potential_kernel', 'field_potential_meeg'}:
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

        has_features = _has_upload("training_features_file", "file-upload-x", "features_train_file")
        has_parameters = _has_upload("training_parameters_file", "file-upload-y", "parameters_train_file")
        if not has_features or not has_parameters:
            flash('Upload both Features Data and Parameters Data for training.', 'error')
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

        has_uploaded_features = _has_upload("features_predict_file", "features_predict", "file-upload-features")
        if not has_uploaded_features and not has_existing_features:
            flash('Upload features data or use an auto-loaded features file.', 'error')
            return redirect(request.referrer or url_for('inference'))

        has_uploaded_model = _has_upload("model_file", "model-file", "file-upload-model")
        has_uploaded_scaler = _has_upload("scaler_file", "scaler-file", "file-upload-scaler")
        has_uploaded_density = _has_upload("density_estimator_file", "density-estimator-file", "file-upload-density-estimator")
        folder_uploads = [f for f in files_obj.getlist("inference_assets_folder") if f and f.filename]
        has_folder_uploads = bool(folder_uploads)
        has_individual_assets = has_uploaded_model or has_uploaded_scaler or has_uploaded_density
        assets_upload_mode = (request.form.get("assets_upload_mode") or "").strip().lower()

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
            base = os.path.basename(folder_upload.filename or "").lower()
            if base.startswith("model") or base == "model.pkl":
                folder_has_model = True
                break
        if not has_uploaded_model and not folder_has_model:
            flash('Upload a model file, or upload a folder that contains a model file (e.g. model.pkl).', 'error')
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

    os.makedirs(temp_uploaded_files, exist_ok=True)

    # If everything is OK, save/prepare the file(s)
    file_paths = {}
    prepared_features_df = None
    empirical_upload_paths = None
    parser_config_obj = None
    if computation_type == "features":
        data_source_kind = (request.form.get("data_source_kind") or "new-simulation").strip()
        empirical_source_mode = (request.form.get("empirical_source_mode") or "upload").strip()

        if data_source_kind == "pipeline":
            existing_path = _validate_feature_existing_path(request.form.get("existing_data_path"))
            copied_name = f"features_data_file_0_{job_id}_{os.path.basename(existing_path)}"
            copied_path = os.path.join(temp_uploaded_files, copied_name)
            shutil.copy2(existing_path, copied_path)
            try:
                normalized_path = _normalize_features_input_path(copied_path, request.form, job_id)
                if normalized_path != copied_path and os.path.exists(copied_path):
                    os.remove(copied_path)
                file_paths["data_file"] = normalized_path
            except Exception as exc:
                flash(f"Failed to prepare selected pipeline file: {exc}", "error")
                return redirect(request.referrer or url_for('features'))

        elif data_source_kind == "new-simulation":
            simulation_file_path = (request.form.get("simulation_file_path") or "").strip()
            if simulation_file_path:
                try:
                    source_path = _validate_simulation_file_path(simulation_file_path)
                    copied_name = f"features_data_file_0_{job_id}_{os.path.basename(source_path)}"
                    copied_path = os.path.join(temp_uploaded_files, copied_name)
                    shutil.copy2(source_path, copied_path)
                    normalized_path = _normalize_features_input_path(copied_path, request.form, job_id)
                    if normalized_path != copied_path and os.path.exists(copied_path):
                        os.remove(copied_path)
                    file_paths["data_file"] = normalized_path
                except Exception as exc:
                    flash(f"Failed to prepare simulation file from file path: {exc}", "error")
                    return redirect(request.referrer or url_for('features'))
            else:
                file = _ensure_files_parsed().get("data_file")
                safe_name = secure_filename(file.filename)
                unique_filename = f"{computation_type}_data_file_0_{job_id}_{safe_name}"
                file_path = os.path.join(temp_uploaded_files, unique_filename)
                file.save(file_path)
                try:
                    normalized_path = _normalize_features_input_path(file_path, request.form, job_id)
                    if normalized_path != file_path and os.path.exists(file_path):
                        os.remove(file_path)
                    file_paths["data_file"] = normalized_path
                except Exception as exc:
                    flash(f"Failed to prepare uploaded simulation file: {exc}", "error")
                    return redirect(request.referrer or url_for('features'))

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
                    source_paths = _collect_empirical_folder_files(request.form.get("empirical_folder_path"))
                    for idx, source_path in enumerate(source_paths):
                        safe_name = os.path.basename(source_path)
                        ext = Path(safe_name).suffix.lower()
                        empirical_upload_paths.append({
                            "name": safe_name,
                            "ext": ext,
                            "path": source_path,
                        })
                    app.logger.warning(
                        "[compute %s] using empirical server-path files=%d",
                        job_id,
                        len(empirical_upload_paths),
                    )
                else:
                    empirical_uploads = [f for f in _ensure_files_parsed().getlist("empirical_files") if f and f.filename]
                    save_started = time.perf_counter()
                    saved_bytes = 0
                    for idx, upload in enumerate(empirical_uploads):
                        safe_name = secure_filename(upload.filename)
                        if not safe_name:
                            continue
                        ext = Path(safe_name).suffix.lower()
                        if ext not in FEATURES_PARSER_FILE_EXTENSIONS:
                            continue
                        unique_filename = f"features_empirical_file_{idx}_{job_id}_{safe_name}"
                        file_path = os.path.join(temp_uploaded_files, unique_filename)
                        save_one_started = time.perf_counter()
                        upload.save(file_path)
                        save_one_ms = (time.perf_counter() - save_one_started) * 1000.0
                        if not os.path.exists(file_path) or os.path.getsize(file_path) <= 0:
                            continue
                        file_size = os.path.getsize(file_path)
                        saved_bytes += file_size
                        file_paths[f"empirical_file_{idx}"] = file_path
                        empirical_upload_paths.append({
                            "name": safe_name,
                            "ext": ext,
                            "path": file_path,
                        })
                        if idx == 0 or (idx + 1) % 25 == 0 or (idx + 1) == len(empirical_uploads):
                            app.logger.warning(
                                "[compute %s] saved empirical file %d/%d (%.2f MB, %.1f ms, cumulative %.2f MB)",
                                job_id,
                                idx + 1,
                                len(empirical_uploads),
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
            except Exception as exc:
                app.logger.exception("[compute %s] empirical staging failed", job_id)
                flash(f"Empirical parser configuration failed: {exc}", "error")
                return redirect(request.referrer or url_for('features'))
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
            file_path = os.path.join(temp_uploaded_files, unique_filename)
            file.save(file_path)
            # Save dictionary with file_key: file_path
            normalized_key = file_key
            if computation_type == "inference":
                normalized_key = inference_file_key_map.get(file_key, file_key)
            file_paths[normalized_key] = file_path
        if computation_type == "inference":
            folder_uploads = [f for f in files_obj.getlist("inference_assets_folder") if f and f.filename]
            for idx, upload in enumerate(folder_uploads):
                raw_name = upload.filename or ""
                base_name = os.path.basename(raw_name)
                safe_name = secure_filename(base_name)
                if not safe_name:
                    continue
                unique_filename = f"{computation_type}_inference_assets_folder_{idx}_{job_id}_{safe_name}"
                file_path = os.path.join(temp_uploaded_files, unique_filename)
                upload.save(file_path)

                lower = safe_name.lower()
                normalized_key = None
                if lower.startswith("model") or lower == "model.pkl":
                    normalized_key = "model_file"
                elif lower.startswith("scaler") or lower == "scaler.pkl":
                    normalized_key = "scaler_file"
                elif "density_estimator" in lower or lower.startswith("density"):
                    normalized_key = "density_estimator_file"

                # Do not override explicit single-file uploads from dedicated controls.
                if normalized_key and normalized_key not in file_paths:
                    file_paths[normalized_key] = file_path
                else:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except OSError:
                        pass
        if computation_type == "inference":
            existing_features_file = (request.form.get("existing_features_file") or "").strip()
            if "features_predict" not in file_paths and existing_features_file:
                available = set(_list_features_data_files())
                if existing_features_file in available:
                    src_path = os.path.join(_features_data_dir(create=False), existing_features_file)
                    if os.path.isfile(src_path):
                        copied_name = f"inference_features_predict_0_{job_id}_{os.path.basename(src_path)}"
                        copied_path = os.path.join(temp_uploaded_files, copied_name)
                        shutil.copy2(src_path, copied_path)
                        file_paths["features_predict"] = copied_path

    data = request.form.to_dict() # Get parameters from form POST
    # Add file information to the data dictionary
    data['file_paths'] = file_paths
    if prepared_features_df is not None:
        data["prepared_features_df"] = prepared_features_df
    if empirical_upload_paths is not None:
        data["empirical_upload_paths"] = empirical_upload_paths
    if parser_config_obj is not None:
        data["parser_config_obj"] = parser_config_obj

    # Store initial status
    job_status[job_id] = {
        "status": "in_progress",
        "progress": 0,
        "start_time": time.time(),
        "estimated_time_remaining": estimated_time_remaining,
        "results": None,
        "error": False,
        "output": "",
        "progress_mode": "manual" if computation_type in {"features", "inference", "inference_training"} else "time",
    }

    # Submit the long-running task according to the computation type.
    # For features, defer start slightly so the redirect response is sent first.
    if computation_type == "features":
        app.logger.warning(
            "[compute %s] enqueue features job after redirect (route elapsed %.1f s)",
            job_id,
            time.perf_counter() - route_started,
        )
        threading.Timer(
            0.05,
            lambda: executor.submit(func, job_id, job_status, data, temp_uploaded_files),
        ).start()
    else:
        executor.submit(func, job_id, job_status, data, temp_uploaded_files)

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
