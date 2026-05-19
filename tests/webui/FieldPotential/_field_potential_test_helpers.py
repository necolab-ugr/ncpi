import ast
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _job_status(job_id="job"):
    return {job_id: {"status": "running", "progress": 0, "output": ""}}


def _write_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


def _simulation_file_paths(output_dir):
    root = Path(output_dir)
    bundle = root / "simulation.pkl"
    if not bundle.is_file():
        legacy_bundle = root / "sim_data.pkl"
        bundle = legacy_bundle if legacy_bundle.is_file() else None
    if bundle is None:
        nested_bundle = next((path for path in root.rglob("simulation.pkl") if path.is_file()), None)
        if nested_bundle is None:
            nested_bundle = next((path for path in root.rglob("sim_data.pkl") if path.is_file()), None)
        if nested_bundle is None:
            nested_bundle = next((path for path in root.rglob("*_simulation_output.pkl") if path.is_file()), None)
        bundle = nested_bundle

    def _pick(name):
        candidate = root / name
        if candidate.is_file():
            return str(candidate)
        nested = next((path for path in root.rglob(name) if path.is_file()), None)
        if nested is not None:
            return str(nested)
        if bundle is not None and bundle.is_file():
            return str(bundle)
        raise FileNotFoundError(f"Missing simulation output file '{name}' in {root}.")

    return {
        "times": _pick("times.pkl"),
        "gids": _pick("gids.pkl"),
        "dt": _pick("dt.pkl"),
        "network": _pick("network.pkl"),
        "population_sizes": _pick("population_sizes.pkl"),
    }


def _stage_simulation_input_files(sim_output):
    source_files = _simulation_file_paths(sim_output["output_dir"])
    staged_root = Path(tempfile.mkdtemp(prefix="fp_inputs_"))
    staged_files = {}
    copied_sources = {}
    for key, source_path in source_files.items():
        source = Path(source_path).resolve()
        cache_key = str(source)
        if cache_key not in copied_sources:
            destination = staged_root / source.name
            if destination.exists():
                destination = staged_root / f"{len(copied_sources)}_{source.name}"
            shutil.copy2(str(source), str(destination))
            copied_sources[cache_key] = str(destination)
        staged_files[key] = copied_sources[cache_key]
    return staged_files


def _make_temp_uploaded_output_dir(prefix="fp_results_"):
    return tempfile.mkdtemp(prefix=prefix)


def _example_input_for_kernel(sim_output, example_paths, sim_file_paths=None):
    model_name = str(sim_output["form_data"].get("sim_model", "")).strip().lower()
    if model_name not in example_paths:
        if "hagen" in model_name:
            model_name = "hagen"
        elif "cavallari" in model_name:
            model_name = "cavallari"
        elif "four" in model_name:
            model_name = "four_area"
    if model_name not in example_paths:
        file_paths = sim_file_paths or _simulation_file_paths(sim_output["output_dir"])
        try:
            with open(file_paths["network"], "rb") as handle:
                network_payload = pickle.load(handle)
        except Exception:
            network_payload = None
        if isinstance(network_payload, list):
            network_payload = network_payload[0] if network_payload else {}
        if isinstance(network_payload, dict):
            if "areas" in network_payload:
                model_name = "four_area"
            elif network_payload.get("model") == "iaf_bw_2003":
                model_name = "cavallari"
            else:
                model_name = "hagen"
    if model_name not in example_paths:
        raise KeyError(f"Unknown simulation model key '{model_name}'.")
    file_paths = sim_file_paths or _simulation_file_paths(sim_output["output_dir"])
    cfg = dict(example_paths[model_name])
    cfg.update({
        "output_sim_path": str(sim_output["output_dir"]),
        "file_paths": {
            "kernel_spike_times_file": file_paths["times"],
            "kernel_population_sizes_file": file_paths["population_sizes"],
            "kernel_network_file": file_paths["network"],
        },
    })
    return cfg


class _FakeEEGModel:
    def __init__(self, sensor_locations, **kwargs):
        sensors = np.asarray(sensor_locations, dtype=float)
        if sensors.ndim == 1:
            sensors = sensors.reshape(1, 3)
        self.sensor_locations = sensors

    def get_transformation_matrix(self, dipole_location):
        loc = np.asarray(dipole_location, dtype=float).reshape(1, 3)
        vec = self.sensor_locations - loc
        dist = np.linalg.norm(vec, axis=1, keepdims=True)
        dist = np.where(dist <= 0.0, 1.0, dist)
        return vec / (dist ** 3)


class _FakeFieldPotential:
    def __init__(self):
        self.nyhead = None

    def compute_proxy(self, method, sim_data, sim_step, excitatory_only=None):
        if isinstance(sim_data, dict) and "FR" in sim_data:
            arr = np.asarray(sim_data["FR"], dtype=float)
            if arr.ndim == 2:
                return np.mean(arr, axis=0)
            return arr.reshape(-1)
        for key in ("AMPA", "GABA", "Vm"):
            if isinstance(sim_data, dict) and key in sim_data:
                arr = np.asarray(sim_data[key], dtype=float)
                if arr.ndim == 2:
                    return np.mean(arr, axis=0)
                return arr.reshape(-1)
        return np.linspace(0.0, 1.0, 32)

    def create_kernel(
        self,
        mc_folder,
        KernelParams,
        biophys,
        dt,
        tstop,
        output_sim_path=None,
        electrodeParameters=None,
        CDM=True,
        probes=None,
        cdm_probe=None,
        mean_nu_X=None,
        Vrest=None,
        t_X=None,
        tau=None,
        g_eff=None,
        n_ext=None,
        weights=None,
    ):
        t = np.linspace(0.1, 1.0, 48)
        vec = np.vstack([t, 2.0 * t, 3.0 * t])
        return {
            "kernel_payload": {
                "KernelApproxCurrentDipoleMoment": vec,
                "CurrentDipoleMoment": vec,
            }
        }

    def compute_cdm_lfp_from_kernels(
        self,
        kernels,
        spike_times,
        cdm_dt,
        cdm_tstop,
        population_sizes=None,
        transient=0.0,
        probe=None,
        component=None,
        mode=None,
        scale=None,
    ):
        dt = max(float(cdm_dt), 1e-6)
        n_times = int(max(16, min(128, round(float(cdm_tstop) / dt))))
        base = np.linspace(0.1, 1.0, n_times)

        areas = []
        if isinstance(spike_times, dict):
            if spike_times and all(isinstance(v, dict) for v in spike_times.values()):
                areas = [str(area).strip() for area in spike_times.keys() if str(area).strip()]
            elif spike_times:
                for key in spike_times.keys():
                    token = str(key)
                    if token.endswith(")") and "(" in token:
                        area = token[token.rfind("(") + 1:-1].strip()
                        if area and area not in areas:
                            areas.append(area)
        if not areas:
            areas = ["global"]

        payload = {}
        for index, area in enumerate(areas):
            scale_area = float(index + 1)
            if component is None:
                signal = np.vstack([
                    base * scale_area,
                    base * scale_area * 1.5,
                    base * scale_area * 2.0,
                ])
            else:
                signal = base * scale_area * float(component + 1)
            key = "P:P" if area == "global" else f"P({area}):P({area})"
            payload[key] = signal
        return payload

    def _normalize_cdm_and_locations(self, cdm, dipole_locations):
        cdm_arr = np.asarray(cdm, dtype=float)
        loc_arr = np.asarray(dipole_locations, dtype=float)
        if loc_arr.ndim == 1:
            loc_arr = loc_arr.reshape(1, 3)

        if cdm_arr.ndim == 2 and cdm_arr.shape[0] == 3:
            return [cdm_arr], [loc_arr[0]]

        if cdm_arr.ndim == 3 and cdm_arr.shape[1] == 3:
            if loc_arr.shape[0] == 1 and cdm_arr.shape[0] > 1:
                loc_arr = np.repeat(loc_arr, cdm_arr.shape[0], axis=0)
            if loc_arr.shape[0] != cdm_arr.shape[0]:
                raise ValueError("Dipole locations must match CDM dipole count.")
            return [cdm_arr[i] for i in range(cdm_arr.shape[0])], [loc_arr[i] for i in range(loc_arr.shape[0])]

        raise ValueError(f"Unsupported CDM shape: {cdm_arr.shape}")

    def _load_eegmegcalc_model(self, model_name):
        return _FakeEEGModel

    def _get_eeg_1020_locations(self):
        return np.asarray([[0.0, 0.0, 0.0]], dtype=float), None


@pytest.fixture()
def fake_field_potential_backend(monkeypatch, compute_utils_module):
    monkeypatch.setattr(compute_utils_module.ncpi, "FieldPotential", _FakeFieldPotential)
    return _FakeFieldPotential


def _assert_finished(job_status, job_id):
    payload = job_status[job_id]
    assert payload["status"] == "finished", payload.get("output", "")
    assert payload.get("error") is False
    return payload


def _assert_failed(job_status, job_id):
    payload = job_status[job_id]
    assert payload["status"] == "failed"
    return payload


def _load_pickle_file(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _extract_simulation_component(payload, field_name):
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and field_name in payload[0]:
            return [item.get(field_name) for item in payload]
        return payload
    if isinstance(payload, dict) and field_name in payload:
        return payload.get(field_name)
    return payload


def _as_trial_list(payload):
    return list(payload) if isinstance(payload, list) else [payload]


def _pick_trial(payload, trial_idx):
    if isinstance(payload, list):
        idx = max(0, min(int(trial_idx), len(payload) - 1))
        return payload[idx]
    return payload


def _flatten_numeric_chunks(value):
    chunks = []

    def _collect(obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            for nested in obj.values():
                _collect(nested)
            return
        if isinstance(obj, (list, tuple)):
            for nested in obj:
                _collect(nested)
            return
        arr = np.asarray(obj).ravel()
        if arr.size == 0:
            return
        try:
            arr = arr.astype(float, copy=False)
        except Exception:
            return
        if arr.size:
            chunks.append(arr)

    _collect(value)
    if not chunks:
        return np.asarray([], dtype=float)
    return np.concatenate(chunks)


def _mean_firing_rate_from_spikes(times, gids, bin_size_ms):
    times_flat = _flatten_numeric_chunks(times)
    gids_flat = _flatten_numeric_chunks(gids)
    if times_flat.size == 0:
        return np.zeros((1, 0), dtype=float)
    if float(bin_size_ms) <= 0.0:
        raise ValueError("bin_size_ms must be > 0.")
    n_units = max(int(np.unique(gids_flat).size), 1) if gids_flat.size > 0 else 1
    t_min = float(np.min(times_flat))
    t_max = float(np.max(times_flat))
    bins = np.arange(t_min, t_max + float(bin_size_ms), float(bin_size_ms))
    hist, _ = np.histogram(times_flat, bins=bins)
    return (hist.astype(float) / float(n_units)).reshape(1, -1)


def _sum_signal_dict(signal_dict):
    total = None
    for value in (signal_dict or {}).values():
        arr = np.asarray(value, dtype=float)
        if total is None:
            total = np.array(arr, copy=True)
        else:
            total = total + arr
    return total


def _assert_nested_allclose(actual, expected, atol=1e-10, rtol=1e-10):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual.keys()) == set(expected.keys())
        for key in expected.keys():
            _assert_nested_allclose(actual[key], expected[key], atol=atol, rtol=rtol)
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple))
        assert len(actual) == len(expected)
        for a_item, e_item in zip(actual, expected):
            _assert_nested_allclose(a_item, e_item, atol=atol, rtol=rtol)
        return
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=atol, rtol=rtol)


def _python_reference_proxy_trial_rows(sim_output, bin_size_ms=1.0):
    sim_files = _simulation_file_paths(sim_output["output_dir"])
    times_payload = _extract_simulation_component(_load_pickle_file(sim_files["times"]), "times")
    gids_payload = _extract_simulation_component(_load_pickle_file(sim_files["gids"]), "gids")
    times_trials = _as_trial_list(times_payload)
    gids_trials = _as_trial_list(gids_payload)
    trial_count = max(len(times_trials), len(gids_trials))

    potential = _FakeFieldPotential()
    rows = []
    for trial_idx in range(trial_count):
        trial_times = _pick_trial(times_payload, trial_idx)
        trial_gids = _pick_trial(gids_payload, trial_idx)
        trial_fr = _mean_firing_rate_from_spikes(trial_times, trial_gids, bin_size_ms)
        area_payload = None
        if (
            isinstance(trial_times, dict)
            and isinstance(trial_gids, dict)
            and all(isinstance(v, dict) for v in trial_times.values())
            and all(isinstance(v, dict) for v in trial_gids.values())
        ):
            area_payload = {}
            for area_name in trial_times.keys():
                if area_name not in trial_gids:
                    continue
                area_fr = _mean_firing_rate_from_spikes(trial_times[area_name], trial_gids[area_name], bin_size_ms)
                area_payload[str(area_name)] = np.asarray(
                    potential.compute_proxy("FR", {"FR": area_fr}, float(bin_size_ms), excitatory_only=None),
                    dtype=float,
                )

        global_proxy = np.asarray(
            potential.compute_proxy("FR", {"FR": trial_fr}, float(bin_size_ms), excitatory_only=None),
            dtype=float,
        )
        rows.append({
            "data": area_payload if area_payload else global_proxy,
            "dt_ms": float(bin_size_ms),
            "decimation_factor": 1,
        })
    return rows


def _python_reference_kernel_trial_rows(sim_output, cdm_dt=0.2, cdm_tstop=100.0, component_axis="z"):
    sim_files = _simulation_file_paths(sim_output["output_dir"])
    spike_times_payload = _extract_simulation_component(_load_pickle_file(sim_files["times"]), "times")
    pop_sizes_payload = _extract_simulation_component(_load_pickle_file(sim_files["population_sizes"]), "population_sizes")
    spike_trials = _as_trial_list(spike_times_payload)
    pop_trials = _as_trial_list(pop_sizes_payload)
    trial_count = max(len(spike_trials), len(pop_trials))
    component = None if component_axis == "xyz" else 2

    potential = _FakeFieldPotential()
    kernels = potential.create_kernel(mc_folder="unused", KernelParams=object(), biophys=[], dt=float(cdm_dt), tstop=float(cdm_tstop))
    rows = []
    for trial_idx in range(trial_count):
        spike_times_trial = _pick_trial(spike_times_payload, trial_idx)
        pop_sizes_trial = _pick_trial(pop_sizes_payload, trial_idx)
        cdm_signals = potential.compute_cdm_lfp_from_kernels(
            kernels,
            spike_times_trial,
            float(cdm_dt),
            float(cdm_tstop),
            population_sizes=pop_sizes_trial,
            transient=0.0,
            probe="KernelApproxCurrentDipoleMoment",
            component=component,
            mode="same",
            scale=float(cdm_dt) * 1e-3,
        )
        rows.append({
            "raw_signals": cdm_signals,
            "sum": _sum_signal_dict(cdm_signals),
            "dt_ms": float(cdm_dt),
            "component_axis": str(component_axis),
        })
    return rows


def _python_reference_kernel_raw_signals_by_probe(sim_output, probe_name, cdm_dt=0.2, cdm_tstop=100.0, component_axis="z"):
    sim_files = _simulation_file_paths(sim_output["output_dir"])
    spike_times_payload = _extract_simulation_component(_load_pickle_file(sim_files["times"]), "times")
    pop_sizes_payload = _extract_simulation_component(_load_pickle_file(sim_files["population_sizes"]), "population_sizes")
    spike_trials = _as_trial_list(spike_times_payload)
    pop_trials = _as_trial_list(pop_sizes_payload)
    trial_count = max(len(spike_trials), len(pop_trials))
    component = None if component_axis == "xyz" else 2

    potential = _FakeFieldPotential()
    kernels = potential.create_kernel(mc_folder="unused", KernelParams=object(), biophys=[], dt=float(cdm_dt), tstop=float(cdm_tstop))
    per_trial_signals = []
    for trial_idx in range(trial_count):
        spike_times_trial = _pick_trial(spike_times_payload, trial_idx)
        pop_sizes_trial = _pick_trial(pop_sizes_payload, trial_idx)
        signals = potential.compute_cdm_lfp_from_kernels(
            kernels,
            spike_times_trial,
            float(cdm_dt),
            float(cdm_tstop),
            population_sizes=pop_sizes_trial,
            transient=0.0,
            probe=str(probe_name),
            component=component,
            mode="same",
            scale=float(cdm_dt) * 1e-3,
        )
        per_trial_signals.append(signals)
    return per_trial_signals


def _coerce_cdm_sum_to_xyz(sum_payload, component_axis="z"):
    payload = sum_payload
    if isinstance(payload, dict):
        payload = _sum_signal_dict(payload)
    arr = np.asarray(payload, dtype=float)
    if arr.ndim == 1:
        axis = str(component_axis or "z").strip().lower()
        idx = 2 if axis not in {"x", "y", "z"} else {"x": 0, "y": 1, "z": 2}[axis]
        components = [np.zeros_like(arr), np.zeros_like(arr), np.zeros_like(arr)]
        components[idx] = arr
        return np.vstack(components)
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.T
    raise ValueError(f"Unsupported CDM payload shape for M/EEG reference: {arr.shape}")


def _python_reference_meeg_rows_from_cdm_payload(cdm_payload, model_name, dipole_location, sensor_locations):
    trials = _as_trial_list(cdm_payload)
    sensor_arr = np.asarray(sensor_locations, dtype=float)
    if sensor_arr.ndim == 1:
        sensor_arr = sensor_arr.reshape(1, 3)
    dipole_arr = np.asarray(dipole_location, dtype=float).reshape(3,)
    if str(model_name) == "FourSphereVolumeConductor":
        dipole_norm = np.linalg.norm(dipole_arr)
        if dipole_norm > 79000.0:
            dipole_arr = dipole_arr * (79000.0 / dipole_norm)
        sensor_norms = np.linalg.norm(sensor_arr, axis=1)
        mask = sensor_norms > 90000.0
        if np.any(mask):
            sensor_arr = np.array(sensor_arr, copy=True)
            sensor_arr[mask] = sensor_arr[mask] * (90000.0 / sensor_norms[mask]).reshape(-1, 1)

    reference_rows = []
    for trial_obj in trials:
        trial_row = dict(trial_obj.iloc[0].to_dict()) if isinstance(trial_obj, pd.DataFrame) else dict(trial_obj)
        cdm_xyz = _coerce_cdm_sum_to_xyz(trial_row.get("sum"), component_axis=trial_row.get("component_axis", "z"))
        model = _FakeEEGModel(sensor_arr)
        M = model.get_transformation_matrix(dipole_arr)
        n_times = cdm_xyz.shape[1]
        if str(model_name) in {"InfiniteHomogeneousVolCondMEG", "SphericallySymmetricVolCondMEG"}:
            data = np.zeros((sensor_arr.shape[0], 3, n_times), dtype=float)
            for sensor_idx in range(sensor_arr.shape[0]):
                scalar = M[sensor_idx] @ cdm_xyz
                data[sensor_idx] = np.tile(scalar, (3, 1))
        else:
            data = np.zeros((sensor_arr.shape[0], n_times), dtype=float)
            for sensor_idx in range(sensor_arr.shape[0]):
                data[sensor_idx] = M[sensor_idx] @ cdm_xyz
        reference_rows.append({
            "data": data,
            "dt_ms": float(trial_row.get("dt_ms")) if trial_row.get("dt_ms") is not None else None,
            "t_start_ms": trial_row.get("t_start_ms"),
            "t_stop_ms": trial_row.get("t_stop_ms"),
        })
    return reference_rows


def _run_meeg_compute(compute_utils_module, job_id, cdm_path, **extra_params):
    status = _job_status(job_id)
    params = {
        "file_paths": {"meeg_cdm_file": str(cdm_path)},
        "meeg_model": "FourSphereVolumeConductor",
        "meeg_model_kwargs": "{}",
        "meeg_forward_mode": "simultaneous_all_dipoles",
        "meeg_dipole_locations": "[[78000.0, 0.0, 0.0]]",
        "meeg_sensor_locations": "[[90000.0, 0.0, 0.0]]",
    }
    params.update(extra_params)
    compute_utils_module.field_potential_meeg_computation(
        job_id,
        status,
        params,
        temp_uploaded_files=str(Path(cdm_path).parent),
    )
    return status


def _parse_locations_literal(raw):
    return np.asarray(ast.literal_eval(raw), dtype=float)
