import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _job_status(job_id="job"):
    return {job_id: {"status": "running", "progress": 0, "output": ""}}


def _simulation_file_paths(output_dir):
    root = Path(output_dir)
    bundle = root / "simulation.pkl"
    if not bundle.is_file():
        legacy_bundle = root / "sim_data.pkl"
        bundle = legacy_bundle if legacy_bundle.is_file() else None

    def _pick(name):
        candidate = root / name
        if candidate.is_file():
            return str(candidate)
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


def _example_input_for_kernel(sim_output, example_paths):
    model_name = str(sim_output["form_data"].get("sim_model", "")).strip().lower()
    if model_name not in example_paths:
        if "hagen" in model_name:
            model_name = "hagen"
        elif "cavallari" in model_name:
            model_name = "cavallari"
        elif "four" in model_name:
            model_name = "four_area"
    if model_name not in example_paths:
        raise KeyError(f"Unknown simulation model key '{model_name}'.")
    file_paths = _simulation_file_paths(sim_output["output_dir"])
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


def _write_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


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


@pytest.mark.parametrize(
    "fixture_name",
    ("hagen_simulation_output", "cavallari_simulation_output", "four_area_simulation_output"),
)
def test_field_potential_proxy_fr_runs_for_all_simulation_examples(
    request,
    compute_utils_module,
    fake_field_potential_backend,
    fixture_name,
):
    sim_output = request.getfixturevalue(fixture_name)
    sim_files = _simulation_file_paths(sim_output["output_dir"])
    job_id = f"proxy_{fixture_name}"
    job_status = _job_status(job_id)

    compute_utils_module.field_potential_proxy_computation(
        job_id,
        job_status,
        {
            "proxy_method": "FR",
            "bin_size": "1.0",
            "file_paths": {
                "times_file": sim_files["times"],
                "gids_file": sim_files["gids"],
                "network_file": sim_files["network"],
                "dt_file": sim_files["dt"],
            },
        },
        temp_uploaded_files=str(sim_output["output_dir"]),
    )

    finished = _assert_finished(job_status, job_id)
    with open(finished["results"], "rb") as handle:
        payload = pickle.load(handle)
    assert isinstance(payload, list) and payload
    first_row = payload[0].iloc[0]
    assert float(first_row["dt_ms"]) == pytest.approx(1.0)
    assert np.asarray(first_row["data"]).size > 0


@pytest.mark.parametrize(
    "fixture_name",
    ("hagen_simulation_output", "cavallari_simulation_output", "four_area_simulation_output"),
)
def test_field_potential_kernel_cdm_z_runs_for_all_examples(
    request,
    compute_utils_module,
    fake_field_potential_backend,
    field_potential_example_paths,
    fixture_name,
):
    sim_output = request.getfixturevalue(fixture_name)
    cfg = _example_input_for_kernel(sim_output, field_potential_example_paths)
    job_id = f"kernel_z_{fixture_name}"
    job_status = _job_status(job_id)

    params = {
        **cfg,
        "cdm_component": "z",
        "cdm_dt": "0.2",
        "cdm_tstop": "100.0",
    }
    compute_utils_module.field_potential_kernel_computation(
        job_id,
        job_status,
        params,
        temp_uploaded_files=str(sim_output["output_dir"]),
    )

    finished = _assert_finished(job_status, job_id)
    with open(finished["results"], "rb") as handle:
        cdm_payload = pickle.load(handle)
    assert isinstance(cdm_payload, list) and cdm_payload
    first_trial = cdm_payload[0]
    assert first_trial.get("component_axis") == "z"
    signal = first_trial.get("sum")
    if isinstance(signal, dict):
        signal = next(iter(signal.values()))
    assert float(np.max(np.abs(np.asarray(signal, dtype=float)))) > 0.0


def test_field_potential_kernel_cdm_xyz_metadata(
    hagen_simulation_output,
    compute_utils_module,
    fake_field_potential_backend,
    field_potential_example_paths,
):
    cfg = _example_input_for_kernel(hagen_simulation_output, field_potential_example_paths)
    job_id = "kernel_xyz_hagen"
    job_status = _job_status(job_id)
    params = {
        **cfg,
        "cdm_component": "xyz",
        "cdm_dt": "0.2",
        "cdm_tstop": "100.0",
    }

    compute_utils_module.field_potential_kernel_computation(
        job_id,
        job_status,
        params,
        temp_uploaded_files=str(hagen_simulation_output["output_dir"]),
    )

    finished = _assert_finished(job_status, job_id)
    with open(finished["results"], "rb") as handle:
        cdm_payload = pickle.load(handle)
    first_trial = cdm_payload[0]
    assert first_trial.get("component_axis") == "xyz"
    assert first_trial.get("component_index") is None
    signal = np.asarray(first_trial["sum"], dtype=float)
    assert signal.ndim == 2 and signal.shape[0] == 3


def test_field_potential_kernel_rejects_invalid_component(
    hagen_simulation_output,
    compute_utils_module,
    fake_field_potential_backend,
    field_potential_example_paths,
):
    cfg = _example_input_for_kernel(hagen_simulation_output, field_potential_example_paths)
    job_id = "kernel_invalid_component"
    job_status = _job_status(job_id)

    compute_utils_module.field_potential_kernel_computation(
        job_id,
        job_status,
        {
            **cfg,
            "cdm_component": "x",
            "cdm_dt": "0.2",
            "cdm_tstop": "100.0",
        },
        temp_uploaded_files=str(hagen_simulation_output["output_dir"]),
    )

    failed = _assert_failed(job_status, job_id)
    assert "CDM component must be 'z' or 'xyz (all)'" in str(failed["error"])


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


def test_field_potential_meeg_reads_z_component_metadata(
    tmp_path,
    compute_utils_module,
    fake_field_potential_backend,
):
    cdm_signal = np.linspace(0.1, 1.0, 32)
    cdm_path = _write_pickle(
        tmp_path / "cdm.pkl",
        [
            {
                "sum": cdm_signal,
                "metadata": {
                    "dt_ms": 0.5,
                    "decimation_factor": 1,
                    "fs_hz": 2000.0,
                    "component_axis": "z",
                    "component_index": 2,
                },
            }
        ],
    )

    status = _run_meeg_compute(compute_utils_module, "meeg_z", cdm_path)
    finished = _assert_finished(status, "meeg_z")
    with open(finished["results"], "rb") as handle:
        payload = pickle.load(handle)
    row = payload[0].iloc[0]
    data = np.asarray(row["data"], dtype=float)
    assert data.shape[0] == 1
    assert float(np.max(np.abs(data))) > 0.0
    assert "z-component-only input" in status["meeg_z"]["output"]


def test_field_potential_meeg_reads_xyz_component_metadata(
    tmp_path,
    compute_utils_module,
    fake_field_potential_backend,
):
    t = np.linspace(0.1, 1.0, 40)
    cdm_signal = np.vstack([t, 2.0 * t, 3.0 * t])
    cdm_path = _write_pickle(
        tmp_path / "cdm.pkl",
        [
            {
                "sum": cdm_signal,
                "metadata": {
                    "dt_ms": 0.5,
                    "decimation_factor": 1,
                    "fs_hz": 2000.0,
                    "component_axis": "xyz",
                },
            }
        ],
    )

    status = _run_meeg_compute(compute_utils_module, "meeg_xyz", cdm_path)
    finished = _assert_finished(status, "meeg_xyz")
    with open(finished["results"], "rb") as handle:
        payload = pickle.load(handle)
    row = payload[0].iloc[0]
    data = np.asarray(row["data"], dtype=float)
    assert data.shape[0] == 1
    assert float(np.max(np.abs(data))) > 0.0
    assert "all xyz components" in status["meeg_xyz"]["output"]


def test_field_potential_meeg_four_area_simultaneous_mode_works(
    tmp_path,
    compute_utils_module,
    fake_field_potential_backend,
):
    area_payload = {
        "frontal": np.linspace(0.1, 0.5, 24),
        "parietal": np.linspace(0.2, 0.6, 24),
        "temporal": np.linspace(0.3, 0.7, 24),
        "occipital": np.linspace(0.4, 0.8, 24),
    }
    cdm_path = _write_pickle(
        tmp_path / "cdm.pkl",
        [
            {
                "sum": area_payload,
                "metadata": {"dt_ms": 1.0, "component_axis": "z"},
            }
        ],
    )

    status = _run_meeg_compute(
        compute_utils_module,
        "meeg_four_area",
        cdm_path,
        meeg_simulation_model="four_area",
        meeg_sensor_locations="[[90000.0, 0.0, 0.0], [0.0, 90000.0, 0.0]]",
    )
    finished = _assert_finished(status, "meeg_four_area")
    with open(finished["results"], "rb") as handle:
        payload = pickle.load(handle)
    data = np.asarray(payload[0].iloc[0]["data"], dtype=float)
    assert data.shape[0] == 2
    assert "fixed dipole locations" in status["meeg_four_area"]["output"]


def test_field_potential_meeg_per_sensor_independent_requires_equal_sensor_and_dipole_counts(
    tmp_path,
    compute_utils_module,
    fake_field_potential_backend,
):
    cdm_path = _write_pickle(
        tmp_path / "cdm.pkl",
        [
            {
                "sum": np.linspace(0.1, 1.0, 32),
                "metadata": {"dt_ms": 1.0, "component_axis": "z"},
            }
        ],
    )

    status = _run_meeg_compute(
        compute_utils_module,
        "meeg_pairs_mismatch",
        cdm_path,
        meeg_forward_mode="per_sensor_independent",
        meeg_dipole_locations="[[78000.0, 0.0, 0.0], [0.0, 78000.0, 0.0]]",
        meeg_sensor_locations="[[90000.0, 0.0, 0.0]]",
    )
    failed = _assert_failed(status, "meeg_pairs_mismatch")
    assert "same number of dipole and sensor locations" in str(failed["error"])


def test_field_potential_meeg_per_sensor_independent_disallowed_for_four_area(
    tmp_path,
    compute_utils_module,
    fake_field_potential_backend,
):
    cdm_path = _write_pickle(
        tmp_path / "cdm.pkl",
        [
            {
                "sum": {
                    "frontal": np.linspace(0.1, 0.5, 16),
                    "parietal": np.linspace(0.1, 0.5, 16),
                    "temporal": np.linspace(0.1, 0.5, 16),
                    "occipital": np.linspace(0.1, 0.5, 16),
                },
                "metadata": {"dt_ms": 1.0, "component_axis": "z"},
            }
        ],
    )

    status = _run_meeg_compute(
        compute_utils_module,
        "meeg_four_area_independent",
        cdm_path,
        meeg_simulation_model="four_area",
        meeg_forward_mode="per_sensor_independent",
        meeg_sensor_locations="[[90000.0, 0.0, 0.0], [0.0, 90000.0, 0.0]]",
    )
    failed = _assert_failed(status, "meeg_four_area_independent")
    assert "not available for the four-area model" in str(failed["error"])
