import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from werkzeug.datastructures import MultiDict

from tests.webui.FieldPotential._field_potential_test_helpers import (
    _FakeEEGModel,
    _FakeFieldPotential,
    _assert_failed,
    _assert_finished,
    _example_input_for_kernel,
    _job_status,
    _make_temp_uploaded_output_dir,
    _run_meeg_compute,
    _stage_simulation_input_files,
    _write_pickle,
)


def test_analysis_plot_warns_when_reducing_vector_signals_to_vector_magnitude(
    field_potential_webui_app_module,
    tmp_path,
):
    payload = [
        pd.DataFrame(
            [
                {
                    "data": np.ones((2, 3, 48), dtype=float),
                    "dt_ms": 1.0,
                    "metadata": {"dt_ms": 1.0},
                }
            ]
        )
    ]
    fp_path = _write_pickle(tmp_path / "meeg.pkl", payload)
    selected_key = f"field_potential::{Path(fp_path).resolve()}"
    form_data = MultiDict([
        ("sim_plot_type", "meeg"),
        ("sim_selected_file_keys", selected_key),
        ("sim_trial_start", "0"),
        ("sim_trial_end", "0"),
        ("sim_time_start", "0"),
        ("sim_time_end", "40"),
        ("sim_freq_min", "0"),
        ("sim_freq_max", "200"),
    ])

    field_potential_webui_app_module._clear_simulation_output_folder_all_files()
    if hasattr(field_potential_webui_app_module, "_clear_analysis_data_files"):
        field_potential_webui_app_module._clear_analysis_data_files()
    if hasattr(field_potential_webui_app_module, "_clear_analysis_selection_mode"):
        field_potential_webui_app_module._clear_analysis_selection_mode()
    if hasattr(field_potential_webui_app_module, "_clear_analysis_selected_simulation_keys"):
        field_potential_webui_app_module._clear_analysis_selected_simulation_keys()

    with field_potential_webui_app_module.app.test_client() as client:
        response = client.post("/analysis/plot/simulation", data=form_data)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Warning: this plot computes vector magnitude (module)" in html


def test_kernel_probe_selection_rejects_simultaneous_cdm_probe_choices(
    hagen_simulation_output,
    compute_utils_module,
    fake_field_potential_backend,
    field_potential_example_paths,
):
    sim_files = _stage_simulation_input_files(hagen_simulation_output)
    cfg = _example_input_for_kernel(hagen_simulation_output, field_potential_example_paths, sim_file_paths=sim_files)
    job_id = "kernel_invalid_probe_selection"
    job_status = _job_status(job_id)
    compute_utils_module.field_potential_kernel_computation(
        job_id,
        job_status,
        {
            **cfg,
            "probe_selection_present": "1",
            "probe_kernel_approx": "true",
            "probe_current_dipole": "true",
            "probe_gauss_cylinder": "false",
            "cdm_component": "z",
        },
        temp_uploaded_files=_make_temp_uploaded_output_dir("kernel_results_"),
    )
    failed = _assert_failed(job_status, job_id)
    assert "Select only one of KernelApproxCurrentDipoleMoment or CurrentDipoleMoment." in str(failed["error"])


def test_meeg_preserves_time_metadata_for_analysis_windowing(
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
                    "t_start_ms": 4000.0,
                    "t_stop_ms": 4016.0,
                    "decimation_factor": 1,
                    "fs_hz": 2000.0,
                    "component_axis": "z",
                    "component_index": 2,
                },
            }
        ],
    )

    status = _run_meeg_compute(
        compute_utils_module,
        "meeg_time_meta",
        cdm_path,
        meeg_sensor_locations="[[90000.0, 0.0, 90000.0]]",
    )
    finished = _assert_finished(status, "meeg_time_meta")
    with open(finished["results"], "rb") as handle:
        payload = pickle.load(handle)
    row = payload[0].iloc[0]
    metadata = row.get("metadata") if isinstance(row, dict) else row["metadata"]
    assert float(row["t_start_ms"]) == pytest.approx(4000.0)
    assert float(row["t_stop_ms"]) == pytest.approx(4016.0)
    assert float(metadata["t_start_ms"]) == pytest.approx(4000.0)
    assert float(metadata["t_stop_ms"]) == pytest.approx(4016.0)


def test_meeg_per_sensor_independent_mode_constraints(
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

    mismatch_status = _run_meeg_compute(
        compute_utils_module,
        "meeg_pairs_mismatch",
        cdm_path,
        meeg_forward_mode="per_sensor_independent",
        meeg_dipole_locations="[[78000.0, 0.0, 0.0], [0.0, 78000.0, 0.0]]",
        meeg_sensor_locations="[[90000.0, 0.0, 0.0]]",
    )
    mismatch_failed = _assert_failed(mismatch_status, "meeg_pairs_mismatch")
    assert "same number of dipole and sensor locations" in str(mismatch_failed["error"])

    four_area_payload = _write_pickle(
        tmp_path / "cdm_four_area.pkl",
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
    four_area_status = _run_meeg_compute(
        compute_utils_module,
        "meeg_four_area_independent",
        four_area_payload,
        meeg_simulation_model="four_area",
        meeg_forward_mode="per_sensor_independent",
        meeg_sensor_locations="[[90000.0, 0.0, 0.0], [0.0, 90000.0, 0.0]]",
    )
    four_area_failed = _assert_failed(four_area_status, "meeg_four_area_independent")
    assert "not available for the four-area model" in str(four_area_failed["error"])


@pytest.mark.parametrize(
    ("model_name", "expected_sensor"),
    (
        ("FourSphereVolumeConductor", np.asarray([[0.0, 0.0, 90000.0]], dtype=float)),
        ("InfiniteVolumeConductor", np.asarray([[0.0, 0.0, 92000.0]], dtype=float)),
        ("InfiniteHomogeneousVolCondMEG", np.asarray([[0.0, 0.0, 92000.0]], dtype=float)),
        ("SphericallySymmetricVolCondMEG", np.asarray([[0.0, 0.0, 92000.0]], dtype=float)),
    ),
)
def test_meeg_four_area_default_sensor_location_is_model_safe_for_all_models(
    tmp_path,
    compute_utils_module,
    monkeypatch,
    model_name,
    expected_sensor,
):
    recorded_sensor_locations = []

    class _RecordingEEGModel(_FakeEEGModel):
        def __init__(self, sensor_locations, **kwargs):
            super().__init__(sensor_locations, **kwargs)
            recorded_sensor_locations.append(np.asarray(sensor_locations, dtype=float))

    class _RecordingFieldPotential(_FakeFieldPotential):
        def _load_eegmegcalc_model(self, _model_name):
            return _RecordingEEGModel

    monkeypatch.setattr(compute_utils_module.ncpi, "FieldPotential", _RecordingFieldPotential)

    cdm_path = _write_pickle(
        tmp_path / "cdm.pkl",
        [
            {
                "sum": {
                    "frontal": np.linspace(0.1, 0.5, 16),
                    "parietal": np.linspace(0.2, 0.6, 16),
                    "temporal": np.linspace(0.3, 0.7, 16),
                    "occipital": np.linspace(0.4, 0.8, 16),
                },
                "metadata": {"dt_ms": 1.0, "component_axis": "z"},
            }
        ],
    )

    status = _run_meeg_compute(
        compute_utils_module,
        f"meeg_four_area_default_sensor_{model_name}",
        cdm_path,
        meeg_model=model_name,
        meeg_simulation_model="four_area",
        meeg_sensor_locations="",
    )
    _assert_finished(status, f"meeg_four_area_default_sensor_{model_name}")
    assert recorded_sensor_locations, "Expected at least one forward-model instantiation."
    np.testing.assert_allclose(recorded_sensor_locations[0], expected_sensor)


def test_meeg_four_area_foursphere_clips_out_of_bound_sensor_and_dipole_locations(
    tmp_path,
    compute_utils_module,
    monkeypatch,
):
    recorded_sensor_locations = []
    recorded_dipole_locations = []

    class _RecordingEEGModel(_FakeEEGModel):
        def __init__(self, sensor_locations, **kwargs):
            super().__init__(sensor_locations, **kwargs)
            recorded_sensor_locations.append(np.asarray(sensor_locations, dtype=float))

        def get_transformation_matrix(self, dipole_location):
            recorded_dipole_locations.append(np.asarray(dipole_location, dtype=float))
            return super().get_transformation_matrix(dipole_location)

    class _RecordingFieldPotential(_FakeFieldPotential):
        def _load_eegmegcalc_model(self, _model_name):
            return _RecordingEEGModel

    monkeypatch.setattr(compute_utils_module.ncpi, "FieldPotential", _RecordingFieldPotential)

    cdm_path = _write_pickle(
        tmp_path / "cdm.pkl",
        [
            {
                "sum": {
                    "frontal": np.linspace(0.1, 0.5, 16),
                    "parietal": np.linspace(0.2, 0.6, 16),
                    "temporal": np.linspace(0.3, 0.7, 16),
                    "occipital": np.linspace(0.4, 0.8, 16),
                },
                "metadata": {"dt_ms": 1.0, "component_axis": "z"},
            }
        ],
    )

    status = _run_meeg_compute(
        compute_utils_module,
        "meeg_four_area_foursphere_clipping",
        cdm_path,
        meeg_model="FourSphereVolumeConductor",
        meeg_simulation_model="four_area",
        meeg_sensor_locations="[[0.0, 0.0, 92000.0]]",
        meeg_model_kwargs="{'radii': [60000.0, 70000.0, 80000.0, 85000.0]}",
    )
    finished = _assert_finished(status, "meeg_four_area_foursphere_clipping")
    assert finished["status"] == "finished"
    assert recorded_sensor_locations and recorded_dipole_locations

    sensor_norms = np.linalg.norm(recorded_sensor_locations[0], axis=1)
    assert np.all(sensor_norms <= 85000.0 + 1e-9)
    dipole_norms = np.linalg.norm(np.vstack(recorded_dipole_locations), axis=1)
    assert np.all(dipole_norms <= 60000.0 + 1e-9)
