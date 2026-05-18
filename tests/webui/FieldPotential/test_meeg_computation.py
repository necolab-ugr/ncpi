import pickle

import numpy as np
import pytest

from tests.webui.FieldPotential._field_potential_test_helpers import (
    _assert_finished,
    _assert_nested_allclose,
    _example_input_for_kernel,
    _job_status,
    _make_temp_uploaded_output_dir,
    _parse_locations_literal,
    _python_reference_meeg_rows_from_cdm_payload,
    _run_meeg_compute,
    _stage_simulation_input_files,
    _write_pickle,
)


def test_field_potential_meeg_webui_matches_python_reference_for_all_simulation_cases(
    field_potential_simulation_cases,
    compute_utils_module,
    fake_field_potential_backend,
    field_potential_example_paths,
):
    forward_models = (
        (
            "FourSphereVolumeConductor",
            "[[78000.0, 0.0, 0.0]]",
            "[[90000.0, 0.0, 90000.0], [0.0, 90000.0, 90000.0]]",
        ),
        (
            "InfiniteHomogeneousVolCondMEG",
            "[[0.0, 0.0, 0.0]]",
            "[[10000.0, 0.0, 0.0], [0.0, 10000.0, 0.0]]",
        ),
    )

    for case in field_potential_simulation_cases:
        sim_output = case["sim_output"]
        sim_files = _stage_simulation_input_files(sim_output)
        cfg = _example_input_for_kernel(sim_output, field_potential_example_paths, sim_file_paths=sim_files)
        kernel_job_id = f"kernel_for_meeg_{case['case_id'].replace(':', '_')}"
        kernel_status = _job_status(kernel_job_id)
        compute_utils_module.field_potential_kernel_computation(
            kernel_job_id,
            kernel_status,
            {
                **cfg,
                "cdm_component": "z",
                "cdm_dt": "0.2",
                "cdm_tstop": "100.0",
                "cdm_decimation_factor": "2",
            },
            temp_uploaded_files=_make_temp_uploaded_output_dir("kernel_results_"),
        )
        kernel_finished = _assert_finished(kernel_status, kernel_job_id)
        cdm_path = kernel_finished["results"]
        with open(cdm_path, "rb") as handle:
            cdm_payload = pickle.load(handle)

        for meeg_model, dipole_locations, sensor_locations in forward_models:
            meeg_job_id = f"meeg_{case['case_id'].replace(':', '_')}_{meeg_model}"
            meeg_status = _run_meeg_compute(
                compute_utils_module,
                meeg_job_id,
                cdm_path,
                meeg_model=meeg_model,
                meeg_dipole_locations=dipole_locations,
                meeg_sensor_locations=sensor_locations,
            )
            meeg_finished = _assert_finished(meeg_status, meeg_job_id)
            with open(meeg_finished["results"], "rb") as handle:
                webui_payload = pickle.load(handle)
            webui_rows = [dict(frame.iloc[0].to_dict()) for frame in webui_payload]
            ref_rows = _python_reference_meeg_rows_from_cdm_payload(
                cdm_payload,
                meeg_model,
                dipole_location=_parse_locations_literal(dipole_locations)[0],
                sensor_locations=_parse_locations_literal(sensor_locations),
            )

            assert len(webui_rows) == len(ref_rows), f"{case['case_id']}::{meeg_model}"
            for webui_row, ref_row in zip(webui_rows, ref_rows):
                _assert_nested_allclose(webui_row["data"], ref_row["data"])
                if ref_row["dt_ms"] is not None:
                    assert float(webui_row["dt_ms"]) == pytest.approx(ref_row["dt_ms"])


def test_field_potential_meeg_uses_xyz_cdm_metadata_when_available(
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

    status = _run_meeg_compute(
        compute_utils_module,
        "meeg_xyz",
        cdm_path,
        meeg_model="FourSphereVolumeConductor",
        meeg_dipole_locations="[[78000.0, 0.0, 0.0]]",
        meeg_sensor_locations="[[90000.0, 0.0, 90000.0]]",
    )
    finished = _assert_finished(status, "meeg_xyz")
    with open(finished["results"], "rb") as handle:
        payload = pickle.load(handle)
    row = payload[0].iloc[0]
    data = np.asarray(row["data"], dtype=float)
    assert data.shape[0] == 1
    assert float(np.max(np.abs(data))) > 0.0
    assert "all xyz components" in status["meeg_xyz"]["output"]
