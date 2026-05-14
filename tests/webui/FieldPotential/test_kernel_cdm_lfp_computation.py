import pickle

import numpy as np
import pytest

from tests.webui.FieldPotential._field_potential_test_helpers import (
    _assert_failed,
    _assert_finished,
    _assert_nested_allclose,
    _example_input_for_kernel,
    _job_status,
    _make_temp_uploaded_output_dir,
    _python_reference_kernel_raw_signals_by_probe,
    _python_reference_kernel_trial_rows,
    _stage_simulation_input_files,
    _sum_signal_dict,
)


def test_field_potential_kernel_cdm_z_and_lfp_match_python_reference_for_all_simulation_cases(
    field_potential_simulation_cases,
    compute_utils_module,
    fake_field_potential_backend,
    field_potential_example_paths,
):
    for case in field_potential_simulation_cases:
        sim_output = case["sim_output"]
        sim_files = _stage_simulation_input_files(sim_output)
        cfg = _example_input_for_kernel(sim_output, field_potential_example_paths, sim_file_paths=sim_files)

        job_id = f"kernel_ref_{case['case_id'].replace(':', '_')}"
        job_status = _job_status(job_id)
        compute_utils_module.field_potential_kernel_computation(
            job_id,
            job_status,
            {
                **cfg,
                "probe_selection_present": "1",
                "probe_kernel_approx": "true",
                "probe_gauss_cylinder": "true",
                "probe_current_dipole": "false",
                "cdm_component": "z",
                "cdm_dt": "0.2",
                "cdm_tstop": "100.0",
            },
            temp_uploaded_files=_make_temp_uploaded_output_dir("kernel_results_"),
        )

        finished = _assert_finished(job_status, job_id)
        with open(finished["results"], "rb") as handle:
            webui_payload = pickle.load(handle)

        py_cdm_rows = _python_reference_kernel_trial_rows(sim_output, cdm_dt=0.2, cdm_tstop=100.0, component_axis="z")
        py_lfp_rows = _python_reference_kernel_raw_signals_by_probe(
            sim_output,
            "GaussCylinderPotential",
            cdm_dt=0.2,
            cdm_tstop=100.0,
            component_axis="z",
        )
        assert len(webui_payload) == len(py_cdm_rows), case["case_id"]
        for webui_trial, py_cdm, py_lfp in zip(webui_payload, py_cdm_rows, py_lfp_rows):
            row = dict(webui_trial)
            assert str(row.get("component_axis")) == "z", case["case_id"]
            assert float(row.get("dt_ms")) == pytest.approx(py_cdm["dt_ms"]), case["case_id"]
            _assert_nested_allclose(row.get("raw_signals"), py_cdm["raw_signals"])
            _assert_nested_allclose(_sum_signal_dict(row.get("raw_signals")), py_cdm["sum"])
            probe_outputs = row.get("probe_outputs") or {}
            assert "GaussCylinderPotential" in probe_outputs, case["case_id"]
            _assert_nested_allclose(probe_outputs["GaussCylinderPotential"].get("raw_signals"), py_lfp)


def test_field_potential_kernel_xyz_component_and_decimation_options(
    field_potential_simulation_scenarios,
    compute_utils_module,
    fake_field_potential_backend,
    field_potential_example_paths,
):
    representative_cases = [
        field_potential_simulation_scenarios["hagen"]["default"],
        field_potential_simulation_scenarios["cavallari"]["default"],
        field_potential_simulation_scenarios["four_area"]["default"],
    ]
    for sim_output in representative_cases:
        sim_files = _stage_simulation_input_files(sim_output)
        cfg = _example_input_for_kernel(sim_output, field_potential_example_paths, sim_file_paths=sim_files)
        job_id = f"kernel_xyz_{sim_output['model']}"
        job_status = _job_status(job_id)
        compute_utils_module.field_potential_kernel_computation(
            job_id,
            job_status,
            {
                **cfg,
                "cdm_component": "xyz",
                "cdm_dt": "0.2",
                "cdm_tstop": "100.0",
                "cdm_decimation_factor": "2",
            },
            temp_uploaded_files=_make_temp_uploaded_output_dir("kernel_results_"),
        )
        finished = _assert_finished(job_status, job_id)
        with open(finished["results"], "rb") as handle:
            cdm_payload = pickle.load(handle)
        first_trial = cdm_payload[0]
        assert first_trial.get("component_axis") == "xyz"
        assert first_trial.get("component_index") is None
        assert int(first_trial.get("decimation_factor", 1)) == 2
        signal = np.asarray(first_trial["sum"], dtype=float)
        assert signal.ndim == 2 and signal.shape[0] == 3


def test_field_potential_kernel_rejects_invalid_component_selection(
    hagen_simulation_output,
    compute_utils_module,
    fake_field_potential_backend,
    field_potential_example_paths,
):
    sim_files = _stage_simulation_input_files(hagen_simulation_output)
    cfg = _example_input_for_kernel(hagen_simulation_output, field_potential_example_paths, sim_file_paths=sim_files)
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
        temp_uploaded_files=_make_temp_uploaded_output_dir("kernel_results_"),
    )

    failed = _assert_failed(job_status, job_id)
    assert "CDM component must be 'z' or 'xyz (all)'" in str(failed["error"])
