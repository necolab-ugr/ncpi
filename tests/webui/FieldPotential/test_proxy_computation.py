import pickle

import numpy as np
import pytest

from tests.webui.FieldPotential._field_potential_test_helpers import (
    _assert_finished,
    _assert_nested_allclose,
    _job_status,
    _make_temp_uploaded_output_dir,
    _python_reference_proxy_trial_rows,
    _stage_simulation_input_files,
)


def test_field_potential_proxy_fr_webui_matches_python_reference_for_all_simulation_cases(
    field_potential_simulation_cases,
    compute_utils_module,
    fake_field_potential_backend,
):
    for case in field_potential_simulation_cases:
        sim_output = case["sim_output"]
        sim_files = _stage_simulation_input_files(sim_output)
        job_id = f"proxy_ref_{case['case_id'].replace(':', '_')}"
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
            temp_uploaded_files=_make_temp_uploaded_output_dir("proxy_results_"),
        )

        finished = _assert_finished(job_status, job_id)
        with open(finished["results"], "rb") as handle:
            webui_payload = pickle.load(handle)
        webui_rows = [dict(frame.iloc[0].to_dict()) for frame in webui_payload]
        python_rows = _python_reference_proxy_trial_rows(sim_output, bin_size_ms=1.0)
        assert len(webui_rows) == len(python_rows), case["case_id"]
        for webui_row, py_row in zip(webui_rows, python_rows):
            assert float(webui_row["dt_ms"]) == pytest.approx(py_row["dt_ms"]), case["case_id"]
            assert int(webui_row.get("decimation_factor", 1)) == int(py_row["decimation_factor"]), case["case_id"]
            _assert_nested_allclose(webui_row["data"], py_row["data"])


def test_field_potential_proxy_fr_decimation_option_updates_metadata_and_shape(
    field_potential_simulation_scenarios,
    compute_utils_module,
    fake_field_potential_backend,
):
    representative_cases = [
        field_potential_simulation_scenarios["hagen"]["default"],
        field_potential_simulation_scenarios["cavallari"]["default"],
        field_potential_simulation_scenarios["four_area"]["default"],
    ]
    for sim_output in representative_cases:
        sim_files = _stage_simulation_input_files(sim_output)
        job_id = f"proxy_decimation_{sim_output['model']}"
        job_status = _job_status(job_id)

        compute_utils_module.field_potential_proxy_computation(
            job_id,
            job_status,
            {
                "proxy_method": "FR",
                "bin_size": "1.0",
                "proxy_decimation_factor": "2",
                "file_paths": {
                    "times_file": sim_files["times"],
                    "gids_file": sim_files["gids"],
                    "network_file": sim_files["network"],
                    "dt_file": sim_files["dt"],
                },
            },
            temp_uploaded_files=_make_temp_uploaded_output_dir("proxy_results_"),
        )

        finished = _assert_finished(job_status, job_id)
        with open(finished["results"], "rb") as handle:
            payload = pickle.load(handle)
        row = payload[0].iloc[0]
        metadata = row.get("metadata") if isinstance(row, dict) else row["metadata"]
        data = np.asarray(row["data"], dtype=float)
        assert int(row["decimation_factor"]) == 2
        assert int(metadata["decimation_factor"]) == 2
        assert float(row["dt_ms"]) == pytest.approx(1.0)
        assert float(metadata["dt_ms"]) == pytest.approx(1.0)
        assert float(metadata["t_start_ms"]) == pytest.approx(0.0)
        assert data.size > 0
