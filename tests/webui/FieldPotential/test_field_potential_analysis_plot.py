import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


@pytest.mark.parametrize(
    ("plot_type", "filename", "payload"),
    (
        (
            "cdm",
            "cdm.pkl",
            [
                pd.DataFrame(
                    [
                        {
                            "sum": np.vstack(
                                [
                                    np.linspace(0.1, 1.0, 48),
                                    np.linspace(0.2, 1.1, 48),
                                    np.linspace(0.3, 1.2, 48),
                                ]
                            ),
                            "dt_ms": 1.0,
                            "metadata": {"dt_ms": 1.0, "component_axis": "xyz"},
                        }
                    ]
                )
            ],
        ),
        (
            "meeg",
            "meeg.pkl",
            [
                pd.DataFrame(
                    [
                        {
                            "data": np.ones((2, 3, 48), dtype=float),
                            "dt_ms": 1.0,
                            "metadata": {"dt_ms": 1.0},
                        }
                    ]
                )
            ],
        ),
    ),
)
def test_analysis_plot_warns_when_reducing_vector_signals_to_z_component(
    field_potential_webui_app_module,
    tmp_path,
    plot_type,
    filename,
    payload,
):
    fp_path = _write_pickle(tmp_path / filename, payload)
    selected_key = f"field_potential::{Path(fp_path).resolve()}"

    with field_potential_webui_app_module.app.test_client() as client:
        response = client.post(
            "/analysis/plot/simulation",
            data={
                "sim_plot_type": plot_type,
                "sim_selected_file_keys": [selected_key],
                "sim_trial_start": "0",
                "sim_trial_end": "0",
                "sim_time_start": "0",
                "sim_time_end": "40",
                "sim_freq_min": "0",
                "sim_freq_max": "200",
            },
        )

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Warning: this plot reduces vector signals to the z-component" in html
