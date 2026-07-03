import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
WEBUI_DIR = REPO_ROOT / "webui"


def _load_webui_app():
    for entry in (str(REPO_ROOT), str(WEBUI_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    for module_name in ("webui.app", "compute_utils", "tmp_paths"):
        if module_name in sys.modules:
            del sys.modules[module_name]
    module = importlib.import_module("webui.app")
    module.refresh_tmp_paths()
    return module


def _write_pickle(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _proxy_frame(**row):
    return pd.DataFrame([row])


def test_field_potential_pipeline_wrapper_expands_four_area_sensors(tmp_path):
    app_module = _load_webui_app()
    proxy_path = tmp_path / "proxy.pkl"
    grid_path = tmp_path / app_module.SIMULATION_GRID_METADATA_FILE
    areas = ("frontal", "parietal", "temporal", "occipital")
    payload = []
    for trial_idx, repeat_idx in ((0, 0), (1, 1)):
        payload.append(_proxy_frame(
            data={area: np.arange(5, dtype=float) + idx + trial_idx for idx, area in enumerate(areas)},
            proxy_method="FR",
            dt_ms=1.0,
            fs_hz=1000.0,
            t_start_ms=2.0,
            trial_index=trial_idx,
            repeat_index=repeat_idx,
            configuration_index=0,
            condition=f"cfg_1__rep_{repeat_idx + 1}__trial_{trial_idx + 1}",
        ))
    _write_pickle(proxy_path, payload)
    _write_pickle(grid_path, {
        "changed_keys": ["J_YX"],
        "trials": [
            {"trial_index": 0, "configuration_index": 0, "repeat_index": 0, "changed": {"J_YX": 1.5}},
            {"trial_index": 1, "configuration_index": 0, "repeat_index": 1, "changed": {"J_YX": 1.5}},
        ],
    })

    normalized = app_module._normalize_features_input_path(
        str(proxy_path),
        {"data_source_kind": "pipeline"},
        "four_area",
        upload_dir=str(tmp_path),
    )
    df = pd.read_pickle(normalized)

    assert df.shape[0] == 8
    assert sorted(df["sensor"].unique().tolist()) == sorted(areas)
    assert set(df["subject_id"].tolist()) == {0}
    assert set(df["epoch"].tolist()) == {"simulation"}
    assert set(df["group"].tolist()) == {"simulation"}
    assert set(df["condition"].tolist()) == {"simulation"}
    assert set(df["configuration"].tolist()) == {"J_YX=1.5"}
    assert set(df["trial"].tolist()) == {0, 1}
    assert "simulation_run" not in df.columns
    assert "configuration_index" not in df.columns
    assert all(np.asarray(value).ndim == 1 for value in df["data"].tolist())
    assert df["fs"].notna().all()


def test_field_potential_pipeline_wrapper_keeps_global_model_rows(tmp_path):
    app_module = _load_webui_app()
    proxy_path = tmp_path / "field_potential_proxy_results_hagen.pkl"
    payload = [
        _proxy_frame(
            data=np.arange(4, dtype=float),
            proxy_method="FR",
            dt_ms=2.0,
            trial_index=0,
            repeat_index=0,
            configuration_index=0,
        ),
        _proxy_frame(
            data=np.arange(6, dtype=float),
            proxy_method="FR",
            dt_ms=2.0,
            trial_index=1,
            repeat_index=0,
            configuration_index=1,
        ),
    ]
    _write_pickle(proxy_path, payload)

    normalized = app_module._normalize_features_input_path(
        str(proxy_path),
        {"data_source_kind": "pipeline"},
        "hagen",
        upload_dir=str(tmp_path),
    )
    df = pd.read_pickle(normalized)

    assert df.shape[0] == 2
    assert df["sensor"].tolist() == ["global", "global"]
    assert df["configuration"].tolist() == ["cfg_1", "cfg_2"]
    assert "simulation_run" not in df.columns
    assert "configuration_index" not in df.columns
    assert df["trial"].tolist() == [0, 0]
    assert df["fs"].tolist() == [500.0, 500.0]
    assert [len(value) for value in df["data"].tolist()] == [4, 6]


def test_field_potential_pipeline_wrapper_applies_parser_metadata_overrides(tmp_path):
    app_module = _load_webui_app()
    proxy_path = tmp_path / "field_potential_proxy_results_cavallari.pkl"
    _write_pickle(proxy_path, [
        _proxy_frame(
            data=np.arange(3, dtype=float),
            proxy_method="FR",
            fs_hz=1000.0,
            trial_index=0,
            repeat_index=0,
            configuration_index=2,
        )
    ])

    normalized = app_module._normalize_features_input_path(
        str(proxy_path),
        {
            "data_source_kind": "pipeline",
            "parser_metadata_condition_source": "__value__",
            "parser_metadata_condition": "custom_condition",
            "parser_metadata_group_source": "configuration",
            "parser_recording_type_source": "__value__",
            "parser_recording_type": "CDM",
            "parser_fs_source": "__numeric__",
            "parser_fs_manual": "250",
        },
        "cavallari",
        upload_dir=str(tmp_path),
    )
    df = pd.read_pickle(normalized)

    assert df.loc[0, "condition"] == "custom_condition"
    assert df.loc[0, "group"] == "cfg_3"
    assert df.loc[0, "recording_type"] == "CDM"
    assert df.loc[0, "fs"] == 250.0


def test_field_potential_pipeline_without_epoching_keeps_continuous_rows(tmp_path):
    app_module = _load_webui_app()
    proxy_path = tmp_path / "field_potential_proxy_results_cavallari.pkl"
    _write_pickle(proxy_path, [
        _proxy_frame(
            data=np.arange(20, dtype=float),
            proxy_method="FR",
            fs_hz=10.0,
            trial_index=0,
            repeat_index=0,
            configuration_index=0,
        )
    ])

    normalized = app_module._normalize_features_input_path(
        str(proxy_path),
        {
            "data_source_kind": "pipeline",
            "parser_data_locator": "data",
            "parser_fs_source": "fs",
            "parser_ch_names_source": "sensor",
        },
        "continuous",
        upload_dir=str(tmp_path),
    )
    df = pd.read_pickle(normalized)

    assert df.shape[0] == 1
    assert df.loc[0, "epoch"] == "simulation"
    assert len(df.loc[0, "data"]) == 20
    assert df.loc[0, "t0"] == 0.0
    assert df.loc[0, "t1"] == 1.9


def test_field_potential_pipeline_epoching_uses_parser_windows(tmp_path):
    app_module = _load_webui_app()
    proxy_path = tmp_path / "field_potential_proxy_results_cavallari.pkl"
    _write_pickle(proxy_path, [
        _proxy_frame(
            data=np.arange(20, dtype=float),
            proxy_method="FR",
            fs_hz=10.0,
            trial_index=0,
            repeat_index=0,
            configuration_index=0,
        )
    ])

    normalized = app_module._normalize_features_input_path(
        str(proxy_path),
        {
            "data_source_kind": "pipeline",
            "parser_data_locator": "data",
            "parser_fs_source": "fs",
            "parser_ch_names_source": "sensor",
            "parser_enable_epoching": "1",
            "parser_epoch_length_s": "0.5",
            "parser_epoch_step_s": "0.5",
        },
        "epoching",
        upload_dir=str(tmp_path),
    )
    df = pd.read_pickle(normalized)

    assert df.shape[0] == 4
    assert df["epoch"].tolist() == [0, 1, 2, 3]
    assert [len(value) for value in df["data"].tolist()] == [5, 5, 5, 5]
    np.testing.assert_array_equal(df.loc[0, "data"], np.arange(5, dtype=float))
    assert df["t0"].tolist() == [0.0, 0.5, 1.0, 1.5]
    assert df["trial"].tolist() == [0, 0, 0, 0]
    assert df["configuration"].tolist() == ["cfg_1", "cfg_1", "cfg_1", "cfg_1"]
    assert "simulation_run" not in df.columns
    assert "configuration_index" not in df.columns


def test_field_potential_pipeline_inspection_exposes_canonical_parser_defaults(tmp_path):
    app_module = _load_webui_app()
    proxy_path = tmp_path / "field_potential_proxy_results_cavallari.pkl"
    _write_pickle(proxy_path, [
        _proxy_frame(
            data=np.arange(3, dtype=float),
            proxy_method="FR",
            fs_hz=1000.0,
            trial_index=0,
            repeat_index=0,
            configuration_index=0,
        )
    ])

    description = app_module._describe_parser_source(str(proxy_path))

    assert description["source_type"] == "dataframe"
    assert description["pipeline_source_type"] == "field_potential"
    assert description["defaults"]["data"] == "data"
    assert description["defaults"]["fs"] == "fs"
    assert description["defaults"]["ch_names"] == "sensor"
    assert description["defaults"]["recording_type"] == "recording_type"
    assert description["defaults"]["subject_id"] == "subject_id"
    assert description["defaults"]["group"] == "group"
    assert description["defaults"]["condition"] == "condition"
    assert "data" in description["candidate_fields"]
    assert "sensor" in description["candidate_fields"]
    assert description["fs_hint_hz"] == 1000.0
