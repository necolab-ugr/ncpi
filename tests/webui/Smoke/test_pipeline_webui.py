"""WebUI pipeline through Flask's test client: simulation upload -> proxy -> features -> inference -> analysis."""
import base64
import io
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _webui_client_helpers import start_job, wait_job

N_TRIALS = 8
N_E, N_I = 20, 5
TSTOP_MS = 1000.0
DT_MS = 0.1
SEED = 0


def _poisson_spikes(rng, n_units, rate_hz, freq_hz, gid_offset):
    """Inhomogeneous Poisson spikes with rate modulated at freq_hz (thinning)."""
    peak = rate_hz * 1.8
    times, gids = [], []
    for unit in range(n_units):
        n = rng.poisson(peak * TSTOP_MS / 1000.0)
        t = np.sort(rng.uniform(0.0, TSTOP_MS, size=n))
        rate = rate_hz * (1.0 + 0.8 * np.sin(2.0 * np.pi * freq_hz * t / 1000.0))
        keep = rng.uniform(size=n) < rate / peak
        times.append(t[keep])
        gids.append(np.full(int(keep.sum()), gid_offset + unit))
    return np.concatenate(times), np.concatenate(gids)


def _simulation_payloads():
    """Multi-trial simulation outputs (one trial per modulation frequency) plus parameters."""
    rng = np.random.default_rng(SEED)
    freqs = np.linspace(4.0, 40.0, N_TRIALS)
    times, gids = [], []
    for f in freqs:
        t_e, g_e = _poisson_spikes(rng, N_E, 20.0, f, 0)
        t_i, g_i = _poisson_spikes(rng, N_I, 40.0, f, N_E)
        times.append({"E": t_e, "I": t_i})
        gids.append({"E": g_e, "I": g_i})
    network = {"X": ["E", "I"], "N_X": [N_E, N_I], "model": "iaf_psc_exp"}
    payloads = {
        "times.pkl": times,
        "gids.pkl": gids,
        "dt.pkl": [DT_MS] * N_TRIALS,
        "tstop.pkl": [TSTOP_MS] * N_TRIALS,
        "network.pkl": [network] * N_TRIALS,
    }
    return payloads, freqs


def _pickle_upload(name, payload):
    return (io.BytesIO(pickle.dumps(payload)), name)


def _read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_webui_full_pipeline(client, webui_module, tmp_path):
    module = webui_module
    payloads, freqs = _simulation_payloads()

    # 1) Upload simulation outputs
    resp = client.post(
        "/upload_sim_files",
        data={"simulation_files": [_pickle_upload(n, p) for n, p in payloads.items()]},
        headers={"Referer": "/simulation/upload_sim"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302
    assert (Path(module.SIMULATION_DATA_DIR) / "simulation.pkl").is_file()

    # 2) Field potential proxy (firing rate)
    job_id = start_job(
        client,
        "field_potential_proxy",
        {"proxy_method": "FR", "bin_size": "1.0"},
        referer="/field_potential/proxy",
    )
    wait_job(client, job_id)
    proxy_path = Path(module.FIELD_POTENTIAL_PROXY_DIR) / "proxy.pkl"
    assert proxy_path.is_file()
    proxy = _read_pickle(proxy_path)
    assert len(proxy) == N_TRIALS

    # 3) Features (catch22) on the proxy produced by the pipeline
    job_id = start_job(
        client,
        "features",
        {
            "select-method": "catch22",
            "data_source_kind": "pipeline",
            "existing_data_path": os.path.realpath(proxy_path),
        },
        referer="/features/methods",
    )
    wait_job(client, job_id)
    features_path = Path(module.FEATURES_DATA_DIR) / "pipeline" / "catch22_features.pkl"
    assert features_path.is_file()
    features_df = _read_pickle(features_path)
    assert len(features_df) == N_TRIALS
    X = np.vstack(features_df["Features"].to_numpy())
    assert X.shape == (N_TRIALS, 22)
    assert np.all(np.isfinite(X))

    # 4) Training (Ridge) with one parameter row per features row, in trial order
    params_path = tmp_path / "parameters.pkl"
    pd.DataFrame({"Parameters": [np.array([f]) for f in freqs]}).to_pickle(params_path)
    job_id = start_job(
        client,
        "inference_training",
        {
            "training_model_name": "Ridge",
            "training_n_splits": "2",
            "training_n_repeats": "1",
            "training_features_source_mode": "server-path",
            "training_features_server_file_path": os.path.realpath(features_path),
            "training_parameters_source_mode": "server-path",
            "training_parameters_server_file_path": os.path.realpath(params_path),
        },
        referer="/inference/new_training",
    )
    wait_job(client, job_id)
    artifacts_dir = Path(module.INFERENCE_TRAINING_DATA_DIR) / "training_artifacts_latest"
    model_path = artifacts_dir / "model.pkl"
    assert model_path.is_file()

    # 5) Predictions with the trained model on the pipeline features
    predict_form = {
        "features_source_mode": "auto-detected",
        "existing_features_file": "pipeline/catch22_features.pkl",
        "model_assets_source": "server-path",
        "inference_model_server_file_path": os.path.realpath(model_path),
    }
    scaler_path = artifacts_dir / "scaler.pkl"
    if scaler_path.is_file():
        predict_form["inference_scaler_server_file_path"] = os.path.realpath(scaler_path)
    job_id = start_job(client, "inference", predict_form, referer="/inference/compute_predictions")
    wait_job(client, job_id)
    predictions_path = Path(module.PREDICTIONS_DATA_DIR) / "predictions.pkl"
    assert predictions_path.is_file()
    predictions_df = _read_pickle(predictions_path)
    assert "Predictions" in predictions_df.columns
    preds = np.asarray(predictions_df["Predictions"].tolist(), dtype=float)
    assert preds.shape[0] == N_TRIALS
    assert np.all(np.isfinite(preds))

    # 6) Analysis: columns of the predictions dataframe and a boxplot
    resp = client.post("/analysis/columns", data={"dataframe_path": os.path.realpath(predictions_path)})
    assert resp.status_code == 200
    assert "Predictions" in resp.get_json()["columns"]

    resp = client.post(
        "/analysis/plot/boxplot",
        data={"boxplot_group_by": "sensor", "boxplot_value_col": "Predictions"},
    )
    assert resp.status_code == 200
    match = re.search(r'id="plot-image" src="data:image/png;base64,([^"]+)"', resp.get_data(as_text=True))
    assert match, "Boxplot page did not include a rendered image."
    assert base64.b64decode(match.group(1)).startswith(b"\x89PNG")
