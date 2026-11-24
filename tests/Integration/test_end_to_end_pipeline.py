"""End-to-end test: simulation -> field potential proxy -> features -> inference."""
import os
import pickle
import textwrap

import numpy as np
import pytest

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
from ncpi import Features, FieldPotential, Inference, Simulation

pytest.importorskip("pycatch22")
pytest.importorskip("sklearn")

N_SIMS = 80
N_UNITS = 4
N_TIMES = 2000
DT_MS = 0.5
SEED = 0

PARAMS_SCRIPT = textwrap.dedent(
    f"""
    import numpy as np

    # Grid of synaptic time constants (ms), one simulation per value
    tau_ms = np.geomspace(1.0, 30.0, {N_SIMS})
    n_units = {N_UNITS}
    n_times = {N_TIMES}
    dt = {DT_MS}
    seed = {SEED}
    """
)

# Toy network: AMPA/GABA currents are Ornstein-Uhlenbeck processes with time constant tau
SIMULATION_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import os
    import pickle
    import sys

    import numpy as np
    from scipy.signal import lfilter

    param_path, output_folder = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location("params", param_path)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    rng = np.random.default_rng(params.seed)

    def ou(tau, shape):
        a = np.exp(-params.dt / tau)
        noise = rng.normal(size=shape)
        return lfilter([np.sqrt(1.0 - a ** 2)], [1.0, -a], noise, axis=-1)

    ampa, gaba = [], []
    shape = (params.n_units, params.n_times)
    for tau in params.tau_ms:
        ampa.append(1.0 + ou(tau, shape))
        gaba.append(-1.0 + ou(tau, shape))

    outputs = {
        "ampa.pkl": ampa,
        "gaba.pkl": gaba,
        "dt.pkl": params.dt,
        "theta.pkl": np.log(params.tau_ms).reshape(-1, 1),
    }
    for name, value in outputs.items():
        with open(os.path.join(output_folder, name), "wb") as f:
            pickle.dump(value, f)
    """
)


def _load(folder, name):
    with open(os.path.join(folder, name), "rb") as f:
        return pickle.load(f)


def test_simulation_features_inference_pipeline(tmp_path):
    param_folder = tmp_path / "params"
    python_folder = tmp_path / "python"
    output_folder = tmp_path / "output"
    result_dir = tmp_path / "inference"
    param_folder.mkdir()
    python_folder.mkdir()
    (param_folder / "network_params.py").write_text(PARAMS_SCRIPT)
    (python_folder / "simulation.py").write_text(SIMULATION_SCRIPT)

    # 1) Simulation
    sim = Simulation(str(param_folder), str(python_folder), str(output_folder))
    sim.simulate("simulation.py", "network_params.py")

    ampa = _load(output_folder, "ampa.pkl")
    gaba = _load(output_folder, "gaba.pkl")
    dt = _load(output_folder, "dt.pkl")
    theta = _load(output_folder, "theta.pkl")
    assert len(ampa) == len(gaba) == N_SIMS
    assert theta.shape == (N_SIMS, 1)

    # 2) Field potential proxy
    fp = FieldPotential()
    proxies = [
        fp.compute_proxy("LRWS", {"AMPA": a, "GABA": g}, dt)
        for a, g in zip(ampa, gaba)
    ]
    assert all(p.shape == (N_TIMES,) for p in proxies)

    # 3) Features
    feats = Features(method="catch22").compute_features(proxies, n_jobs=1)
    X = np.asarray(feats, dtype=float)
    assert X.shape == (N_SIMS, 22)
    assert np.all(np.isfinite(X))

    # 4) Inference: train on a random 80 % split, predict on the rest
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N_SIMS)
    n_train = int(0.8 * N_SIMS)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    inf = Inference("Ridge", hyperparams={"alpha": 1.0})
    inf.add_simulation_data(X[train_idx], theta[train_idx])
    inf.train(
        param_grid=None,
        n_splits=3,
        n_repeats=1,
        result_dir=str(result_dir),
        scaler=True,
        seed=SEED,
    )
    for name in ("model.pkl", "scaler.pkl"):
        assert (result_dir / name).is_file()

    preds = inf.predict(X[test_idx], result_dir=str(result_dir), scaler=True, n_jobs=1)
    y_pred = np.asarray(preds, dtype=float).reshape(-1)
    y_true = theta[test_idx, 0]
    assert y_pred.shape == y_true.shape
    assert np.all(np.isfinite(y_pred))

    # Parameter is recovered from features of held-out simulations
    r = np.corrcoef(y_true, y_pred)[0, 1]
    assert r > 0.8, f"Pearson r between true and predicted log(tau) too low: {r:.3f}"
