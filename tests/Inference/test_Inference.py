import numpy as np
import pytest
import pickle
from ncpi import Inference


@pytest.fixture(scope="function")
def tmp_result_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return str(d)


def _make_linear_data(n=40, d=2, theta_dim=1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(float)
    W = rng.normal(size=(d, theta_dim)).astype(float)
    Y = X @ W + 0.05 * rng.normal(size=(n, theta_dim))
    if theta_dim == 1:
        Y = Y.reshape(-1, 1)
    return X, Y


def test_sklearn_ridge_train_predict_shapes(tmp_result_dir):
    X, Y = _make_linear_data(n=30, d=2, theta_dim=1, seed=1)

    inf = Inference("Ridge", hyperparams={"alpha": 1.0})
    inf.add_simulation_data(X, Y)

    inf.train(param_grid=None, n_splits=2, n_repeats=1, result_dir=tmp_result_dir, seed=0)

    # predict with 2D input
    preds = inf.predict(X[:5], result_dir=tmp_result_dir)
    assert isinstance(preds, list)
    assert len(preds) == 5
    assert all(isinstance(p, float) or isinstance(p, list) for p in preds)

    # predict with 1D list that should be interpreted as ONE sample with 2 features
    one = inf.predict(X[0], result_dir=tmp_result_dir)
    assert len(one) == 1

    # predict with 1D list of scalars for 1-feature samples: should become (N,1)
    X1 = np.array([0.1, 0.2, 0.3], dtype=float)
    inf1 = Inference("Ridge", hyperparams={"alpha": 1.0})
    inf1.add_simulation_data(X[:, :1], Y)
    inf1.train(param_grid=None, n_splits=2, n_repeats=1, result_dir=tmp_result_dir, seed=0)
    preds1 = inf1.predict(X1, result_dir=tmp_result_dir)
    assert len(preds1) == 3


def test_sklearn_random_forest_ensemble_predict(tmp_result_dir):
    X, Y = _make_linear_data(n=40, d=3, theta_dim=1, seed=2)

    inf = Inference("RandomForestRegressor", hyperparams={"n_estimators": 10})
    inf.add_simulation_data(X, Y)

    # simple "param_grid" with 1 candidate: will train fold-ensemble and save list of models
    inf.train(
        param_grid=[{"n_estimators": 10, "max_depth": 3}],
        n_splits=2,
        n_repeats=2,
        result_dir=tmp_result_dir,
        seed=0,
    )

    preds = inf.predict(X[:7], result_dir=tmp_result_dir)
    assert len(preds) == 7


def test_predict_empty_returns_empty_list(tmp_result_dir):
    X, Y = _make_linear_data(n=20, d=2, theta_dim=1, seed=10)
    inf = Inference("Ridge", hyperparams={"alpha": 1.0})
    inf.add_simulation_data(X, Y)
    inf.train(param_grid=None, n_splits=2, n_repeats=1, result_dir=tmp_result_dir, seed=0)

    out = inf.predict(np.empty((0, 2)), result_dir=tmp_result_dir)
    assert out == []


def test_predict_rejects_ragged_object_array(tmp_result_dir):
    X, Y = _make_linear_data(n=20, d=2, theta_dim=1, seed=11)
    inf = Inference("Ridge", hyperparams={"alpha": 1.0})
    inf.add_simulation_data(X, Y)
    inf.train(param_grid=None, n_splits=2, n_repeats=1, result_dir=tmp_result_dir, seed=0)

    ragged = [np.array([1.0, 2.0]), np.array([1.0])]  # inconsistent lengths => dtype=object
    with pytest.raises(ValueError):
        inf.predict(ragged, result_dir=tmp_result_dir)


def test_predict_nan_rows_return_nan_row_template(tmp_result_dir):
    X, Y = _make_linear_data(n=30, d=2, theta_dim=1, seed=12)
    inf = Inference("Ridge", hyperparams={"alpha": 1.0})
    inf.add_simulation_data(X, Y)
    inf.train(param_grid=None, n_splits=2, n_repeats=1, result_dir=tmp_result_dir, seed=0)

    Xq = X[:3].copy()
    Xq[1, 0] = np.nan
    preds = inf.predict(Xq, result_dir=tmp_result_dir)

    assert len(preds) == 3
    assert np.isfinite(preds[0])
    # scalar regression -> might return float nan OR [nan]
    assert np.isnan(preds[1]) or (isinstance(preds[1], list) and all(np.isnan(v) for v in preds[1]))
    assert np.isfinite(preds[2])


def test_pickle_roundtrip_preserves_predict(tmp_result_dir, tmp_path):
    X, Y = _make_linear_data(n=25, d=2, theta_dim=1, seed=13)
    inf = Inference("Ridge", hyperparams={"alpha": 1.0})
    inf.add_simulation_data(X, Y)
    inf.train(param_grid=None, n_splits=2, n_repeats=1, result_dir=tmp_result_dir, seed=0)

    p = tmp_path / "inf.pkl"
    p.write_bytes(pickle.dumps(inf))
    inf2 = pickle.loads(p.read_bytes())

    preds = inf2.predict(X[:5], result_dir=tmp_result_dir)
    assert len(preds) == 5


@pytest.mark.parametrize("sbi_model", ["NPE", "NLE", "NRE"])
def test_sbi_train_predict_and_sample(tmp_result_dir, sbi_model):
    # Keep this tiny so tests run fast.
    X, Y = _make_linear_data(n=25, d=2, theta_dim=2, seed=3)

    import torch
    from sbi.utils import BoxUniform

    low = torch.tensor([-5.0, -5.0])
    high = torch.tensor([5.0, 5.0])
    prior = BoxUniform(low=low, high=high)

    hyper = {
        "prior": prior,
        "inference_kwargs": {"device": "cpu"},
        "build_posterior_kwargs": {},
        # default num_samples used by predict() if not passed
        "num_samples": 200,
    }

    inf = Inference(sbi_model, hyperparams=hyper)
    inf.add_simulation_data(X, Y)

    train_params = {
        "max_num_epochs": 1,
        "training_batch_size": 16,
        "learning_rate": 1e-3,
        "show_train_summary": False,
    }

    inf.train(
        param_grid=None,
        n_splits=2,
        n_repeats=1,
        train_params=train_params,
        result_dir=tmp_result_dir,
        seed=0,
        sbi_eval_num_posterior_samples=30,
        sbi_eval_batch_size=8,
    )

    # Predict posterior mean for multiple observations (must return one theta per row)
    preds = inf.predict(X[:4], result_dir=tmp_result_dir, num_posterior_samples=30, sbi_batch_size=8)
    assert isinstance(preds, list)
    assert len(preds) == 4
    for p in preds:
        p = np.asarray(p)
        assert p.shape == (2,)
        assert np.all(np.isfinite(p))

    # Sample posterior for a single observation
    samples = inf.sample_posterior(X[0], num_samples=40, result_dir=tmp_result_dir)
    assert samples.shape == (40, 2)
    assert np.all(np.isfinite(samples))
