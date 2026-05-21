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


def test_predict_rejects_non_boolean_scaler_flag(tmp_result_dir):
    X, Y = _make_linear_data(n=20, d=2, theta_dim=1, seed=14)
    inf = Inference("Ridge", hyperparams={"alpha": 1.0})
    inf.add_simulation_data(X, Y)
    inf.train(param_grid=None, n_splits=2, n_repeats=1, result_dir=tmp_result_dir, seed=0)

    with pytest.raises(TypeError):
        inf.predict(X[:3], result_dir=tmp_result_dir, scaler="yes")


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
    if sbi_model == "NRE":
        # NRE classifier_nn does not support flow-based estimators like "nsf"
        hyper["estimator_kwargs"] = {"estimator": "mlp"}

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

    # Predict posterior samples for multiple observations
    samples = inf.predict(X[:4], result_dir=tmp_result_dir, num_posterior_samples=30, sbi_batch_size=8)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (30, 4, 2)
    assert np.all(np.isfinite(samples))

    # Sample posterior for a single observation (returns S x theta_dim)
    samples1 = inf.predict(X[0], result_dir=tmp_result_dir, num_posterior_samples=40, sbi_batch_size=8)
    assert samples1.shape == (40, 2)
    assert np.all(np.isfinite(samples1))


def test_sbi_predict_empty_input_returns_empty_sample_tensor(tmp_result_dir):
    X, Y = _make_linear_data(n=20, d=2, theta_dim=2, seed=5)

    import torch
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=torch.tensor([-5.0, -5.0]), high=torch.tensor([5.0, 5.0]))
    hyper = {
        "prior": prior,
        "inference_kwargs": {"device": "cpu"},
        "build_posterior_kwargs": {},
    }

    inf = Inference("NPE", hyperparams=hyper)
    inf.add_simulation_data(X, Y)
    inf.train(
        param_grid=None,
        n_splits=2,
        n_repeats=1,
        train_params={"max_num_epochs": 1, "show_train_summary": False},
        result_dir=tmp_result_dir,
        seed=0,
        sbi_eval_num_posterior_samples=10,
        sbi_eval_batch_size=4,
    )

    empty = inf.predict(np.empty((0, 2)), result_dir=tmp_result_dir, num_posterior_samples=15)
    assert isinstance(empty, np.ndarray)
    assert empty.shape == (15, 0, 2)


def test_train_rejects_nonpositive_sbi_eval_knobs(tmp_result_dir):
    X, Y = _make_linear_data(n=20, d=2, theta_dim=1, seed=15)
    inf = Inference("Ridge", hyperparams={"alpha": 1.0})
    inf.add_simulation_data(X, Y)

    with pytest.raises(ValueError):
        inf.train(
            param_grid=None,
            n_splits=2,
            n_repeats=1,
            result_dir=tmp_result_dir,
            seed=0,
            sbi_eval_num_posterior_samples=0,
            sbi_eval_batch_size=8,
        )

    with pytest.raises(ValueError):
        inf.train(
            param_grid=None,
            n_splits=2,
            n_repeats=1,
            result_dir=tmp_result_dir,
            seed=0,
            sbi_eval_num_posterior_samples=10,
            sbi_eval_batch_size=0,
        )


def test_sbi_param_grid_stores_posterior_ensemble_and_predicts(tmp_result_dir):
    X, Y = _make_linear_data(n=24, d=2, theta_dim=2, seed=4)

    import torch
    from sbi.utils import BoxUniform

    low = torch.tensor([-5.0, -5.0])
    high = torch.tensor([5.0, 5.0])
    prior = BoxUniform(low=low, high=high)

    hyper = {
        "prior": prior,
        "inference_kwargs": {"device": "cpu"},
        "build_posterior_kwargs": {},
    }

    inf = Inference("NPE", hyperparams=hyper)
    inf.add_simulation_data(X, Y)

    train_params = {
        "max_num_epochs": 1,
        "training_batch_size": 16,
        "learning_rate": 1e-3,
        "show_train_summary": False,
    }

    inf.train(
        param_grid=[{"estimator_kwargs": {"estimator": "nsf"}}],
        n_splits=2,
        n_repeats=1,
        train_params=train_params,
        result_dir=tmp_result_dir,
        seed=0,
        sbi_eval_num_posterior_samples=20,
        sbi_eval_batch_size=8,
    )

    with open(f"{tmp_result_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)

    assert isinstance(model, list)
    assert len(model) == 2  # n_splits * n_repeats
    assert all(hasattr(p, "sample") for p in model)

    samples = inf.predict(X[:3], result_dir=tmp_result_dir, num_posterior_samples=30, sbi_batch_size=8)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (30, 3, 2)
    assert np.all(np.isfinite(samples))


@pytest.mark.parametrize("sbi_model", ["NPE", "NLE", "NRE"])
def test_sbi_native_workflow_and_pairplot_smoke(sbi_model):
    """
    Native sbi integration smoke test (not via ncpi.Inference):
      1) simulate theta/x
      2) train inference object
      3) build posterior
      4) sample posterior for x_o
      5) render sbi.analysis.pairplot
    """
    import torch
    from sbi.analysis import pairplot
    from sbi.inference import NPE, NLE, NRE
    from sbi.utils import BoxUniform

    torch.manual_seed(0)
    n = 96
    theta_dim = 2
    x_dim = 2

    prior = BoxUniform(
        low=-5.0 * torch.ones(theta_dim),
        high=5.0 * torch.ones(theta_dim),
    )

    theta = prior.sample((n,)).to(torch.float32)

    # Simple simulator: x = theta + Gaussian noise
    def simulate(theta_batch):
        return theta_batch + 0.1 * torch.randn(theta_batch.shape[0], x_dim)

    x = simulate(theta).to(torch.float32)
    x_o = x[0:1]

    inference_cls = {"NPE": NPE, "NLE": NLE, "NRE": NRE}[sbi_model]
    inference = inference_cls(prior=prior)
    density_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=1,
        training_batch_size=32,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(density_estimator)
    posterior_theta = posterior.sample((40,), x=x_o, show_progress_bars=False)

    assert posterior_theta.shape == (40, theta_dim)
    assert torch.isfinite(posterior_theta).all()

    fig, ax = pairplot(posterior_theta)
    assert fig is not None
    assert ax is not None

    import matplotlib.pyplot as plt
    plt.close(fig)
