import numpy as np
import pytest
import pickle
import os

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
from ncpi.Inference import Inference


class _FailBatchedPosterior:
    """Test double: fail in sample_batched(), succeed in single-row sample()."""

    sample_calls = 0
    sample_batched_calls = 0

    def sample_batched(self, *args, **kwargs):
        type(self).sample_batched_calls += 1
        raise AssertionError("simulated batched spline discriminant failure")

    def sample(self, sample_shape=(), x=None, **kwargs):
        import torch

        type(self).sample_calls += 1
        size = int(torch.Size(sample_shape).numel())
        theta_dim = 2
        return torch.zeros((size, theta_dim), dtype=torch.float32)


class _PartiallyFailingPosterior:
    """Test double: row-wise sampling fails for selected rows only."""

    sample_calls = 0
    sample_batched_calls = 0

    def sample_batched(self, *args, **kwargs):
        type(self).sample_batched_calls += 1
        raise AssertionError("simulated batched spline discriminant failure")

    def sample(self, sample_shape=(), x=None, **kwargs):
        import torch

        type(self).sample_calls += 1
        if float(x[0, 0].item()) < 0.0:
            raise AssertionError("simulated single-row spline discriminant failure")

        size = int(torch.Size(sample_shape).numel())
        theta_dim = 2
        return torch.ones((size, theta_dim), dtype=torch.float32)


class _DirectSamplerPosterior:
    """Test double: supports direct proposal sampling without rejection loop."""

    sample_batched_calls = 0

    class _Estimator:
        @staticmethod
        def sample(sample_shape, condition):
            import torch

            n = int(torch.Size(sample_shape).numel())
            b = int(condition.shape[0])
            base = condition[:, :2].to(torch.float32)
            return base.unsqueeze(0).repeat(n, 1, 1)

    class _Prior:
        @staticmethod
        def log_prob(theta):
            import torch

            return torch.where(theta[:, 0] >= 0.0, torch.zeros_like(theta[:, 0]), -torch.inf)

    def __init__(self):
        self.posterior_estimator = self._Estimator()
        self.prior = self._Prior()

    def sample_batched(self, *args, **kwargs):
        type(self).sample_batched_calls += 1
        raise AssertionError("should not call sample_batched in direct-sampler mode")

    def sample(self, *args, **kwargs):
        raise AssertionError("should not call sample() in direct-sampler mode")


class _DirectSamplerBatchAssertionPosterior:
    """Test double: direct batched sampling asserts, row-wise partially succeeds."""

    class _Estimator:
        @staticmethod
        def sample(sample_shape, condition):
            import torch

            if int(condition.shape[0]) > 1:
                raise AssertionError("simulated direct batched spline discriminant failure")
            if float(condition[0, 0].item()) < 0.0:
                raise AssertionError("simulated direct single-row spline discriminant failure")
            n = int(torch.Size(sample_shape).numel())
            return torch.ones((n, 2), dtype=torch.float32)

    class _Prior:
        @staticmethod
        def log_prob(theta):
            import torch

            return torch.zeros(theta.shape[0], dtype=torch.float32, device=theta.device)

    def __init__(self):
        self.posterior_estimator = self._Estimator()
        self.prior = self._Prior()

    def sample(self, *args, **kwargs):
        raise AssertionError("should not call posterior.sample() in direct-sampler mode")

    def sample_batched(self, *args, **kwargs):
        raise AssertionError("should not call posterior.sample_batched() in direct-sampler mode")


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


def _require_sbi():
    pytest.importorskip("torch")
    pytest.importorskip("sbi")


def _sbi_sampling_kwargs(model_name: str, n_samples: int, *, show_progress_bars: bool = False):
    if model_name == "NPE":
        return {
            "sample_shape": (int(n_samples),),
            "max_sampling_batch_size": 10_000,
            "show_progress_bars": bool(show_progress_bars),
        }
    return {
        "sample_shape": (int(n_samples),),
        "method": "slice_np_vectorized",
        "thin": 1,
        "num_chains": 1,
        "num_workers": 1,
        "show_progress_bars": bool(show_progress_bars),
    }


def _native_sbi_samples(
        sbi_model: str,
        *,
        prior,
        theta,
        x,
        x_query,
        train_params: dict,
        sample_kwargs: dict,
        estimator_name: str | None = None,
):
    from sbi.inference import NPE, NLE, NRE

    inference_cls = {"NPE": NPE, "NLE": NLE, "NRE": NRE}[sbi_model]
    if estimator_name is None:
        inference = inference_cls(prior=prior)
    elif sbi_model in {"NPE", "NLE"}:
        inference = inference_cls(prior=prior, density_estimator=estimator_name)
    else:
        inference = inference_cls(prior=prior, classifier=estimator_name)
    density_estimator = inference.append_simulations(theta, x).train(**train_params)
    posterior = inference.build_posterior(density_estimator)

    kwargs = dict(sample_kwargs)
    kwargs["x"] = x_query

    if x_query.ndim >= 2 and x_query.shape[0] > 1 and hasattr(posterior, "sample_batched"):
        samples = posterior.sample_batched(**kwargs)
    else:
        samples = posterior.sample(**kwargs)

    return samples


def _ncpi_sbi_samples(
        sbi_model: str,
        *,
        prior,
        theta,
        x,
        x_query,
        train_params: dict,
        sample_kwargs: dict,
        result_dir: str,
        estimator_kwargs: dict | None = None,
        seed: int = 0,
):
    hyper = {
        "prior": prior,
        "inference_kwargs": {"device": "cpu"},
        "build_posterior_kwargs": {},
    }
    if estimator_kwargs is not None:
        hyper["estimator_kwargs"] = dict(estimator_kwargs)

    inf = Inference(sbi_model, hyperparams=hyper)
    inf.add_simulation_data(x.detach().cpu().numpy(), theta.detach().cpu().numpy())
    inf.train(
        param_grid=None,
        n_splits=2,
        n_repeats=1,
        train_params=train_params,
        result_dir=result_dir,
        seed=seed,
        sbi_eval_sampling_kwargs=sample_kwargs,
    )
    return inf.predict(
        x_query.detach().cpu().numpy(),
        result_dir=result_dir,
        sbi_eval_sampling_kwargs=sample_kwargs,
    )


def _assert_samples_close(native_samples, ncpi_samples, *, atol_mean=0.45, atol_std=0.45):
    native_np = native_samples.detach().cpu().numpy()
    ncpi_np = np.asarray(ncpi_samples)
    assert native_np.shape == ncpi_np.shape
    assert np.all(np.isfinite(native_np))
    assert np.all(np.isfinite(ncpi_np))

    mean_delta = float(np.max(np.abs(native_np.mean(axis=0) - ncpi_np.mean(axis=0))))
    std_delta = float(np.max(np.abs(native_np.std(axis=0) - ncpi_np.std(axis=0))))
    assert mean_delta < atol_mean
    assert std_delta < atol_std


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
        sbi_eval_sampling_kwargs=_sbi_sampling_kwargs(
            sbi_model,
            30,
            show_progress_bars=False,
        ),
    )

    # Predict posterior samples for multiple observations
    samples = inf.predict(
        X[:4],
        result_dir=tmp_result_dir,
        sbi_eval_sampling_kwargs=_sbi_sampling_kwargs(
            sbi_model,
            30,
            show_progress_bars=False,
        ),
    )
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (30, 4, 2)
    assert np.all(np.isfinite(samples))

    # Sample posterior for a single observation (returns S x theta_dim)
    samples1 = inf.predict(
        X[0],
        result_dir=tmp_result_dir,
        sbi_eval_sampling_kwargs=_sbi_sampling_kwargs(
            sbi_model,
            40,
            show_progress_bars=False,
        ),
    )
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
        sbi_eval_sampling_kwargs=_sbi_sampling_kwargs("NPE", 10, show_progress_bars=False),
    )

    empty = inf.predict(
        np.empty((0, 2)),
        result_dir=tmp_result_dir,
        sbi_eval_sampling_kwargs=_sbi_sampling_kwargs("NPE", 15, show_progress_bars=False),
    )
    assert isinstance(empty, np.ndarray)
    assert empty.shape == (15, 0, 2)


def test_sbi_predict_all_invalid_rows_returns_empty_sample_tensor_shape(tmp_result_dir):
    X, Y = _make_linear_data(n=20, d=2, theta_dim=2, seed=16)

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
        sbi_eval_sampling_kwargs=_sbi_sampling_kwargs("NPE", 10, show_progress_bars=False),
    )

    X_bad = np.full((3, 2), np.nan, dtype=float)
    out = inf.predict(
        X_bad,
        result_dir=tmp_result_dir,
        sbi_eval_sampling_kwargs=_sbi_sampling_kwargs("NPE", 12, show_progress_bars=False),
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (12, 0, 2)


def test_sbi_train_predict_allow_missing_sample_shape_sampling_kwargs(tmp_result_dir):
    _require_sbi()
    X, Y = _make_linear_data(n=20, d=2, theta_dim=2, seed=15)

    import torch
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=torch.tensor([-5.0, -5.0]), high=torch.tensor([5.0, 5.0]))
    inf = Inference("NPE", hyperparams={"prior": prior})
    inf.add_simulation_data(X, Y)

    inf.train(
        param_grid=None,
        n_splits=2,
        n_repeats=1,
        train_params={"max_num_epochs": 1, "show_train_summary": False},
        result_dir=tmp_result_dir,
        seed=0,
        sbi_eval_sampling_kwargs={"show_progress_bars": False},
    )

    preds = inf.predict(
        X[:3],
        result_dir=tmp_result_dir,
        sbi_eval_sampling_kwargs={"show_progress_bars": False},
    )
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1, 3, 2)
    assert np.all(np.isfinite(preds))


def test_sbi_predict_fallbacks_to_rowwise_after_batched_assertion(tmp_result_dir):
    _require_sbi()
    import torch
    from sbi.utils import BoxUniform

    X, Y = _make_linear_data(n=20, d=2, theta_dim=2, seed=31)
    prior = BoxUniform(low=torch.tensor([-5.0, -5.0]), high=torch.tensor([5.0, 5.0]))
    inf = Inference("NPE", hyperparams={"prior": prior})
    inf.theta = Y

    _FailBatchedPosterior.sample_calls = 0
    _FailBatchedPosterior.sample_batched_calls = 0

    posterior_path = os.path.join(tmp_result_dir, "posterior.pkl")
    with open(posterior_path, "wb") as f:
        pickle.dump(_FailBatchedPosterior(), f)

    out = inf.predict(
        X[:4],
        result_dir=tmp_result_dir,
        scaler=False,
        sbi_eval_sampling_kwargs={"sample_shape": (6,), "show_progress_bars": False},
    )

    assert isinstance(out, np.ndarray)
    assert out.shape == (6, 4, 2)
    assert np.all(np.isfinite(out))
    assert _FailBatchedPosterior.sample_batched_calls == 1
    assert _FailBatchedPosterior.sample_calls == 4


def test_sbi_predict_rowwise_fallback_keeps_partial_rows_as_nan(tmp_result_dir):
    _require_sbi()
    import torch
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=torch.tensor([-5.0, -5.0]), high=torch.tensor([5.0, 5.0]))
    inf = Inference("NPE", hyperparams={"prior": prior})
    inf.theta = np.zeros((5, 2), dtype=float)

    _PartiallyFailingPosterior.sample_calls = 0
    _PartiallyFailingPosterior.sample_batched_calls = 0

    posterior_path = os.path.join(tmp_result_dir, "posterior.pkl")
    with open(posterior_path, "wb") as f:
        pickle.dump(_PartiallyFailingPosterior(), f)

    x_query = np.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]], dtype=float)
    out = inf.predict(
        x_query,
        result_dir=tmp_result_dir,
        scaler=False,
        sbi_eval_sampling_kwargs={"sample_shape": (4,), "show_progress_bars": False},
    )

    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 3, 2)
    assert np.all(np.isfinite(out[:, 0, :]))
    assert np.all(np.isnan(out[:, 1, :]))
    assert np.all(np.isfinite(out[:, 2, :]))
    assert _PartiallyFailingPosterior.sample_batched_calls == 1
    assert _PartiallyFailingPosterior.sample_calls == 3


def test_sbi_predict_reject_outside_prior_false_bypasses_rejection_sampler(tmp_result_dir):
    _require_sbi()
    import torch
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=torch.tensor([-5.0, -5.0]), high=torch.tensor([5.0, 5.0]))
    inf = Inference("NPE", hyperparams={"prior": prior})
    inf.theta = np.zeros((5, 2), dtype=float)

    _DirectSamplerPosterior.sample_batched_calls = 0
    posterior_path = os.path.join(tmp_result_dir, "posterior.pkl")
    with open(posterior_path, "wb") as f:
        pickle.dump(_DirectSamplerPosterior(), f)

    x_query = np.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]], dtype=float)
    out = inf.predict(
        x_query,
        result_dir=tmp_result_dir,
        scaler=False,
        sbi_eval_sampling_kwargs={
            "sample_shape": (4,),
            "show_progress_bars": False,
            "reject_outside_prior": False,
            "ncpi_nan_outside_prior": True,
        },
    )

    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 3, 2)
    assert np.all(np.isfinite(out[:, 0, :]))
    assert np.all(np.isnan(out[:, 1, :]))
    assert np.all(np.isfinite(out[:, 2, :]))
    assert _DirectSamplerPosterior.sample_batched_calls == 0


def test_sbi_predict_direct_sampler_batch_assertion_falls_back_rowwise(tmp_result_dir):
    _require_sbi()
    import torch
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=torch.tensor([-5.0, -5.0]), high=torch.tensor([5.0, 5.0]))
    inf = Inference("NPE", hyperparams={"prior": prior})
    inf.theta = np.zeros((5, 2), dtype=float)

    posterior_path = os.path.join(tmp_result_dir, "posterior.pkl")
    with open(posterior_path, "wb") as f:
        pickle.dump(_DirectSamplerBatchAssertionPosterior(), f)

    x_query = np.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]], dtype=float)
    out = inf.predict(
        x_query,
        result_dir=tmp_result_dir,
        scaler=False,
        sbi_eval_sampling_kwargs={
            "sample_shape": (4,),
            "show_progress_bars": False,
            "reject_outside_prior": False,
            "ncpi_nan_outside_prior": True,
        },
    )

    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 3, 2)
    assert np.all(np.isfinite(out[:, 0, :]))
    assert np.all(np.isnan(out[:, 1, :]))
    assert np.all(np.isfinite(out[:, 2, :]))


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
        sbi_eval_sampling_kwargs=_sbi_sampling_kwargs("NPE", 20, show_progress_bars=False),
    )

    with open(f"{tmp_result_dir}/posterior.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{tmp_result_dir}/inference.pkl", "rb") as f:
        inference_obj = pickle.load(f)
    with open(f"{tmp_result_dir}/density_estimator.pkl", "rb") as f:
        density_estimator = pickle.load(f)

    assert isinstance(model, list)
    assert len(model) == 2  # n_splits * n_repeats
    assert all(hasattr(p, "sample") for p in model)
    assert isinstance(inference_obj, list) and len(inference_obj) == 2
    assert isinstance(density_estimator, list) and len(density_estimator) == 2

    samples = inf.predict(
        X[:3],
        result_dir=tmp_result_dir,
        sbi_eval_sampling_kwargs=_sbi_sampling_kwargs("NPE", 30, show_progress_bars=False),
    )
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


def test_sbi_pairing_tutorial_npe_single_observation(tmp_path):
    """
    Pairing test (tutorial-like): compare native sbi NPE vs ncpi.Inference NPE.
    """
    _require_sbi()
    import random
    import torch
    from sbi.utils import BoxUniform

    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_dims = 2
    num_sims = 300

    prior = BoxUniform(low=torch.zeros(num_dims), high=torch.ones(num_dims))
    theta = prior.sample((num_sims,)).to(torch.float32)
    x = (theta + torch.randn_like(theta) * 0.1).to(torch.float32)
    x_o = torch.tensor([0.5, 0.5], dtype=torch.float32)

    train_params = {
        "max_num_epochs": 1,
        "training_batch_size": 64,
        "show_train_summary": False,
    }
    sample_kwargs = _sbi_sampling_kwargs("NPE", 100, show_progress_bars=False)

    torch.manual_seed(seed)
    native_samples = _native_sbi_samples(
        "NPE",
        prior=prior,
        theta=theta,
        x=x,
        x_query=x_o,
        train_params=train_params,
        sample_kwargs=sample_kwargs,
    )

    torch.manual_seed(seed)
    ncpi_samples = _ncpi_sbi_samples(
        "NPE",
        prior=prior,
        theta=theta,
        x=x,
        x_query=x_o,
        train_params=train_params,
        sample_kwargs=sample_kwargs,
        result_dir=str(tmp_path / "pair_npe"),
        seed=seed,
    )

    _assert_samples_close(native_samples, ncpi_samples)


@pytest.mark.parametrize("sbi_model", ["NPE", "NLE", "NRE"])
def test_sbi_pairing_tutorial_models_single_and_batch(tmp_path, sbi_model):
    """
    Pairing test (tutorial-like): compare native sbi and ncpi for NPE/NLE/NRE.
    Verifies both single-observation and batched-observation sampling.
    """
    _require_sbi()
    import random
    import torch
    from sbi.utils import BoxUniform

    seed = 11
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_dims = 2
    num_sims = 250

    prior = BoxUniform(low=torch.zeros(num_dims), high=torch.ones(num_dims))
    theta = prior.sample((num_sims,)).to(torch.float32)
    x = (theta + torch.randn_like(theta) * 0.1).to(torch.float32)
    x_o = torch.tensor([0.5, 0.5], dtype=torch.float32)
    x_batch = x[:4]

    train_params = {
        "max_num_epochs": 1,
        "training_batch_size": 64,
        "show_train_summary": False,
    }
    sample_kwargs = _sbi_sampling_kwargs(sbi_model, 120, show_progress_bars=False)
    estimator_kwargs = {"estimator": "mlp"} if sbi_model == "NRE" else None

    torch.manual_seed(seed)
    native_single = _native_sbi_samples(
        sbi_model,
        prior=prior,
        theta=theta,
        x=x,
        x_query=x_o,
        train_params=train_params,
        sample_kwargs=sample_kwargs,
        estimator_name=(None if estimator_kwargs is None else estimator_kwargs["estimator"]),
    )
    torch.manual_seed(seed)
    ncpi_single = _ncpi_sbi_samples(
        sbi_model,
        prior=prior,
        theta=theta,
        x=x,
        x_query=x_o,
        train_params=train_params,
        sample_kwargs=sample_kwargs,
        result_dir=str(tmp_path / f"pair_single_{sbi_model}"),
        estimator_kwargs=estimator_kwargs,
        seed=seed,
    )
    _assert_samples_close(native_single, ncpi_single)

    torch.manual_seed(seed + 1)
    native_batch = _native_sbi_samples(
        sbi_model,
        prior=prior,
        theta=theta,
        x=x,
        x_query=x_batch,
        train_params=train_params,
        sample_kwargs=sample_kwargs,
        estimator_name=(None if estimator_kwargs is None else estimator_kwargs["estimator"]),
    )
    torch.manual_seed(seed + 1)
    ncpi_batch = _ncpi_sbi_samples(
        sbi_model,
        prior=prior,
        theta=theta,
        x=x,
        x_query=x_batch,
        train_params=train_params,
        sample_kwargs=sample_kwargs,
        result_dir=str(tmp_path / f"pair_batch_{sbi_model}"),
        estimator_kwargs=estimator_kwargs,
        seed=seed,
    )
    _assert_samples_close(native_batch, ncpi_batch, atol_mean=0.55, atol_std=0.55)


@pytest.mark.parametrize("sbi_model", ["NPE", "NLE", "NRE"])
def test_sbi_pairing_modified_hyperparams_and_sampling_kwargs(tmp_path, sbi_model):
    """
    Pairing test with modified model hyperparameters and sbi_eval_sampling_kwargs.
    """
    _require_sbi()
    import random
    import torch
    from sbi.utils import BoxUniform

    seed = 19
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_dims = 2
    num_sims = 220

    prior = BoxUniform(low=torch.zeros(num_dims), high=torch.ones(num_dims))
    theta = prior.sample((num_sims,)).to(torch.float32)
    x = (theta + torch.randn_like(theta) * 0.1).to(torch.float32)
    x_o = torch.tensor([0.25, 0.75], dtype=torch.float32)

    train_params = {
        "max_num_epochs": 1,
        "training_batch_size": 32,
        "learning_rate": 1e-3,
        "show_train_summary": False,
    }

    if sbi_model == "NPE":
        estimator_kwargs = {"estimator": "maf"}
        sample_kwargs = {
            "sample_shape": (90,),
            "max_sampling_batch_size": 64,
            "show_progress_bars": False,
        }
    elif sbi_model == "NLE":
        estimator_kwargs = {"estimator": "maf"}
        sample_kwargs = {
            "sample_shape": (90,),
            "method": "slice_np_vectorized",
            "thin": 1,
            "num_chains": 1,
            "num_workers": 1,
            "show_progress_bars": False,
        }
    else:
        estimator_kwargs = {"estimator": "mlp"}
        sample_kwargs = {
            "sample_shape": (90,),
            "method": "slice_np_vectorized",
            "thin": 1,
            "num_chains": 1,
            "num_workers": 1,
            "show_progress_bars": False,
        }

    torch.manual_seed(seed)
    native_samples = _native_sbi_samples(
        sbi_model,
        prior=prior,
        theta=theta,
        x=x,
        x_query=x_o,
        train_params=train_params,
        sample_kwargs=sample_kwargs,
        estimator_name=estimator_kwargs["estimator"],
    )
    torch.manual_seed(seed)
    ncpi_samples = _ncpi_sbi_samples(
        sbi_model,
        prior=prior,
        theta=theta,
        x=x,
        x_query=x_o,
        train_params=train_params,
        sample_kwargs=sample_kwargs,
        result_dir=str(tmp_path / f"pair_modified_{sbi_model}"),
        estimator_kwargs=estimator_kwargs,
        seed=seed,
    )

    _assert_samples_close(native_samples, ncpi_samples, atol_mean=0.6, atol_std=0.6)
