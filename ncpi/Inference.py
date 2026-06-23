import os
import random
import numpy as np
import inspect
import pickle
from ncpi import tools

# --- Prediction multiprocessing helpers (worker-global state) ---
_PRED_MODEL = None
_PRED_SCALER = None

# --- Sklearn CV multiprocessing helpers (worker-global state) ---
_CV_BASE_MODEL = None
_CV_X = None
_CV_Y = None
_CV_PARAMS = None
_CV_N_SPLITS = None
_CV_SEED = None
_CV_DEFAULT_VERBOSE = None


def _prediction_worker_init(model, scaler):
    """Initializer: runs once per worker process (sklearn-only)."""
    global _PRED_MODEL, _PRED_SCALER
    _PRED_MODEL = model
    _PRED_SCALER = scaler


def _predict_one(x):
    """Predict one sample in a worker process; return None for invalid rows."""
    global _PRED_MODEL, _PRED_SCALER

    x = np.asarray(x)

    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(1, -1)

    if _PRED_SCALER is not None:
        x = _PRED_SCALER.transform(x)

    if not np.all(np.isfinite(x)):
        return None

    model = _PRED_MODEL
    if isinstance(model, list):
        preds = [np.asarray(m.predict(x)) for m in model]
        y = np.mean(preds, axis=0)
    else:
        y = np.asarray(model.predict(x))

    y0 = np.asarray(y)[0]
    if y0.ndim == 0:
        return float(y0)
    return y0.astype(float).tolist()


def _cv_worker_init(base_model, X, Y, params, n_splits, seed, default_verbose):
    """Initializer for sklearn CV workers (runs once per process)."""
    global _CV_BASE_MODEL, _CV_X, _CV_Y, _CV_PARAMS, _CV_N_SPLITS, _CV_SEED, _CV_DEFAULT_VERBOSE
    _CV_BASE_MODEL = base_model
    _CV_X = X
    _CV_Y = Y
    _CV_PARAMS = params
    _CV_N_SPLITS = n_splits
    _CV_SEED = seed
    _CV_DEFAULT_VERBOSE = default_verbose


def _train_sklearn_cv_fold(task):
    """Train one sklearn CV fold and return (fold_i, mse, trained_model)."""
    fold_i, tr, te = task

    # Local import in worker process keeps module-level imports lightweight.
    from sklearn.base import clone as sklearn_clone

    repeat_id = fold_i // _CV_N_SPLITS
    repeat_seed = _CV_SEED + repeat_id

    m = sklearn_clone(_CV_BASE_MODEL)
    p = dict(_CV_PARAMS)
    if "random_state" in m.get_params():
        p["random_state"] = repeat_seed
    if _CV_DEFAULT_VERBOSE is not None and "verbose" in m.get_params() and "verbose" not in p:
        p["verbose"] = _CV_DEFAULT_VERBOSE
    m.set_params(**p)

    y_tr = _CV_Y[tr]
    if y_tr.ndim == 2 and y_tr.shape[1] == 1:
        y_tr = y_tr.ravel()
    m.fit(_CV_X[tr], y_tr)

    pred = np.asarray(m.predict(_CV_X[te]))

    y_te = _CV_Y[te]
    if y_te.ndim == 2 and y_te.shape[1] == 1:
        y_te = y_te.ravel()
    if np.asarray(pred).ndim > 1 and y_te.ndim == 1:
        pred = np.asarray(pred).ravel()

    mse = float(np.mean((pred - y_te) ** 2))
    return fold_i, mse, m


class Inference:
    """Parameter inference using sklearn regressors or SBI methods (NPE, NLE, NRE)."""

    # Supported amortized SBI models
    SBI_MODELS = ("NPE", "NLE", "NRE")

    # Class-level caches so we don't re-scan sklearn each instance
    _SKLEARN_READY = False
    _SKLEARN_REGRESSORS = None  # dict[str, type]

    def __init__(self, model: str, hyperparams: dict | None = None):
        """Initialize inference backend and validate model/hyperparameter inputs."""
        if not isinstance(model, str):
            raise TypeError("model must be a string.")
        model = model.strip()

        if hyperparams is not None and not isinstance(hyperparams, dict):
            raise TypeError("hyperparams must be a dict or None.")
        self.hyperparams = hyperparams

        self.model_name = model

        # Decide backend using registry-style checks
        if model in self.SBI_MODELS:
            self.backend = "sbi"
            self._init_sbi_modules()
        else:
            self.backend = "sklearn"
            self._init_sklearn_modules()
            self._ensure_sklearn_regressor_cache()

            if model not in self._SKLEARN_REGRESSORS:
                valid_sklearn = sorted(self._SKLEARN_REGRESSORS.keys())
                raise ValueError(
                    f"'{model}' is not valid. Use a sklearn regressor or an SBI model from {list(self.SBI_MODELS)}.\n"
                    f"Example sklearn regressors: {valid_sklearn[:10]}{' ...' if len(valid_sklearn) > 10 else ''}"
                )

        # Training data
        self.features = None
        self.theta = None

    # -----------------------------
    # Module initialization
    # -----------------------------

    def _init_sklearn_modules(self):
        """Load sklearn and runtime helpers needed by the sklearn backend."""
        if not tools.ensure_module("sklearn", package="scikit-learn", version_spec="==1.5.0"):
            raise ImportError("scikit-learn==1.5.0 is required (import name: 'sklearn').")

        self.RepeatedKFold = tools.dynamic_import("sklearn.model_selection", "RepeatedKFold")
        self.all_estimators = tools.dynamic_import("sklearn.utils", "all_estimators")
        self.RegressorMixin = tools.dynamic_import("sklearn.base", "RegressorMixin")

        self.multiprocessing = tools.dynamic_import("multiprocessing")
        self.tqdm_inst = tools.ensure_module("tqdm")
        self.tqdm = tools.dynamic_import("tqdm", "tqdm") if self.tqdm_inst else None


    def _ensure_sklearn_regressor_cache(self):
        """Populate a class-level cache that maps sklearn regressor names to classes."""
        if self.__class__._SKLEARN_READY and self.__class__._SKLEARN_REGRESSORS is not None:
            return

        reg_map = {}
        for name, cls in self.all_estimators():
            if inspect.isclass(cls) and issubclass(cls, self.RegressorMixin):
                reg_map[name] = cls

        self.__class__._SKLEARN_REGRESSORS = reg_map
        self.__class__._SKLEARN_READY = True


    def _init_sbi_modules(self):
        """Load torch plus only the SBI inference/builder callables required by the active SBI model."""
        if not tools.ensure_module("sbi", package="sbi", version_spec="==0.24.0"):
            raise ImportError("sbi==0.24.0 is required.")
        if not tools.ensure_module("torch", package="torch", raise_on_error=False):
            raise ImportError("PyTorch ('torch') is required but not importable.")

        # sklearn is necessary for RepeatedKFold
        if not tools.ensure_module("sklearn", package="scikit-learn", version_spec="==1.5.0"):
            raise ImportError("scikit-learn==1.5.0 is required (import name: 'sklearn').")

        if self.model_name not in self.SBI_MODELS:
            raise ValueError(f"Unsupported SBI model '{self.model_name}'.")

        self.RepeatedKFold = tools.dynamic_import("sklearn.model_selection", "RepeatedKFold")
        self.torch = tools.dynamic_import("torch")
        self.sbi_inference_cls = tools.dynamic_import("sbi.inference", self.model_name)

        builder_name_by_model = {
            "NPE": "posterior_nn",
            "NLE": "likelihood_nn",
            "NRE": "classifier_nn",
        }
        builder_name = builder_name_by_model[self.model_name]
        self.sbi_builder_factory = tools.dynamic_import("sbi.neural_nets", builder_name)

    # -----------------------------
    # Posterior sampling (SBI)
    # -----------------------------

    def _coerce_sample_shape(self, sample_shape):
        """Normalize sample_shape into torch.Size, accepting scalar ints as (int,)."""
        if isinstance(sample_shape, (int, np.integer)):
            sample_shape = (int(sample_shape),)
        return self.torch.Size(sample_shape)

    def _sample_posterior(
            self,
            posterior_obj,
            *,
            x=None,
            **sbi_eval_sampling_kwargs,
    ):
        """
        Draw posterior samples for one observation or a batch of observations.

        Any kwargs accepted by the underlying SBI posterior `sample(...)` or
        `sample_batched(...)` can be forwarded via `sbi_eval_sampling_kwargs`.
        This method injects/overrides `x` from its explicit argument and uses
        `sample_batched(...)` when `x` is a batch (batch size > 1).
        """
        kwargs = dict(sbi_eval_sampling_kwargs or {})
        reject_outside_prior = kwargs.pop("reject_outside_prior", None)
        nan_outside_prior = kwargs.pop("ncpi_nan_outside_prior", True)
        force_direct_sampler = kwargs.pop("ncpi_force_direct_sampler", False)
        kwargs["x"] = x
        if "sample_shape" in kwargs:
            kwargs["sample_shape"] = self._coerce_sample_shape(kwargs["sample_shape"])
        else:
            kwargs["sample_shape"] = self.torch.Size(())

        batch_size = None
        x_arg = kwargs.get("x", None)
        if x_arg is not None:
            x_shape = getattr(x_arg, "shape", None)
            if x_shape is not None:
                ndim = len(x_shape)
                if ndim >= 2:
                    batch_size = int(x_shape[0])
                else:
                    batch_size = 1
            else:
                batch_size = 1

        use_batched = bool(batch_size is not None and batch_size > 1)
        use_direct_sampler = bool(force_direct_sampler) or (reject_outside_prior is False)

        if use_direct_sampler:
            samples = self._sample_posterior_without_rejection(
                posterior_obj,
                kwargs,
                nan_outside_prior=bool(nan_outside_prior),
            )
            if samples.ndim == 2:
                samples = samples.unsqueeze(1)
            return samples

        if use_batched:
            sample_batched = getattr(posterior_obj, "sample_batched", None)
            if not callable(sample_batched):
                raise AttributeError(
                    f"Posterior object of type {type(posterior_obj)} does not expose "
                    "'sample_batched', required for batched observations."
                )
            try:
                samples = sample_batched(**kwargs)
            except AssertionError as exc:
                # Some NPE/flow combinations can hit numeric assertions in batched
                # spline inversion. Retry robustly row-by-row.
                samples = self._sample_posterior_rowwise(posterior_obj, kwargs, exc)
        else:
            samples = posterior_obj.sample(**kwargs)

        if samples.ndim == 2:
            samples = samples.unsqueeze(1)
        return samples

    def _sample_posterior_without_rejection(self, posterior_obj, kwargs, *, nan_outside_prior=True):
        """
        Draw directly from posterior_estimator.sample(condition=x), bypassing
        rejection sampling loops. Optionally mask out-of-prior samples as NaN.
        """
        estimator = getattr(posterior_obj, "posterior_estimator", None)
        proposal_sample = getattr(estimator, "sample", None)
        if not callable(proposal_sample):
            raise RuntimeError(
                "Direct proposal sampling requested, but posterior object does not expose "
                "posterior_estimator.sample(...)."
            )

        sample_shape = kwargs.get("sample_shape", self.torch.Size(()))
        x_arg = kwargs.get("x", None)
        if x_arg is None:
            raise ValueError("Direct proposal sampling requires a conditioning observation 'x'.")

        def _normalize_direct_shape(t):
            if t.ndim == 2:
                t = t.unsqueeze(1)
            elif t.ndim > 3:
                t = t.reshape(-1, t.shape[-2], t.shape[-1])
            if t.ndim != 3:
                raise RuntimeError(
                    f"Unexpected direct-proposal sample shape: {tuple(t.shape)}. "
                    "Expected (S, B, theta_dim) or (S, theta_dim)."
                )
            return t

        x_shape = getattr(x_arg, "shape", None)
        batch_size = int(x_shape[0]) if (x_shape is not None and len(x_shape) >= 2) else 1

        try:
            samples = proposal_sample(sample_shape, condition=x_arg)
            samples = _normalize_direct_shape(samples)
        except AssertionError as exc:
            # Fall back to row-wise direct proposal sampling.
            if batch_size <= 1:
                theta_dim = self._infer_theta_dim_for_sbi_samples(posterior_obj)
                if theta_dim is None:
                    raise RuntimeError(
                        "Direct SBI proposal sampling failed for a single row and "
                        "could not infer output dimensionality for NaN placeholder."
                    ) from exc
                samples = self.torch.full(
                    (int(self._coerce_sample_shape(sample_shape).numel()), 1, int(theta_dim)),
                    float("nan"),
                    dtype=self.torch.float32,
                    device=x_arg.device,
                )
                print("Warning: direct SBI proposal sampling failed for one row; returning NaN samples.")
            else:
                n_samples = int(self._coerce_sample_shape(sample_shape).numel())
                theta_dim = self._infer_theta_dim_for_sbi_samples(posterior_obj)
                row_samples = [None] * batch_size
                first_valid = None
                failed_rows = []

                for row_idx in range(batch_size):
                    x_row = x_arg[row_idx:row_idx + 1]
                    try:
                        one = proposal_sample(sample_shape, condition=x_row)
                        one = _normalize_direct_shape(one)
                    except AssertionError as row_exc:
                        failed_rows.append((row_idx, row_exc))
                        continue

                    if one.shape[0] != n_samples:
                        raise RuntimeError(
                            f"Direct SBI row-wise sample count mismatch: expected {n_samples}, got {int(one.shape[0])}."
                        )
                    if int(one.shape[1]) != 1:
                        raise RuntimeError(
                            f"Direct SBI row-wise sample batch mismatch: expected 1, got {int(one.shape[1])}."
                        )

                    if theta_dim is None:
                        theta_dim = int(one.shape[-1])
                    elif int(one.shape[-1]) != int(theta_dim):
                        raise RuntimeError(
                            f"Inconsistent theta dimension in direct SBI fallback: expected {theta_dim}, got {int(one.shape[-1])}."
                        )

                    row_samples[row_idx] = one
                    if first_valid is None:
                        first_valid = one

                if theta_dim is None:
                    first_row_exc = failed_rows[0][1] if failed_rows else exc
                    raise RuntimeError(
                        "Direct SBI proposal sampling failed for all rows and could not infer output dimensionality."
                    ) from first_row_exc

                if first_valid is not None:
                    dtype = first_valid.dtype
                    device = first_valid.device
                else:
                    dtype = self.torch.float32
                    device = x_arg.device

                for row_idx, entry in enumerate(row_samples):
                    if entry is not None:
                        continue
                    row_samples[row_idx] = self.torch.full(
                        (n_samples, 1, int(theta_dim)),
                        float("nan"),
                        dtype=dtype,
                        device=device,
                    )

                if failed_rows:
                    preview = ",".join(str(idx) for idx, _ in failed_rows[:10])
                    suffix = "..." if len(failed_rows) > 10 else ""
                    print(
                        f"Warning: direct SBI row-wise fallback produced NaN samples for "
                        f"{len(failed_rows)}/{batch_size} rows (indices: {preview}{suffix})."
                    )

                samples = self.torch.cat(row_samples, dim=1)

        if nan_outside_prior:
            prior = getattr(posterior_obj, "prior", None)
            prior_log_prob = getattr(prior, "log_prob", None)
            if callable(prior_log_prob):
                try:
                    flat = samples.reshape(-1, samples.shape[-1])
                    logp = prior_log_prob(flat)
                    if hasattr(logp, "ndim") and int(logp.ndim) > 1:
                        logp = logp.reshape(logp.shape[0], -1).sum(dim=1)
                    finite = self.torch.isfinite(logp).reshape(samples.shape[0], samples.shape[1])
                    invalid = ~finite
                    if bool(invalid.any()):
                        samples = samples.clone()
                        samples[invalid] = float("nan")
                        bad = int(invalid.sum().item())
                        total = int(invalid.numel())
                        print(
                            f"Warning: direct SBI proposal sampling generated out-of-prior draws "
                            f"for {bad}/{total} sample-row entries; replaced with NaN."
                        )
                except Exception:
                    # Best-effort masking only; do not fail prediction because prior
                    # support checking is unavailable for a specific prior type.
                    pass

        return samples

    def _infer_theta_dim_for_sbi_samples(self, posterior_obj):
        """Best-effort theta dimensionality inference for SBI fallback outputs."""
        if self.theta is not None:
            y = np.asarray(self.theta)
            if y.ndim == 1:
                return 1
            if y.ndim >= 2 and y.shape[1] > 0:
                return int(y.shape[1])

        prior = getattr(posterior_obj, "prior", None)
        event_shape = getattr(prior, "event_shape", None)
        if event_shape is not None:
            try:
                n = int(np.prod(tuple(event_shape))) if len(tuple(event_shape)) > 0 else 1
                if n > 0:
                    return n
            except Exception:
                pass

        return None

    def _sample_posterior_rowwise(self, posterior_obj, kwargs, original_exc):
        """Fallback path when batched SBI sampling fails numerically."""
        x_arg = kwargs.get("x", None)
        x_shape = getattr(x_arg, "shape", None)
        if x_arg is None or x_shape is None or len(x_shape) < 2 or int(x_shape[0]) <= 1:
            raise original_exc

        base_kwargs = dict(kwargs)
        base_kwargs.pop("x", None)

        sample_shape = kwargs.get("sample_shape", self.torch.Size(()))
        n_samples = int(self._coerce_sample_shape(sample_shape).numel())
        if n_samples <= 0:
            raise ValueError("SBI fallback received non-positive sample count.")

        n_rows = int(x_shape[0])
        row_samples = [None] * n_rows
        theta_dim = self._infer_theta_dim_for_sbi_samples(posterior_obj)
        first_valid = None
        failed_rows = []

        for row_idx in range(n_rows):
            x_row = x_arg[row_idx:row_idx + 1]
            try:
                one = posterior_obj.sample(x=x_row, **base_kwargs)
            except AssertionError as row_exc:
                failed_rows.append((row_idx, row_exc))
                continue

            if one.ndim == 2:
                one = one.unsqueeze(1)
            elif one.ndim > 3:
                one = one.reshape(-1, one.shape[-2], one.shape[-1])

            if one.ndim != 3:
                raise RuntimeError(
                    f"Unexpected SBI single-row sample shape: {tuple(one.shape)}. "
                    "Expected (S, 1, theta_dim)."
                )

            if one.shape[0] != n_samples:
                raise RuntimeError(
                    f"SBI single-row sample count mismatch: expected {n_samples}, got {int(one.shape[0])}."
                )

            if theta_dim is None:
                theta_dim = int(one.shape[-1])
            elif int(one.shape[-1]) != int(theta_dim):
                raise RuntimeError(
                    f"Inconsistent theta dimension during SBI fallback: expected {theta_dim}, got {int(one.shape[-1])}."
                )

            row_samples[row_idx] = one
            if first_valid is None:
                first_valid = one

        if theta_dim is None:
            first_row_exc = failed_rows[0][1] if failed_rows else original_exc
            raise RuntimeError(
                "SBI posterior sampling failed in both batched and single-row modes. "
                "Could not infer output dimensionality to continue with NaN placeholders."
            ) from first_row_exc

        if first_valid is not None:
            sample_dtype = first_valid.dtype
            sample_device = first_valid.device
        else:
            sample_dtype = self.torch.float32
            sample_device = x_arg.device

        for row_idx, entry in enumerate(row_samples):
            if entry is not None:
                continue
            row_samples[row_idx] = self.torch.full(
                (n_samples, 1, int(theta_dim)),
                float("nan"),
                dtype=sample_dtype,
                device=sample_device,
            )

        if failed_rows:
            preview = ",".join(str(idx) for idx, _ in failed_rows[:10])
            suffix = "..." if len(failed_rows) > 10 else ""
            print(
                f"Warning: SBI row-wise fallback produced NaN samples for "
                f"{len(failed_rows)}/{n_rows} rows (indices: {preview}{suffix})."
            )

        return self.torch.cat(row_samples, dim=1)


    # -----------------------------
    # Pickling
    # -----------------------------

    def __getstate__(self):
        """Remove non-pickleable dynamically imported modules/callables."""
        state = self.__dict__.copy()

        # Drop modules and dynamic callables (they will be re-imported)
        drop_keys = {
            "RepeatedKFold",
            "all_estimators", "RegressorMixin",
            "multiprocessing", "tqdm", "torch",
            "sbi_inference_cls", "sbi_builder_factory",
        }
        for k in drop_keys:
            state.pop(k, None)

        # Drop any imported module objects if present
        for k in list(state.keys()):
            if isinstance(state[k], type(os)):
                del state[k]

        return state


    def __setstate__(self, state):
        """Restore instance state and re-import backend-specific dynamic dependencies."""
        self.__dict__.update(state)

        # Validate and normalize backend/model_name metadata before re-init.
        model_name = getattr(self, "model_name", None)
        backend = getattr(self, "backend", None)

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Invalid pickled Inference object: missing or invalid 'model_name'.")
        model_name = model_name.strip()
        self.model_name = model_name

        if backend is None:
            # Backward compatibility: infer backend for older pickles.
            backend = "sbi" if model_name in self.SBI_MODELS else "sklearn"
        elif backend not in {"sbi", "sklearn"}:
            raise ValueError(
                f"Invalid pickled Inference object: unsupported backend '{backend}'."
            )

        if backend == "sbi" and model_name not in self.SBI_MODELS:
            raise ValueError(
                f"Invalid pickled Inference object: backend='sbi' is incompatible with model '{model_name}'."
            )
        if backend == "sklearn" and model_name in self.SBI_MODELS:
            raise ValueError(
                f"Invalid pickled Inference object: backend='sklearn' is incompatible with SBI model '{model_name}'."
            )

        self.backend = backend

        # Re-init only what the normalized backend requires.
        if backend == "sbi":
            self._init_sbi_modules()
        else:
            self._init_sklearn_modules()
            self._ensure_sklearn_regressor_cache()


    # -----------------------------
    # Main methods
    # -----------------------------

    def add_simulation_data(self, features, parameters, *, append: bool = False, copy: bool = False):
        """
        Add (features, parameters) pairs to the training data.

        Parameters
        ----------
        features : array-like
            Feature matrix. Shape (n_samples,) or (n_samples, n_features).
        parameters : array-like
            Parameter matrix/array. Shape (n_samples,) or (n_samples, n_params).
        append : bool, default=False
            If True, concatenate onto existing self.features/self.theta. If False, overwrite.
        copy : bool, default=False
            If True, store copies. If False, store views when possible.
        """

        # Convert early; allows lists/torch tensors/etc. (if they implement __array__)
        X = np.asarray(features)
        Y = np.asarray(parameters)

        if X.ndim not in (1, 2):
            raise ValueError(f"features must be 1D or 2D, got shape {X.shape}.")
        if Y.ndim not in (1, 2):
            raise ValueError(f"parameters must be 1D or 2D, got shape {Y.shape}.")

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Features and parameters must have the same number of rows; got {X.shape[0]} and {Y.shape[0]}."
            )

        # Build a row-wise finite mask (works for both 1D and 2D)
        X_finite = np.isfinite(X).all(axis=-1) if X.ndim == 2 else np.isfinite(X)
        Y_finite = np.isfinite(Y).all(axis=-1) if Y.ndim == 2 else np.isfinite(Y)
        mask = X_finite & Y_finite

        # If everything was filtered out, fail loudly (prevents silent training on empty data)
        if not np.any(mask):
            raise ValueError("All rows contain NaN/Inf in features or parameters; nothing to add.")

        X = X[mask]
        Y = Y[mask]

        # Normalize to 2D (n_samples, n_features/n_params)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if copy:
            X = X.copy()
            Y = Y.copy()

        if append and self.features is not None and self.theta is not None:
            # Defensive checks: feature/param dimensionality must match to concatenate
            if self.features.shape[1] != X.shape[1]:
                raise ValueError(
                    f"Cannot append: feature dimension mismatch ({self.features.shape[1]} vs {X.shape[1]})."
                )
            if self.theta.shape[1] != Y.shape[1]:
                raise ValueError(
                    f"Cannot append: parameter dimension mismatch ({self.theta.shape[1]} vs {Y.shape[1]})."
                )
            self.features = np.concatenate([self.features, X], axis=0)
            self.theta = np.concatenate([self.theta, Y], axis=0)
        else:
            self.features = X
            self.theta = Y


    def initialize_sbi(self, hyperparams: dict):
        """
        Initialize an sbi inference trainer (NPE / NLE / NRE) from a configuration dict.

        Required
        --------
        prior:
            A torch.distributions.Distribution (or Distribution-like object supported by sbi).

        Optional
        --------
        inference_kwargs:
            Dict forwarded to the NPE/NLE/NRE constructor (e.g., show_progress_bars).

        estimator (advanced):
            A callable density-estimator/classifier builder (e.g., the object returned by
            sbi.neural_nets.posterior_nn / likelihood_nn / classifier_nn). If provided, it
            will be used directly.

        estimator_kwargs (recommended):
            Declarative network configuration used to build the estimator *inside* this method.
            Expected keys:
                - "estimator": str, e.g. "nsf", "maf", "mdn" (default: "nsf")
                - plus any kwargs accepted by the corresponding sbi builder, e.g.
                hidden_features, num_transforms, embedding_net, etc.

        build_posterior_kwargs:
            Dict stored on self for later calls to inference.build_posterior(...).
        """
        # Ensure this instance is configured for SBI.
        if self.backend != "sbi":
            raise ValueError("initialize_sbi() is only valid when backend='sbi'.")

        if not isinstance(hyperparams, dict):
            raise TypeError("hyperparams must be a dict.")

        # Prior is mandatory for SBI.
        prior = hyperparams.get("prior", None)
        if prior is None:
            raise ValueError("Missing required hyperparams['prior'].")

        # Select SBI method and its corresponding class (NPE/NLE/NRE).
        model_key = self.model_name  # "NPE" | "NLE" | "NRE"
        inference_cls = getattr(self, "sbi_inference_cls", None)
        if inference_cls is None:
            raise RuntimeError(
                "SBI modules not initialized. Construct with an SBI model so _init_sbi_modules() runs."
            )

        # Optional configs.
        inference_kwargs = hyperparams.get("inference_kwargs", {}) or {}
        estimator_kwargs = hyperparams.get("estimator_kwargs", {}) or {}
        build_posterior_kwargs = hyperparams.get("build_posterior_kwargs", {}) or {}

        if not isinstance(inference_kwargs, dict):
            raise TypeError("hyperparams['inference_kwargs'] must be a dict if provided.")
        if not isinstance(estimator_kwargs, dict):
            raise TypeError("hyperparams['estimator_kwargs'] must be a dict if provided.")
        if not isinstance(build_posterior_kwargs, dict):
            raise TypeError("hyperparams['build_posterior_kwargs'] must be a dict if provided.")

        # ------------------------------------------------------------------
        # Build the estimator:
        #   - If user provides a callable estimator builder, use it directly.
        #   - Otherwise, build it from 'estimator_kwargs' using sbi's canonical builders.
        # ------------------------------------------------------------------
        estimator_callable = hyperparams.get("estimator", None)

        if callable(estimator_callable):
            builder = estimator_callable
        else:
            # Work on a copy to avoid mutating the caller's dict.
            ekw = dict(estimator_kwargs)
            estimator_name = ekw.pop("estimator", "nsf")

            if not isinstance(estimator_name, str):
                raise TypeError("estimator must be a string like 'nsf', 'maf', or 'mdn'.")

            builder_factory = getattr(self, "sbi_builder_factory", None)
            if builder_factory is None:
                raise RuntimeError(
                    "SBI neural-net builder is not initialized. Construct with an SBI model so _init_sbi_modules() runs."
                )

            if model_key not in self.SBI_MODELS:
                raise ValueError(f"Unsupported SBI model '{model_key}'.")
            builder = builder_factory(model=estimator_name, **ekw)

        # ------------------------------------------------------------------
        # Instantiate the inference trainer with the correct keyword argument.
        # ------------------------------------------------------------------
        if model_key in ("NPE", "NLE"):
            inference = inference_cls(prior=prior, density_estimator=builder, **inference_kwargs)
        else:  # NRE
            inference = inference_cls(prior=prior, classifier=builder, **inference_kwargs)

        # Cache build_posterior kwargs for later posterior construction.
        self._sbi_build_posterior_kwargs = build_posterior_kwargs

        return inference



    def train(
            self,
            param_grid=None,
            *,
            n_splits: int = 10,
            n_repeats: int = 10,
            train_params: dict | None = None,
            result_dir: str = "data",
            scaler=None,
            seed: int = 0,
            sklearn_verbose: int | None = None,
            sbi_eval_sampling_kwargs: dict | None = None,
    ):
        """
        Train and (optionally) hyperparameter-search, then save artifacts to disk.

        Philosophy:
          - If param_grid is provided: train models for every fold for each candidate config,
            pick best config by mean CV MSE, and SAVE ALL fold-trained models for that best config.
          - If param_grid is None: train a single model on all data (no fold ensemble).

        Parameters
        ----------
        param_grid : list of dict or None
            If provided, a list of hyperparameter dicts to evaluate using cross-validation.
            Each dict is merged into self.hyperparams for each candidate.
        n_splits : int
            Number of CV splits (K-folds).
        n_repeats : int
            Number of CV repeats.
        train_params : dict or None
            Additional training parameters for SBI models (e.g. max_num_epochs).
        result_dir : str
            Directory where artifacts are saved:
            - sklearn: model.pkl (single model or fold-model list) and optionally scaler.pkl
            - sbi: inference.pkl, density_estimator.pkl, posterior.pkl (each single object or list),
              and optionally scaler.pkl
        scaler : fitted transformer or None
            If provided, used to scale features before training.
        seed : int
            Random seed for reproducibility.
        sklearn_verbose : int or None
            Default sklearn estimator verbosity level when the estimator exposes a
            ``verbose`` parameter and it is not explicitly set in model/grid hyperparameters.
            ``0`` disables verbose output; positive values enable it.
        sbi_eval_sampling_kwargs : dict or None
            Extra kwargs forwarded to `_sample_posterior(...)` during SBI CV evaluation.
            Any keyword arguments supported by the underlying SBI posterior sampling
            API are accepted. `x` is managed internally and must not be provided.
        Returns
        -------
        None
            This method saves trained artifacts to files and does not return objects.
        """

        train_params = train_params or {}
        if not isinstance(train_params, dict):
            raise TypeError("train_params must be a dict or None.")
        if sbi_eval_sampling_kwargs is None:
            sbi_eval_sampling_kwargs = {}
        if not isinstance(sbi_eval_sampling_kwargs, dict):
            raise TypeError("sbi_eval_sampling_kwargs must be a dict or None.")
        if "x" in sbi_eval_sampling_kwargs:
            raise ValueError(
                "sbi_eval_sampling_kwargs must not include 'x'; "
                "it is managed internally."
            )
        if self.backend == "sbi" and "sample_shape" in sbi_eval_sampling_kwargs:
            try:
                sample_shape_eval = self._coerce_sample_shape(sbi_eval_sampling_kwargs["sample_shape"])
            except Exception as exc:
                raise ValueError(
                    "sbi_eval_sampling_kwargs['sample_shape'] must be an int, tuple/list, or torch.Size."
                ) from exc
            if sample_shape_eval.numel() <= 0:
                raise ValueError("sbi_eval_sampling_kwargs['sample_shape'] must define at least one sample.")
        if sklearn_verbose is not None:
            try:
                sklearn_verbose = int(sklearn_verbose)
            except (TypeError, ValueError):
                raise ValueError(f"sklearn_verbose must be an integer or None, got: {sklearn_verbose}")
            sklearn_verbose = max(0, sklearn_verbose)

        # --------- Validate data ----------
        if self.features is None or len(self.features) == 0:
            raise ValueError("No features provided. Call add_simulation_data(...) first.")
        if self.theta is None or len(self.theta) == 0:
            raise ValueError("No parameters provided. Call add_simulation_data(...) first.")

        X = np.asarray(self.features)
        Y = np.asarray(self.theta)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"features/theta row mismatch: {X.shape[0]} vs {Y.shape[0]}.")

        # --------- Scale (do not mutate self.features) ----------
        fitted_scaler = None
        if scaler is True:
            from sklearn.preprocessing import StandardScaler
            fitted_scaler = StandardScaler()
        elif scaler is not None and not isinstance(scaler, (bool, np.bool_)):
            fitted_scaler = scaler

        if fitted_scaler is not None:
            fitted_scaler.fit(X)
            X = fitted_scaler.transform(X)

        all_splits = None
        total_folds = None
        if param_grid is not None:
            if n_splits < 2:
                raise ValueError("n_splits must be >= 2.")
            if n_repeats < 1:
                raise ValueError("n_repeats must be >= 1.")
            if X.shape[0] < n_splits:
                raise ValueError(f"n_samples ({X.shape[0]}) must be >= n_splits ({n_splits}).")

            if not tools.ensure_module("sklearn", package="scikit-learn", version_spec="==1.5.0"):
                raise ImportError("scikit-learn==1.5.0 is required (import name: 'sklearn').")

            repeated_kfold_cls = getattr(self, "RepeatedKFold", None)
            if repeated_kfold_cls is None:
                repeated_kfold_cls = tools.dynamic_import("sklearn.model_selection", "RepeatedKFold")

            splitter = repeated_kfold_cls(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
            all_splits = list(splitter.split(X))
            total_folds = len(all_splits)

        # =============== SKLEARN ===============
        if self.backend == "sklearn":
            RegressorClass = self._SKLEARN_REGRESSORS.get(self.model_name)
            if RegressorClass is None:
                raise ValueError(f"Unknown sklearn regressor '{self.model_name}'.")

            base_params = self.hyperparams or {}
            if not isinstance(base_params, dict):
                raise TypeError("For sklearn backend, hyperparams must be a dict or None.")

            sklearn_clone = tools.dynamic_import("sklearn.base", "clone")
            base_model = RegressorClass(**base_params)
            base_model_params = base_model.get_params()

            default_verbose = None
            if "verbose" in base_model_params and "verbose" not in base_params and sklearn_verbose is not None:
                current_verbose = base_model_params.get("verbose")
                if isinstance(current_verbose, (bool, np.bool_)):
                    default_verbose = bool(sklearn_verbose > 0)
                else:
                    default_verbose = int(sklearn_verbose)
                base_model.set_params(verbose=default_verbose)
                print(f"Enabled sklearn verbose logging: verbose={default_verbose}")

            if param_grid is not None:
                print("Starting hyperparameter search with cross-validation...")
                if not isinstance(param_grid, list) or not all(isinstance(d, dict) for d in param_grid):
                    raise ValueError("param_grid must be a list of dicts.")

                best_score = np.inf
                best_params = None
                best_fold_models = None

                for params in param_grid:
                    print(f"Evaluating params: {params}")
                    fold_models = []
                    fold_scores = []
                    fold_tasks = [(fold_i, tr, te) for fold_i, (tr, te) in enumerate(all_splits)]
                    max_workers = min(total_folds, os.cpu_count() or 1)

                    if max_workers > 1:
                        print(f"  Running folds in parallel across {max_workers} worker(s)")
                        ctx = self.multiprocessing.get_context("spawn")
                        with ctx.Pool(
                            processes=max_workers,
                            initializer=_cv_worker_init,
                            initargs=(base_model, X, Y, params, n_splits, seed, default_verbose),
                        ) as pool:
                            fold_results = pool.map(_train_sklearn_cv_fold, fold_tasks)
                    else:
                        _cv_worker_init(base_model, X, Y, params, n_splits, seed, default_verbose)
                        fold_results = [_train_sklearn_cv_fold(task) for task in fold_tasks]

                    for fold_i, mse, m in sorted(fold_results, key=lambda x: x[0]):
                        print(f"  Fold {fold_i+1}/{total_folds}")
                        fold_scores.append(mse)
                        fold_models.append(m)

                    mean_mse = float(np.mean(fold_scores))
                    if mean_mse < best_score:
                        best_score = mean_mse
                        best_params = dict(params)
                        best_fold_models = fold_models

                if best_fold_models is None:
                    raise ValueError("No best hyperparameters found.")
                else:
                    print(f"Best params: {best_params} | mean CV MSE: {best_score:.6f}")

                model = best_fold_models  # <- ensemble list

            else:
                print("Training single sklearn model on data...")
                # single model on full data
                model = sklearn_clone(base_model)
                if "random_state" in model.get_params():
                    model.set_params(random_state=seed)

                y_all = Y
                if y_all.ndim == 2 and y_all.shape[1] == 1:
                    y_all = y_all.ravel()

                model.fit(X, y_all)

        # =============== SBI ===============
        elif self.backend == "sbi":
            if self.hyperparams is None or not isinstance(self.hyperparams, dict) or self.hyperparams.get(
                    "prior") is None:
                raise ValueError("For SBI models you must provide hyperparams including a non-None 'prior'.")

            torch = self.torch
            X_t = torch.from_numpy(X.astype(np.float32))
            Y_t = torch.from_numpy(Y.astype(np.float32))
            base_cfg = dict(self.hyperparams)

            if param_grid is not None:
                print("Starting hyperparameter search with cross-validation...")
                if not isinstance(param_grid, list) or not all(isinstance(d, dict) for d in param_grid):
                    raise ValueError("param_grid must be a list of dicts.")

                best_score = np.inf
                best_cfg_delta = None
                best_fold_artifacts = None  # list[(inference, density_estimator, posterior)]

                for params in param_grid:
                    print(f"Evaluating params: {params}")
                    cfg = dict(base_cfg)
                    cfg.update(params)

                    fold_artifacts = []
                    fold_scores = []

                    for fold_i, (tr, te) in enumerate(all_splits):
                        print(f"  Fold {fold_i+1}/{total_folds}")
                        repeat_id = fold_i // n_splits
                        repeat_seed = seed + repeat_id

                        # seed once per repeat (consistent with sklearn)
                        torch.manual_seed(repeat_seed)
                        np.random.seed(repeat_seed)
                        random.seed(repeat_seed)

                        inf = self.initialize_sbi(cfg)
                        inf.append_simulations(Y_t[tr], X_t[tr])
                        de = inf.train(**train_params)

                        build_kwargs = getattr(self, "_sbi_build_posterior_kwargs", {}) or {}
                        posterior = inf.build_posterior(de, **build_kwargs)

                        # posterior mean MSE over test fold (batched)
                        te_idx = np.asarray(te)
                        total = 0.0
                        xb = X_t[te_idx]
                        yb = Y_t[te_idx]

                        samples = self._sample_posterior(
                            posterior,
                            x=xb,
                            **sbi_eval_sampling_kwargs,
                        )
                        if samples.ndim > 3:
                            samples = samples.reshape(-1, samples.shape[-2], samples.shape[-1])

                        mean = samples.mean(dim=0)  # [B, theta_dim]
                        mse = torch.mean((mean - yb) ** 2).item()
                        total += mse * te_idx.shape[0]
                        fold_mse = total / te_idx.shape[0]
                        fold_scores.append(fold_mse)
                        fold_artifacts.append((inf, de, posterior))

                    mean_mse = float(np.mean(fold_scores))
                    if mean_mse < best_score:
                        best_score = mean_mse
                        best_cfg_delta = dict(params)
                        best_fold_artifacts = fold_artifacts

                if best_fold_artifacts is None:
                    raise ValueError("No best hyperparameters found.")
                else:
                    print(f"Best params: {best_cfg_delta} | mean CV MSE: {best_score:.6f}")

                inf_artifact = [inf_obj for (inf_obj, _, _) in best_fold_artifacts]
                density_estimator_artifact = [de for (_, de, _) in best_fold_artifacts]
                posterior_artifact = [posterior for (_, _, posterior) in best_fold_artifacts]

            else:
                # single SBI model on full data
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                inf = self.initialize_sbi(base_cfg)
                inf.append_simulations(Y_t, X_t)
                print("Starting single-trial SBI training (inf.train)...")
                density_estimator = inf.train(**train_params)
                print("Finished single-trial SBI training (inf.train).")
                
                build_kwargs = getattr(self, "_sbi_build_posterior_kwargs", {}) or {}
                posterior = inf.build_posterior(density_estimator, **build_kwargs)

                inf_artifact = inf
                density_estimator_artifact = density_estimator
                posterior_artifact = posterior
                

        else:
            raise RuntimeError(f"Unknown backend '{self.backend}'.")

        # --------- Save artifacts ----------
        os.makedirs(result_dir, exist_ok=True)

        if fitted_scaler is not None:
            with open(os.path.join(result_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(fitted_scaler, f)
            print(f"Scaler saved at '{result_dir}/scaler.pkl'")

        if self.backend == "sklearn":
            with open(os.path.join(result_dir, "model.pkl"), "wb") as f:
                pickle.dump(model, f)
            print(f"Model saved at '{result_dir}/model.pkl'")
        else:
            with open(os.path.join(result_dir, "inference.pkl"), "wb") as f:
                pickle.dump(inf_artifact, f)
            print(f"Inference object saved at '{result_dir}/inference.pkl'")

            with open(os.path.join(result_dir, "density_estimator.pkl"), "wb") as f:
                pickle.dump(density_estimator_artifact, f)
            print(f"Density estimator saved at '{result_dir}/density_estimator.pkl'")

            with open(os.path.join(result_dir, "posterior.pkl"), "wb") as f:
                pickle.dump(posterior_artifact, f)
            print(f"Posterior saved at '{result_dir}/posterior.pkl'")


    def predict(
            self,
            features,
            *,
            result_dir: str = "data",
            scaler: bool = False,
            # sklearn multiprocessing knobs
            n_jobs: int | None = None,
            chunksize: int | None = None,
            start_method: str = "spawn",
            # SBI knobs
            sbi_eval_sampling_kwargs: dict | None = None,
    ):
        """
        Predict parameters from input features using the active backend.

        The method loads trained artifacts from ``result_dir``:
        - sklearn backend: ``model.pkl`` (single regressor or CV ensemble list).
        - sbi backend: ``posterior.pkl`` (single posterior or CV ensemble list).
        - optionally ``scaler.pkl`` when ``scaler=True``.

        Parameters
        ----------
        features : array-like
            Input observations. Accepted shapes:
            - scalar: interpreted as one sample with one feature, ``(1, 1)``.
            - 1D: interpreted using inferred feature dimension:
              ``(N, 1)`` when expected feature count is 1, otherwise ``(1, D)``.
            - 2D: interpreted as ``(n_samples, n_features)``.
            Ragged/object inputs that cannot be converted to a rectangular numeric
            array raise ``ValueError``.
        result_dir : str, default="data"
            Directory containing serialized training artifacts
            (``model.pkl``/``posterior.pkl`` and optionally ``scaler.pkl``).
        scaler : bool, default=False
            If ``True``, load and apply ``scaler.pkl`` before prediction/sampling.
            If ``False``, no scaling is applied.
        n_jobs : int | None, default=None
            sklearn only. Number of worker processes used for multiprocessing
            prediction. ``None`` uses all available CPUs.
        chunksize : int | None, default=None
            sklearn only. Chunk size passed to ``Pool.imap``. ``None`` selects an
            automatic value based on ``n_jobs`` and batch size.
        start_method : {"spawn", "fork", "forkserver"}, default="spawn"
            sklearn only. Multiprocessing context start method.
        sbi_eval_sampling_kwargs : dict | None, default=None
            SBI only. Extra keyword arguments forwarded to posterior sampling via
            ``_sample_posterior(...)``. The ``x`` argument is managed internally
            and must not be provided here. ``sample_shape`` may be an ``int``,
            tuple/list, or ``torch.Size`` and must define at least one sample.

        Returns
        -------
        sklearn backend
            list
                Per-input predictions. Each element is:
                - ``float`` for scalar output, or
                - ``list[float]`` for multi-output regression.
                Non-finite input rows are returned as NaN placeholders
                (shape follows inferred parameter dimensionality).
        sbi backend
            np.ndarray
                Posterior samples as float array.
                - For multiple input rows: shape ``(S, B, theta_dim)``.
                - For one finite input row: shape ``(S, theta_dim)``.

                ``S`` is the requested number of samples computed from
                ``sample_shape`` (product of dimensions; default is 1).
                ``B`` is the number of finite input rows.

                If ``posterior.pkl`` contains an ensemble (list of posteriors),
                the total requested ``S`` samples are distributed as evenly as
                possible across members and concatenated along the sample axis.

                Non-finite rows are excluded from SBI output (no NaN row
                placeholders are inserted).

        Raises
        ------
        FileNotFoundError
            If required artifacts are missing from ``result_dir``.
        TypeError
            If argument types are invalid (e.g., non-bool ``scaler``,
            non-dict ``sbi_eval_sampling_kwargs``).
        ValueError
            For invalid shapes/values (e.g., unsupported feature rank, invalid
            ``sample_shape``, or forbidden ``x`` in SBI kwargs).
        """

        scaler_path = os.path.join(result_dir, "scaler.pkl")
        if not isinstance(scaler, (bool, np.bool_)):
            raise TypeError("scaler must be a boolean: True=load scaler.pkl, False=do not scale.")
        if sbi_eval_sampling_kwargs is None:
            sbi_eval_sampling_kwargs = {}
        if not isinstance(sbi_eval_sampling_kwargs, dict):
            raise TypeError("sbi_eval_sampling_kwargs must be a dict or None.")
        if "x" in sbi_eval_sampling_kwargs:
            raise ValueError(
                "sbi_eval_sampling_kwargs must not include 'x'; "
                "it is managed internally."
            )
        if self.backend == "sbi" and "sample_shape" in sbi_eval_sampling_kwargs:
            try:
                sample_shape_pred = self._coerce_sample_shape(sbi_eval_sampling_kwargs["sample_shape"])
            except Exception as exc:
                raise ValueError(
                    "sbi_eval_sampling_kwargs['sample_shape'] must be an int, tuple/list, or torch.Size."
                ) from exc
            if sample_shape_pred.numel() <= 0:
                raise ValueError("sbi_eval_sampling_kwargs['sample_shape'] must define at least one sample.")

        def _requested_num_samples_from_kwargs(kwargs: dict) -> int:
            shape = kwargs.get("sample_shape", ())
            size = self._coerce_sample_shape(shape)
            n_samples = int(size.numel())
            if n_samples <= 0:
                raise ValueError(
                    "sbi_eval_sampling_kwargs['sample_shape'] must define at least one sample."
                )
            return n_samples

        if self.backend == "sklearn":
            model_path = os.path.join(result_dir, "model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at '{model_path}'. Train first.")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        elif self.backend == "sbi":
            posterior_path = os.path.join(result_dir, "posterior.pkl")
            if not os.path.exists(posterior_path):
                raise FileNotFoundError(f"Posterior not found at '{posterior_path}'. Train first.")
            with open(posterior_path, "rb") as f:
                posterior_obj = pickle.load(f)
        else:
            raise RuntimeError(f"Unknown backend '{self.backend}'.")

        if scaler:
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    "scaler=True but scaler.pkl not found in result_dir."
                )
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        else:
            scaler = None


        # -------- Infer expected number of features (for 1D input disambiguation) --------
        expected_n_features = None

        # 1) scaler is best (it was fit on training X)
        if scaler is not None and not isinstance(scaler, (bool, np.bool_)) and hasattr(scaler, "n_features_in_"):
            expected_n_features = int(scaler.n_features_in_)

        # 2) sklearn model(s) sometimes store n_features_in_
        if expected_n_features is None and self.backend == "sklearn":
            m0 = model[0] if isinstance(model, list) and len(model) > 0 else model
            if hasattr(m0, "n_features_in_"):
                expected_n_features = int(m0.n_features_in_)

        # 3) fallback to in-memory training data shape, if available
        if expected_n_features is None and self.features is not None:
            Xtrain = np.asarray(self.features)
            if Xtrain.ndim == 2:
                expected_n_features = int(Xtrain.shape[1])
            elif Xtrain.ndim == 1:
                expected_n_features = 1

        # -------- Normalize input shape  --------
        X = np.asarray(features)

        # Disallow ragged/object arrays (usually means inconsistent row lengths).
        # If numpy produced dtype=object, try one strict numeric conversion; if it still fails, raise.
        if X.dtype == object:
            try:
                X = np.asarray(features, dtype=float)
            except Exception as e:
                raise ValueError(
                    "features looks like a ragged / object array (inconsistent sample lengths). "
                    "Provide a rectangular array-like of shape (n_samples, n_features) or (n_features,)."
                ) from e

        if X.ndim == 0:
            # scalar -> single sample, single feature
            X = X.reshape(1, 1)

        elif X.ndim == 1:
            # Ambiguous case: could be (n_samples,) with 1 feature OR (n_features,) for one sample.
            # Use expected_n_features when we can.
            if expected_n_features == 1:
                # Treat as N samples of 1 feature: [x1, x2, ...] -> (N, 1)
                X = X.reshape(-1, 1)
            else:
                # Treat as single sample with n_features: [f1, f2, ...] -> (1, D)
                X = X.reshape(1, -1)

        elif X.ndim == 2:
            # Already (n_samples, n_features)
            pass

        else:
            raise ValueError(f"features must be scalar, 1D, or 2D; got shape {X.shape}.")

        # Prepare NaN row template (best-effort)
        theta_dim = None
        if self.theta is not None:
            Y = np.asarray(self.theta)
            if Y.ndim == 1:
                theta_dim = 1
            elif Y.ndim == 2:
                theta_dim = Y.shape[1]

        n = X.shape[0]
        if n == 0:
            if self.backend == "sbi":
                s = _requested_num_samples_from_kwargs(sbi_eval_sampling_kwargs)
                td = 0 if theta_dim is None else int(theta_dim)
                return np.empty((s, 0, td), dtype=float)
            return []

        # Identify finite rows early (used for SBI; sklearn worker checks too)
        finite_mask = np.isfinite(X).all(axis=1)

        nan_row = [np.nan] if (theta_dim in (None, 1)) else [np.nan] * theta_dim

        # ---------------- SKLEARN: multiprocessing + ensemble ----------------
        if self.backend == "sklearn":
            # IMPORTANT: do NOT scale X here; _predict_one scales inside workers.
            if n_jobs is None:
                num_cpus = os.cpu_count() or 1
            else:
                num_cpus = int(n_jobs)
                if num_cpus <= 0:
                    raise ValueError("n_jobs must be a positive integer.")

            if chunksize is None:
                chunksize_local = max(1, n // (num_cpus * 8))
            else:
                chunksize_local = int(chunksize)
                if chunksize_local <= 0:
                    raise ValueError("chunksize must be a positive integer.")

            if start_method not in {"spawn", "fork", "forkserver"}:
                raise ValueError("start_method must be one of: spawn, fork, forkserver.")

            ctx = self.multiprocessing.get_context(start_method)
            with ctx.Pool(
                    processes=num_cpus,
                    initializer=_prediction_worker_init,
                    initargs=(model, scaler),
            ) as pool:
                it = pool.imap(_predict_one, X, chunksize=chunksize_local)
                if self.tqdm_inst:
                    it = self.tqdm(it, total=n, desc="Computing predictions")
                preds = list(it)

            return [p if p is not None else nan_row for p in preds]

        # ---------------- SBI: posterior sampling  ----------------
        if self.backend != "sbi":
            raise RuntimeError(f"Unknown backend '{self.backend}'.")

        # In the SBI path, posterior.pkl stores either:
        #   - a single posterior, or
        #   - a list of posteriors (CV ensemble).
        posteriors = posterior_obj if isinstance(posterior_obj, list) else [posterior_obj]
        if len(posteriors) == 0:
            raise ValueError("Loaded SBI model ensemble is empty.")

        for i, p in enumerate(posteriors):
            if not hasattr(p, "sample"):
                raise TypeError(
                    f"SBI model at index {i} does not look like a posterior "
                    f"(missing 'sample' method): {type(p)}"
                )

        if scaler is not None and not isinstance(scaler, (bool, np.bool_)) and np.any(finite_mask):
            X2 = X.copy()
            X2[finite_mask] = scaler.transform(X[finite_mask])
            X = X2
            finite_mask = np.isfinite(X).all(axis=1)

        if not np.any(finite_mask):
            s = _requested_num_samples_from_kwargs(sbi_eval_sampling_kwargs)
            td = 0 if theta_dim is None else int(theta_dim)
            return np.empty((s, 0, td), dtype=float)

        torch = self.torch

        Xf = X[finite_mask]
        if Xf.dtype != np.float32:
            Xf = Xf.astype(np.float32, copy=False)
        Xf_t = torch.from_numpy(Xf)

        total_samples = _requested_num_samples_from_kwargs(sbi_eval_sampling_kwargs)

        # Batched posterior sampling
        with torch.no_grad():
            xb = Xf_t
            n_post = len(posteriors)

            if n_post == 1:
                samples = self._sample_posterior(
                    posteriors[0],
                    x=xb,
                    **sbi_eval_sampling_kwargs,
                )
                if samples.ndim > 3:
                    samples = samples.reshape(-1, samples.shape[-2], samples.shape[-1])
            else:
                base = total_samples // n_post
                rem = total_samples % n_post
                per_model_counts = [base + (1 if j < rem else 0) for j in range(n_post)]

                parts = []
                for posterior_obj, n_samples in zip(posteriors, per_model_counts):
                    if n_samples <= 0:
                        continue
                    kw = dict(sbi_eval_sampling_kwargs)
                    kw["sample_shape"] = (int(n_samples),)
                    sb = self._sample_posterior(
                        posterior_obj,
                        x=xb,
                        **kw,
                    )
                    if sb.ndim > 3:
                        sb = sb.reshape(-1, sb.shape[-2], sb.shape[-1])
                    parts.append(sb)

                if not parts:
                    raise RuntimeError("No posterior samples drawn from ensemble.")
                samples = torch.cat(parts, dim=0)

        if X.shape[0] == 1 and np.isfinite(X).all():
            return samples[:, 0, :].detach().cpu().numpy()

        return samples.detach().cpu().numpy()
