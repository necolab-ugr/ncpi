import os
import random
import numpy as np
import inspect
import pickle
from ncpi import tools

# --- Prediction multiprocessing helpers (worker-global state) ---
_PRED_MODEL = None
_PRED_SCALER = None


def _prediction_worker_init(model, scaler):
    """Initializer: runs once per worker process (sklearn-only)."""
    global _PRED_MODEL, _PRED_SCALER
    _PRED_MODEL = model
    _PRED_SCALER = scaler


def _predict_one(x):
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


class Inference:
    """Parameter inference using sklearn regressors or SBI methods (NPE, NLE, NRE)."""

    # Supported amortized SBI models
    SBI_MODELS = ("NPE", "NLE", "NRE")

    # Class-level caches so we don't re-scan sklearn each instance
    _SKLEARN_READY = False
    _SKLEARN_REGRESSORS = None  # dict[str, type]

    def __init__(self, model: str, hyperparams: dict | None = None):
        if not isinstance(model, str):
            raise TypeError("model must be a string.")
        model = model.strip()

        if hyperparams is not None and not isinstance(hyperparams, dict):
            raise TypeError("hyperparams must be a dict or None.")
        self.hyperparams = hyperparams

        # Init sklearn stuff (and build cache once)
        self._init_sklearn_modules()
        self._ensure_sklearn_regressor_cache()

        # Decide backend using registry-style checks
        if model in self.SBI_MODELS:
            self.backend = "sbi"
        elif model in self._SKLEARN_REGRESSORS:
            self.backend = "sklearn"
        else:
            valid_sklearn = sorted(self._SKLEARN_REGRESSORS.keys())
            raise ValueError(
                f"'{model}' is not valid. Use a sklearn regressor or an SBI model from {list(self.SBI_MODELS)}.\n"
                f"Example sklearn regressors: {valid_sklearn[:10]}{' ...' if len(valid_sklearn) > 10 else ''}"
            )

        self.model_name = model

        # Import SBI only if needed
        if self.backend == "sbi":
            self._init_sbi_modules()
        else:
            self._set_sbi_attrs_to_none()

        # Training data
        self.features = None
        self.theta = None

    # -----------------------------
    # Module initialization
    # -----------------------------
    def _init_sklearn_modules(self):
        if not tools.ensure_module("sklearn", package="scikit-learn", version_spec="==1.5.0"):
            raise ImportError("scikit-learn==1.5.0 is required (import name: 'sklearn').")

        self.RepeatedKFold = tools.dynamic_import("sklearn.model_selection", "RepeatedKFold")
        self.all_estimators = tools.dynamic_import("sklearn.utils", "all_estimators")
        self.RegressorMixin = tools.dynamic_import("sklearn.base", "RegressorMixin")

        self.multiprocessing = tools.dynamic_import("multiprocessing")
        self.tqdm_inst = tools.ensure_module("tqdm")
        self.tqdm = tools.dynamic_import("tqdm", "tqdm") if self.tqdm_inst else None


    def _ensure_sklearn_regressor_cache(self):
        """Build a {name: class} mapping once per process."""
        if self.__class__._SKLEARN_READY and self.__class__._SKLEARN_REGRESSORS is not None:
            return

        reg_map = {}
        for name, cls in self.all_estimators():
            if inspect.isclass(cls) and issubclass(cls, self.RegressorMixin):
                reg_map[name] = cls

        self.__class__._SKLEARN_REGRESSORS = reg_map
        self.__class__._SKLEARN_READY = True


    def _init_sbi_modules(self):
        if not tools.ensure_module("sbi", package="sbi", version_spec="==0.24.0"):
            raise ImportError("sbi==0.24.0 is required.")
        if not tools.ensure_module("torch", package="torch", raise_on_error=False):
            raise ImportError("PyTorch ('torch') is required but not importable.")

        self.torch = tools.dynamic_import("torch")
        self.NPE = tools.dynamic_import("sbi.inference", "NPE")
        self.NLE = tools.dynamic_import("sbi.inference", "NLE")
        self.NRE = tools.dynamic_import("sbi.inference", "NRE")

        # single registry defined once per instance (small)
        self.SBI_REGISTRY = {"NPE": self.NPE, "NLE": self.NLE, "NRE": self.NRE}

        self.posterior_nn = tools.dynamic_import("sbi.neural_nets", "posterior_nn")
        self.likelihood_nn = tools.dynamic_import("sbi.neural_nets", "likelihood_nn")
        self.classifier_nn = tools.dynamic_import("sbi.neural_nets", "classifier_nn")


    def _set_sbi_attrs_to_none(self):
        self.torch = None
        self.SBI_REGISTRY = None
        self.posterior_nn = None
        self.likelihood_nn = None
        self.classifier_nn = None
        self.NPE = None
        self.NLE = None
        self.NRE = None


    # -----------------------------
    # Pickling
    # -----------------------------

    def __getstate__(self):
        """Remove non-pickleable dynamically imported modules/callables."""
        state = self.__dict__.copy()

        # Drop modules and dynamic callables (they will be re-imported)
        drop_keys = {
            "RepeatedKFold", "all_estimators", "RegressorMixin",
            "multiprocessing", "tqdm", "torch",
            "NPE", "NLE", "NRE",
            "posterior_nn", "likelihood_nn", "classifier_nn",
            "SBI_REGISTRY",
        }
        for k in drop_keys:
            state.pop(k, None)

        # Drop any imported module objects if present
        for k in list(state.keys()):
            if isinstance(state[k], type(os)):
                del state[k]

        return state


    def __setstate__(self, state):
        self.__dict__.update(state)

        # Always re-init sklearn tooling + cache
        self._init_sklearn_modules()
        self._ensure_sklearn_regressor_cache()

        # Re-init SBI only if needed
        model_name = getattr(self, "model_name", None)
        backend = getattr(self, "backend", None)

        is_sbi = (backend == "sbi") or (isinstance(model_name, str) and model_name in self.SBI_MODELS)
        if is_sbi:
            self._init_sbi_modules()
        else:
            self._set_sbi_attrs_to_none()


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
        Initialize a sbi inference trainer (NPE / NLE / NRE) using the provided hyperparameters.

        Required:
          - prior: torch.distributions.Distribution (or Distribution-like supported by sbi)

        Optional:
          - estimator: callable builder returned by sbi.neural_nets.*_nn(...) OR None
              If None, sbi will use its internal default estimator (recommended to start).
          - estimator_kwargs: dict forwarded to the default builder for the chosen method
              NPE -> posterior_nn, NLE -> likelihood_nn, NRE -> classifier_nn
          - inference_kwargs: dict forwarded to NPE/NLE/NRE constructor (e.g. device="cpu")
          - build_posterior_kwargs: dict stored on self for later build_posterior() call
        """

        if self.backend != "sbi":
            raise ValueError("initialize_sbi() is only valid when backend='sbi'.")

        if not isinstance(hyperparams, dict):
            raise TypeError("hyperparams must be a dict.")
        prior = hyperparams.get("prior", None)
        if prior is None:
            raise ValueError("Missing required key: hyperparams['prior'].")

        model_key = self.model_name  # "NPE" | "NLE" | "NRE"
        if not self.SBI_REGISTRY or model_key not in self.SBI_REGISTRY:
            raise RuntimeError(
                "SBI modules not initialized. Construct with an SBI model so _init_sbi_modules() runs."
            )

        inference_cls = self.SBI_REGISTRY[model_key]

        inference_kwargs = hyperparams.get("inference_kwargs", {}) or {}
        estimator_kwargs = hyperparams.get("estimator_kwargs", {}) or {}
        build_posterior_kwargs = hyperparams.get("build_posterior_kwargs", {}) or {}

        if not isinstance(inference_kwargs, dict):
            raise TypeError("hyperparams['inference_kwargs'] must be a dict if provided.")
        if not isinstance(estimator_kwargs, dict):
            raise TypeError("hyperparams['estimator_kwargs'] must be a dict if provided.")
        if not isinstance(build_posterior_kwargs, dict):
            raise TypeError("hyperparams['build_posterior_kwargs'] must be a dict if provided.")

        # Either user provides a ready-to-use estimator builder (flexible interface),
        # or we build one using the canonical sbi builders.
        estimator = hyperparams.get("estimator", None)
        if estimator is not None and not callable(estimator):
            raise TypeError("hyperparams['estimator'] must be callable (or None).")

        # If user didn’t provide estimator or estimator_kwargs, choose defaults.
        if estimator is None:
            if model_key in ("NPE", "NLE"):
                estimator = "maf"
            elif model_key == "NRE":
                estimator = "resnet"
            else:
                raise ValueError(f"Unsupported SBI model '{model_key}'.")

        # Instantiate the trainer with the correct keyword per method.
        if model_key in ("NPE", "NLE"):
            inference = inference_cls(prior=prior, density_estimator=estimator, **inference_kwargs)
        elif model_key == "NRE":
            inference = inference_cls(prior=prior, classifier=estimator, **inference_kwargs)
        else:
            raise ValueError(f"Unsupported SBI model '{model_key}'.")

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
            sbi_eval_num_posterior_samples: int = 2000,
            sbi_eval_batch_size: int = 256,
    ):
        """
        Train and (optionally) hyperparameter-search.

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
            Directory where to save model.pkl (and scaler.pkl / density_estimator.pkl if applicable).
        scaler : fitted transformer or None
            If provided, used to scale features before training.
        seed : int
            Random seed for reproducibility.
        sbi_eval_num_posterior_samples : int
            Number of posterior samples to draw per observation when evaluating SBI models during CV.
        sbi_eval_batch_size : int
            Batch size when evaluating SBI models during CV.
        Returns
        -------
        model
            The trained model(s):
              - sklearn: single model or list of fold models if param_grid was used
              - sbi: single inference object or list of fold inference objects if param_grid was used
        """

        train_params = train_params or {}
        if not isinstance(train_params, dict):
            raise TypeError("train_params must be a dict or None.")

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

        finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        if not np.any(finite_mask):
            raise ValueError("All rows contain NaN/Inf in features or parameters; nothing to train on.")
        X = X[finite_mask]
        Y = Y[finite_mask]

        # --------- Scale (do not mutate self.features) ----------
        fitted_scaler = None
        if scaler is not None:
            fitted_scaler = scaler
            fitted_scaler.fit(X)
            X = fitted_scaler.transform(X)

        # CV splitter used only for param_grid
        rkf = self.RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        total_folds = n_splits * n_repeats

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

                    for fold_i, (tr, te) in enumerate(rkf.split(X)):
                        print(f"  Fold {fold_i+1}/{total_folds}")
                        repeat_id = fold_i // n_splits
                        repeat_seed = seed + repeat_id

                        # fresh estimator each fold
                        m = sklearn_clone(base_model)

                        # fold config (never mutate the source dict)
                        p = dict(params)
                        if "random_state" in m.get_params():
                            p["random_state"] = repeat_seed

                        m.set_params(**p)

                        y_tr = Y[tr]
                        if y_tr.ndim == 2 and y_tr.shape[1] == 1:
                            y_tr = y_tr.ravel()
                        m.fit(X[tr], y_tr)

                        pred = np.asarray(m.predict(X[te]))

                        y_te = Y[te]
                        if y_te.ndim == 2 and y_te.shape[1] == 1:
                            y_te = y_te.ravel()
                        if np.asarray(pred).ndim > 1 and y_te.ndim == 1:
                            pred = np.asarray(pred).ravel()

                        mse = float(np.mean((pred - y_te) ** 2))
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
                density_estimator = None

            else:
                print("Training single sklearn model on full data...")
                # single model on full data
                model = sklearn_clone(base_model)
                if "random_state" in model.get_params():
                    model.set_params(random_state=seed)

                y_all = Y
                if y_all.ndim == 2 and y_all.shape[1] == 1:
                    y_all = y_all.ravel()

                model.fit(X, y_all)
                density_estimator = None

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
                best_fold_pairs = None  # list[(inference, density_estimator)]

                for params in param_grid:
                    print(f"Evaluating params: {params}")
                    cfg = dict(base_cfg)
                    cfg.update(params)

                    fold_pairs = []
                    fold_scores = []

                    for fold_i, (tr, te) in enumerate(rkf.split(X)):
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
                        n_te = te_idx.shape[0]
                        for i in range(0, n_te, sbi_eval_batch_size):
                            idx = te_idx[i:i + sbi_eval_batch_size]
                            xb = X_t[idx]
                            yb = Y_t[idx]

                            samples = posterior.sample(
                                (sbi_eval_num_posterior_samples,),
                                x=xb,
                                show_progress_bars=False,
                            )  # [S, B, theta_dim]
                            mean = samples.mean(dim=0)  # [B, theta_dim]
                            mse = torch.mean((mean - yb) ** 2).item()
                            total += mse * idx.shape[0]

                        fold_mse = total / n_te
                        fold_scores.append(fold_mse)
                        fold_pairs.append((inf, de))

                    mean_mse = float(np.mean(fold_scores))
                    if mean_mse < best_score:
                        best_score = mean_mse
                        best_cfg_delta = dict(params)
                        best_fold_pairs = fold_pairs

                if best_fold_pairs is None:
                    raise ValueError("No best hyperparameters found.")
                else:
                    print(f"Best params: {best_cfg_delta} | mean CV MSE: {best_score:.6f}")

                model = [inf for (inf, _) in best_fold_pairs]
                density_estimator = [de for (_, de) in best_fold_pairs]

            else:
                # single SBI model on full data
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                inf = self.initialize_sbi(base_cfg)
                inf.append_simulations(Y_t, X_t)
                de = inf.train(**train_params)

                model = inf
                density_estimator = de

        else:
            raise RuntimeError(f"Unknown backend '{self.backend}'.")

        # --------- Save artifacts ----------
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved at '{result_dir}/model.pkl'")

        if fitted_scaler is not None:
            with open(os.path.join(result_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(fitted_scaler, f)
            print(f"Scaler saved at '{result_dir}/scaler.pkl'")

        if self.backend == "sbi":
            with open(os.path.join(result_dir, "density_estimator.pkl"), "wb") as f:
                pickle.dump(density_estimator, f)
            print(f"Density estimator saved at '{result_dir}/density_estimator.pkl'")

        return model


    def predict(
            self,
            features,
            *,
            result_dir: str = "data",
            scaler=None,
            # SBI knobs
            num_posterior_samples: int | None = None,
            sbi_batch_size: int = 256,
    ):
        """
        Predict parameters.

        Ensemble philosophy preserved:
          - if loaded model.pkl is a list -> average predictions across all members
          - if loaded model.pkl is a single model -> return its prediction

        Returns a Python list containing:
          - sklearn: per-row list[float] for multi-output or float for scalar output
          - sbi: per-row np.ndarray (theta_dim,)
          - invalid rows -> NaN row (list of NaNs or np.nan)
        """

        model_path = os.path.join(result_dir, "model.pkl")
        scaler_path = os.path.join(result_dir, "scaler.pkl")
        density_estimator_path = os.path.join(result_dir, "density_estimator.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at '{model_path}'. Train first.")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        if scaler is None and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        # -------- Infer expected number of features (for 1D input disambiguation) --------
        expected_n_features = None

        # 1) scaler is best (it was fit on training X)
        if scaler is not None and hasattr(scaler, "n_features_in_"):
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

        # -------- Normalize input shape (this is what changes vs your current version) --------
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

        n = X.shape[0]
        if n == 0:
            return []

        # Identify finite rows early (used for SBI; sklearn worker checks too)
        finite_mask = np.isfinite(X).all(axis=1)

        # Prepare NaN row template (best-effort)
        theta_dim = None
        if self.theta is not None:
            Y = np.asarray(self.theta)
            if Y.ndim == 1:
                theta_dim = 1
            elif Y.ndim == 2:
                theta_dim = Y.shape[1]

        nan_row = [np.nan] if (theta_dim in (None, 1)) else [np.nan] * theta_dim

        # ---------------- SKLEARN: multiprocessing + ensemble ----------------
        if self.backend == "sklearn":
            # IMPORTANT: do NOT scale X here; _predict_one scales inside workers.
            num_cpus = os.cpu_count() or 1
            chunksize = max(1, n // (num_cpus * 8))

            ctx = self.multiprocessing.get_context("spawn")
            with ctx.Pool(
                    processes=num_cpus,
                    initializer=_prediction_worker_init,
                    initargs=(model, scaler),
            ) as pool:
                it = pool.imap(_predict_one, X, chunksize=chunksize)
                if self.tqdm_inst:
                    it = self.tqdm(it, total=n, desc="Computing predictions")
                preds = list(it)

            return [p if p is not None else nan_row for p in preds]

        # ---------------- SBI ----------------
        if self.backend != "sbi":
            raise RuntimeError(f"Unknown backend '{self.backend}'.")

        if not os.path.exists(density_estimator_path):
            raise FileNotFoundError(f"density_estimator.pkl not found at '{density_estimator_path}'.")

        with open(density_estimator_path, "rb") as f:
            density_estimator = pickle.load(f)

        # Apply scaler for SBI here (no workers)
        if scaler is not None and np.any(finite_mask):
            X2 = X.copy()
            X2[finite_mask] = scaler.transform(X[finite_mask])
            X = X2

        # Re-check finiteness after scaling
        finite_mask = np.isfinite(X).all(axis=1)

        out = [nan_row for _ in range(n)]
        if not np.any(finite_mask):
            return out

        # Choose samples count
        if num_posterior_samples is None:
            if isinstance(self.hyperparams, dict):
                num_posterior_samples = int(self.hyperparams.get("num_samples", 5000))
            else:
                num_posterior_samples = 5000
        if num_posterior_samples <= 0:
            raise ValueError("num_posterior_samples must be > 0.")

        torch = self.torch
        build_kwargs = getattr(self, "_sbi_build_posterior_kwargs", {}) or {}

        # Build posterior(s)
        if isinstance(model, list):
            if not isinstance(density_estimator, list) or len(density_estimator) != len(model):
                raise ValueError("Ensemble SBI requires density_estimator to be a list matching model length.")
            posteriors = [model[i].build_posterior(density_estimator[i], **build_kwargs) for i in range(len(model))]
        else:
            posteriors = model.build_posterior(density_estimator, **build_kwargs)

        Xf = X[finite_mask]
        Xf_t = torch.from_numpy(Xf.astype(np.float32))

        def posterior_mean_single(posterior_obj, x_single_t):
            # x_single_t: shape [1, D]
            s = posterior_obj.sample(
                (num_posterior_samples,),
                x=x_single_t,
                show_progress_bars=False
            )  # [S, 1, theta_dim] or [S, theta_dim] depending on posterior
            s = s.reshape(s.shape[0], -1) if s.ndim == 2 else s[:, 0, :]  # -> [S, theta_dim]
            return s.mean(dim=0)  # [theta_dim]

        means = []
        for i in range(Xf_t.shape[0]):
            xi = Xf_t[i:i + 1]  # [1, D]
            if isinstance(posteriors, list):
                mm = [posterior_mean_single(p, xi) for p in posteriors]
                mean_i = torch.stack(mm, dim=0).mean(dim=0)
            else:
                mean_i = posterior_mean_single(posteriors, xi)
            means.append(mean_i.detach().cpu().numpy())

        means = np.stack(means, axis=0)  # [n_finite, theta_dim]

        finite_idx = np.where(finite_mask)[0]
        for k, idx in enumerate(finite_idx):
            out[idx] = means[k]

        return out


    def sample_posterior(
            self,
            x,
            *,
            num_samples: int = 10_000,
            result_dir: str = "data",
            scaler=None,
            cache_posterior: bool = False,
    ):
        """
        Sample from the posterior for a single observation x (SBI only).

        Parameters
        ----------
        x : array-like
            Shape (n_features,) or (1, n_features)
        num_samples : int
            Number of posterior samples.
        result_dir : str
            Where model.pkl / density_estimator.pkl are stored.
        scaler : fitted transformer or None
            If None, will load scaler.pkl if present.
        cache_posterior : bool
            If True, pickles posterior.pkl (optional; off by default to avoid side effects).

        Returns
        -------
        np.ndarray
            If single posterior: shape (num_samples, theta_dim)
            If ensemble posterior list: shape (num_samples * n_members, theta_dim) stacked
        """
        if self.backend != "sbi":
            raise ValueError("sample_posterior() is only valid for backend='sbi'.")

        if num_samples <= 0:
            raise ValueError("num_samples must be > 0.")

        model_path = os.path.join(result_dir, "model.pkl")
        scaler_path = os.path.join(result_dir, "scaler.pkl")
        density_estimator_path = os.path.join(result_dir, "density_estimator.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at '{model_path}'. Train first.")
        if not os.path.exists(density_estimator_path):
            raise FileNotFoundError(f"Density estimator not found at '{density_estimator_path}'. Train first.")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(density_estimator_path, "rb") as f:
            density_estimator = pickle.load(f)

        if scaler is None and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        # Normalize x to (1, n_features)
        x_np = np.asarray(x)
        if x_np.ndim == 0:
            x_np = x_np.reshape(1, 1)
        elif x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        elif x_np.ndim != 2 or x_np.shape[0] != 1:
            raise ValueError(f"x must be shape (n_features,) or (1, n_features), got {x_np.shape}.")

        if not np.isfinite(x_np).all():
            raise ValueError("x contains NaN/Inf; cannot sample posterior.")

        if scaler is not None:
            x_np = scaler.transform(x_np)

        torch = self.torch
        x_t = torch.from_numpy(x_np.astype(np.float32))

        build_kwargs = getattr(self, "_sbi_build_posterior_kwargs", {}) or {}

        # Build posterior(s)
        if isinstance(model, list):
            if not isinstance(density_estimator, list) or len(density_estimator) != len(model):
                raise ValueError("If model is a list, density_estimator must be a list of same length.")
            posterior = [model[i].build_posterior(density_estimator[i], **build_kwargs) for i in range(len(model))]
        else:
            posterior = model.build_posterior(density_estimator, **build_kwargs)

        if cache_posterior:
            with open(os.path.join(result_dir, "posterior.pkl"), "wb") as f:
                pickle.dump(posterior, f)
            print(f"Posterior saved at '{result_dir}/posterior.pkl'")

        # Draw samples
        if isinstance(posterior, list):
            # Stack samples from ensemble members
            samples = [p.sample((num_samples,), x=x_t, show_progress_bars=False).detach().cpu().numpy()
                       for p in posterior]
            return np.vstack(samples)
        else:
            samples = posterior.sample((num_samples,), x=x_t, show_progress_bars=False)
            return samples.detach().cpu().numpy()