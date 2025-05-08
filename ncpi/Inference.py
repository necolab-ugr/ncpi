import os
import pickle
import numpy as np
import random
from ncpi import tools


class Inference:
    """
    General-purpose class for inferring parameters from simulated or observed features using either
    Bayesian inference (with SBI) or traditional regression (with sklearn).

    Attributes
    ----------
    model : list
        Name and backend library of the chosen model (e.g., ['NPE', 'sbi'] or ['RandomForestRegressor', 'sklearn']).
    hyperparams : dict
        Dictionary of hyperparameters passed to the selected model.
    features : np.ndarray
        Input feature matrix used for training or prediction.
    theta : np.ndarray
        Output parameter matrix (target values to infer).

    Methods
    -------
    __init__(model, hyperparams=None)
        Initializes the class with a model and hyperparameters.
    add_simulation_data(features, parameters)
        Adds training data (features and targets).
    initialize_sbi(hyperparams)
        Prepares an SBI inference method (only for sbi-based models).
    train(param_grid=None, n_splits=10, n_repeats=10, train_params={...})
        Trains the model using either sbi or sklearn depending on configuration.
    predict(features)
        Predicts parameters for new input features.
    sample_posterior(x, num_samples=10000)
        Samples from the posterior (only for sbi-based models).
    """


    def __init__(self, model, hyperparams=None):
        # Ensure required libraries are available
        if not tools.ensure_module("sklearn"):
            raise ImportError('sklearn is not installed.')
        self.RepeatedKFold = tools.dynamic_import("sklearn.model_selection", "RepeatedKFold")
        self.StandardScaler = tools.dynamic_import("sklearn.preprocessing", "StandardScaler")
        self.all_estimators = tools.dynamic_import("sklearn.utils", "all_estimators")
        self.RegressorMixin = tools.dynamic_import("sklearn.base", "RegressorMixin")

        if model in ['NPE', 'NLE', 'NRE']:
            if not tools.ensure_module("sbi"):
                raise ImportError('sbi is not installed.')
            if not tools.ensure_module("torch"):
                raise ImportError('torch is not installed.')

            # Dynamic imports for SBI components
            self.NPE = tools.dynamic_import("sbi.inference", "NPE")
            self.NLE = tools.dynamic_import("sbi.inference", "NLE")
            self.NRE = tools.dynamic_import("sbi.inference", "NRE")
            self.posterior_nn = tools.dynamic_import("sbi.neural_nets", "posterior_nn")
            self.likelihood_nn = tools.dynamic_import("sbi.neural_nets", "likelihood_nn")
            self.classifier_nn = tools.dynamic_import("sbi.neural_nets", "classifier_nn")
            self.BoxUniform = tools.dynamic_import("sbi.utils", "BoxUniform")
            self.torch = tools.dynamic_import("torch")
            self.model = [model, 'sbi']
        else:
            regressors = [e for e in self.all_estimators() if issubclass(e[1], self.RegressorMixin)]
            if model not in [r[0] for r in regressors]:
                raise ValueError(f"{model} not valid.")
            self.model = [model, 'sklearn']

        # Handle multiprocessing support
        if not tools.ensure_module("pathos"):
            self.pathos_inst = False
            self.multiprocessing = tools.dynamic_import("multiprocessing")
        else:
            self.pathos_inst = True
            self.pathos = tools.dynamic_import("pathos", "pools")

        # Handle tqdm progress bar support
        if not tools.ensure_module("tqdm"):
            self.tqdm_inst = False
        else:
            self.tqdm_inst = True
            self.tqdm = tools.dynamic_import("tqdm", "tqdm")

        if hyperparams is not None:
            if not isinstance(hyperparams, dict):
                raise ValueError('Hyperparameters must be a dictionary.')
            self.hyperparams = hyperparams
        else:
            self.hyperparams = {}

        self.features = []
        self.theta = []

        # Limit torch threads for performance
        if model in ['NPE', 'NLE', 'NRE']:
            torch_threads = int(os.cpu_count()/2)
            self.torch.set_num_threads(torch_threads)

    def add_simulation_data(self, features, parameters):
        """
        Adds features and parameters to the training dataset.
        """
        if type(features) is not np.ndarray:
            raise ValueError('X must be a numpy array.')
        if type(parameters) is not np.ndarray:
            raise ValueError('Y must be a numpy array.')
        if features.shape[0] != parameters.shape[0]:
            raise ValueError('Shape mismatch.')
        mask = np.all(np.isfinite(features), axis=1) & np.all(np.isfinite(parameters), axis=1)
        self.features = features[mask]
        self.theta = parameters[mask]

    def initialize_sbi(self, hyperparams):
        """
        Initializes SBI inference method (NPE, NLE, NRE) with proper estimators.
        """
        inference_type = self.model[0].lower()
        if 'density_estimator' not in hyperparams:
            raise ValueError('Missing density_estimator.')
        est = hyperparams['density_estimator']
        model = est['model']
        hidden = est['hidden_features']
        transforms = est.get('num_transforms', 5)

        if inference_type == 'npe':
            estimator_fn = self.posterior_nn(model=model, hidden_features=hidden, num_transforms=transforms)
            inference = self.NPE(prior=hyperparams['prior'], density_estimator=estimator_fn)
        elif inference_type == 'nle':
            estimator_fn = self.likelihood_nn(model=model, hidden_features=hidden, num_transforms=transforms)
            inference = self.NLE(prior=hyperparams['prior'], density_estimator=estimator_fn)
        elif inference_type == 'nre':
            estimator_fn = self.classifier_nn(model=model, hidden_features=hidden)
            inference = self.NRE(prior=hyperparams['prior'], ratio_estimator=estimator_fn)
        else:
            raise ValueError(f"Tipo {inference_type} no reconocido.")

        return inference

    def train(self, param_grid=None, n_splits=10, n_repeats=10, train_params=None):
        """
        Trains the model using SBI or sklearn.
        """
        if train_params is None:
            train_params = {'learning_rate': 0.0005, 'training_batch_size': 256}
        if self.model[1] == 'sbi':
            model = self.initialize_sbi(self.hyperparams)
        else:
            regressors = [e for e in self.all_estimators() if issubclass(e[1], self.RegressorMixin)]
            cl = str([r[1] for r in regressors if r[0] == self.model[0]][0]).split('.')[1]
            exec(f'from sklearn.{cl} import {self.model[0]}')
            model = eval(f'{self.model[0]}')(**self.hyperparams) if self.hyperparams else eval(f'{self.model[0]}')()

        if len(self.features) == 0 or len(self.theta) == 0:
            raise ValueError('No data.')

        scaler = self.StandardScaler()
        scaler.fit(self.features)
        self.features = scaler.transform(self.features)

        if self.model[1] == 'sbi':
            if self.theta.ndim == 1:
                self.theta = self.theta.reshape(-1, 1)
            model.append_simulations(
                self.torch.from_numpy(self.theta.astype(np.float32)),
                self.torch.from_numpy(self.features.astype(np.float32))
            )
            # Extract training parameters
            learning_rate = train_params.get("learning_rate", 0.0005)
            training_batch_size = train_params.get("training_batch_size", 100)
            density_estimator = model.train(learning_rate=learning_rate, training_batch_size=training_batch_size)
        else:
            model.fit(self.features, self.theta)

        if not os.path.exists('data'):
            os.makedirs('data')
        with open('data/model.pkl', 'wb') as file:
            pickle.dump(model, file)
        with open('data/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
        if self.model[1] == 'sbi':
            with open('data/density_estimator.pkl', 'wb') as file:
                pickle.dump(density_estimator, file)

    def predict(self, features):
        """
        Predicts parameters from features using the trained model.
        """
        def process_batch(batch):
            if self.model[1] == 'sbi':
                idx, feats, scaler, model, posterior = batch
            else:
                idx, feats, scaler, model = batch
            preds = []
            num_samples = self.hyperparams.get("num_samples", 5000)
            for f in feats:
                f = scaler.transform(f.reshape(1, -1))
                if np.all(np.isfinite(f)):
                    if self.model[1] == 'sklearn':
                        preds.append(model.predict(f)[0])
                    else:
                        x_o = self.torch.from_numpy(f.astype(np.float32))
                        s = posterior.sample((num_samples,), x=x_o, show_progress_bars=False)
                        preds.append(np.mean(s.numpy(), axis=0))
                else:
                    preds.append([np.nan]*self.theta.shape[1])
            return idx, preds

        if not os.path.exists('data/model.pkl'):
            raise ValueError('Model not trained.')

        with open('data/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        if self.model[1] == 'sbi':
            with open('data/density_estimator.pkl', 'rb') as f:
                density_estimator = pickle.load(f)
            posterior = model.build_posterior(density_estimator)

        if type(features) is not np.ndarray:
            raise ValueError('features must be numpy array.')
        features = np.stack(features)

        if self.model[1] == 'sbi':
            batch_args = [(0, features, scaler, model, posterior)]
            results = [process_batch(batch) for batch in batch_args]
        else:
            batch_size = len(features) // os.cpu_count()
            if batch_size == 0:
                batch_size = 1
            batches = [(i, features[i:i + batch_size]) for i in range(0, len(features), batch_size)]
            pool_class = self.pathos.ProcessPool if self.pathos_inst else self.multiprocessing.Pool
            batch_args = [(i, batch, scaler, model) for i, batch in batches]
            with pool_class(os.cpu_count()) as pool:
                imap_results = pool.imap(process_batch, batch_args)
                results = list(self.tqdm(imap_results, total=len(batches))) if self.tqdm_inst else list(imap_results)

        results.sort(key=lambda x: x[0])
        predictions = [p for _, ps in results for p in ps]
        return predictions

    def sample_posterior(self, x, num_samples=10000):
        """
        Returns samples from the posterior given an observation x.
        """
        if not os.path.exists('data/model.pkl'):
            raise ValueError('Model not trained.')

        with open('data/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('data/density_estimator.pkl', 'rb') as f:
            density_estimator = pickle.load(f)

        posterior = model.build_posterior(density_estimator)

        x = scaler.transform(x.reshape(1, -1))
        x_tensor = self.torch.from_numpy(x.astype(np.float32))
        samples = posterior.sample((num_samples,), x=x_tensor)

        return samples.numpy()
