import os
import pickle
import numpy as np
import random
from ncpi import tools


class Inference:
    """
    General-purpose class for inferring parameters from simulated or observed features using either
    Bayesian inference (with SBI) or regression (with sklearn).

    Attributes
    ----------
    model : list
        Name and backend library of the chosen model (e.g., ['NPE', 'sbi'] or ['RandomForestRegressor', 'sklearn']).
    hyperparams : dict
        Dictionary of hyperparameters passed to the selected model.
    features : np.ndarray
        Input feature array used for training.
    theta : np.ndarray
        Parameter array (target values to infer).

    Methods
    -------
    __init__(model, hyperparams=None)
        Initializes the class with a model and hyperparameters.
    add_simulation_data(features, parameters)
        Adds training data (features and target parameters).
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
        """
        Initializes the Inference class with the specified model and hyperparameters.

        Parameters
        ----------
        model : str
            Name of the machine-learning model to use. It can be any of the regression models from sklearn or NPE, NLE,
            NRE from SBI.
        hyperparams : dict, optional
            Dictionary of hyperparameters of the model. The default is None.
        """

        # Ensure that sklearn is installed
        if not tools.ensure_module("sklearn"):
            raise ImportError('sklearn is not installed. Please install it to use the Inference class.')
        self.RepeatedKFold = tools.dynamic_import("sklearn.model_selection", "RepeatedKFold")
        self.StandardScaler = tools.dynamic_import("sklearn.preprocessing", "StandardScaler")
        self.all_estimators = tools.dynamic_import("sklearn.utils", "all_estimators")
        self.RegressorMixin = tools.dynamic_import("sklearn.base", "RegressorMixin")

        # Ensure that sbi and torch are installed
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

        # Check if pathos is installed. If not, use the default Python multiprocessing library
        if not tools.ensure_module("pathos"):
            self.pathos_inst = False
            self.multiprocessing = tools.dynamic_import("multiprocessing")
        else:
            self.pathos_inst = True
            self.pathos = tools.dynamic_import("pathos", "pools")

        # Check if tqdm is installed
        if not tools.ensure_module("tqdm"):
            self.tqdm_inst = False
        else:
            self.tqdm_inst = True
            self.tqdm = tools.dynamic_import("tqdm", "tqdm")

        # Assert that model is a string
        if type(model) is not str:
            raise ValueError('Model must be a string.')

        # Check if model is in the list of regression models from sklearn, or it is one of the SBI methods
        regressors = [estimator for estimator in self.all_estimators() if issubclass(estimator[1], self.RegressorMixin)]
        sbi_models = ['NPE', 'NLE', 'NRE']
        if model not in [regressor[0] for regressor in regressors] + sbi_models:
            raise ValueError(f'{model} not in the list of machine-learning models from sklearn or SBI (NPE, NLE, NRE).')

        # Set model and library
        self.model = [model, 'sbi'] if model in sbi_models else [model, 'sklearn']

        # Check if hyperparameters is a dictionary
        if hyperparams is not None:
            if type(hyperparams) is not dict:
                raise ValueError('Hyperparameters must be a dictionary.')
            # Set hyperparameters
            self.hyperparams = hyperparams
        else:
            self.hyperparams = None

        # Initialize features and parameters
        self.features = []
        self.theta = []

        # Set the number of threads used by PyTorch for SBI models
        if model in ['NPE', 'NLE', 'NRE']:
            torch_threads = int(os.cpu_count()/2)
            self.torch.set_num_threads(torch_threads)


    def add_simulation_data(self, features, parameters):
        """
        Method to add features and parameters to the training data.

        Parameters
        ----------
        features : np.ndarray
            Features.
        parameters : np.ndarray
            Parameters to infer.
        """

        # Assert that features and parameters are numpy arrays
        if type(features) is not np.ndarray:
            raise ValueError('X must be a numpy array.')
        if type(parameters) is not np.ndarray:
            raise ValueError('Y must be a numpy array.')

        # Assert that features and parameters have the same number of rows
        if features.shape[0] != parameters.shape[0]:
            raise ValueError('Features and parameters must have the same number of rows.')

        # Create a mask to identify rows without NaN or Inf values
        if features.ndim == 1 and parameters.ndim == 1:
            mask = np.isfinite(features) & np.isfinite(parameters)
        elif features.ndim == 1:
            mask = np.isfinite(features) & np.all(np.isfinite(parameters), axis=1)
        elif parameters.ndim == 1:
            mask = np.all(np.isfinite(features), axis=1) & np.isfinite(parameters)
        else:
            mask = np.all(np.isfinite(features), axis=1) & np.all(np.isfinite(parameters), axis=1)

        # Apply the mask to filter out rows with NaN or Inf values
        features = features[mask]
        parameters = parameters[mask]

        # Stack features and parameters
        features = np.stack(features)
        parameters = np.stack(parameters)

        # Reshape features if your data has a single feature
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        # Add features and parameters to training data
        self.features = features
        self.theta = parameters

    def initialize_sbi(self, hyperparams):
        """
        Initializes the SBI inference method (NPE, NLE, or NRE) using the appropriate neural estimator.

        Parameters
        ----------
        hyperparams : dict
            Dictionary of hyperparameters required to set up the inference method.
            Must include:
                - 'prior': the prior distribution over parameters (BoxUniform or similar)
                - 'density_estimator': a dictionary containing:
                    - 'model': the neural network type (e.g., 'nsf', 'maf', etc.)
                    - 'hidden_features': number of hidden units per layer
                    - 'num_transforms': number of normalizing flow transformations (only used for NPE and NLE)

        Returns
        -------
        inference : NPE, NLE, or NRE object
            A configured SBI inference object ready for appending simulations and training.
        """
        inference_type = self.model[0].lower()

        if 'density_estimator' not in hyperparams:
            raise ValueError('Missing density_estimator.')
        if 'hidden_features' not in hyperparams['density_estimator']:
            raise ValueError('Missing hidden_features.')
        if 'num_transforms' not in hyperparams['density_estimator'] and inference_type in ['npe', 'nle']:
            raise ValueError('Missing num_transforms.')
        if 'model' not in hyperparams['density_estimator']:
            raise ValueError('Missing model.')
        if 'prior' not in hyperparams:
            raise ValueError('Missing prior.')

        est = hyperparams['density_estimator']
        model = est['model']
        hidden = est['hidden_features']
        if inference_type in ['npe', 'nle']:
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
            raise ValueError(f"'{self.model[0]} is not a valid SBI model. Choose from NPE, NLE, or NRE.")

        return inference



    def train(self, param_grid=None, n_splits=10, n_repeats=10, train_params={'learning_rate': 0.0005,
                                                                              'training_batch_size': 256}):
        """
        Method to train the model.

        Parameters
        ----------
        param_grid : list of dictionaries, optional
            List of dictionaries of hyperparameters to search over. The default
            is None (no hyperparameter search).
        n_splits : int, optional
            Number of splits for RepeatedKFold cross-validation. The default is 10.
        n_repeats : int, optional
            Number of repeats for RepeatedKFold cross-validation. The default is 10.
        train_params : dict, optional
            Dictionary of training parameters for SBI.
        """

        # Import the sklearn model
        if self.model[1] == 'sklearn':
            regressors = [estimator for estimator in self.all_estimators() if issubclass(estimator[1], self.RegressorMixin)]
            pos = np.where(np.array([regressor[0] for regressor in regressors]) == self.model[0])[0][0]
            cl = str(regressors[pos][1]).split('.')[1]
            exec(f'from sklearn.{cl} import {self.model[0]}')

        # Initialize model with default hyperparameters
        if self.hyperparams is None:
            if self.model[1] == 'sklearn':
                model = eval(f'{self.model[0]}')()
            elif self.model[1] == 'sbi':
                model = self.initialize_sbi({'prior': None, 'density_estimator':  {'model': "maf", 'hidden_features': 10,
                                                                                    'num_transforms': 2}})

        # Initialize model with user-defined hyperparameters
        else:
            if self.model[1] == 'sklearn':
                model = eval(f'{self.model[0]}')(**self.hyperparams)
            elif self.model[1] == 'sbi':
                model = self.initialize_sbi(self.hyperparams)

        # Check if features and parameters are not empty
        if len(self.features) == 0:
            raise ValueError('No features provided.')
        if len(self.theta) == 0:
            raise ValueError('No parameters provided.')

        # Initialize the StandardScaler
        scaler = self.StandardScaler()

        # Fit the StandardScaler
        scaler.fit(self.features)

        # Transform the features
        self.features = scaler.transform(self.features)

        # Remove Nan and Inf values from features
        if self.features.ndim == 1:
            mask = np.isfinite(self.features)
        else:
            mask = np.all(np.isfinite(self.features), axis=1)
        self.features = self.features[mask]
        self.theta = self.theta[mask]

        # Search for the best hyperparameters using RepeatedKFold cross-validation and grid search if param_grid is
        # provided
        if param_grid is not None:
            # Assert that param_grid is a list
            if type(param_grid) is not list:
                raise ValueError('param_grid must be a list.')

            # Loop over each set of hyperparameters
            best_score = np.inf
            best_config = None
            best_fits = None
            for params in param_grid:
                print(f'\n\n--> Hyperparameters: {params}')

                # Initialize RepeatedKFold (added random_state for reproducibility)
                rkf = self.RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

                # Loop over each repeat and fold
                mean_scores = []
                fits = []
                for repeat_idx, (train_index, test_index) in enumerate(rkf.split(self.features)):
                    # Print info of repeat and fold
                    print('\n') if self.model[1] == 'sbi' else None
                    print(f'\rRepeat {repeat_idx // n_splits + 1}, Fold {repeat_idx % n_splits + 1}', end='', flush=True)
                    print('\n') if self.model[1] == 'sbi' else None

                    # Split the data
                    X_train, X_test = self.features[train_index], self.features[test_index]
                    Y_train, Y_test = self.theta[train_index], self.theta[test_index]

                    if self.model[1] == 'sklearn':
                        # Set the random state for reproducibility
                        params['random_state'] = repeat_idx // n_splits
                        # Update parameters
                        model.set_params(**params)

                        # Fit the model
                        model.fit(X_train, Y_train)
                        fits.append(model)

                        # Predict the parameters
                        Y_pred = model.predict(X_test)

                        # Compute the mean squared error
                        mse = np.mean((Y_pred - Y_test) ** 2)

                        # Append the mean squared error
                        mean_scores.append(mse)

                    if self.model[1] == 'sbi':
                        # Set the seeds for reproducibility
                        self.torch.manual_seed(repeat_idx)
                        random.seed(repeat_idx)

                        # Re-initialize the SBI object with the new configuration
                        model = self.initialize_sbi(params)

                        # Ensure theta is a 2D array
                        if Y_train.ndim == 1:
                            Y_train = Y_train.reshape(-1, 1)

                        # Append simulations
                        model.append_simulations(
                            self.torch.from_numpy(Y_train.astype(np.float32)),
                            self.torch.from_numpy(X_train.astype(np.float32))
                        )

                        # Train the neural density estimator
                        density_estimator = model.train(**train_params)
                        fits.append([model, density_estimator])

                        # Build the posterior
                        posterior = model.build_posterior(density_estimator)

                        # Loop over all test samples
                        for i in range(len(X_test)):
                            # Sample the posterior
                            x_o = self.torch.from_numpy(np.array(X_test[i], dtype=np.float32).reshape(1, -1))
                            posterior_samples = posterior.sample((5000,), x=x_o, show_progress_bars=False)
                            pred = np.mean(posterior_samples.numpy(), axis=0)
                            # Compute the mean squared error
                            mse = np.mean((pred[0] - Y_test[i]) ** 2)
                            # Append the mean squared error
                            mean_scores.append(mse)

                # Compute the mean of the mean squared errors
                if np.mean(mean_scores) < best_score:
                    best_score = np.mean(mean_scores)
                    best_config = params
                    best_fits = fits

            # Update the model with the best hyperparameters
            if best_config is not None:
                if self.model[1] == 'sklearn':
                    model = best_fits
                if self.model[1] == 'sbi':
                    model = [best_fits[i][0] for i in range(len(best_fits))]
                    density_estimator = [best_fits[i][1] for i in range(len(best_fits))]
                print(f'\n\n--> Best hyperparameters: {best_config}\n')
            else:
                raise ValueError('\nNo best hyperparameters found.\n')

        # Fit the model using all the data
        else:
            if self.model[1] == 'sklearn':
                model.fit(self.features, self.theta)

            if self.model[1] == 'sbi':
                # Ensure theta is a 2D array
                if self.theta.ndim == 1:
                    self.theta = self.theta.reshape(-1, 1)

                # Append simulations
                model.append_simulations(
                    self.torch.from_numpy(self.theta.astype(np.float32)),
                    self.torch.from_numpy(self.features.astype(np.float32))
                )

                # Extract training parameters
                learning_rate = train_params.get("learning_rate", 0.0005)
                training_batch_size = train_params.get("training_batch_size", 256)

                # Train the neural density estimator
                density_estimator = model.train(learning_rate=learning_rate, training_batch_size=training_batch_size)

        # Save the best model and the StandardScaler
        if not os.path.exists('data'):
            os.makedirs('data')
        with open('data/model.pkl', 'wb') as file:
            pickle.dump(model, file)
        with open('data/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        # Save also the density estimator if the model is SBI
        if self.model[1] == 'sbi':
            with open('data/density_estimator.pkl', 'wb') as file:
                pickle.dump(density_estimator, file)

    def predict(self, features):
        """
        Method to predict the parameters.

        Parameters
        ----------
        features : np.ndarray
            Features.

        Returns
        -------
        predictions : list
            List of predictions.
        """

        def process_batch(batch):
            """
            Function to compute predictions from a batch of features.

            Parameters
            ----------
            batch: tuple
                Tuple containing the batch of features, the StandardScaler and the model (and the posterior if the model
                is SBI).

            Returns
            -------
            predictions: list
                List of predictions
            """
            if self.model[1] == 'sbi':
                batch_index, feat_batch, scaler, model, posterior = batch
            else:
                batch_index, feat_batch, scaler, model = batch
            predictions = []
            for feat in feat_batch:
                # Transform the features
                feat = scaler.transform(feat.reshape(1, -1))

                # Check that feat has no NaN or Inf values
                if np.all(np.isfinite(feat)):
                    # Predict the parameters
                    if self.model[1] == 'sklearn':
                        if type(model) is list:
                            pred = np.mean([m.predict(feat) for m in model], axis=0)
                        else:
                            pred = model.predict(feat)
                        predictions.append(pred[0])

                    if self.model[1] == 'sbi':
                        # Sample the posterior
                        x_o = self.torch.from_numpy(np.array(feat, dtype=np.float32))
                        if self.hyperparams is not None:
                            num_samples = self.hyperparams.get("num_samples", 5000)
                        else:
                            num_samples = 5000

                        if type(posterior) is list:
                            posterior_samples = [post.sample((num_samples,), x=x_o, show_progress_bars=False) for post in posterior]
                            # Compute the mean of the posterior samples
                            pred = np.mean([np.mean(post.numpy(), axis=0) for post in posterior_samples], axis=0)
                        else:
                            posterior_samples = posterior.sample((num_samples,), x=x_o, show_progress_bars=False)
                            # Compute the mean of the posterior samples
                            pred = np.mean(posterior_samples.numpy(), axis=0)
                        predictions.append(pred)

                else:
                    predictions.append([np.nan for _ in range(self.theta.shape[1])])

            # Return the predictions
            return batch_index, predictions

        # Assert that the model has been trained
        if not os.path.exists('data/model.pkl'):
            raise ValueError('Model has not been trained.')

        # Load the best model and the StandardScaler
        with open('data/model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('data/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        if self.model[1] == 'sbi':
            with open('data/density_estimator.pkl', 'rb') as file:
                density_estimator = pickle.load(file)
                # Build the posterior
                if type(density_estimator) is list:
                    posterior = [model[i].build_posterior(density_estimator[i]) for i in range(len(density_estimator))]
                else:
                    posterior = model.build_posterior(density_estimator)

        # Assert that features is a numpy array
        if type(features) is not np.ndarray:
            raise ValueError('features must be a numpy array.')

        # Stack features
        features = np.stack(features)

        # Split the data into batches using the number of available CPUs
        num_cpus = os.cpu_count()
        if self.model[1] == 'sbi':
            batch_size = len(features)  # to avoid memory issues
        else:
            batch_size = len(features) // num_cpus
        if batch_size == 0:
            batch_size = 1
        batches = [(i, features[i:i + batch_size]) for i in range(0, len(features), batch_size)]

        # Choose the appropriate parallel processing library
        pool_class = self.pathos.ProcessPool if self.pathos_inst else self.multiprocessing.Pool

        # Prepare batch arguments based on model type
        use_posterior = self.model[1] == 'sbi'
        batch_args = [(ii, batch, scaler, model, posterior) if use_posterior else (ii, batch, scaler, model) for
                      ii, batch in batches]

        # Compute predictions in parallel if model is not SBI
        if self.model[1] == 'sbi':
            results = [process_batch(batch_arg) for batch_arg in batch_args]
        else:
            with pool_class(num_cpus) as pool:
                imap_results = pool.imap(process_batch, batch_args)
                results = list(
                    self.tqdm(imap_results, total=len(batches), desc="Computing predictions")) if self.tqdm_inst else list(
                    imap_results)

        # Sort the predictions based on the original index
        results.sort(key=lambda x: x[0])
        predictions = [pred for _, batch_preds in results for pred in batch_preds]

        return predictions

    def sample_posterior(self, x, num_samples=10000):
        """
        Sample from the posterior distribution for a given observation.

        Parameters
        ----------
        x : np.ndarray
            Observed feature vector (1D array).
        num_samples : int, optional
            Number of posterior samples to draw. Default is 10000.

        Returns
        -------
        np.ndarray
            Array of posterior samples.
        """
        if not os.path.exists('data/model.pkl'):
            raise ValueError('Model not trained.')

        with open('data/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('data/density_estimator.pkl', 'rb') as f:
            density_estimator = pickle.load(f)

        if type(density_estimator) is list:
            posterior = [model[i].build_posterior(density_estimator[i]) for i in range(len(density_estimator))]
        else:
            posterior = model.build_posterior(density_estimator)

        x = scaler.transform(x.reshape(1, -1))
        x_tensor = self.torch.from_numpy(x.astype(np.float32))

        if isinstance(posterior, list):
            samples = [p.sample((num_samples,), x=x_tensor).numpy() for p in posterior]
            return np.vstack(samples)
        else:
            samples = posterior.sample((num_samples,), x=x_tensor)
            return samples.numpy()