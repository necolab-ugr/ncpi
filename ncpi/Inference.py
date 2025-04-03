import os
import pickle
import numpy as np
import random
from ncpi import tools


class Inference:
    """
    Class for inferring cortical circuit parameters from features of field potential recordings.

    Attributes
    ----------
    model : list
        List of the model name and the library where it is implemented.
    hyperparams : dict
        Dictionary of hyperparameters of the model.
    features : np.ndarray
        Features.
    theta : np.ndarray
        Parameters to infer.

    Methods
    -------
    __init__(model, hyperparams=None)
        Initializes the Inference class with the specified model and hyperparameters.
    add_training_data(features, parameters)
        Adds features and parameters to the training data.
    initialize_training_data()
        Initializes the training data.
    train(param_grid=None, n_splits=10, n_repeats=10)
        Trains the model using the provided training data.
    predict(features)
        Predicts the parameters for the given features.
    """

    def __init__(self, model, hyperparams=None):
        """
        Initializes the Inference class with the specified model and hyperparameters.

        Parameters
        ----------
        model : str
            Name of the machine-learning model to use. It can be any of the regression models from sklearn or 'SNPE' from sbi.
        hyperparams : dict, optional
            Dictionary of hyperparameters of the model. The default is None.

        Raises
        ------
        ValueError
            If model is not a string.
            If model is not in the list of regression models from sklearn or 'SNPE'.
            If hyperparameters is not a dictionary.
        """

        # Ensure that sklearn is installed
        if not tools.ensure_module("sklearn"):
            raise ImportError('sklearn is not installed. Please install it to use the Inference class.')
        self.RepeatedKFold = tools.dynamic_import("sklearn.model_selection", "RepeatedKFold")
        self.StandardScaler = tools.dynamic_import("sklearn.preprocessing", "StandardScaler")
        self.all_estimators = tools.dynamic_import("sklearn.utils", "all_estimators")
        self.RegressorMixin = tools.dynamic_import("sklearn.base", "RegressorMixin")

        # Ensure that sbi and torch are installed
        if model == 'SNPE':
            if not tools.ensure_module("sbi"):
                raise ImportError('sbi is not installed. Please install it to use SNPE.')
            self.SNPE = tools.dynamic_import("sbi.inference", "SNPE")
            self.posterior_nn = tools.dynamic_import("sbi.utils", "posterior_nn")

            if not tools.ensure_module("torch"):
                raise ImportError('torch is not installed. Please install it to use SNPE.')
            self.torch = tools.dynamic_import("torch")

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

        # Check if model is in the list of regression models from sklearn, or it is SNPE
        regressors = [estimator for estimator in self.all_estimators() if issubclass(estimator[1], self.RegressorMixin)]
        if model not in [regressor[0] for regressor in regressors] + ['SNPE']:
            raise ValueError(f'{model} not in the list of machine-learning models from sklearn or sbi libraries that '
                             f'can be used for inference.')

        # Set model and library
        self.model = [model, 'sbi'] if model == 'SNPE' else [model, 'sklearn']

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

        # Set the number of threads used by PyTorch
        if model == 'SNPE':
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
        Method to initialize the SNPE model with the specified hyperparameters. The hyperparameters must contain, at
        least, the density_estimator and prior. The density_estimator must contain the keys 'hidden_features',
        'num_transforms' and 'model'.

        Parameters
        ----------
        hyperparams : dict
            Dictionary of hyperparameters of the model.

        Returns
        -------
        model : sbi.inference.snpe.SNPE
            SNPE model.
        """
        # Check if density_estimator is in hyperparams
        if 'density_estimator' in hyperparams:
            # Check that density_estimator is a dictionary and contain the keys 'hidden_features',
            # 'num_transforms' and 'model'
            if type(hyperparams['density_estimator']) is not dict:
                raise ValueError('density_estimator must be a dictionary.')
            if 'hidden_features' not in hyperparams['density_estimator']:
                raise ValueError('hidden_features must be in density_estimator.')
            if 'num_transforms' not in hyperparams['density_estimator']:
                raise ValueError('num_transforms must be in density_estimator.')
            if 'model' not in hyperparams['density_estimator']:
                raise ValueError('model must be in density_estimator.')
            # Initialize the posterior neural network
            density_estimator_build_fun = self.posterior_nn(
                model=hyperparams['density_estimator']['model'],
                hidden_features=hyperparams['density_estimator']['hidden_features'],
                num_transforms=hyperparams['density_estimator']['num_transforms']
            )
        else:
            raise ValueError('density_estimator must be in hyperparams.')
        # Check that prior is in hyperparams
        if 'prior' not in hyperparams:
            raise ValueError('prior must be in hyperparams.')
        # Create a dict with the remaining hyperparameters
        others = {key: value for key, value in hyperparams.items() if key not in ['density_estimator', 'prior']}
        # Initialize SNPE
        model = self.SNPE(
            prior=hyperparams['prior'],
            density_estimator=density_estimator_build_fun,
            **others
        )

        return model


    def train(self, param_grid=None, n_splits=10, n_repeats=10, train_params={'learning_rate': 0.0005}):
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
            Dictionary of training parameters for SNPE.
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
                model = self.SNPE(prior=None)

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

                        # Re-initialize the SNPE object with the new configuration
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
                # if self.model[1] == 'sklearn':
                #     model.set_params(**best_config)
                # if self.model[1] == 'sbi':
                #     model = self.initialize_sbi(best_config)

                if self.model[1] == 'sklearn':
                    model = best_fits
                if self.model[1] == 'sbi':
                    model = [best_fits[i][0] for i in range(len(best_fits))]
                    density_estimator = [best_fits[i][1] for i in range(len(best_fits))]

                # print best hyperparameters
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

                # Train the neural density estimator
                density_estimator = model.train(**train_params)

        # Save the best model and the StandardScaler
        if not os.path.exists('data'):
            os.makedirs('data')
        with open('data/model.pkl', 'wb') as file:
            pickle.dump(model, file)
        with open('data/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        # Save also the density estimator if the model is SNPE
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
            Function to process a batch of features.

            Parameters
            ----------
            batch: tuple
                Tuple containing the batch of features, the StandardScaler and the model (and the posterior if the model
                is SNPE).

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
                        if type(posterior) is list:
                            posterior_samples = [post.sample((5000,), x=x_o, show_progress_bars=False) for post in
                                                 posterior]
                            # Compute the mean of the posterior samples
                            pred = np.mean([np.mean(post.numpy(), axis=0) for post in posterior_samples], axis=0)
                        else:
                            posterior_samples = posterior.sample((5000,), x=x_o, show_progress_bars=False)
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
            batch_size = len(features) # to avoid memory issues
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

        # Compute features in parallel
        with pool_class(num_cpus) as pool:
            imap_results = pool.imap(process_batch, batch_args)
            results = list(
                self.tqdm(imap_results, total=len(batches), desc="Computing predictions")) if self.tqdm_inst else list(
                imap_results)

        # Sort the predictions based on the original index
        results.sort(key=lambda x: x[0])
        predictions = [pred for _, batch_preds in results for pred in batch_preds]

        return predictions
