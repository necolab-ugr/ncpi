import os
import pickle
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from sbi.inference import SNPE
import torch


class Inference(object):
    """
    Class for inferring cortical circuit parameters from features of
    electrophysiological recordings.
    """

    def __init__(self, model, hyperparams=None):
        """
        Constructor method.

        Parameters
        ----------
        model : str
            Name of the machine-learning model to use. It can be any of the regression models from sklearn or 'SNPE'
            from sbi.
        hyperparams : dict, optional
            Dictionary of hyperparameters of the model. The default is None.
        """

        # Assert that model is a string
        if type(model) is not str:
            raise ValueError('Model must be a string.')

        # Check if model is in the list of regression models from sklearn, or it is SNPE
        regressors = [estimator for estimator in all_estimators() if issubclass(estimator[1], RegressorMixin)]
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

        # Stack features and parameters
        features = np.stack(features)
        parameters = np.stack(parameters)

        # Add features and parameters to training data
        self.features = features
        self.theta = parameters

    def train(self, param_grid=None, n_splits=10, n_repeats=10):
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
        """

        # Initialize model with default hyperparameters
        if self.hyperparams is None:
            if self.model[1] == 'sklearn':
                # Import and initialize the model
                exec(f'from sklearn.linear_model import {self.model[0]}')
                model = eval(f'{self.model[0]}')()
            elif self.model == 'SNPE':
                model = SNPE(prior=None, logging_level='ERROR')  # Does logging_level='ERROR' work?

        # Initialize model with user-defined hyperparameters
        else:
            if self.model[1] == 'sklearn':
                # Import and initialize the model
                exec(f'from sklearn.linear_model import {self.model[0]}')
                model = eval(f'{self.model[0]}')(**self.hyperparams)
            elif self.model == 'SNPE':
                # Add first logging_level to hyperparams
                self.hyperparams['logging_level'] = 'ERROR'  # Does logging_level='ERROR' work?
                model = SNPE(**self.hyperparams)

        # Check if features and parameters are not empty
        if len(self.features) == 0:
            raise ValueError('No features provided.')
        if len(self.theta) == 0:
            raise ValueError('No parameters provided.')

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the StandardScaler
        scaler.fit(self.features)

        # Transform the features
        self.features = scaler.transform(self.features)

        # Search for the best hyperparameters using RepeatedKFold cross-validation and grid search if param_grid is
        # provided
        if param_grid is not None:
            # Assert that param_grid is a list
            if type(param_grid) is not list:
                raise ValueError('param_grid must be a list.')

            # Loop over each set of hyperparameters
            best_score = np.inf
            best_config = None
            for params in param_grid:
                print(f'\nHyperparameters: {params}')
                # Initialize RepeatedKFold
                rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

                # Loop over each repeat and fold
                mean_scores = []
                for repeat_idx, (train_index, test_index) in enumerate(rkf.split(self.features)):
                    print(f'\rRepeat {repeat_idx // 10 + 1}, Fold {repeat_idx % 10 + 1}', end='', flush=True)
                    # Split the data
                    X_train, X_test = self.features[train_index], self.features[test_index]
                    Y_train, Y_test = self.theta[train_index], self.theta[test_index]

                    if self.model[1] == 'sklearn':
                        # Update parameters
                        model.set_params(**params)

                        # Fit the model
                        model.fit(X_train, Y_train)

                        # Predict the parameters
                        Y_pred = model.predict(X_test)

                        # Compute the mean squared error
                        mse = np.mean((Y_pred - Y_test) ** 2)

                        # Append the mean squared error
                        mean_scores.append(mse)

                    if self.model == 'SNPE':
                        # Re-initialize the SNPE object with the new configuration
                        model = SNPE(**params)

                        # Ensure theta is a 2D array with a single column
                        if Y_train.ndim == 1:
                            Y_train = Y_train.reshape(-1, 1)

                        # Append simulations
                        model.append_simulations(
                            torch.from_numpy(Y_train.astype(np.float32)),
                            torch.from_numpy(X_train.astype(np.float32))
                        )

                        # Train the neural density estimator
                        density_estimator = model.train()

                        # Build the posterior
                        posterior = model.build_posterior(density_estimator)

                        # Loop over all test samples
                        for i in range(len(X_test)):
                            # Sample the posterior
                            x_o = torch.from_numpy(np.array(X_test[i], dtype=np.float32))
                            posterior_samples = posterior.sample((5000,), x=x_o)
                            pred = np.mean(posterior_samples.numpy(), axis=0)
                            # Compute the mean squared error
                            mse = np.mean((pred - Y_test[i]) ** 2)
                            # Append the mean squared error
                            mean_scores.append(mse)

                # Compute the mean of the mean squared errors
                if np.mean(mean_scores) < best_score:
                    best_score = np.mean(mean_scores)
                    best_config = params

            # Update the model with the best hyperparameters
            if best_config is not None:
                if self.model[1] == 'sklearn':
                    model.set_params(**best_config)
                if self.model == 'SNPE':
                    model = SNPE(**best_config)
                # print best hyperparameters
                print(f'\nBest hyperparameters: {best_config}')
            else:
                raise ValueError('No best hyperparameters found.')

        # Fit the model using all the data
        if self.model[1] == 'sklearn':
            model.fit(self.features, self.theta)

        if self.model == 'SNPE':
            # Ensure theta is a 2D array with a single column
            if self.theta.ndim == 1:
                self.theta = self.theta.reshape(-1, 1)

            # Append simulations
            model.append_simulations(
                torch.from_numpy(self.theta.astype(np.float32)),
                torch.from_numpy(self.features.astype(np.float32))
            )

            # Train the neural density estimator
            density_estimator = model.train()

        # Save the best model and the StandardScaler
        if not os.path.exists('data'):
            os.makedirs('data')
        with open('data/model.pkl', 'wb') as file:
            pickle.dump(model, file)
        with open('data/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        # Save also the density estimator if the model is SNPE
        if self.model == 'SNPE':
            with open('data/density_estimator.pkl', 'wb') as file:
                pickle.dump(density_estimator, file)

    def predict(self, features):
        """
        Method to predict the parameters.
        """

        # Load the best model and the StandardScaler
        with open('data/model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('data/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        if self.model == 'SNPE':
            with open('data/density_estimator.pkl', 'rb') as file:
                density_estimator = pickle.load(file)

        # Assert that features is a numpy array
        if type(features) is not np.ndarray:
            raise ValueError('features must be a numpy array.')

        # Stack features
        features = np.stack(features)

        # Predict the parameters
        predictions = []
        for feat in features:
            # Transform the features
            feat = scaler.transform(feat.reshape(1, -1))

            # Predict the parameters
            if self.model[1] == 'sklearn':
                pred = model.predict(feat)
            if self.model == 'SNPE':
                # Build the posterior
                posterior = model.build_posterior(density_estimator)

                # Sample the posterior
                x_o = torch.from_numpy(np.array(feat, dtype=np.float32))
                posterior_samples = posterior.sample((5000,), x=x_o)

                # Compute the mean of the posterior samples
                pred = np.mean(posterior_samples.numpy(), axis=0)

            # Append the predictions
            predictions.append(pred[0])

        # Return the predictions
        return predictions
