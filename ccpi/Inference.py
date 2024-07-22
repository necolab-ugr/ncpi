import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sbi.inference import SNPE


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
            Name of the machine-learning model to use. Options are 'Ridge',
            'MLPRegressor', and 'SNPE'.
        hyperparams : dict, optional
            Dictionary of hyperparameters of the model. The default is None.
        """

        # Assert that model is a string
        if type(model) is not str:
            raise ValueError('Model must be a string.')

        # Check if model is in the list of machine-learning models
        self.model_list = {'Ridge', 'MLPRegressor', 'SNPE'}
        if model not in self.model_list:
            raise ValueError(f'{model} not in the list of machine-learning models.')

        # Set model
        self.model = model

        # Check if hyperparameters is a dictionary
        if hyperparams is not None:
            if type(hyperparams) is not dict:
                raise ValueError('Hyperparameters must be a dictionary.')
            self.hyperparams = hyperparams
        else:
            self.hyperparams = None

        # Initialize X and Y training data
        self.features = []
        self.theta = []

    def append_simulation_data(self, X, Y):
        """
        Method to append simulation data.

        Parameters
        ----------
        X : np.ndarray
            Features.
        Y : np.ndarray
            Parameters to infer.
        """

        # Assert that X and Y are numpy arrays
        if type(X) is not np.ndarray:
            raise ValueError('X must be a numpy array.')
        if type(Y) is not np.ndarray:
            raise ValueError('Y must be a numpy array.')

        # Assert that X and Y have the same number of rows
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows.')

        # Stack X and Y
        X = np.stack(X)
        Y = np.stack(Y)

        # Append X and Y to training data
        self.features = X
        self.theta = Y

    def train(self, param_grid=None):
        """
        Method to train the model.

        Parameters
        ----------
        param_grid : list of dictionaries, optional
            List of dictionaries of hyperparameters to search over. The default
            is None (no hyperparameter search).
        """

        # Initialize model with default hyperparameters
        if self.hyperparams is None:
            if self.model == 'Ridge':
                model = Ridge(alpha=1.0)
            elif self.model == 'MLPRegressor':
                model = MLPRegressor(random_state=0,
                                     max_iter=500,
                                     tol=1e-1,
                                     n_iter_no_change=2,
                                     verbose=False)
            elif self.model == 'SNPE':
                model = SNPE(prior=None)

        # Initialize model with user-defined hyperparameters
        else:
            if self.model == 'Ridge':
                model = Ridge(**self.hyperparams)
            elif self.model == 'MLPRegressor':
                model = MLPRegressor(**self.hyperparams)
            elif self.model == 'SNPE':
                model = SNPE(**self.hyperparams)

        # Check if X_train and Y_train are not empty
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

        # Perform repeated grid search if param_grid is provided
        if param_grid is not None:
            # Loop over each set of hyperparameters
            best_score = np.inf
            best_config = None
            for params in param_grid:
                # Initialize RepeatedKFold
                rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)

                # Loop over each repeat and fold
                mean_scores = []
                for train_index, test_index in rkf.split(self.features):
                    # Split the data
                    X_train, X_test = self.features[train_index], self.features[test_index]
                    Y_train, Y_test = self.theta[train_index], self.theta[test_index]

                    if self.model == 'Ridge' or self.model == 'MLPRegressor':
                        # Update parameters using set_params
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
                        # pass the simulated data to the inference object
                        model.append_simulations(
                            torch.from_numpy(np.array(Y_train, dtype=np.float32)),
                            torch.from_numpy(np.array(X_train, dtype=np.float32)))

                        # Train the neural density estimator
                        density_estimator = model.train()

                        # Build the posterior
                        posterior = model.build_posterior(density_estimator)

                        # Sample the posterior and compute the parameter recovery error
                        for sample in range(X_test.shape[0]):
                            # x_o and theta_o values
                            x_o = torch.from_numpy(np.array(
                                X_test[sample], dtype=np.float32))
                            # do not consider samples with inf or nan values
                            if (np.isinf(x_o).sum() < 1) and (np.isnan(x_o).sum() < 1):
                                theta_o = torch.from_numpy(np.array(Y_test[sample], dtype=np.float32))
                                # sample the posterior
                                posterior_samples = posterior.sample((5000,), x=x_o)

                                # compute the parameter recovery error (PRE),
                                # defined as the average absolute error using all
                                # values from posterior
                                absdiff = (posterior_samples - theta_o).abs()
                                avg_error = np.array(absdiff.mean(axis=0))

                                # Append the average error
                                mean_scores.append(avg_error)

                # Compute the mean of the mean squared errors
                if np.mean(mean_scores) < best_score:
                    best_score = np.mean(mean_scores)
                    best_config = params

            # Update the model with the best hyperparameters
            model.set_params(**best_config)

        # Fit the model using all the data
        if self.model == 'Ridge' or self.model == 'MLPRegressor':
            model.fit(self.features, self.theta)
            best_model = model

        if self.model == 'SNPE':
            # pass all data to the inference object
            model.append_simulations(
                torch.from_numpy(np.array(self.theta, dtype=np.float32)),
                torch.from_numpy(np.array(self.features, dtype=np.float32)))

            # Train the neural density estimator
            density_estimator = model.train()
            best_model = density_estimator

        # Save the best model and the StandardScaler
        if not os.path.exists('data'):
            os.makedirs('data')
        with open('data/best_model.pkl', 'wb') as file:
            pickle.dump(best_model, file)
        with open('data/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

    def predict(self, features):
        """
        Method to predict the parameters.
        """

        # Load the best model and the StandardScaler
        with open('data/best_model.pkl', 'rb') as file:
            best_model = pickle.load(file)
        with open('data/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

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
            if self.model == 'Ridge' or self.model == 'MLPRegressor':
                pred = best_model.predict(feat)
            if self.model == 'SNPE':
                # Build the posterior
                posterior = best_model.build_posterior()

                # Sample the posterior
                x_o = torch.from_numpy(np.array(feat, dtype=np.float32))
                posterior_samples = posterior.sample((5000,), x=x_o)

                # Compute the mean of the posterior samples
                pred = np.mean(posterior_samples.numpy(), axis=0)

            # Append the predictions
            predictions.append(pred[0])

        # Return the predictions
        return predictions
