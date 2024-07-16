import os
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


class Inference(object):
    '''
    Class for inferring circuit model parameters from features of
    electrophysiological recordings.
    '''

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_model(self):
        """Train a regression model to predict the parameters of the model."""
        # check if there is a best estimator already saved
        if os.path.exists('data/best_estimator.pkl'):
            with open('data/best_estimator.pkl', 'rb') as file:
                return pickle.load(file)
        else:
            print('No best estimator found. Training the regression model from scratch.')

            # Initialize the MLPRegressor
            mlp = MLPRegressor(random_state=0,
                               max_iter=500,
                               tol=1e-1,
                               n_iter_no_change=2,
                               verbose=False)
            # set up the parameter grid
            param_grid = {
                'hidden_layer_sizes': [(25,), (25, 25), (50,), (50, 50)]
            }

            # initialize GridSearchCV
            grid_search = GridSearchCV(mlp,
                                       param_grid,
                                       cv=10,
                                       scoring='neg_mean_squared_error',
                                       verbose=3,
                                       n_jobs=-1)

            # fit GridSearchCV to the data
            grid_search.fit(self.X, self.y)

            # save and return the best model
            if not os.path.exists('data'):
                os.makedirs('data')
            with open('data/best_estimator.pkl', 'wb') as file:
                pickle.dump(grid_search.best_estimator_, file)
            return grid_search.best_estimator_