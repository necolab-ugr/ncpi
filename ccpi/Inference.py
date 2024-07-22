import numpy as np
import pickle
import os
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sbi.inference import SNPE

class inference(object):
    '''
    Class for inferring cortical circuit parameters from features of
    electrophysiological recordings.

    Parameters:
    framework (str): Name of framework to use. 
                     'scikit-learn' or 'SBI'.
    model (str): Name of model to use depending of the framework.
                    'Perceptron', 'MLPRegressor' or 'SNPE'.
    hyperparams (dict): Dictionary of hyperparameters for the model.
    '''

    def __init__(self, framework, model, hyperparams = None):
        self.framework_list = {
            'scikit-learn': ['Perceptron', 'MLPRegressor'],
            'SBI': ['SNPE']
        }

        self.framework_str = framework
        self.model_str = model
        self.framework = None
        self.model = None
        self.hyperparams = hyperparams
        self.X_train = []
        self.Y_train = []

    def train_data(self, x, y):
        '''
        Method to set training data.

        Parameters:
        x (np.ndarray): Training data.
        y (np.ndarray): Labels for training data.
        '''
        self.X_train = x
        self.Y_train = y
    


    def set_model(self):
        '''
        Method to set model with hyperparameters.
        '''

        models = {
            'scikit-learn': [Perceptron, MLPRegressor],
            'SBI': [SNPE]
        }  
        model_index = self.framework_list[self.framework_str].index(self.model_str)
        if self.hyperparams == None:
            print(f'\nNo parameters given. Setting default parameters.')    
            self.model = models[self.framework_str][model_index]()
        else: 
            print(f'\nSetting {self.framework_str} model with {self.model_str} model and parameters:')
            for key, value in self.hyperparams.items():
                print(f'\t{key}: {value}')
            self.model = models[self.framework_str][model_index](**self.hyperparams)
                


    def check_data(self, x):
        '''
        Method to check if data is in the correct format for the model.

        Parameters:
        x (np.ndarray): Data to check.
        '''

        if self.framework_str == 'scikit-learn':
            if type(x) is not np.ndarray:
                x = np.array(x)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return x
        else:
            print(f'{self.model_str} model not implemented in check_training_data model. Returning data as is.')
            return x
        
    def check_existing_model(self, model):
        '''
        Check if model already exists. 
        If it does, ask user if they want to load the existing model.

        Parameters:
        model (str): Name of model to check for.

        Returns:
        bool: True if model exists. 
              False if model does not exist or want to create a new one.
        '''

        if os.path.exists(f'/data/{model}.pkl'):
            print(f'{model} already exists.')
            input('Load existing model? (y/n): ')
            if input == 'n':
                print('Removing existing model...')
                os.remove(f'/data/{model}.pkl')
                print('Creating new model...')
                return False
            else:
                with open(f'/data/{model}.pkl', 'rb') as file:
                    self.model = pickle.load(file)
                return True
        else:
            print(f'{model} does not exist.')
            print('Creating new model...')   
            return False     

    def train(self, gridsearch = False, param_grid = None):
        '''
        Method to train the model. It saves the model in a pickle file.

        Parameters:
        gridsearch (bool): If True, perform GridSearchCV.
        param_grid (dict): Dictionary of parameters for GridSearchCV.

        '''

        # Check if GridSearchCV is selected and if the model already exists
        if gridsearch and self.model_str in self.framework_list['sklearn']:
            new_model = self.check_existing_model('best_estimator')
            if new_model == False:
                # If there ir no existing model, set the model
                print('\nPerforming GridSearchCV...')
                if param_grid == None:
                    print('No parameters given for GridSearchCV. Setting default parameters.')
                    param_grid = {
                        'hidden_layer_sizes': [
                            (25,), 
                            (25, 25,), 
                            (50,),
                            (50, 50)
                            ],
                    }
                
                # Show parameters for GridSearchCV
                print('Setting parameters for GridSearchCV:')
                for key, value in param_grid.items():
                    print(f'\t{key}: {value}')
                
                grid_search = GridSearchCV(self.model, 
                                param_grid,
                                cv=10,
                                scoring='neg_mean_squared_error',
                                verbose=3,
                                n_jobs=-1)
            
                # Training data
                grid_search.fit(self.X_train, self.Y_train)
                self.model = grid_search.best_estimator_
                print('GridSearchCV completed')

                if not os.path.exists('data'):
                    os.makedirs('data')
                with open('data/best_estimator.pkl', 'wb') as file:
                    pickle.dump(self.model, file)

        elif gridsearch == False:
            # Check if model already exists
            new_model = self.check_existing_model(self.model_str)
            if new_model == False:
                self.set_model()

                print(f'\nSelected {self.framework_str} model with {self.model_str} model')
                print('\nTraining model...')

                self.X_train = self.check_data(self.X_train)
                self.Y_train = np.array(self.Y_train)

            self.model.fit(self.X_train, self.Y_train)

            with open(f'data/{self.model_str}.pkl', 'wb') as file:
                pickle.dump(self.model, file)

        print('Training completed')



    def predict(self, x):
        '''
        Method to perform inference.

        Parameters:
        x (np.ndarray): Data for inference.

        Returns:
        np.ndarray: Predictions from the model.
        '''
        print('\nPerforming inference...')

        x = self.check_data(x)
        predictions = self.model.predict(x)

        print('Inference completed')
        return predictions
