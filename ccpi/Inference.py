import numpy as np
from sklearn.linear_model import Perceptron

class inference(object):
    '''
    Class for inferring cortical circuit parameters from features of
    electrophysiological recordings.
    '''

    def __init__(self, model):

        self.model = model()
        self.X = []
        self.Y = []

        def append_simulation_data(self, x, y):
            self.X.append(x)
            self.Y.append(y)

        def train(self):
            self.X = np.array(self.X)
            self.Y = np.array(self.Y)
            self.model.fit(self.X, self.Y)