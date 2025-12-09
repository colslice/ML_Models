"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        distances = utils.euclidean_dist_squared(X_hat, self.X)
        predictions = []
       
        for dist_row in distances:
            nearest_neighbors = np.argsort(dist_row)[:self.k]
            labels = np.array([self.y[i] for i in nearest_neighbors])
            predictions.append(utils.mode(labels))
        
        return predictions
