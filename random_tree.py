from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:

    def __init__(self, num_trees, max_depth):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.forest = []

    def fit(self, X, y):
        self.forest = []
        for _ in range(self.num_trees):
            tree = RandomTree(max_depth=self.max_depth)
            tree.fit(X, y)
            self.forest.append(tree)

    def predict(self, X_pred):
        
        predictions = np.array([tree.predict(X_pred) for tree in self.forest])
        
        n = X_pred.shape[0]
        winners = np.empty(n, dtype=predictions.dtype)
        
        for i in range(n):
            votes, counts = np.unique(predictions[:, i], return_counts=True)
            winners[i] = votes[np.argmax(counts)]

        return winners

                        




