from __future__ import annotations # For type hinting my own class!
import numpy as np
from copy import deepcopy

from DecisionTreeRegressor import DecisionTreeRegressor

class GradientBoostedTreesRegressor():
    
    def __init__(self,
        num_trees: int = 5,
        learning_rate: float = 0.01,
        min_samples_per_node: int = 5,
        max_depth: int = np.inf,
        loss_measure: str = 'mse',
        impurity_measure: str = 'variance',
        num_targets: int = None
    ) -> None:
        
        # Gradient Boosted Trees Hyperparameters
        self.num_trees = num_trees
        self.learning_rate = learning_rate

        # Decision Tree Hyperparameters
        self.min_samples_per_node = min_samples_per_node
        self.max_depth = max_depth
        self.num_targets = num_targets

        # Impurity Functions
        self.impurity_measure = impurity_measure

        # The Forest
        self.forest = []

    def build_forest(self,
        X: np.ndarray,
        y: np.ndarray
    ) -> GradientBoostedTreesRegressor:
        
        # Initialize the targets
        y_t = np.array(deepcopy(y), np.float64)

        # Gradient Boosting is a progressive process so we can not parallelize
        num_samples = y_t.shape[0]
        for t in range(self.num_trees):

            # Train a decision tree
            tree = DecisionTreeRegressor(
                min_samples_per_node = self.min_samples_per_node,
                max_depth = self.max_depth,
                impurity_measure = self.impurity_measure,
                # num_targets = self.num_targets
            ).build_tree(X, y_t)

            # Store the decision tree
            self.forest.append(tree)

            # Compute the new targets
            # Ideally, we want to do the inference part in parallel
            y_pred = np.array([self.predict(X[i]) for i in range(num_samples)])
            y_t -= self.learning_rate * y_pred

        return self

    def predict(self,
        X: np.ndarray
    ) -> float:
        
        pred = self.forest[0].predict(X)
        curr_num_trees = len(self.forest)
        for t in range(1, curr_num_trees):
            pred += self.learning_rate * self.forest[t].predict(X)

        return pred
    
    def early_stoping(self):
        pass