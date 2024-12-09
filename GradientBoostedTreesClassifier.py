from __future__ import annotations # For type hinting my own class!
import numpy as np
from copy import deepcopy

from DecisionTreeRegressor import DecisionTreeRegressor

class GradientBoostedTreesClassifier():
    
    def __init__(self,
        num_trees: int = 5,
        learning_rate: float = 0.01,
        min_samples_per_node: int = 5,
        max_depth: int = np.inf,
        # loss_measure: str = 'entropy',
        impurity_measure: str = 'entropy',
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
    ) -> GradientBoostedTreesClassifier:
        
        # Initialize targets and Turn labels into One-hot vectors
        num_samples = X.shape[0]
        y_t = np.zeros((num_samples, self.num_targets))
        y_t[np.arange(num_samples), y] = 1

        # Gradient Boosting is a progressive process so we can not parallelize
        num_samples = y_t.shape[0]
        for t in range(self.num_trees):

            # Train a multi-output decision tree
            tree = DecisionTreeRegressor(
                min_samples_per_node = self.min_samples_per_node,
                max_depth = self.max_depth,
                # impurity_measure = self.impurity_measure,
                num_targets = self.num_targets
            ).build_tree(X, y_t)

            # Store the decision tree
            self.forest.append(tree)

            # Compute the new targets
            # Ideally, we want to do the inference part in parallel
            y_pred = np.array([self.predict_proba(X[i]).reshape(-1) for i in range(num_samples)])
            y_t -= self.learning_rate * y_pred

        return self

    def predict_logits(self,
        X: np.ndarray
    ) -> np.ndarray:
        
        logits = self.forest[0].predict(X)
        curr_num_trees = len(self.forest)
        for t in range(1, curr_num_trees):
            logits += self.learning_rate * self.forest[t].predict(X)

        return logits
    
    def predict_proba(self,
        X: np.ndarray
    ) -> np.ndarray:
        
        logits = self.predict_logits(X)
        proba = self.softmax(logits)

        return proba
    
    def predict(self,
        X: np.ndarray
    ) -> np.ndarray:

        logits = self.predict_logits(X)
        if len(logits.shape) == 1:
            logits = logits.reshape(1, -1)

        pred = np.argmax(logits, axis=1)

        return pred
    

    def softmax(self,
        logits: np.ndarray
    ) -> np.ndarray:
        
        if len(logits.shape) == 1:
            logits = logits.reshape(1, -1)

        numerator = np.exp(logits)
        denominator = np.sum(numerator, axis=1)
        sm = numerator / denominator

        return sm

    def early_stoping(self):
        pass