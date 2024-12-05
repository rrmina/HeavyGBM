import numpy as np
from typing import Tuple, Union

from DecisionTreeClassifier import DecisionTreeClassifier

class RandomForestClassifier():
    def __init__(self,
        num_trees: int = 5,
        min_samples_per_node: int = 5,
        max_depth: int = np.inf,
        impurity_measure: str = 'variance',
        num_targets: int = None
    ) -> None:

        # Random Forest Hyperparameters
        self.num_trees = num_trees

        # Decision Tree Hyperparameters
        self.min_samples_per_node = min_samples_per_node
        self.max_depth = max_depth
        self.num_targets = num_targets

        # Impurity Functions
        self.impurity_measure = impurity_measure
        
        # The Forest
        self.forest = []

    def build_forest(self, X, y):

        # Instead of using for-loop. We can do the bagging and training in parallel
        for _ in range(self.num_trees):

            # Sample with replacement
            X_sampled, y_sampled = self.sample_dataset_with_replacement(X, y)

            # Train a decision tree
            tree = DecisionTreeClassifier(
                min_samples_per_node = self.min_samples_per_node,
                max_depth = self.max_depth,
                impurity_measure = self.impurity_measure,
                num_targets = self.num_targets
            ).build_tree(X_sampled, y_sampled)

            # Store the decision tree
            self.forest.append(tree)

        return self
    
    # Currently, inference can only predict for one sample at a time
    def predict(self, 
        X: np.ndarray
    ) -> int:
        
        # Instead of using for-loop, we can do the inference in parallel
        pred = 0
        for i in range(self.num_trees):
            pred += self.forest[i].predict(X)

        pred = pred / self.num_trees

        # Half Round-up Trick
        pred = int(pred + 0.5)

        return pred

    def sample_dataset_with_replacement(self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        num_samples = X.shape[0]
        sample_indexes = np.random.choice(
            np.arange(num_samples),
            size=num_samples,
            replace=True
        )

        sampled_dataset = (X[sample_indexes, :], y[sample_indexes])

        return sampled_dataset
