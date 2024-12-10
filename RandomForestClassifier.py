from __future__ import annotations # For type hinting my own class!
import numpy as np
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

from DecisionTreeClassifier import DecisionTreeClassifier

class RandomForestClassifier():
    def __init__(self,
        num_trees: int = 5,
        min_samples_per_node: int = 5,
        max_depth: int = np.inf,
        impurity_measure: str = 'gini',
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
    
    def build_forest(self,
        X: np.ndarray,
        y: np.ndarray
    ) -> RandomForestClassifier:

        # Instead of using for-loop. We can do the bagging and training in parallel
        for _ in range(self.num_trees):

            # Sample with replacement
            X_sampled, y_sampled = self.sample_dataset_with_replacement(X, y)

            # Missing step is to take only sqrt(num_features) as sampled data
            # [WIP] Only bag for sqrt(num_features) of features for each dataset
            # We would need a mapping in of feature_index in training and inference
            # Alternatively, we could assign names to each feature for more readable indexing 
            # ^ in other words, use Pandas!
            # pass
            # pass

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
    
    def build_forest_experimental(self,
        X: np.ndarray,
        y: np.ndarray
    ) -> RandomForestClassifier:

        # Defining a new function for parallel execution
        def train_tree():
            # Sample with replacement
            X_sampled, y_sampled = self.sample_dataset_with_replacement(X, y)

            # Train a decision tree
            tree = DecisionTreeClassifier(
                min_samples_per_node = self.min_samples_per_node,
                max_depth = self.max_depth,
                impurity_measure = self.impurity_measure,
                num_targets = self.num_targets
            ).build_tree(X_sampled, y_sampled)

            return tree
        
        # Execute in parallel
        with ThreadPoolExecutor() as executor:
            self.forest = list(
                executor.map(lambda _: train_tree(), range(self.num_trees))
            )

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
