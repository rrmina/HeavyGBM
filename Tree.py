from __future__ import annotations # For type hinting my own class!

import numpy as np
from typing import Tuple, Union

class DecisionTree():
    def __init__(self,
        min_samples_per_node = 5,
        max_depth = 5,
        impurity_measure = 'gini',
        num_targets = None
    ) -> None:

        # Decision Tree Hyperparameters
        self.min_samples_per_node = min_samples_per_node
        self.max_depth = max_depth

        # Impurity Functions
        self.impurity_measure = impurity_measure
        impurity_functions = {
            'gini': self._compute_gini_impurity,
            'entropy': self._compute_entropy,
            'variance': self._compute_variance
        }
        self.impurity_function = impurity_functions[impurity_measure]

        # Number of targets for counting purpose
        self.num_targets =  num_targets # to be determined

        # Class Variables
        self.depth = 0

        # Node Variables
        self.left_node = None
        self.right_node = None
        self.splitting_feature = None
        self.splitting_threshold = None

    # Currently writing for training only
    # I will ammend this to enable prediction
    def build_tree(self, 
        X: np.ndarray, 
        y: np.ndarray,
        depth: int = 0
    ) -> DecisionTree:
        
        # Store depth
        self.depth = depth

        # Check stopping criterion
        if self.check_stopping_criterion(X, y) is True:
            return None

        # Get the Decision Split and the Data Split
        data_splits, feature_index, threshold = self.find_best_split(X, y)
        if data_splits is None:
            return None
        self.splitting_feature = feature_index
        self.splitting_threshold = threshold

        # Build Left and Right (Children) Tree
        (X_left, y_left), (X_right, y_right) = data_splits 
        self.left_node = DecisionTree(
            min_samples_per_node = self.min_samples_per_node,
            max_depth = self.max_depth,
            impurity_measure = self.impurity_measure,
            num_targets = self.num_targets
        ).build_tree(X_left, y_left, depth + 1)     # Add 1 to depth

        self.right_node = DecisionTree(
            min_samples_per_node = self.min_samples_per_node,
            max_depth = self.max_depth,
            impurity_measure = self.impurity_measure,
            num_targets = self.num_targets
        ).build_tree(X_right, y_right, depth + 1)   # Add 1 to depth

        # Return the Node
        return self

    def check_stopping_criterion(self,
        X: np.ndarray, 
        y: np.ndarray
    ) -> bool:
        
        # Check max depth
        if self.depth > self.max_depth:
            print("Stopping with max_depth: ", self.max_depth)
            return True
        
        # Check minimum number of samples per node
        if X.shape[0] <= self.min_samples_per_node:
            print("Stopping with min_samples_per_node: ", X.shape[0])
            return True
        
        return False

    def find_best_split(self,
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],    # Left Data and Right Data
        Union[int, str],                                                        # Feature index or feature name 
        float                                                                   # Threshold
    ]:
        
        # Find the best split
        best_split = None
        best_impurity = np.inf
        best_threshold = None

        num_samples, num_feautures = X.shape[0], X.shape[1]
        for i in range(num_feautures):

            # Can be improved by using buckets / histograms
            # Find the threshold with the least impurity
            thresholds = np.unique(X[:, i])      
            for threshold in thresholds:
                left_data, right_data = self.split_data(X, i, y, threshold)
                X_left, y_left = left_data
                X_right, y_right = right_data

                # Skip the lowest and highest values because no split
                if (X_left.shape[0] == 0) or (X_right.shape[0] == 0):
                    continue

                # Compute impurity
                left_impurity = self.impurity_function(y_left)
                right_impurity = self.impurity_function(y_right)
                impurity = (y_left.shape[0] / y.shape[0]) * left_impurity + \
                    (y_right.shape[0] / y.shape[0]) * right_impurity
                
                # Compare with the best impurity
                if impurity < best_impurity:
                    best_split = (left_data, right_data)
                    best_impurity = impurity
                    best_threshold = threshold

        # Normally we would compare the parent's impurities to the 
        # children's impurities. But mathematically, the children's impurities 
        # will always be less or equal to the parent's impurities

        # Return (data split, feature/feature name, feature threshold)
        if best_split is None:
            return None, None, None
        return best_split, i, threshold

    
    def split_data(self,
        X: np.ndarray,
        i: Union[int, str],
        y: np.ndarray,
        threshold: float
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: 
        
        # Split mask according to a specific feature
        X_i = X[:, i]
        left_indices = np.where(X_i <= threshold)
        right_indices = np.where(X_i > threshold)
        
        # Apply split mask to the whole dataset
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        return (X_left, y_left), (X_right, y_right)


    ######################################################################################################
    #
    #                        Impurity Methods | Supposedly methods of a Node Class
    #
    ######################################################################################################
    def _compute_target_dist(self, 
        y: np.ndarray
    ) -> float:
        
        # Count the number of samples per target
        # TO BE IMPROVED
        target_dist = []
        for i in range(self.num_targets):
            target_dist.append(np.mean((y == i)*1))

        return target_dist

    # For Classification
    def _compute_gini_impurity(self, 
        y: np.ndarray
    ) -> float:
        
        target_dist = np.array(self._compute_target_dist(y))
        gini_impurity = np.sum(target_dist * (1-target_dist))

        return gini_impurity
    
    # For Classification
    def _compute_entropy(self,
        y: np.ndarray
    ) -> float:
        
        target_dist = np.array(self._compute_target_dist(y))
        target_dist = target_dist[target_dist > 0]                  # Remove zero probabilities
        entropy = -1 * np.sum(target_dist * np.log2(target_dist))   # Zero prob results to log undef

        return entropy
    
    # For Regression
    def _compute_variance(self,
        y: np.ndarray
    ) -> float:
        
        num_samples = y.shape[0]
        if num_samples == 0:
            return 0
        
        insides = y - np.mean(y)
        squares = np.power(insides, 2)
        variance = np.mean(squares)

        return variance
    