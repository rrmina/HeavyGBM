from __future__ import annotations # For type hinting my own class!

import numpy as np
from typing import Tuple, Union

class DecisionTreeRegressor():
    def __init__(self,
        min_samples_per_node: int = 5,
        max_depth: int = 5,
        impurity_measure: str = 'variance',
        num_targets: int = 1    # Set to more than 1 if multi-output regressor
    ) -> None:

        # Decision Tree Hyperparameters
        self.min_samples_per_node = min_samples_per_node
        self.max_depth = max_depth
        self.num_targets = num_targets

        # Impurity Functions
        self.impurity_measure = impurity_measure
        impurity_functions = {
            'variance': self._compute_variance
        }
        self.impurity_function = impurity_functions[impurity_measure]

        # And Prediction Storage Functions
        pred_storage_functions = {
            'variance': self._store_regression_pred
        }
        self.pred_storage_function = pred_storage_functions[impurity_measure]

        # Class Variables
        self.depth = 1

        # Node Variables
        self.left_node = None
        self.right_node = None
        self.splitting_feature = None
        self.splitting_threshold = None
        self.pred = None

    ######################################################################################################
    #
    #                                         Training Methods
    #
    ######################################################################################################

    def build_tree(self, 
        X: np.ndarray, 
        y: np.ndarray,
        depth: int = 1
    ) -> DecisionTreeRegressor:
        
        # Ensure y is a 2D array (general support for multi-output regression)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Store depth
        self.depth = depth

        # Check stopping criterion
        if self.check_stopping_criterion(X, y) is True:
            # Store the prediction
            self.pred = self.pred_storage_function(y)
            return self

        # Get the Decision Split and the Data Split
        data_splits, feature_index, threshold = self.find_best_split(X, y)
        if data_splits is None:
            # Store the prediction
            self.pred = self.pred_storage_function(y)
            return self
        self.splitting_feature = feature_index
        self.splitting_threshold = threshold

        # Store the prediction
        self.pred = self.pred_storage_function(y)

        # Build Left and Right (Children) Tree
        (X_left, y_left), (X_right, y_right) = data_splits 
        self.left_node = DecisionTreeRegressor(
            min_samples_per_node = self.min_samples_per_node,
            max_depth = self.max_depth,
            impurity_measure = self.impurity_measure
        ).build_tree(X_left, y_left, depth + 1)     # Add 1 to depth

        self.right_node = DecisionTreeRegressor(
            min_samples_per_node = self.min_samples_per_node,
            max_depth = self.max_depth,
            impurity_measure = self.impurity_measure
        ).build_tree(X_right, y_right, depth + 1)   # Add 1 to depth

        # Return the Node
        return self

    def check_stopping_criterion(self,
        X: np.ndarray, 
        y: np.ndarray
    ) -> bool:
        
        # Check max depth
        if self.depth > self.max_depth:
            # print("Stopping with max_depth: ", self.max_depth)
            return True
        
        # Check minimum number of samples per node
        if X.shape[0] <= self.min_samples_per_node:
            # print("Stopping with min_samples_per_node: ", X.shape[0])
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
        best_feature = None

        # Total Sums of y and y^2
        total_target_sum_y = np.sum(y, axis=0)          # Sum for each target
        total_target_sum_y2 = np.sum((y**2), axis=0)     # Sum of square for each target

        # Loop through features
        num_samples, num_features = X.shape[0], X.shape[1]
        for i in range(num_features):

            # Sorted samples by feature values
            X_i = X[:, i]
            sorted_indices = np.argsort(X_i)
            X_sorted = X[sorted_indices]
            thresholds = X_sorted[:, i]
            y_sorted = y[sorted_indices]

            # Track cumulative sums
            target_left_sum_y = 0
            target_left_sum_y2 = 0

            # Loop though thresholds
            # TO BE IMPROVED by using buckets / histograms
            # Find the threshold with the least impurity
            for j in range(num_samples-1):
                target_left_sum_y += y_sorted[j]
                target_left_sum_y2 += y_sorted[j]**2
                target_right_sum_y = total_target_sum_y - target_left_sum_y
                target_right_sum_y2 = total_target_sum_y2 - target_left_sum_y2

                # Skip identical thresholds
                if (j < (num_samples - 1)): # Only check up to 2nd to the last sample
                    if (thresholds[j] == thresholds[j+1]):
                        continue

                # Computer impurities
                left_weight = ((j + 1) / num_samples)
                right_weight = ((num_samples - j - 1) / num_samples)
                left_impurity = self.impurity_function(target_left_sum_y2, target_left_sum_y, j+1)
                right_impurity = self.impurity_function(target_right_sum_y2, target_right_sum_y, num_samples - j - 1)
                impurity = left_weight * left_impurity + right_weight * right_impurity

                # Compare with the best impurity
                if impurity < best_impurity:
                    left_mask, right_mask = sorted_indices[:(j+1)], sorted_indices[(j+1):]
                    left_data = ( X[left_mask], y[left_mask] )
                    right_data = ( X[right_mask], y[right_mask] )
                    best_split = (left_data, right_data)
                    best_impurity = impurity
                    best_feature = i
                    best_threshold = thresholds[j]

        # If no best split
        # Return (data split, feature/feature name, feature threshold)
        if best_split is None:
            return None, None, None

        # If same number of samples after best split, return None
        # TO BE IMPROVED
        # Do we need this?
        num_samples_left = best_split[0][0].shape[0]
        num_samples_right = best_split[1][0].shape[0]
        if (num_samples_left == 0) or (num_samples_right == 0):
            print("im here")
            return None, None, None

        # Normally we would compare the parent's impurities to the 
        # children's impurities. But mathematically, the children's impurities 
        # will always be less or equal to the parent's impurities

        return best_split, best_feature, best_threshold
    
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
    #                               Prediction and Prediction Storage Methods
    #
    ######################################################################################################    

    def predict(self, 
        X: np.ndarray
    ) -> np.ndarray:
        
        # If node is a leaf
        if self.left_node is None and self.right_node is None:
            return self.pred
        
        # Else traverse the tree by recursion
        is_left = X[self.splitting_feature] <= self.splitting_threshold
        if is_left:
            return self.left_node.predict(X)
        else:
            return self.right_node.predict(X)

    def _store_regression_pred(self, 
        y: np.ndarray   # float
    ) -> float:
        
        best_pred = np.mean(y, axis=0)  # Compute mean for each target
        
        return best_pred

    ######################################################################################################
    #
    #                                     Visualization Methods
    #
    ######################################################################################################    

    def visualize_decision_nodes(self, 
        depth: int = 0
    ) -> None:

        # Indentation to represent tree depth
        indent = " " * (4 * depth)

        # If the node is a leaf, print the prediction
        if self.left_node is None and self.right_node is None:
            print(f"{indent}Leaf Node: Prediction = {self.pred}")
            return

        # Print the current node's feature and threshold
        print(f"{indent}Node: Feature[{self.splitting_feature}] <= {self.splitting_threshold}")

        # Recur for left and right children
        if self.left_node is not None:
            print(f"{indent}  Left:")
            self.left_node.visualize_decision_nodes(depth + 1)
        
        if self.right_node is not None:
            print(f"{indent}  Right:")
            self.right_node.visualize_decision_nodes(depth + 1)

    ######################################################################################################
    #
    #                    Impurity Methods (Fast) | Supposedly methods of a Node Class
    #
    ######################################################################################################
    
    # For Regression
    # No Speedup
    # To be implemented
    def _compute_variance(self,
        target_sum_y2: float,
        target_sum_y: float,
        n
    ) -> float:
        
        # Biased Variance
        variance_per_target = (target_sum_y2 / n) - (target_sum_y / n) ** 2
        aggregated_variance = np.mean(variance_per_target)

        # Unbiased Variance
        # variance_per_target = (target_sum_y2 / n) - ((n-1) / n) * (target_sum_y / n) ** 2
        # aggregated_variance = np.mean(variance_per_target)

        return aggregated_variance

    ######################################################################################################
    #
    #                    Impurity Methods (Naive) | Supposedly methods of a Node Class
    #
    ######################################################################################################
    
    # For Regression
    def _compute_variance_naive(self,
        y: np.ndarray
    ) -> float:
        
        num_samples = y.shape[0]
        if num_samples == 0:
            return 0
        
        variance = np.mean( (y - np.mean(y)) ** 2 )

        return variance
    