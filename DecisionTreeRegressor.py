from __future__ import annotations # For type hinting my own class!

import numpy as np
from typing import Tuple, Union, Optional
from copy import deepcopy

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
        self.null_direction = None

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
        data_splits, feature_index, threshold, null_direction = self.find_best_split(X, y)
        if data_splits is None:
            # Store the prediction and null direction
            self.pred = self.pred_storage_function(y)
            self.null_direction = null_direction
            return self
        self.splitting_feature = feature_index
        self.splitting_threshold = threshold

        # Store the prediction and null direction
        self.pred = self.pred_storage_function(y)
        self.null_direction = null_direction

        # Build Left and Right (Children) Tree
        (X_left, y_left), (X_right, y_right) = data_splits 
        self.left_node = DecisionTreeRegressor(
            min_samples_per_node = self.min_samples_per_node,
            max_depth = self.max_depth,
            impurity_measure = self.impurity_measure,
            num_targets = self.num_targets
        ).build_tree(X_left, y_left, depth + 1)     # Add 1 to depth

        self.right_node = DecisionTreeRegressor(
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
        float,                                                                  # Threshold
        Union[None, str]                                                        # Null direction / Optional[str]
    ]:
        
        # Find the best split
        best_split = None
        best_impurity = np.inf
        best_threshold = None
        best_feature = None
        best_null_direction = None

        # Total Sums of y and y^2
        sum_y = np.sum(y, axis=0)               # Sum for each target
        sum_y2 = np.sum((y**2), axis=0)         # Sum of square for each target

        # Loop through features
        num_samples, num_features = X.shape[0], X.shape[1]
        for i in range(num_features):

            # Get the NULL and Non-null dataset
            X_i = X[:, i]
            null_idx = np.where(np.isnan(X_i))[0]
            nonnull_idx = list(set(range(num_samples)) - set(null_idx))

            # Separate the NULL and Non=null dataset
            X_i_null, X_null, y_null = X_i[null_idx], X[null_idx], y[null_idx]
            X_i_nonnull, X_nonnull, y_nonnull = X_i[nonnull_idx], X[nonnull_idx], y[nonnull_idx]
            num_null, num_nonnull = X_i_null.shape[0], X_i_nonnull.shape[0]

            # Sort the thresholdss of Non-NULL X_i
            sorted_indices = np.argsort(X_i_nonnull)
            thresholds, y_sorted = X_i_nonnull[sorted_indices], y_nonnull[sorted_indices]

            # Check if there are null feature values and set the correct direction pools
            null_directions = ['left', 'right'] if num_null > 0 else [None]

            # Precompute the Total Sums of y and y^2 of null values
            nullsum_y = np.array([0.]*self.num_targets) if num_null == 0 else np.sum(y_null, axis=0) 
            nullsum_y2 = np.array([0.]*self.num_targets) if num_null == 0 else np.sum(y_null**2, axis=0) 

            # Loop though null spliting directions
            for null_direction in null_directions:

                # Left sum trackers
                left_sum_y = nullsum_y if null_direction == 'left' else np.array([0.]*self.num_targets)
                left_sum_y2 = nullsum_y2 if null_direction == 'left' else np.array([0.]*self.num_targets)

                # Right sum trackers
                right_sum_y = deepcopy(sum_y) - nullsum_y if null_direction == 'left' else deepcopy(sum_y)
                right_sum_y2 = deepcopy(sum_y2) - nullsum_y2 if null_direction == 'left' else deepcopy(sum_y2)

                # Loop though non-null values
                for j in range(num_nonnull):

                    # Skip identical thresholds
                    # Skip the first element because there is no j-1 element
                    if (j < num_nonnull) and j != 0:    # Only check up to the 2nd to the last sample
                        if thresholds[j-1] == thresholds[j]:
                            # Target Sum Trackers
                            # Moved the sum trackers to the end and continues of for loop to accomodate Null edge-cases
                            left_sum_y += y_sorted[j]
                            left_sum_y2 += y_sorted[j]**2
                            right_sum_y = sum_y - left_sum_y
                            right_sum_y2 = sum_y2 - left_sum_y2
                            continue

                    # Compute left and right samples
                    left_num_samples = num_null + j if null_direction =='left' else j
                    right_num_samples = num_samples - left_num_samples

                    # # Skip empty splits
                    # if left_num_samples == 0 or right_num_samples == 0:
                    #     continue

                    # Compute impurities
                    left_impurity = self.impurity_function(left_sum_y2, left_sum_y, left_num_samples)
                    right_impurity = self.impurity_function(right_sum_y2, right_sum_y, right_num_samples)
                    impurity = (
                        left_num_samples / num_samples * left_impurity +
                        right_num_samples / num_samples * right_impurity
                    )

                    # Compare with the best impurity
                    if impurity < best_impurity:
                        left_mask, right_mask = sorted_indices[:j], sorted_indices[j:]

                        # Prepare the data split
                        if null_direction == 'left':
                            left_data = (
                                np.concatenate([X_null, X_nonnull[left_mask]], axis=0),     # X : [n, f] || [n, f]
                                np.concatenate([y_null, y_nonnull[left_mask]], axis=0)      # y : [n, ] || [n, ]
                            )
                            right_data = (X_nonnull[right_mask], y_nonnull[right_mask])
                        else:
                            left_data = (X_nonnull[left_mask], y_nonnull[left_mask])
                            right_data = (
                                np.concatenate([X_null, X_nonnull[right_mask]], axis=0),     # X : [n, f] || [n, f]
                                np.concatenate([y_null, y_nonnull[right_mask]], axis=0)      # y : [n, ] || [n, ]
                            )

                        best_split = (left_data, right_data)
                        best_impurity = impurity
                        best_feature = i
                        best_threshold = thresholds[j]
                        best_null_direction = null_direction

                    # Target Sum Trackers
                    # Moved the sum trackers to the end of for loop to accomodate Null edge-cases
                    left_sum_y += y_sorted[j]
                    left_sum_y2 += y_sorted[j]**2
                    right_sum_y = sum_y - left_sum_y
                    right_sum_y2 = sum_y2 - left_sum_y2

        # If no best split
        # Return (data split, feature/feature name, feature threshold)
        if best_split is None:
            return None, None, None, None

        # If same number of samples after best split, return None
        # TO BE IMPROVED
        # Do we need this?
        num_samples_left = best_split[0][0].shape[0]
        num_samples_right = best_split[1][0].shape[0]
        if (num_samples_left == 0) or (num_samples_right == 0):
            return None, None, None, None

        # Normally we would compare the parent's impurities to the 
        # children's impurities. But mathematically, the children's impurities 
        # will always be less or equal to the parent's impurities

        return best_split, best_feature, best_threshold, best_null_direction
    
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
        
        X_i = X[self.splitting_feature]
        
        # If null value
        if np.isnan(X_i):
            if self.null_direction is None:
                pass  # Proceed to real numbers
                        # Might be wrong!
            else:
                if self.null_direction == 'left':
                    return self.left_node.predict(X)
                else:   # Right
                    return self.right_node.predict(X)

        # Else traverse the tree by recursion
        is_left = X_i <= self.splitting_threshold
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

        # Print the current node's feature and threshold, and null direction
        print(f"{indent}Node: Feature[{self.splitting_feature}] <= {self.splitting_threshold} | Null Direction: {self.null_direction}")

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
        
        # If empty set (edge-case)
        if n == 0:
            return 0

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
    