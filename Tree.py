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

        # And Prediction Storage Functions
        pred_storage_functions = {
            'gini': self._store_classification_pred,
            'entropy': self._store_classification_pred,
            'variance': self._store_regression_pred
        }
        self.pred_storage_function = pred_storage_functions[impurity_measure]

        # Number of targets for counting purpose
        self.num_targets =  num_targets # to be determined

        # Class Variables
        self.depth = 0

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
            self.pred = self.pred_storage_function(y)
            return self

        # Get the Decision Split and the Data Split
        data_splits, feature_index, threshold = self.find_best_split(X, y)
        if data_splits is None:
            return None
        self.splitting_feature = feature_index
        self.splitting_threshold = threshold

        # Store the prediction
        self.pred = self.pred_storage_function(y)

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
        best_feature = None

        num_samples, num_feautures = X.shape[0], X.shape[1]
        for i in range(num_feautures):

            # TO BE IMPROVED by using buckets / histograms
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
                    best_feature = i
                    best_threshold = threshold

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

    def predict(self, X):
        
        # If node is a leaf
        if self.left_node is None and self.right_node is None:
            return self.pred
        
        # Else traverse the tree by recursion
        is_left = X[self.splitting_feature] <= self.splitting_threshold
        if is_left:
            return self.left_node.predict(X)
        else:
            return self.right_node.predict(X)

    def _store_classification_pred(self, y):
        target_dist = self._compute_target_dist(y)
        best_class = np.argmax(target_dist)

        return best_class

    def _store_regression_pred(self, y):
        best_pred = np.mean(y)
        
        return best_pred


    ######################################################################################################
    #
    #                                     Visualization Methods
    #
    ######################################################################################################    

    def visualize_decision_nodes(self, depth: int = 0):

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
    