from __future__ import annotations  # For type hinting my own class!
from DecisionTreeRegressor import DecisionTreeRegressor
import numpy as np
from typing import List, Tuple, Dict
from copy import copy
from multiprocessing import Pool

# DCG function
def DCG(
    y: np.ndarray
) -> float:
    
    n = y.shape[0]
    denominators = np.log2(np.arange(n) + 2)
    y_ = y.reshape(denominators.shape)
    dcg = np.sum(y_ / denominators)
    return dcg

# Parallelized LambdaMART Class
class LambdaMART():
    def __init__(self,
        num_trees: int = 5,
        learning_rate: float = 0.01,
        min_samples_per_node: int = 5,
        max_depth: int = 5,
        impurity_measure: str = 'variance',
        num_targets: int = 1
    ) -> None:
        
        # Grandient Boosting Hyperparameters
        self.num_trees = num_trees
        self.learning_rate = learning_rate

        # Decision Tree Hyperparameters
        self.min_samples_per_node = min_samples_per_node
        self.max_depth = max_depth
        self.num_targets = num_targets

        # Impurity Measure
        self.impurity_measure = impurity_measure

        # the forest
        self.forest = []

    def group_queries(self,
        q: List[int],
        x: np.ndarray,
        y: np.ndarray
    ) -> Dict[np.ndarray]:
        
        # Temporary concatenation
        temp = np.concatenate([x, y.reshape(-1,1)], axis=1)
        idxs = [0] + q
        grouped_xy = {}
        for qid in range(1, len(q) + 1):
            left = sum(idxs[:qid])
            right = sum(idxs[:(qid + 1)])
            grouped_xy[qid-1] = temp[left:right]

        return grouped_xy

    def get_pairs(self,
        y: np.ndarray
    ) -> List[Tuple[int, int]]:
    
        n_q = y.shape[0]
        pairs = [(i, j) for i in range(n_q) for j in range(i + 1, n_q) if y[i] > y[j]]

        return pairs

    def swap_y(self,
        y: np.ndarray,
        i: int, 
        j: int
    ) -> np.ndarray:
        
        y_copy = copy(y)
        y_copy[i], y_copy[j] = y_copy[j], y_copy[i]

        return y_copy

    def compute_lambda_for_group(self,
        group_xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # Numbero of samples per group
        n_q = group_xy.shape[0]  
        grouped_x, grouped_y = group_xy[:, :-1], group_xy[:, -1]
        grouped_y = grouped_y.reshape(-1)

        # Ideal Ranking and iDCG
        sorted_idx = np.argsort(grouped_y)[::-1]
        ideal_x, ideal_y = grouped_x[sorted_idx], grouped_y[sorted_idx]
        iDCG = DCG(ideal_y)

        # Compute the model scores for each group item (parallelizable)
        scores = np.array([self.predict(ideal_x[i]) for i in range(n_q)])
        scores = scores.reshape(-1)

        # Initialize lambdas per group
        lambda_per_group = np.zeros(n_q)

        # Pairwise lambda computation
        pairs = self.get_pairs(ideal_y)
        for pair in pairs:
            i, j = pair
            
            # Compute deltaDCG
            swapped_y = self.swap_y(ideal_y, i, j)
            nDCG = DCG(swapped_y) / iDCG
            delta_dcg = np.abs(nDCG - 1)

            # Compute BCE gradient
            score_diff = scores[i] - scores[j]
            lambda_ij = 1/(1 + np.exp(score_diff)) * delta_dcg

            # Accumulate Gradients
            lambda_per_group[i] += lambda_ij
            lambda_per_group[j] -= lambda_ij

        return lambda_per_group, ideal_x, ideal_y

    def evaluate_ndcg_of_group(self,
        group_xy: np.ndarray
    ) -> float:
        
        # Number of samples per group
        n_q = group_xy.shape[0]  
        x, y = group_xy[:, :-1], group_xy[:, -1]
        y = y.reshape(-1)

        # Predict scores for the group
        scores = np.array([self.predict(x[i]) for i in range(n_q)])
        scores = scores.reshape(-1)

        # Sort indices based on scores (predicted ranking)
        sorted_score_idx = np.argsort(scores)[::-1]
        
        # Compute DCG for the predicted ranking
        dcg = DCG(y[sorted_score_idx])
        
        # Compute iDCG (ideal ranking by sorting true relevance scores in descending order)
        ideal_y = np.sort(y)[::-1]
        idcg = DCG(ideal_y)

        # Avoid division by zero if iDCG is zero
        if idcg == 0:
            return 0.0

        # Return NDCG
        return dcg / idcg

    def build_forest(self,
        q: List[int],
        x: np.ndarray,
        y: np.ndarray
    ) -> LambdaMART:
        
        num_samples = y.shape[0]
        M = len(q)  # num groups
        for t in range(self.num_trees):
            print(f'Training Tree {t + 1}')
            # Precompute Lambdas
            lambdas = []
            ideal_xs = []
            ideal_ys = []

            # Group the queries
            grouped_xy = self.group_queries(q, x, y)

            # Parallelize the lambda computation for each group
            print('Precomputing Lambdas')
            with Pool() as pool:
                # Pass grouped data to pool.map
                results = pool.map(self.compute_lambda_for_group, list(grouped_xy.values()))

            # Unpack results
            for lambda_per_group, ideal_x, ideal_y in results:
                lambdas.append(lambda_per_group)
                ideal_xs.append(ideal_x)
                ideal_ys.append(ideal_y)

            # Flatten lambdas, x, y
            flattened_lambda = np.concatenate(lambdas, axis=0)
            flattened_x = np.concatenate(ideal_xs, axis=0)
            flattened_y = np.concatenate(ideal_ys, axis=0)

            # Train Regressor on Lambdas
            print('Training Regressor')
            tree = DecisionTreeRegressor(
                min_samples_per_node=self.min_samples_per_node,
                max_depth=self.max_depth,
                impurity_measure=self.impurity_measure,
                num_targets=1
            ).build_tree(flattened_x, flattened_lambda)

            # Store the decision tree
            self.forest.append(tree)

            # Repoint the new x, y
            x = flattened_x
            y = flattened_y

            # (Optional) Evaluate nDCG
            eval_grouped_xy = self.group_queries(q, x, y)
            print('Evaluating NDCG')
            with Pool() as pool:
                results = pool.map(self.evaluate_ndcg_of_group, list(eval_grouped_xy.values()))

            print(f'NDCG: {np.mean(results)}')
            print("="*40)

    def predict(self, X: np.ndarray) -> float:
        if len(self.forest) == 0:
            return 0

        pred = self.forest[0].predict(X)
        curr_num_trees = len(self.forest)
        for t in range(1, curr_num_trees):
            pred += self.learning_rate * self.forest[t].predict(X)

        return pred
