# copyright: https://github.com/enesozeren/machine_learning_from_scratch/blob/main/decision_trees/decision_tree.py
# for personal study purposes only

import numpy as np
from decision_tree import DecisionTree

class AdaBoostClassifer():
    """
    AdaBoost 
    """

    def __init__(self, n_base_learner=10) -> None:
        """
        n_base_learner: # of base learners in the model (base learners are DecisionTree with max_depth=1)
        """
        self.n_base_learner = n_base_learner
        return

    def _calcualte_amount_of_say(self, base_learner: DecisionTree, X: np.array, y: np.array) -> float:
        K = self.label_count
        preds = base_learner.predict(X)
        err = 1 - np.sum(preds==y) / preds.shape[0]
        amount_of_say = np.log((1 - err) / err) + np.log(K - 1)
        return amount_of_say

    def _fit_base_learner(self, X_bootstrapped: np.array, y_bootstrapped: np.array) -> DecisionTree:
        """
        Train a DecisionTree with depth 1 and return the model
        """
        base_learner = DecisionTree(max_depth=1)
        base_learner.fit(X_bootstrapped, y_bootstrapped)
        base_learner.amount_of_say = self._calcualte_amount_of_say(base_learner, self.X_train, self.y_train)
        return base_learner

    def _update_dataset(self, sample_weights: np.array) -> tuple:
        """
        Calculate bootstrapped samples wrt sample weights
        """
        n_samples = self.X_train.shape[0]
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
        X_bootstrapped = self.X_train[bootstrap_indices]
        y_bootstrapped = self.y_train[bootstrap_indices]
        return X_bootstrapped, y_bootstrapped

    def _calculate_sample_weights(self, base_learner: DecisionTree) -> np.array:
        preds = base_learner.predict(self.X_train)
        matches = (preds == self.y_train)
        not_matches = (~matches).astype(int)
        sample_weights = 1 / self.X_train.shape[0] * np.exp(base_learner.amount_of_say * not_matches)
        return sample_weights
    
    def fit(self, X_train: np.array, y_train: np.array) -> None:
        pass

    def _predict_scores_with_base_learner(self, X: np.array) -> list:
        pass

    def predict_proba(self, X: np.array) -> np.array:
        pass

    def predict(self, X: np.array) -> np.array:
        pass

    def 
