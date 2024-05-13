# copyright: https://github.com/enesozeren/machine_learning_from_scratch/blob/main/decision_trees/decision_tree.py
# for personal study purposes only


import numpy as np
from decision_tree import DecisionTree

class RandomForestClassifier():
    """
    Random Forest Classifier
    """

    def __init__(
            self,
            n_base_learner=10,
            max_depth=5,
            min_samples_leaf=1,
            min_information_gain=0,
            numb_of_features_splitting=None,
            boostrap_sample_size=None,
    ) -> None:
        self.n_base_learner = n_base_learner
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.boostrap_sample_size = boostrap_sample_size

    def _create_bootstrap_samples(self, X, Y) -> tuple:
        """
        Create bootstrap samples for each base learner
        """
        bootstrap_samples_X = []
        bootstrap_samples_Y = []

        for i in range(self.n_base_learner):
            
            if not self.boostrap_sample_size:
                self.boostrap_sample_size = X.shape[0]
            
            sampled_idx = np.random.choice(X.shape[0], size=self.boostrap_sample_size, replace=True)
            bootstrap_samples_X.append(X[sampled_idx])
            bootstrap_samples_Y.append(Y[sampled_idx])
        
        return bootstrap_samples_X, bootstrap_samples_Y


    def fit(self, X_train: np.array, Y_train: np.array) -> None:
        """"
        Train the model with given X and Y datsets
        """

        bootstrap_samples_X, bootstrap_samples_Y = self._create_bootstrap_samples(X_train, Y_train)

        self.base_learner_list = []
        for base_learner_idx in range(self.n_base_learner):
            base_learner = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_information_gain=self.min_information_gain,
                numb_of_features_splitting=self.numb_of_features_splitting,
            )
            base_learner.fit(
                bootstrap_samples_X[base_learner_idx],
                bootstrap_samples_Y[base_learner_idx],
            )
            self.base_learner_list.append(base_learner)

        # calculate feature importance
        self.feature_importances = self._calculate_rf_feature_importance(self.base_learner_list)
        return
    

    def _predict_proba_with_base_learners(self, X_set: np.array) -> list:
        """
        Predict probabilities for one row (one observation)
        """
        pred_prob_list = []
        for base_learner in self.base_learner_list:
            pred_prob_list.append(base_learner.predict_proba(X_set))
        return pred_prob_list


    def predict_proba(self, X_set: np.array) -> list:
        """
        Predict probabilities for all rows
        """
        pred_probs = []
        base_learners_pred_probs = self._predict_proba_with_base_learners(X_set)

        # average the predicted probabilities of base learners
        for obs in range(X_set.shape[0]):
            base_learner_probs_for_obs = [a[obs] for a in base_learners_pred_probs]

            # calculate the average for each index
            obs_average_pred_probs = np.mean(base_learner_probs_for_obs, axis=0)
            pred_probs.append(obs_average_pred_probs)

        return pred_probs



    def predict(self, X_set: np.array) -> np.array:
        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        return preds

    def _calculate_rf_feature_importance(self, base_learners):
        feature_importance_dict_list = []
        for base_learner in base_learners:
            feature_importance_dict_list.append(base_learner.feature_importances)
        
        feature_importance_list = [list(x.values()) for x in feature_importance_dict_list]
        average_feature_importance = np.mean(feature_importance_list, axis=0)

        return average_feature_importance


