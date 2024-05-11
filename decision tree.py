# copyright: https://github.com/enesozeren/machine_learning_from_scratch/blob/main/decision_trees/decision_tree.py
# for personal study purposes only

import numpy as np
from collections import Counter
from treenode import TreeNode

class DecisionTree():
    """
    Decision Tree Classifier
    """

    def __init__(
            self,
            max_depth=4,
            min_samples_leaf=1,
            min_information_gain=0.0,
            numb_of_features_splitting=None,
            amount_of_say=None,
    ) -> None:
        """
        setting class hyperparameters
        @max_depth: (int) max depth of the tree
        @min_samples_leaf: (int) min # of samples required to be in a leaf
        @min_information_gain: (float) min info gain required to make a further split
        @num_of_features_splitting: (str) 
            when splitting 
            if sqrt then sqrt(# of features) considered
            if log then log(# of features) considered
            else all features are considered
        @amount_of_say: (float) used for Adaboost algorithm
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say

    def _entropy(self, class_probabilities: list) -> float:
        return sum(
            [-p * np.log2(p) for p in class_probabilities if p > 0]
        )

    def _class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels: list) -> float:
        return self._entropy(self._class_probabilities(labels))

    def _partition_entropy(self, subsets: list) -> float:
        """
        @subsets: list of label lists (eg [[1,0,0]], [[1,1,1]])
        """
        total_count = sum([len(subset) for subset in subsets])
        return sum(
            [self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets]
        )

    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2

    def _select_feature_to_use(self, data: np.array) -> list:
        """
        Randomly select the features to use while splitting wrt hyperparameter numb_of_features_splitting
        """
        feature_idx = list(range(data.shape[1]-1)) # indices of all features
        if self.numb_of_features_splitting == 'sqrt':
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.numb_of_features_splitting == 'log':
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            feature_idx_to_use = feature_idx
        return feature_idx_to_use


    def _find_best_split(self, data: np.array) -> tuple:
        """
        Find the best split (with lowest entropy) given data
        Return: 2 splitted groups and split information
        """
        min_part_entropy = 1e9
        feature_idx_to_use = self._select_feature_to_use(data)

        for idx in feature_idx_to_use:
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25)) # chop feature into 4 quartiles
            for feature_val in feature_vals:
                g1, g2 = self._split(data, idx, feature_val)
                part_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]]) # last column stores Y
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2
        
        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy
    


    def _find_label_probs(self, data: np.array) -> np.array:
        
        labels_as_integers = data[:, -1].astype(int)
        total_labels = len(labels_as_integers) # total number of labels
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float) # ratios (probabilities) for each label

        # populate the label_probabilities array based on specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers==i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels
        
        return label_probabilities
    


    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        """
        Recursive, depth first creation algorithm
        """
        
        # check if max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        
        # find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)

        # find label probs for the node
        label_probabilities = self._find_label_probs(data)

        # calculate information gain
        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        
        # create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # check min_samples_leaf
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # check min_information_gain
        elif information_gain < self.min_information_gain:
            return node
        
        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)

        return node


    def _predict_one_sample(self, X: np.array) -> np.array:
        """
        Return prediction for 1 dim array
        """

        node = self.tree
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return pred_probs
    

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Train the model with X and Y dataset
        """

        # concat features and label
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate([X_train, np.reshape(Y_train, (-1, 1))], axis=1)

        # create the tree
        self.tree = self._create_tree(data=train_data, current_depth=0)

        # calculate feature importance
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)

        # normalize feature importance values
        self.feature_importances = {k: v / total for total in (sum(self.feature_importances.values()), ) for k, v in self.feature_importances.items()}

        return
    

    def predict_proba(self, X_set: np.array) -> np.array:
        """
        Return the predicted probs for a given data set
        """
        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """
        Return the predicted labels for a given data set
        """
        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        return preds

    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '->' + node.node_def())
            self._print_recursive(node.right, level + 1)
        return

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)
        return

    def _calculate_feature_importance(self, node):
        """
        Calculate the feature importance by visiting each node in the tree recursively
        """
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)
        return 
    


if __name__ == '__main__':

    """
    toy dataset:
        x1: height in m
        x2: weight in kg
        y: diabetes or not

        y = bmi (kg/m2) + N(0, 5) >= 32
    """
    N = 500
    x1 = np.random.normal(loc=1.75, scale=0.2, size=N)
    x2 = np.random.normal(loc=70, scale=20, size=N)
    eps = np.random.normal(loc=0, scale=5, size=N)

    bmi = x2 / (x1**2)
    y_raw = bmi + eps # add noise to bmi

    X_train = np.column_stack((x1, x2))
    Y_train = np.array([1 if y >= 32 else 0 for y in y_raw])

    print(y_raw)
    
    decision_tree = DecisionTree()
    decision_tree.train(X_train, Y_train)
    decision_tree.print_tree()

    