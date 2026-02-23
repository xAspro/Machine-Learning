import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Sample data
# # Loves Popcorn, Loves Soda, Age, Loves Movie
# # True,True,7,False
# # True,False,12,False
# # False,True,18,True
# # False,True,35,True
# # True,True,38,True
# # True,False,50,False
# # False,False,83,False

# feature_names = ["Loves Popcorn", "Loves Soda", "Age", "Loves Movie"]
# data = np.array([
#     [1, 1, 7, 0],
#     [1, 0, 12, 0],
#     [0, 1, 18, 1],
#     [0, 1, 35, 1],
#     [1, 1, 38, 1],
#     [1, 0, 50, 0],
#     [0, 0, 83, 0]
# ])


# Random data set for testing the decision tree implementation
URL = "https://raw.githubusercontent.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/master/diabetes_dataset.csv"
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 10

df = pd.read_csv(URL)


print(df.head())

feature_names = df.columns[:-1].tolist()
data = df.values


class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        prediction=None,
        gini=None,
        num_samples=None,
        num_samples_per_class=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)


def best_split(X, y, min_samples_leaf=1):
    total_samples, num_features = X.shape
    best_gini = float('inf')
    best_feature_index = None
    best_threshold = None

    for feature_index in range(num_features):
        unique_values = np.unique(X[:, feature_index])
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        for threshold in thresholds:
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold

            if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                continue

            if len(y[left_indices]) < min_samples_leaf or len(y[right_indices]) < min_samples_leaf:
                continue

            gini_left = gini_impurity(y[left_indices])
            gini_right = gini_impurity(y[right_indices])
            gini_split = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / total_samples

            if gini_split < best_gini:
                best_gini = gini_split
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold

def build_tree(X, y, depth=0, max_depth=3, num_classes=None, min_samples_leaf=1):
    if num_classes is None:
        # On the first call, set num_classes based on full dataset
        num_classes = len(np.unique(y))

    # Leaf node: all labels the same
    # Doesnt work if min_samples_leaf is set to a value greater than 1, 
    # because it will stop splitting before all labels are the same
    if len(set(y)) == 1:
        return Node(
            prediction=y[0],
            gini=0.0,
            num_samples=len(y),
            num_samples_per_class=np.bincount(y, minlength=num_classes)
        )
    
    if depth >= max_depth:
        majority_class = np.bincount(y).argmax()
        return Node(
            prediction=majority_class,
            gini=gini_impurity(y),
            num_samples=len(y),
            num_samples_per_class=np.bincount(y, minlength=num_classes)
        )
    
    feature_index, threshold = best_split(X, y, min_samples_leaf=min_samples_leaf)

    if feature_index is None:
        majority_class = np.bincount(y).argmax()
        return Node(
            prediction=majority_class,
            gini=gini_impurity(y),
            num_samples=len(y),
            num_samples_per_class=np.bincount(y, minlength=num_classes)
        )
    
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold

    left_subtree = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth, num_classes, min_samples_leaf)
    right_subtree = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth, num_classes, min_samples_leaf)

    return Node(
        feature_index=feature_index,
        threshold=threshold,
        left=left_subtree,
        right=right_subtree,
        gini=gini_impurity(y),
        num_samples=len(y),
        num_samples_per_class=np.bincount(y, minlength=num_classes)
    )


def print_tree(node, feature_names, depth=0):
    indent = "|   " * depth

    counts_str = ", ".join(
        f"class {i}: {count}"
        for i, count in enumerate(node.num_samples_per_class)
    )

    if node.prediction is not None:
        print(indent + f"Leaf:")
        print(indent)
        print(indent + f"  Predict: {node.prediction}")
        print(indent)
        print(indent + f"  Gini: {node.gini:.3f}")
        print(indent + f"  Samples: {node.num_samples}")
        print(indent + f"  Counts: {counts_str}")
        return

    feature_name = feature_names[node.feature_index]

    print(indent + f"{feature_name} <= {node.threshold:.2f}")
    # print(indent + f"  Gini: {node.gini:.3f}")
    # print(indent + f"  Samples: {node.num_samples}")
    # print(indent + f"  Counts: {counts_str}")

    print_tree(node.left, feature_names, depth + 1)

    print(indent + f"{feature_name} > {node.threshold:.2f}")
    print_tree(node.right, feature_names, depth + 1)

def gini_tree(node, total_samples):
    if node.prediction is not None:
        return node.gini * node.num_samples / total_samples
    left_gini = gini_tree(node.left, total_samples)
    right_gini = gini_tree(node.right, total_samples)
    return left_gini + right_gini

def predict(node, sample):
    if node.prediction is not None:
        return node.prediction
    if sample[node.feature_index] <= node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)
    


X = data[:, :-1]
y = data[:, -1].astype(int)
tree = build_tree(X, y, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF)
predictions = np.array([predict(tree, sample) for sample in X])
accuracy = np.mean(predictions == y)


print("\nDecision Tree Structure:")
print_tree(tree, feature_names)
print("\nGini impurity of the whole dataset:", gini_impurity(y))
print("Gini impurity of the tree:", gini_tree(tree, len(y)))
print(f"Tree accuracy: {accuracy:.4f}")

print()
