import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================Constants============================
# Random data set for testing the decision tree implementation
URL = "https://raw.githubusercontent.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/master/diabetes_dataset.csv"
TASK = 'classification'     # Change it to 'regression' if you need regression tree
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 10

class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        prediction=None,
        impurity=None,
        num_samples=None,
        num_samples_per_class=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.impurity = impurity
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class

def variance(y):
    return np.var(y)

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

def impurity(y, task='classification'):
    if task == 'regression':
        return variance(y)
    else:
        return gini_impurity(y)


def best_split(X, y, min_samples_leaf=1, n_features=None, task='classification'):
    total_samples, num_features = X.shape
    best_impurity_score = float('inf')
    best_feature_index = None
    best_threshold = None

    if n_features is None:
        feature_indices = np.arange(num_features)
    else:
        if n_features > num_features:
            raise ValueError("n_features cannot be greater than the number of features in X")
        feature_indices = np.random.choice(
            num_features,
            size=n_features,
            replace=False
        )

    for feature_index in feature_indices:
        unique_values = np.unique(X[:, feature_index])
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        for threshold in thresholds:
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold

            if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                continue

            if len(y[left_indices]) < min_samples_leaf or len(y[right_indices]) < min_samples_leaf:
                continue

            # gini_left = gini_impurity(y[left_indices])
            # gini_right = gini_impurity(y[right_indices])
            # gini_split = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / total_samples

            impurity_left = impurity(y[left_indices], task=task)
            impurity_right = impurity(y[right_indices], task=task)
            impurity_split = (len(y[left_indices]) * impurity_left + len(y[right_indices]) * impurity_right) / total_samples

            if impurity_split < best_impurity_score:
                best_impurity_score = impurity_split
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold, best_impurity_score

def build_tree(X, y, depth=0, max_depth=3, num_classes=None, min_samples_leaf=1, n_features=None, task='classification', MIN_IMPURITY_DECREASE=1e-6):
    if num_classes is None and task == 'classification':
        # On the first call, set num_classes based on full dataset
        num_classes = len(np.unique(y))

    impurity_score = impurity(y, task=task)

    if impurity_score == 0.0:
        if task == 'classification':
            return Node(
                prediction=y[0],
                impurity=0.0,
                num_samples=len(y),
                num_samples_per_class=np.bincount(y, minlength=num_classes)
            )
        
        return Node(
            prediction=np.mean(y),
            impurity=0.0,
            num_samples=len(y)
        )
    
    if max_depth is not None and depth >= max_depth:
        if task == 'classification':
            majority_class = np.bincount(y).argmax()
            return Node(
                prediction=majority_class,
                impurity=impurity_score,
                num_samples=len(y),
                num_samples_per_class=np.bincount(y, minlength=num_classes)
            )

        return Node(
            prediction=np.mean(y),
            impurity=impurity_score,
            num_samples=len(y)
        )
    
    feature_index, threshold, best_impurity_score = best_split(X, y, min_samples_leaf=min_samples_leaf, n_features=n_features, task=task)

    if feature_index is None or (impurity_score - best_impurity_score) <= MIN_IMPURITY_DECREASE:
        if task == 'classification':
            majority_class = np.bincount(y).argmax()
            return Node(
                prediction=majority_class,
                impurity=impurity_score,
                num_samples=len(y),
                num_samples_per_class=np.bincount(y, minlength=num_classes)
            )

        return Node(
            prediction=np.mean(y),
            impurity=impurity_score,
            num_samples=len(y)
        )
    
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold

    left_subtree = build_tree(
        X[left_indices],
        y[left_indices],
        depth + 1,
        max_depth,
        num_classes,
        min_samples_leaf,
        n_features,
        task=task
    )
    right_subtree = build_tree(
        X[right_indices], 
        y[right_indices], 
        depth + 1, 
        max_depth, 
        num_classes, 
        min_samples_leaf,
        n_features,
        task=task
    )

    if task == 'classification':
        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree,
            impurity=impurity_score,
            num_samples=len(y),
            num_samples_per_class=np.bincount(y, minlength=num_classes)
        )
    
    return Node(
        feature_index=feature_index,
        threshold=threshold,
        left=left_subtree,
        right=right_subtree,
        impurity=impurity_score,
        num_samples=len(y)
    )


def print_tree(node, feature_names, depth=0, task='classification'):
    indent = "|   " * depth

    if task == 'classification':
        counts_str = ", ".join(
            f"class {i}: {count}"
            for i, count in enumerate(node.num_samples_per_class)
        )

    if node.prediction is not None:
        print(indent + f"Leaf:")
        print(indent)
        print(indent + f"  Predict: {node.prediction}")
        print(indent)
        print(indent + f"  Impurity: {node.impurity:.3f}")
        print(indent + f"  Samples: {node.num_samples}")
        if task == 'classification':
            print(indent + f"  Counts: {counts_str}")
        return

    feature_name = feature_names[node.feature_index]

    print(indent + f"{feature_name} <= {node.threshold:.2f}")
    # print(indent + f"  Impurity: {node.impurity:.3f}")
    # print(indent + f"  Samples: {node.num_samples}")
    # print(indent + f"  Counts: {counts_str}")

    print_tree(node.left, feature_names, depth + 1, task=task)

    print(indent + f"{feature_name} > {node.threshold:.2f}")
    print_tree(node.right, feature_names, depth + 1, task=task)

def impurity_tree(node, total_samples):
    if node.prediction is not None:
        return node.impurity * node.num_samples / total_samples
    left_impurity = impurity_tree(node.left, total_samples)
    right_impurity = impurity_tree(node.right, total_samples)
    return left_impurity + right_impurity

def predict(node, sample):
    if node.prediction is not None:
        return node.prediction
    if sample[node.feature_index] <= node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)
    
def predict_batch(node, X):
    if node.prediction is not None:
        return np.full(X.shape[0], node.prediction)
    
    left_indices = X[:, node.feature_index] <= node.threshold
    right_indices = ~left_indices

    predictions = np.empty(X.shape[0], dtype=float)
    predictions[left_indices] = predict_batch(node.left, X[left_indices])
    predictions[right_indices] = predict_batch(node.right, X[right_indices])
    return predictions
    
def evaluate_model(y_true, y_pred, task='classification'):
    if task == 'classification':
        accuracy = np.mean(y_true == y_pred)
        return {
            "accuracy": accuracy
        }
    else:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
    
def get_config(**kwargs):
    config = {
        "url": URL,
        "task": TASK,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF
    }

    for key in kwargs:
        if key not in config:
            raise ValueError(f"Invalid config key: {key}")
        
    config.update(kwargs)
    return config

def load_data(config):
    df = pd.read_csv(config["url"])
    print(df.head())

    feature_names = df.columns[:-1].tolist()
    data = df.values

    X = data[:, :-1]
    if config["task"] == "classification":
        y = data[:, -1].astype(int)
    else:
        y = data[:, -1].astype(float)
    
    return X, y, feature_names
    
def main():
    config = get_config()
    X, y, feature_names = load_data(config)


    tree = build_tree(X, y, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, task=config["task"])
    predictions = np.array([predict(tree, sample) for sample in X])


    print("\nDecision Tree Structure:")
    print_tree(tree, feature_names, task=config["task"])
    print("\nImpurity of the whole dataset:", impurity(y, task=config["task"]))
    print("Impurity of the tree:", impurity_tree(tree, len(y)))

    metrics = evaluate_model(y, predictions, task=config["task"])
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


    print()

if __name__ == "__main__":
    main()
