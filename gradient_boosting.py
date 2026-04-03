import numpy as np
import decision_tree
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

TASK = 'classification'  # Change to 'regression' for regression tasks
config = decision_tree.get_config(task=TASK)

X, y, feature_names = decision_tree.load_data(config)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

def predict_classes(logits, cutoff=0.5):
    p = sigmoid(logits)
    return (p >= cutoff).astype(int)

def loss_function_reg(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_loss_reg(y_true, y_pred):
    return -2 * (y_true - y_pred)

def loss_function_class(y_true, logits, eps=1e-12):
    p = sigmoid(logits)
    return -np.mean((y_true * np.log(p + eps)) + ((1 - y_true) * np.log(1 - p + eps)))

def gradient_loss_class(y_true, logits):
    y_pred = sigmoid(logits)
    return y_pred - y_true



def gradient_boosting(X, y, n_trees=100, lr=0.1, max_depth=3, min_samples_leaf=1, eps=1e-12):
    n_samples, n_features = X.shape
    trees = []
    if config['task'] == 'classification':
        print("\nUsing classification loss and gradient")
        p = np.clip(np.mean(y), eps, 1 - eps)
        initial_pred = logit(p)
        y_pred = np.full(n_samples, initial_pred)
        gradient_loss = gradient_loss_class
    else:
        print("\nUsing regression loss and gradient")
        initial_pred = np.mean(y)
        y_pred = np.full(n_samples, initial_pred)
        gradient_loss = gradient_loss_reg

    for i in range(n_trees):
        residuals = -gradient_loss(y, y_pred)
        tree = decision_tree.build_tree(
            X, 
            residuals, 
            # feature_names, 
            max_depth=max_depth, 
            min_samples_leaf=min_samples_leaf, 
            task='regression')
        update = decision_tree.predict_batch(tree, X)
        y_pred += lr * update
        trees.append(tree)
        if np.max(np.abs(update)) < 1e-6:
            break

    return initial_pred, trees

def evaluate_gradient_boosting(trees, X_test, y_test, initial_pred, lr=0.1):
    y_pred = np.full_like(y_test, initial_pred, dtype=float)
    for tree in trees:
        y_pred += lr * decision_tree.predict_batch(tree, X_test)

    if config['task'] == 'classification':
        y_pred_classes = predict_classes(y_pred)
        cm = confusion_matrix(y_test, y_pred_classes)
        acc = accuracy_score(y_test, y_pred_classes)
        prec = precision_score(y_test, y_pred_classes)
        rec = recall_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes)
        class_report = classification_report(y_test, y_pred_classes)
        return {
            'confusion_matrix': cm,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'classification_report': class_report
        }
    else:
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2_score = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        return {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2_score
        }

def print_tree_simple(node, feature_names, depth=0):
    indent = "  " * depth

    # Leaf node
    if node.prediction is not None:
        print(f"{indent}Leaf -> value: {node.prediction:.4f}")
        return

    feature = feature_names[node.feature_index]
    threshold = node.threshold

    print(f"{indent}if {feature} <= {threshold:.4f}:")
    print_tree_simple(node.left, feature_names, depth + 1)

    print(f"{indent}else:  # {feature} > {threshold:.4f}")
    print_tree_simple(node.right, feature_names, depth + 1)


def main():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    initial_pred, trees = gradient_boosting(X_train, y_train, n_trees=100, lr=0.1, max_depth=3, min_samples_leaf=1)

    # # Uncomment this if you want to see the trees
    # for i, tree in enumerate(trees):
    #     if i % 10 == 9:
    #         print(f"\ntree {i+1}:")
    #         print_tree_simple(tree, feature_names)

    results = evaluate_gradient_boosting(trees, X_test, y_test, initial_pred, lr=0.1)

    print("Gradient Boosting Results:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value}")


if __name__ == "__main__":
    main()