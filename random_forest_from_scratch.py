import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import decision_tree
import itertools
from sklearn.model_selection import train_test_split

# # Data
# # Chest Pain, Good Blood Circulation, Blocked Arteries, Weight, Heart Disease
# # True, True, True, 180, True
# # False, False, False, 125, False
# # True, False, True, 167, True
# # True, True, False, 210, False

# column_names = ["Chest Pain", "Good Blood Circulation", "Blocked Arteries", "Weight", "Heart Disease"]
# data = np.array([
#     [1, 1, 1, 180, 1],
#     [0, 0, 0, 125, 0],
#     [1, 0, 1, 167, 1],
#     [1, 1, 0, 210, 0]
# ])

# Random data set for testing the decision tree implementation
URL = "https://raw.githubusercontent.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/master/diabetes_dataset.csv"

df = pd.read_csv(URL)
print(df.head())

feature_names = df.columns[:-1].tolist()
data = df.values

X = data[:, :-1]
y = data[:, -1].astype(int)

def bootstrapping(data, n_samples=None):
    if n_samples is None:
        n_samples = data.shape[0]
    n = data.shape[0]
    indices = np.random.choice(n, size=n_samples, replace=True)
    return data[indices]

def random_forest(data, n_trees=10, n_samples=None, n_features=None, max_depth=decision_tree.MAX_DEPTH, min_samples_leaf=decision_tree.MIN_SAMPLES_LEAF):
    trees = []
    for i in range(n_trees):
        # if (i + 1) % 50 == 0:
        #     print(f"\nBuilding tree {i+1}/{n_trees}\ndata.shape[1]: {data.shape[1]}, n_samples: {n_samples}, n_features: {n_features}, max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}")
        bootstrapped_data = bootstrapping(data, n_samples)
        tree = decision_tree.build_tree(bootstrapped_data[:, :-1], bootstrapped_data[:, -1].astype(int), max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_features=n_features)
        trees.append(tree)
    return trees

def predict_forest(trees, sample):
    predictions = [decision_tree.predict(tree, sample) for tree in trees]
    # print("Prediction: ", np.bincount(predictions, minlength=2))
    return np.bincount(predictions).argmax().astype(int)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp],
                        [fn, tp]])

def specificity(cm):
    tn, fp = cm[0]
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def sensitivity(cm):
    fn, tp = cm[1]
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def precision(cm):
    fp, tp = cm[0][1], cm[1][1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def f1_score(cm):
    prec = precision(cm)
    sens = sensitivity(cm)
    return 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0

def evaluate_random_forest(n_trees=100, n_features=3, max_depth=5, min_samples_leaf=5, return_metrics=False):
    print("\nBuilding random forest...")
    print(f"n_trees: {n_trees}, n_features: {n_features}, max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    data_train = np.hstack((X_train, y_train.reshape(-1, 1)))
    forest = random_forest(data_train, n_trees=n_trees, n_features=n_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    predictions = np.array([predict_forest(forest, sample) for sample in X_test])
    accuracy = np.mean(predictions == y_test)
    # print(f"Random Forest Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    # print(f"Specificity: {specificity(cm):.4f}")
    # print(f"Sensitivity: {sensitivity(cm):.4f}")
    # print(f"Precision: {precision(cm):.4f}")
    # print(f"F1 Score: {f1_score(cm):.4f}")

    if return_metrics:
        return {
            "accuracy": accuracy,
            "specificity": specificity(cm),
            "sensitivity": sensitivity(cm),
            "precision": precision(cm),
            "f1_score": f1_score(cm)
        }
    
def main():

    # After multiple runs, these values seem to give the best results
    n_tree_values = [100]
    n_feature_values = [3]
    max_depth_values = [5]
    min_samples_leaf_values = [5]


    results = []

    for n_trees, n_features, max_depth, min_samples_leaf in itertools.product(
        n_tree_values,
        n_feature_values,
        max_depth_values,
        min_samples_leaf_values
    ):
        result = evaluate_random_forest(
            n_trees=n_trees,
            n_features=n_features,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            return_metrics=True
        )

        results.append({
            "n_trees": n_trees,
            "n_features": n_features,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            **result
        })

    results_df = pd.DataFrame(results)
    print("\nRandom Forest Hyperparameter Tuning Results:")
    print(results_df.sort_values("f1_score", ascending=False).reset_index(drop=True)[:200])

if __name__ == "__main__":
    main()

    
