import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import decision_tree
import itertools
from sklearn.model_selection import train_test_split


config = decision_tree.get_config()

X, y, feature_names = decision_tree.load_data(config)

# y = X[:, -1]
# X = X[:, :-1]
# feature_names = feature_names[:-1]

def bootstrapping(data, n_samples=None, return_oob_indices=False):
    if n_samples is None:
        n_samples = data.shape[0]
    n = data.shape[0]
    indices = np.random.choice(n, size=n_samples, replace=True)
    oob_indices = np.setdiff1d(np.arange(n), indices)
    if return_oob_indices:
        return data[indices], oob_indices
    return data[indices]

def random_forest(data, n_trees=10, n_samples=None, n_features=None, max_depth=decision_tree.MAX_DEPTH, min_samples_leaf=decision_tree.MIN_SAMPLES_LEAF, task='classification', return_oob_list=False):
    trees = []
    oob_indices_list = []
    for i in range(n_trees):
        # if (i + 1) % 50 == 0:
        #     print(f"\nBuilding tree {i+1}/{n_trees}\ndata.shape[1]: {data.shape[1]}, n_samples: {n_samples}, n_features: {n_features}, max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}")
        
        if return_oob_list:
            bootstrapped_data, oob_indices = bootstrapping(data, n_samples, return_oob_indices=return_oob_list)
            oob_indices_list.append(oob_indices)
        else:
            bootstrapped_data = bootstrapping(data, n_samples)
        bootstrapped_x = bootstrapped_data[:, :-1]
        if task == 'classification':
            bootstrapped_y = bootstrapped_data[:, -1].astype(int)
        else:
            bootstrapped_y = bootstrapped_data[:, -1].astype(float)

        tree = decision_tree.build_tree(bootstrapped_x, bootstrapped_y, max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_features=n_features, task=task)
        trees.append(tree)
    
    if return_oob_list:
        return trees, oob_indices_list
    return trees

def predict_forest(trees, sample, task='classification'):
    predictions = [decision_tree.predict(tree, sample) for tree in trees]
    if task == 'classification':
        return np.bincount(predictions).argmax()
    return np.mean(predictions)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(mse_value):
    return np.sqrt(mse_value)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0



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

def evaluate_oob_score(forest, oob_indices_list, data, task='classification'):
    n = data.shape[0]
    oob_predictions = [[] for _ in range(n)]

    X = data[:, :-1]
    Y = data[:, -1]

    # for tree, oob_indices in zip(forest, oob_indices_list):
    #     for index in oob_indices:
    #         prediction = decision_tree.predict(tree, X[index])
    #         oob_predictions[index].append(prediction)

    for tree, oob_indices in zip(forest, oob_indices_list):
        if len(oob_indices) == 0:
            continue

        preds = decision_tree.predict_batch(tree, X[oob_indices])

        for idx, pred in zip(oob_indices, preds):
            oob_predictions[idx].append(pred)


    final_oob_predictions = []
    final_oob_true = []
    for i, prediction in enumerate(oob_predictions):
        if len(prediction) == 0:
            final_oob_predictions.append(np.nan)
        elif task == 'classification':
            prediction = np.bincount(prediction).argmax()
            # prediction = np.bincount(prediction.astype(int)).argmax()
        else:
            prediction = np.mean(prediction)
        final_oob_true.append(Y[i])
        final_oob_predictions.append(prediction)

    final_oob_predictions = np.array(final_oob_predictions, dtype=float)
    final_oob_true = np.array(final_oob_true)

    valid_mask = ~np.isnan(final_oob_predictions)
    final_oob_predictions = final_oob_predictions[valid_mask]
    final_oob_true = final_oob_true[valid_mask]

    if task == 'classification':
        cm = confusion_matrix(final_oob_true, final_oob_predictions)
        return {
            "oob_accuracy": np.mean(final_oob_true == final_oob_predictions),
            "oob_confusion_matrix": cm,
            "oob_specificity": specificity(cm),
            "oob_sensitivity": sensitivity(cm),
            "oob_precision": precision(cm),
            "oob_f1_score": f1_score(cm)
        }
    else:
        return {
            "oob_mse": mse(final_oob_true, final_oob_predictions),
            "oob_rmse": rmse(mse(final_oob_true, final_oob_predictions)),
            "oob_r2": r2_score(final_oob_true, final_oob_predictions)
        }
    

def evaluate_random_forest(X_train, Y_train, X_dev, Y_dev, n_trees=100, n_features=3, max_depth=5, min_samples_leaf=5, return_metrics=False, task='classification', perform_oob_evaluation=False):
    print("\nBuilding random forest...")
    print(f"n_trees: {n_trees}, n_features: {n_features}, max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}\n\n")

    oob_score = None

    data_train = np.hstack((X_train, Y_train.reshape(-1, 1)))
    if perform_oob_evaluation:
        forest, oob_indices_list = random_forest(data_train, n_trees=n_trees, n_features=n_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf, task=task, return_oob_list=True)
        oob_score = evaluate_oob_score(forest, oob_indices_list, data_train, task=task)
    else:
        forest = random_forest(data_train, n_trees=n_trees, n_features=n_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf, task=task)

    predictions = np.array([predict_forest(forest, sample, task=task) for sample in X_dev])

    if task == 'classification':
        accuracy = np.mean(predictions == Y_dev)
        # print(f"Random Forest Accuracy: {accuracy:.4f}")

        cm = confusion_matrix(Y_dev, predictions)
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
                "f1_score": f1_score(cm),
                **(oob_score if perform_oob_evaluation else {})
            }
        
    else:
        mse_value = mse(Y_dev, predictions)
        rmse_value = rmse(mse_value)
        r2 = r2_score(Y_dev, predictions)

        print(f"Random Forest MSE: {mse_value:.4f}")
        print(f"Random Forest RMSE: {rmse_value:.4f}")
        print(f"Random Forest R2 Score: {r2:.4f}")

        if return_metrics:
            return {
                "mse": mse_value,
                "rmse": rmse_value,
                "r2": r2,
                **(oob_score if perform_oob_evaluation else {})
            }

    
def main():

    perform_oob_evaluation = True

    if config["task"] == 'classification':
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.2, stratify=y_train_dev)
    else:
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.2)

    n_tree_values = [50, 100, 200]
    n_feature_values = [2, 3, 4]
    max_depth_values = [3, 4, 5, 6]
    min_samples_leaf_values = [1, 5, 10, 15, 20]

    # n_tree_values = [100]
    # n_feature_values = [3]
    # max_depth_values = [5]
    # min_samples_leaf_values = [5]


    results = []

    for n_trees, n_features, max_depth, min_samples_leaf in itertools.product(
        n_tree_values,
        n_feature_values,
        max_depth_values,
        min_samples_leaf_values
    ):
        result = evaluate_random_forest(
            X_train=X_train,
            Y_train=y_train,
            X_dev=X_dev,
            Y_dev=y_dev,
            n_trees=n_trees,
            n_features=n_features,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            task=config["task"],
            return_metrics=True,
            perform_oob_evaluation=perform_oob_evaluation
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
    if config['task'] == 'classification':
        print(results_df.sort_values("f1_score", ascending=False).reset_index(drop=True)[:200])

        # Show the final model's performance on the test set
        best_params = results_df.sort_values("f1_score", ascending=False).iloc[0]
    else:
        print(results_df.sort_values("r2", ascending=False).reset_index(drop=True)[:200])

        # Show the final model's performance on the test set
        best_params = results_df.sort_values("r2", ascending=False).iloc[0]

    print("\nBest Hyperparameters:")
    print(best_params)
    print("\nEvaluating best model on test set...")
    final_result = evaluate_random_forest(
        X_train=X_train,
        Y_train=y_train,
        X_dev=X_test,
        Y_dev=y_test,
        n_trees=int(best_params["n_trees"]),
        n_features=int(best_params["n_features"]),
        max_depth=int(best_params["max_depth"]),
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        task=config['task'],
        return_metrics=True,
        perform_oob_evaluation=False
    )

    print("\nFinal Model Performance on Test Set:")
    if config['task'] == 'classification':
        print('Accuracy: {:.4f}'.format(final_result["accuracy"]))
        print('Specificity: {:.4f}'.format(final_result["specificity"]))
        print('Sensitivity: {:.4f}'.format(final_result["sensitivity"]))
        print('Precision: {:.4f}'.format(final_result["precision"]))
        print('F1 Score: {:.4f}'.format(final_result["f1_score"]))
    else:
        print(f"Random Forest MSE: {final_result['mse']:.4f}")
        print(f"Random Forest RMSE: {final_result['rmse']:.4f}")
        print(f"Random Forest R2 Score: {final_result['r2']:.4f}")

    # Note: Havent implemented CV and hence the results might fluctuate a lot.
    # This is just to understand the working of random forest anyways.
    # I will implement CV when writing random forest using sklearn.

if __name__ == "__main__":
    main()

    
