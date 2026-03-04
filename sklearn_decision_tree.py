import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score

# Random data set for testing the decision tree implementation
URL = "https://raw.githubusercontent.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/master/diabetes_dataset.csv"

df = pd.read_csv(URL)
print(df.head())

feature_names = df.columns[:-1].tolist()
data = df.values

X = data[:, :-1]
y = data[:, -1].astype(int)

tree = DecisionTreeClassifier(criterion='gini')
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_leaf': [1, 5, 10, 15, 20]
}

grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=4,
    scoring='accuracy',
    n_jobs=1,
    verbose=3,
    refit=False,
    return_train_score=True
)

grid_search.fit(X, y)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")


results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values("rank_test_score").reset_index(drop=True)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)

print(results[[
    "params",
    "mean_train_score",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]].head(15))


def simple_model(results, max_tol=0.01):
    best_std = results["std_test_score"].min()
    print(f"\nBest std_test_score: {best_std:.4f}")
    tol = max(max_tol, best_std)

    top_score = results["mean_test_score"].max()
    top_models = results[results["mean_test_score"] >= top_score - tol].copy()

    top_models.loc[:, "depth_val"] = top_models["params"].apply(lambda x: float('inf') if x["max_depth"] is None else x["max_depth"])
    top_models.loc[:, "leaf_val"] = top_models["params"].apply(lambda x: x["min_samples_leaf"])

    simple_models = top_models.sort_values(["depth_val","leaf_val","mean_test_score"], ascending=[True, False, False])
    simplest_model_params = simple_models.iloc[0]["params"]
    print("\nTop models within tolerance:")
    print(simple_models[[
        "params",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
    ]])

    return simplest_model_params

# Using the simplest model from the grid search results to train the final decision tree
simplest_model_params = simple_model(results)
print("\nChoosing simplest model params:", simplest_model_params)
final_model = DecisionTreeClassifier(**simplest_model_params)
final_model.fit(X, y)


# # Uncomment to see the tree rules and the tree plot
# tree_rules = export_text(final_model, feature_names=feature_names, show_weights=True)
# print("\nFinal model tree rules:")
# print(tree_rules)

# plt.figure(figsize=(12,8), dpi=300)
# plot_tree(final_model, feature_names=feature_names, class_names=True, filled=True)
# plt.savefig("tree.pdf")
# plt.show()


tree = final_model.tree_
N = tree.n_node_samples[0]

is_leaf = tree.children_left == -1

final_gini = np.sum(tree.impurity[is_leaf] * tree.n_node_samples[is_leaf]) / N
print(f"Final Gini impurity of the tree: {final_gini:.4f}")
print()

cross_val_scores = cross_val_score(final_model, X, y, cv=10, scoring='accuracy')
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean cross-validation accuracy: {cross_val_scores.mean():.4f}")
print(f"Standard deviation of cross-validation accuracy: {cross_val_scores.std():.4f}")