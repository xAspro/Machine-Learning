import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score

import sys



# Random data set for testing the decision tree implementation
URL = "https://raw.githubusercontent.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/master/diabetes_dataset.csv"

df = pd.read_csv(URL)
print(df.head())

feature_names = df.columns[:-1].tolist()
data = df.values

X = data[:, :-1]
y = data[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
# best_tree = grid_search.best_estimator_
# test_predictions = best_tree.predict(X_test)
# test_accuracy = accuracy_score(y_test, test_predictions)
# print(f"Test set accuracy with best parameters: {test_accuracy:.4f}")


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
]].head(5))



# Using the second best parameter because of its low std and same test score
second_best_params = results.iloc[1]["params"]

print("Choosing Second best params:", second_best_params)

# Train new model using them
second_model = DecisionTreeClassifier(
    random_state=42,
    **second_best_params
)

second_model.fit(X_train, y_train)
# tree_rules = export_text(second_model, feature_names=feature_names, show_weights=True)
# print("\nSecond best tree rules:")
# print(tree_rules)

# plt.figure(figsize=(12,8), dpi=300)
# plot_tree(second_model, feature_names=feature_names, class_names=True, filled=True)
# plt.savefig("tree.pdf")
# plt.show()


tree = second_model.tree_
N = tree.n_node_samples[0]

is_leaf = tree.children_left == -1

final_gini = np.sum(tree.impurity[is_leaf] * tree.n_node_samples[is_leaf]) / N
print(f"Final Gini impurity of the tree: {final_gini:.4f}")
print()

test_predictions = second_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test set accuracy with second-best parameters: {test_accuracy:.4f}\n")




# sklearn_tree_full = DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='gini', min_samples_leaf=MIN_SAMPLES_LEAF)
# sklearn_tree_full.fit(X, y)

# sklearn_predictions = sklearn_tree_full.predict(X)
# sklearn_accuracy = accuracy_score(y, sklearn_predictions)



# tree_rules = export_text(sklearn_tree_full, feature_names=feature_names)
# # print("\nSklearn tree rules:")
# # print(tree_rules)

# plt.figure(figsize=(12,8))
# plot_tree(sklearn_tree_full, feature_names=feature_names, class_names=True, filled=True)
# # plt.show()


# # print(f"\nDecision tree using sklearn:")
# # print(f"Tree structure:")
# # print(sklearn_tree_full)
# # print(f"Sklearn tree accuracy: {sklearn_accuracy:.4f}")
# # print(f"Sklearn tree feature importances: {sklearn_tree_full.feature_importances_}")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# sklearn_tree_train_split = DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='gini', min_samples_leaf=MIN_SAMPLES_LEAF)
# sklearn_tree_train_split.fit(X_train, y_train)
# sklearn_test_predictions = sklearn_tree_train_split.predict(X_test)
# sklearn_test_accuracy = accuracy_score(y_test, sklearn_test_predictions)
# # print("Trying 80-20 train-test split:")
# # print(f"Sklearn tree test accuracy: {sklearn_test_accuracy:.4f}")

# sklearn_tree_cv = DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='gini', min_samples_leaf=MIN_SAMPLES_LEAF)
# cv_scores = cross_val_score(sklearn_tree_cv, X, y, cv=5)
# # print("Trying 5-fold cross-validation:")
# # print(f"Sklearn tree cross-validation scores: {cv_scores}")
# # print(f"Sklearn tree average cross-validation accuracy: {cv_scores.mean():.4f}")

# mean = np.mean(cv_scores)
# std = np.std(cv_scores)

# # print(f"Mean CV accuracy: {mean:.4f}")
# # print(f"CV score std dev: {std:.4f}")

# print()
# print("The parameters used for the decision tree are:")
# print(f"Max depth: {MAX_DEPTH}")
# print(f"Min samples leaf: {MIN_SAMPLES_LEAF}")
# print("The three accuracy are:")
# print(f"Full tree accuracy: {sklearn_accuracy:.4f}")
# print(f"Train-test split accuracy: {sklearn_test_accuracy:.4f}")
# print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")
# print(f"Cross-validation accuracy std dev: {std:.4f}")