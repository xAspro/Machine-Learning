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

print("\nChoosing Second best params:", second_best_params)

# Train new model using them
second_model = DecisionTreeClassifier(
    random_state=42,
    **second_best_params
)

second_model.fit(X_train, y_train)

# # Uncomment to see the tree rules and the tree plot
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