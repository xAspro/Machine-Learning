import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score


# Random data set for testing the decision tree implementation
URL = "https://raw.githubusercontent.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/master/diabetes_dataset.csv"
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 10

df = pd.read_csv(URL)


print(df.head())

feature_names = df.columns[:-1].tolist()
data = df.values

X = data[:, :-1]
y = data[:, -1].astype(int)


sklearn_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='gini', min_samples_leaf=MIN_SAMPLES_LEAF)
sklearn_tree.fit(X, y)

sklearn_predictions = sklearn_tree.predict(X)
sklearn_accuracy = accuracy_score(y, sklearn_predictions)



tree_rules = export_text(sklearn_tree, feature_names=feature_names)
print("\nSklearn tree rules:")
print(tree_rules)

plt.figure(figsize=(12,8))
plot_tree(sklearn_tree, feature_names=feature_names, class_names=True, filled=True)
plt.show()


print(f"\nDecision tree using sklearn:")
print(f"Tree structure:")
print(sklearn_tree)
print(f"Sklearn tree accuracy: {sklearn_accuracy:.4f}")