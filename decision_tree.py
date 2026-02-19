import numpy as np

# Loves Popcorn, Loves Soda, Age, Loves Movie
# True,True,7,False
# True,False,12,False
# False,True,18,True
# False,True,35,True
# True,True,38,True
# True,False,50,False
# False,False,83,False

column_names = ["Loves Popcorn", "Loves Soda", "Age", "Loves Movie"]
data = np.array([
    [1, 1, 7, 0],
    [1, 0, 12, 0],
    [0, 1, 18, 1],
    [0, 1, 35, 1],
    [1, 1, 38, 1],
    [1, 0, 50, 0],
    [0, 0, 83, 0]
])


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, prediction=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

def gini_impurity(y):
    print("Calculating Gini impurity for labels:", y)
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    print("Classes:", classes)
    print("counts:", counts)
    print("total count:", counts.sum())
    print("total count check (should match total count):", len(y))
    print("probabilities:", probabilities)
    return 1 - np.sum(probabilities ** 2)

print("Gini impurity of the whole dataset:", gini_impurity(data[:, -1]))

## NEED TO CHECK IF THIS WILL WORK FOR NON CATEGORICAL Y TOO

## BUT ANYWAYS< FOR CATEGORICAL Y
## MAKE A CATEGORY VARIABLE FOR EACH FEATURE
## IF THE CATEGORY TYPE IS CATEGORICAL, IT IS SIMPLE
## IF THE CATEGORY TYPE IS NUMERIC, THEN WE CAN CHECK ALL POSSIBLE THRESHOLDS