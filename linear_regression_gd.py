# Linear Regression & Multiple Linear Regression Example

import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

wt = np.array([0.9, 1.8, 2.4, 3.5, 3.9, 4.4, 5.1, 5.6, 6.3])
ht = np.array([1.4, 2.6, 1.0, 3.7, 5.5, 3.2, 3.0, 4.9, 6.3])
age = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26])


def plot_2d(features, feature_names, target, target_name, w, b):   # Plotting function
    plt.scatter(features[0], target, color='blue', label='Data Points')
    plt.plot(features[0], w * features[0] + b, color='red', label='Regression Line')
    plt.xlabel(feature_names[0])
    plt.ylabel(target_name)
    plt.title('Linear Regression Example')
    plt.legend()
    plt.show()

def plot_3d(features, feature_names, target, target_name, w1, w2, b):   # 3D Plotting function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[0], features[1], target, color='blue', label='Data Points')

    x_surf = np.linspace(min(features[0]), max(features[0]), 10)
    y_surf = np.linspace(min(features[1]), max(features[1]), 10)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = w1 * x_surf + w2 * y_surf + b

    ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(target_name)
    ax.set_title('Multi-Linear Regression Example')
    plt.show()

def prepare_features(features):  # Feature Scaling
    X = np.column_stack(features)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def SSE(X, y, w, b):    # Sum of Squared Errors
    y_pred = X @ w + b
    return np.sum((y - y_pred) ** 2)

def SS_total(y):   # Total Sum of Squares
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean) ** 2)
    return sst

def compute_cost(X, y, w, b):   # Mean Squared Error Cost Function
    m = X.shape[0]
    cost = SSE(X, y, w, b) / (2 * m)
    return cost

def gradient_descent(X, y, w, b, learning_rate, num_iterations, tol=1e-12):   # Gradient Descent Algorithm
    m = X.shape[0]

    y_pred = X @ w + b
    error = y_pred - y
    prev_cost = np.sum(error**2) / (2*m)

    for _ in range(num_iterations):
        dw = (1/m) * (X.T @ error)
        db = (1/m) * np.sum(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        y_pred = X @ w + b
        error = y_pred - y

        current_cost = np.sum(error**2) / (2*m)

        if abs(prev_cost - current_cost) < tol:
            print("\t\t\t\tConvergence reached.")
            return w, b

        if not np.isfinite(current_cost):
            print("Divergence detected. Stopping Gradient Descent.")
            return w, b

        prev_cost = current_cost

    print("\t\t\t\tMax iterations reached.")
    return w, b

def R2_score(x, y, w, b):   # R-squared Calculation
    sse = SSE(x, y, w, b)
    sst = SS_total(y)

    assert sst != 0, "Total Sum of Squares (SST) cannot be zero."

    r2 = 1 - (sse / sst)
    return r2

def f_score(x, y, w, b):   # F-score Calculation
    sse = SSE(x, y, w, b)
    sst = SS_total(y)

    n = x.shape[0]
    p = x.shape[1]

    assert sst != 0, "Total Sum of Squares (SST) cannot be zero."
    assert n - p - 1 != 0, "Denominator in F-score calculation cannot be zero."

    msr = (sst - sse) / p
    mse = sse / (n - p - 1)

    f_score_value = msr / mse
    return f_score_value

def p_value(f_score_value, n, p):   # P-value Calculation
    p_val = 1 - f.cdf(f_score_value, p, n - p - 1)
    return p_val

def run_regression(features, target, learning_rate=0.1, num_iterations=500000): # Main Regression Function
    X_scaled, mean, std = prepare_features(features)
    
    w_scaled = np.zeros(X_scaled.shape[1])
    b_scaled = 0.0
    
    w_scaled, b_scaled = gradient_descent(X_scaled, target, w_scaled, b_scaled, learning_rate, num_iterations)

    w = w_scaled / std
    b = b_scaled - np.sum((w_scaled * mean) / std)

    X = np.column_stack(features)

    cost = compute_cost(X, target, w, b)
    r2 = R2_score(X, target, w, b)
    f = f_score(X, target, w, b)
    p = p_value(f, X.shape[0], X.shape[1])
    
    return X, w, b, cost, r2, f, p

def auto_plot(features, feature_names, target, target_name, w=None, b=None, title_suffix=""):   # Automatic Plotting based on feature count
    if len(features) == 1:
        if w is None: w = 0
        if b is None: b = 0
        plot_2d(features, feature_names, target, target_name, w, b)
    elif len(features) == 2:
        if w is None: w = np.zeros(2)
        if b is None: b = 0
        plot_3d(features, feature_names, target, target_name, w[0], w[1], b)

def main(): # Main Execution Function
    cases = [
        {"features": [wt, age], "feature_names": ["wt", "age"], "target": ht, "target_name": "ht", "plot": lambda w, b: plot_3d(wt, age, ht, w1=w[0], w2=w[1], b=b)},
        {"features": [wt], "feature_names": ["wt"], "target": ht, "target_name": "ht", "plot": lambda w, b: plot_2d(wt, ht, w=w[0], b=b)},
        {"features": [age], "feature_names": ["age"], "target": ht, "target_name": "ht", "plot": lambda w, b: plot_2d(age, ht, w=w[0], b=b)},
    ]

    for case in cases:
        auto_plot(case["features"], case["feature_names"], case["target"], case["target_name"], title_suffix=" (Before Regression)")
        
        X, w, b, cost, r2, f, p = run_regression(case["features"], case["target"])
        
        print("Weights:", w)
        print("Bias:", b)
        print("Final Cost:", cost)
        print()
        print("R-squared:", r2)
        print("F-score:", f)
        print("P-value:", p)
        print()
        print("Final Function is ")
        print(case["target_name"], " =", " + ".join([f"{w[i]:.6f}*{case['feature_names'][i]}" for i in range(len(w))]) + f" + {b:.6f}")
        print("-" * 40)
        
        auto_plot(case["features"], case["feature_names"], case["target"], case["target_name"], w=w, b=b, title_suffix=" (After Regression)")



if __name__ == "__main__":
    main()