import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# Random data set for testing the decision tree implementation
URL = "https://raw.githubusercontent.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/master/diabetes_dataset.csv"

df = pd.read_csv(URL)
print(df.head())
print()

feature_names = df.columns[:-1].tolist()
data = df.values

X = data[:, :-1]
Y = data[:, -1].astype(int)

print(f"Class distribution:\n{pd.Series(Y).value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

def filter_row(grid, chosen_params):
    mask = np.ones(len(grid), dtype=bool)

    for key, value in chosen_params.items():
        if isinstance(value, float):
            mask &= np.isclose(grid[key], value)
        else:
            mask &= (grid[key] == value)

    row = grid[mask]

    return row.iloc[[0]] if not row.empty else None

def compute_grid_search():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

    param_grid = [
    {
        'log_reg__C': np.logspace(-3, 3, 20),
        'log_reg__penalty': ['l1'],
        'log_reg__solver': ['saga']
    },
    {
        'log_reg__C': np.logspace(-3, 3, 20),
        'log_reg__penalty': ['l2'],
        'log_reg__solver': ['saga']
    },
    {
            'log_reg__C': np.logspace(-3, 3, 20),
        'log_reg__penalty': ['elasticnet'],
        'log_reg__solver': ['saga'],
        'log_reg__l1_ratio': [0.1, 0.5, 0.9]
    }
    ]

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='accuracy',
        n_jobs=1,
        verbose=1,
        return_train_score=True,
        error_score='raise'
    )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values("rank_test_score").reset_index(drop=True)

    print(results[[
        "params",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]].head())
    print()

    params_df = results['params'].apply(pd.Series)

    final_df = pd.concat(
        [params_df, results[['mean_train_score', 'mean_test_score', 'std_test_score', 'rank_test_score']]],
        axis=1
    )
    final_df_sorted = final_df.sort_values(
        by=['mean_test_score', 'std_test_score', 'mean_train_score'],
        ascending=[False, True, False],
        ignore_index=True
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(final_df_sorted)

    return final_df_sorted



# Uncomment to run the grid search and see the results
final_df_sorted = compute_grid_search()
best_row = final_df_sorted.iloc[[0]]

# Since most of the hyperparameter values are giving me the same test score
# And I am not choosing fixed random state for my train test split
# I am considering one of the simpler models with
# penalty = 'l2', C = 0.336
# Note: I could have considered just the top model and created a model out of it
# However, that could give me a complex model (elasticnet), which is not giving me
# any better result than the one I am currently using below
# The scores are more or less the same. The difference is insignificant compared to the std.

chosen_params = {
    'log_reg__C': 0.335982,
    'log_reg__penalty': 'l2',
    'log_reg__solver': 'saga'
}

comparison_df = pd.concat([best_row, filter_row(final_df_sorted, chosen_params)])
comparison_df['model_type'] = ['Best Model', 'Chosen Model']
print("\n\nComparison of Best Model and Chosen Simpler Model:")
print(comparison_df)
print()

# Final model with chosen hyperparameters
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(
        C=chosen_params['log_reg__C'],
        penalty=chosen_params['log_reg__penalty'],
        solver=chosen_params['log_reg__solver'],
        max_iter=5000  # increase for safety with saga
    ))
])

# Fit on training data
final_pipeline.fit(X_train, y_train)

# Evaluate
train_accuracy = final_pipeline.score(X_train, y_train)
test_accuracy = final_pipeline.score(X_test, y_test)

print(f"Final Train Accuracy: {train_accuracy:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, final_pipeline.predict(X_test)))

print("\nConfusion Matrix on Test Set:")
print(confusion_matrix(y_test, final_pipeline.predict(X_test)))