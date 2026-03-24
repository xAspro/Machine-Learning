import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import time

# Random data set for testing the decision tree implementation
URL = "https://raw.githubusercontent.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/master/diabetes_dataset.csv"

df = pd.read_csv(URL)
print(df.head())

feature_names = df.columns[:-1].tolist()
data = df.values

X = data[:, :-1]
y = data[:, -1].astype(int)

def evaluate_random_forest(model, Y_test, y_pred):
    cm = confusion_matrix(Y_test, y_pred)
    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    return cm, acc, prec, rec, f1

def simple_model(results, tol=0.01):

    top_score = results['mean_test_score'].max()
    simple_results = results[results['mean_test_score'] >= top_score - tol].copy()
    simple_results['param_max_depth'] = simple_results['param_max_depth'].fillna(np.inf)
    simple_results = simple_results.sort_values(['param_max_depth', 'param_max_features', 'param_min_samples_leaf', 
                                                    'param_n_estimators', 'mean_test_score'], ascending=[True, True, False, True, False])

    print("\nSimple Models:")
    print(simple_results[[
        'param_max_depth',
        'param_max_features',
        'param_min_samples_leaf',
        'param_n_estimators',
        'mean_test_score',
        'std_test_score'
    ]])

    simplest_model = simple_results.iloc[0]
    print("\nSelected Simple Model:")
    print(simplest_model[[
        'param_max_depth',
        'param_max_features',
        'param_min_samples_leaf',
        'param_n_estimators',
        'mean_test_score',
        'std_test_score'
    ]])

    return simplest_model

def compute_grid_search(X, Y):
    params_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': [2, 3, 4, 5],
        'max_depth': [3, 4, 5, 6],
        'min_samples_leaf': [1, 5, 10, 15, 20],
    }

    # params_grid = {
    #     'n_estimators': [100],
    #     'max_features': [3],
    #     'max_depth': [5],
    #     'min_samples_leaf': [5],
    # }

    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X,Y, test_size=0.2, stratify=Y)

    model = RandomForestClassifier(bootstrap=True)
    grid = GridSearchCV(
        estimator=model,
        param_grid=params_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid.fit(X_train_dev, y_train_dev)
    

    cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "param_max_depth",
        "param_max_features",
        "param_min_samples_leaf",
        "param_n_estimators",
    ]

    results = (
        pd.DataFrame(grid.cv_results_)[cols]
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )

    print(results.head(10))

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation F1 Score: {grid.best_score_:.4f}")

    simplest_model_params = simple_model(results)
    final_model = RandomForestClassifier(
        n_estimators=int(simplest_model_params["param_n_estimators"]),
        max_features=int(simplest_model_params["param_max_features"]),
        max_depth=int(simplest_model_params["param_max_depth"]) if simplest_model_params["param_max_depth"] != np.inf else None,
        min_samples_leaf=int(simplest_model_params["param_min_samples_leaf"])
    )

    final_model.fit(X_train_dev, y_train_dev)
    y_pred = final_model.predict(X_test)
    cm, acc, prec, rec, f1 = evaluate_random_forest(final_model, y_test, y_pred)

    print("\nFinal Model Performance on Test Set:")
    print('Confusion Matrix:\n', cm)
    print('Accuracy: {:.4f}'.format(acc))
    print('Precision: {:.4f}'.format(prec))
    print('Recall: {:.4f}'.format(rec))
    print('F1 Score: {:.4f}'.format(f1))

    print()
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))



if __name__ == "__main__":
    start_time = time.time()
    compute_grid_search(X, y)
    end_time = time.time()
    print(f"Grid Search Time: {end_time - start_time:.2f} seconds")