import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_regression, make_classification


def load_synthetic_data(task='classification'):
    if task == 'classification':
        X, y = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_informative=10, 
            n_redundant=10, 
            n_clusters_per_class=1, 
            flip_y=0.1,
            random_state=42
            )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        X, y, coef = make_regression(n_samples=1000, n_features=5, noise=20, coef=True, random_state=42)
        print("True coefficients:", coef)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, task):
    if task == 'classification':
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        return {
            'confusion_matrix': cm,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'classification_report': class_report
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 =r2_score(y_true, y_pred)
        return {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2
        }

def gradient_boositing(X_train, y_train, X_test, y_test, param_grid, task='classification'):
    if task == 'classification':
        grid = GridSearchCV(
            estimator=GradientBoostingClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            error_score='raise'
        )

    else:
        grid = GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            error_score='raise'
        )

    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    print("Best parameters:", best_params)
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    results = evaluate_model(y_test, y_pred, task=task)

    return results

def main():
    task = 'classification'  # Change to 'regression' for regression tasks
    X_train, X_test, y_train, y_test = load_synthetic_data(task=task)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_leaf': [1,5]
    }

    results = gradient_boositing(X_train, y_train, X_test, y_test, param_grid, task=task)
    for metric, value in results.items():
        print(f"{metric}:\n\t{value}\n")

if __name__ == "__main__":
    main()