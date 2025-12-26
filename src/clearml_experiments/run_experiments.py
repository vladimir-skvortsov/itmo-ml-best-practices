from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from clearml import Task
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_model(model_type: str, params: dict[str, Any]) -> Any:
    """Get model instance by type."""
    models = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "decision_tree": DecisionTreeClassifier,
        "naive_bayes": GaussianNB,
        "adaboost": AdaBoostClassifier,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](**params)


# Experiment configurations
EXPERIMENT_CONFIGS = [
    # Logistic Regression variations
    {
        "name": "lr_l1",
        "model": "logistic_regression",
        "params": {"penalty": "l1", "C": 1.0, "solver": "saga", "max_iter": 1000},
    },
    {
        "name": "lr_l2",
        "model": "logistic_regression",
        "params": {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    },
    {
        "name": "lr_strong_reg",
        "model": "logistic_regression",
        "params": {"penalty": "l2", "C": 0.1, "solver": "lbfgs", "max_iter": 1000},
    },
    # Random Forest variations
    {
        "name": "rf_shallow",
        "model": "random_forest",
        "params": {"n_estimators": 50, "max_depth": 5, "random_state": 42},
    },
    {
        "name": "rf_deep",
        "model": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 15, "random_state": 42},
    },
    {
        "name": "rf_balanced",
        "model": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
    },
    # Gradient Boosting variations
    {
        "name": "gb_conservative",
        "model": "gradient_boosting",
        "params": {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 3},
    },
    {
        "name": "gb_aggressive",
        "model": "gradient_boosting",
        "params": {"n_estimators": 150, "learning_rate": 0.2, "max_depth": 5},
    },
    {
        "name": "gb_balanced",
        "model": "gradient_boosting",
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    },
    # SVM variations
    {"name": "svm_linear", "model": "svm", "params": {"kernel": "linear", "C": 1.0}},
    {"name": "svm_rbf", "model": "svm", "params": {"kernel": "rbf", "C": 1.0}},
    {
        "name": "svm_poly",
        "model": "svm",
        "params": {"kernel": "poly", "degree": 3, "C": 1.0},
    },
    # Other algorithms
    {"name": "knn_5", "model": "knn", "params": {"n_neighbors": 5}},
    {"name": "knn_10", "model": "knn", "params": {"n_neighbors": 10}},
    {"name": "decision_tree", "model": "decision_tree", "params": {"max_depth": 10}},
    {"name": "naive_bayes", "model": "naive_bayes", "params": {}},
    {"name": "adaboost", "model": "adaboost", "params": {"n_estimators": 50}},
]


def run_experiment(
    config: dict[str, Any],
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    project_name: str = "iris-experiments",
) -> dict[str, Any]:
    """Run a single experiment with ClearML tracking."""

    # Initialize ClearML task
    task = Task.init(
        project_name=project_name,
        task_name=config["name"],
        tags=[config["model"], "experiment"],
    )

    # Log configuration
    task.connect(config, name="experiment_config")

    print(f"\nRunning experiment: {config['name']}")
    print(f"Model: {config['model']}")
    print(f"Parameters: {config['params']}")

    # Train model
    model = get_model(config["model"], config["params"])
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    # Log metrics
    task.get_logger().report_single_value("train_accuracy", train_accuracy)
    task.get_logger().report_single_value("test_accuracy", test_accuracy)
    task.get_logger().report_single_value("train_f1", train_f1)
    task.get_logger().report_single_value("test_f1", test_f1)
    task.get_logger().report_single_value("overfit_gap", train_accuracy - test_accuracy)

    # Save model
    output_dir = Path("models") / "experiments" / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pkl"
    joblib.dump(model, model_path)

    # Upload model to ClearML
    task.upload_artifact("model", artifact_object=model_path)

    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Overfit Gap: {train_accuracy - test_accuracy:.4f}")

    # Close task
    task.close()

    return {
        "name": config["name"],
        "model": config["model"],
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "overfit_gap": float(train_accuracy - test_accuracy),
    }


def main() -> None:
    """Run all experiments."""

    # Create parent task for experiment tracking
    parent_task = Task.init(
        project_name="iris-experiments",
        task_name="experiment_suite",
        task_type=Task.TaskTypes.controller,
    )

    print("=" * 60)
    print("Running ClearML Experiments Suite")
    print("=" * 60)

    # Load and prepare data
    data_path = "data/raw/iris.csv"
    print(f"\nLoading data from {data_path}")
    df = pd.read_csv(data_path)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    target_col = "Species"
    if df[target_col].dtype == "object":
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        parent_task.upload_artifact("label_mapping", dict(enumerate(le.classes_)))

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Run all experiments
    results = []
    for config in EXPERIMENT_CONFIGS:
        try:
            result = run_experiment(config, X_train, X_test, y_train, y_test)
            results.append(result)
        except Exception as e:
            print(f"Error in experiment {config['name']}: {e}")
            continue

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_accuracy", ascending=False)

    print("\n" + "=" * 60)
    print("Experiment Results Summary")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save results
    output_dir = Path("reports") / "clearml_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "experiment_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Upload results to parent task
    parent_task.upload_artifact("experiment_results", artifact_object=results_path)

    # Log best model
    best_experiment = results_df.iloc[0]
    print(f"\nBest Experiment: {best_experiment['name']}")
    print(f"Model: {best_experiment['model']}")
    print(f"Test Accuracy: {best_experiment['test_accuracy']:.4f}")

    parent_task.get_logger().report_single_value(
        "best_test_accuracy", best_experiment["test_accuracy"]
    )

    print("\nView all experiments in ClearML Web UI: http://localhost:8080")

    parent_task.close()


if __name__ == "__main__":
    main()
