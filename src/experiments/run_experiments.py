"""
Run multiple experiments with different configurations.
"""

from typing import Any

import click
import mlflow
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

EXPERIMENT_CONFIGS = [
    # Logistic Regression variations
    {
        "name": "logistic_l1",
        "model": "logistic",
        "params": {"penalty": "l1", "solver": "saga", "C": 1.0},
    },
    {
        "name": "logistic_l2",
        "model": "logistic",
        "params": {"penalty": "l2", "solver": "lbfgs", "C": 1.0},
    },
    {
        "name": "logistic_strong_reg",
        "model": "logistic",
        "params": {"penalty": "l2", "C": 0.1},
    },
    # Random Forest variations
    {
        "name": "rf_shallow",
        "model": "random_forest",
        "params": {"n_estimators": 50, "max_depth": 5},
    },
    {
        "name": "rf_medium",
        "model": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 10},
    },
    {
        "name": "rf_deep",
        "model": "random_forest",
        "params": {"n_estimators": 200, "max_depth": 20},
    },
    {
        "name": "rf_many_trees",
        "model": "random_forest",
        "params": {"n_estimators": 500, "max_depth": 10},
    },
    # Gradient Boosting variations
    {
        "name": "gbm_slow",
        "model": "gradient_boosting",
        "params": {"n_estimators": 100, "learning_rate": 0.01},
    },
    {
        "name": "gbm_fast",
        "model": "gradient_boosting",
        "params": {"n_estimators": 50, "learning_rate": 0.1},
    },
    {
        "name": "gbm_optimal",
        "model": "gradient_boosting",
        "params": {"n_estimators": 100, "learning_rate": 0.05},
    },
    # SVM variations
    {"name": "svm_linear", "model": "svm", "params": {"kernel": "linear", "C": 1.0}},
    {"name": "svm_rbf", "model": "svm", "params": {"kernel": "rbf", "C": 1.0}},
    # Other algorithms
    {"name": "knn_3", "model": "knn", "params": {"n_neighbors": 3}},
    {"name": "knn_5", "model": "knn", "params": {"n_neighbors": 5}},
    {"name": "decision_tree", "model": "decision_tree", "params": {"max_depth": 10}},
    {"name": "naive_bayes", "model": "naive_bayes", "params": {}},
    {"name": "adaboost", "model": "adaboost", "params": {"n_estimators": 50}},
]


def get_model(model_type: str, params: dict[str, Any], random_state: int = 42) -> Any:
    """Get model instance by type."""
    params = {**params, "random_state": random_state}

    if model_type == "logistic":
        return LogisticRegression(**params, max_iter=1000)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    elif model_type == "svm":
        return SVC(**params, probability=True)
    elif model_type == "knn":
        params.pop("random_state", None)
        return KNeighborsClassifier(**params)
    elif model_type == "decision_tree":
        return DecisionTreeClassifier(**params)
    elif model_type == "naive_bayes":
        params.pop("random_state", None)
        return GaussianNB()
    elif model_type == "adaboost":
        return AdaBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@click.command()
@click.option("--data-path", default="data/raw/iris.csv", help="Path to dataset")
@click.option(
    "--experiment-name", default="ml-experiments", help="MLflow experiment name"
)
@click.option("--target-col", default="Species", help="Target column name")
def main(data_path: str, experiment_name: str, target_col: str) -> None:
    """Run multiple experiments with different configurations."""
    # Load data
    df = pd.read_csv(data_path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    X = df.drop(columns=[target_col]).values
    y_raw = df[target_col].values

    # Encode target if categorical
    from sklearn.preprocessing import LabelEncoder

    if y_raw.dtype == object or isinstance(y_raw[0], str):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = y_raw

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set experiment
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()

    click.echo(f"Running {len(EXPERIMENT_CONFIGS)} experiments...")
    click.echo(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples\n")

    results = []

    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        click.echo(f"[{i}/{len(EXPERIMENT_CONFIGS)}] Running: {config['name']}...")

        with mlflow.start_run(run_name=config["name"]):
            # Log config
            mlflow.log_param("config_name", config["name"])
            mlflow.log_param("algorithm", config["model"])

            # Train model
            model_type: str = config["model"]  # type: ignore[assignment]
            model_params: dict[str, Any] = config["params"]  # type: ignore[assignment]
            model = get_model(model_type, model_params)
            model.fit(X_train, y_train)

            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("test_accuracy", test_score)
            mlflow.log_metric("overfit_gap", train_score - test_score)

            results.append(
                {
                    "name": config["name"],
                    "model": config["model"],
                    "train_acc": train_score,
                    "test_acc": test_score,
                    "gap": train_score - test_score,
                }
            )

            click.echo(f"  Train: {train_score:.4f}, Test: {test_score:.4f}")

    click.echo("\n" + "=" * 60)
    click.echo("Experiments completed!")
    click.echo("=" * 60)

    # Show summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_acc", ascending=False)

    click.echo("\nTop 5 models by test accuracy:")
    for _, row in results_df.head(5).iterrows():
        click.echo(
            f"  {row['name']:20s} - Test: {row['test_acc']:.4f}, Gap: {row['gap']:.4f}"
        )

    click.echo("\nView results in MLflow UI: http://localhost:5000")


if __name__ == "__main__":
    main()
