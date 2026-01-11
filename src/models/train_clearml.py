import json
from pathlib import Path
from typing import Any, cast

import click
import joblib
import pandas as pd
from clearml import Task
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def get_model(model_type: str, params: dict[str, Any]) -> Any:
    """Get model instance by type."""
    models = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "svm": SVC,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](**params)


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default="data/raw/iris.csv",
    help="Path to dataset",
)
@click.option(
    "--model-type",
    type=click.Choice(
        ["logistic_regression", "random_forest", "gradient_boosting", "svm"]
    ),
    default="random_forest",
    help="Type of model to train",
)
@click.option("--target-col", default="Species", help="Target column name")
@click.option("--test-size", default=0.2, help="Test set size")
@click.option("--random-state", default=42, help="Random state for reproducibility")
@click.option(
    "--project-name", default="iris-classification", help="ClearML project name"
)
@click.option("--task-name", default=None, help="ClearML task name")
def main(
    data_path: str,
    model_type: str,
    target_col: str,
    test_size: float,
    random_state: int,
    project_name: str,
    task_name: str,
) -> None:
    """Train model with ClearML tracking."""

    # Initialize ClearML Task
    task_name = task_name or f"train_{model_type}"
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        tags=[model_type, "training"],
    )

    # Log parameters
    task.connect(
        {
            "data_path": data_path,
            "model_type": model_type,
            "target_col": target_col,
            "test_size": test_size,
            "random_state": random_state,
        }
    )

    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Drop ID column if exists
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Encode target if categorical
    if df[target_col].dtype == "object":
        print(f"Encoding target column: {target_col}")
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

        # Log label mapping
        label_mapping = dict(enumerate(le.classes_))
        task.upload_artifact("label_mapping", label_mapping)

    # Split features and target
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Train-test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Model parameters
    model_params = {
        "logistic_regression": {"max_iter": 1000, "random_state": random_state},
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": random_state,
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": random_state,
        },
        "svm": {"kernel": "rbf", "C": 1.0, "random_state": random_state},
    }

    params = model_params[model_type]
    model_params_dict = cast(dict[str, Any], params)
    task.connect(params, name="model_parameters")

    # Train model
    print(f"Training {model_type} model...")
    model = get_model(model_type, model_params_dict)
    model.fit(X_train, y_train)

    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    train_precision = precision_score(y_train, y_train_pred, average="weighted")
    test_precision = precision_score(y_test, y_test_pred, average="weighted")
    train_recall = recall_score(y_train, y_train_pred, average="weighted")
    test_recall = recall_score(y_test, y_test_pred, average="weighted")

    # Log metrics (ClearML will automatically create plots)
    task.get_logger().report_single_value("train_accuracy", train_accuracy)
    task.get_logger().report_single_value("test_accuracy", test_accuracy)
    task.get_logger().report_single_value("train_f1", train_f1)
    task.get_logger().report_single_value("test_f1", test_f1)
    task.get_logger().report_single_value("train_precision", train_precision)
    task.get_logger().report_single_value("test_precision", test_precision)
    task.get_logger().report_single_value("train_recall", train_recall)
    task.get_logger().report_single_value("test_recall", test_recall)
    task.get_logger().report_single_value("cv_mean", cv_scores.mean())
    task.get_logger().report_single_value("cv_std", cv_scores.std())

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    task.get_logger().report_confusion_matrix(
        "Confusion Matrix",
        "Test Set",
        iteration=0,
        matrix=cm,
    )

    # Classification report
    report = classification_report(y_test, y_test_pred, output_dict=True)
    task.upload_artifact("classification_report", report)

    print("\nTraining Results:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Train F1: {train_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")

    # Save model
    output_dir = Path("models") / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Register model in ClearML
    task.upload_artifact("model", artifact_object=model_path)

    # Save metrics
    metrics = {
        "model_type": model_type,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "train_precision": float(train_precision),
        "test_precision": float(test_precision),
        "train_recall": float(train_recall),
        "test_recall": float(test_recall),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    task.upload_artifact("metrics", artifact_object=metrics_path)

    print(f"Metrics saved to {metrics_path}")
    print(f"\nView results in ClearML: {task.get_output_log_web_page()}")

    # Mark task as completed
    task.close()


if __name__ == "__main__":
    main()
