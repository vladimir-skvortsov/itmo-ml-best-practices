"""
Train machine learning model with MLflow tracking.
"""
from pathlib import Path
from typing import Any, Dict, Tuple

import click
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.models.mlflow_config import MLflowConfig


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(data_path)
    click.echo(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def prepare_data(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    click.echo(f"Train set: {X_train.shape[0]} samples")
    click.echo(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def train_model(
    model_type: str, X_train: np.ndarray, y_train: np.ndarray, **params: Any
) -> Any:
    """Train a classification model."""
    if model_type == "logistic":
        model = LogisticRegression(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    click.echo(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    click.echo("Training completed")

    return model


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "f1": f1_score(y_test, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    click.echo("\nModel Performance:")
    for metric_name, value in metrics.items():
        click.echo(f"  {metric_name}: {value:.4f}")

    return metrics


def plot_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, save_path: str
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(
    y_test: np.ndarray, y_pred_proba: np.ndarray, save_path: str
) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default="data/raw/iris.csv",
    help="Path to training data",
)
@click.option(
    "--model-type",
    type=click.Choice(["logistic", "random_forest"]),
    default="random_forest",
    help="Type of model to train",
)
@click.option(
    "--test-size", type=float, default=0.2, help="Proportion of data for testing"
)
@click.option("--random-state", type=int, default=42, help="Random seed")
@click.option(
    "--n-estimators", type=int, default=100, help="Number of trees (Random Forest)"
)
@click.option("--max-depth", type=int, default=10, help="Maximum tree depth")
@click.option("--run-name", type=str, default=None, help="MLflow run name")
@click.option(
    "--register-model",
    type=str,
    default=None,
    help="Register model with this name in Model Registry",
)
def main(
    data_path: str,
    model_type: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    run_name: str,
    register_model: str,
) -> None:
    """Train a model with MLflow tracking."""
    # Initialize MLflow
    mlflow_config = MLflowConfig(experiment_name="ml-best-practices")

    # Prepare parameters
    if model_type == "random_forest":
        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }
    else:
        model_params = {
            "max_iter": 1000,
            "random_state": random_state,
        }

    # Start MLflow run
    with mlflow_config.start_run(
        run_name=run_name,
        tags={
            "model_type": model_type,
            "data_path": data_path,
        },
    ):
        # Load and prepare data
        df = load_data(data_path)
        X_train, X_test, y_train, y_test = prepare_data(
            df, test_size=test_size, random_state=random_state
        )

        # Log data parameters
        mlflow_config.log_params(
            {
                "data_path": data_path,
                "test_size": test_size,
                "random_state": random_state,
                "n_samples": len(df),
                "n_features": X_train.shape[1],
            }
        )

        # Train model
        model = train_model(model_type, X_train, y_train, **model_params)

        # Log model parameters
        mlflow_config.log_params(
            {
                "model_type": model_type,
                **model_params,
            }
        )

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        mlflow_config.log_metrics(metrics)

        # Create and log visualizations
        plots_dir = Path("reports/figures")
        plots_dir.mkdir(parents=True, exist_ok=True)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        cm_path = plots_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, str(cm_path))
        mlflow_config.log_artifact(str(cm_path))

        roc_path = plots_dir / "roc_curve.png"
        plot_roc_curve(y_test, y_pred_proba, str(roc_path))
        mlflow_config.log_artifact(str(roc_path))

        # Save classification report
        report = classification_report(y_test, y_pred)
        report_path = plots_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow_config.log_artifact(str(report_path))

        # Log model
        mlflow_config.log_model(
            model,
            artifact_path="model",
            registered_model_name=register_model,
        )

        # Save model locally as well
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{model_type}_model.joblib"
        joblib.dump(model, model_path)
        click.echo(f"\nModel saved to: {model_path}")

        # Get run info
        run = mlflow.active_run()
        click.echo(f"\nMLflow Run ID: {run.info.run_id}")
        click.echo(f"Experiment ID: {run.info.experiment_id}")
        click.echo(f"Artifact URI: {run.info.artifact_uri}")

        if register_model:
            click.echo(f"\nModel registered as: {register_model}")


if __name__ == "__main__":
    main()
