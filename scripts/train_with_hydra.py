"""
Train ML model using Hydra configuration.
"""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def get_model(model_type: str, params: dict[str, Any]) -> Any:
    """Get model instance by type."""
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    elif model_type == "svm":
        return SVC(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model_with_hydra(
    train_data: str,
    metadata_file: str,
    model_file: str,
    metrics_file: str,
    model_name: str,
) -> None:
    """Train model using Hydra configuration."""
    print(f"Training model: {model_name}")

    config_dir = str(Path.cwd() / "config" / "models")
    config_name = model_name

    print(f"Config directory: {config_dir}")
    print(f"Config name: {config_name}")

    GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)

        print(f"Model type: {cfg.model.type}")
        print(f"Parameters: {dict(cfg.parameters)}")

        df = pd.read_csv(train_data)

        with open(metadata_file) as f:
            metadata = json.load(f)

        target_col = metadata["target_col"]
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        model = get_model(cfg.model.type, dict(cfg.parameters))

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:3000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("snakemake-pipeline")
        mlflow.sklearn.autolog()

        with mlflow.start_run(run_name=f"{model_name}_{cfg.model.name}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", cfg.model.type)
            mlflow.log_param("hydra_config", config_name)

            if cfg.training.cross_validation:
                cv_scores = cross_val_score(model, X, y, cv=cfg.training.cv_folds)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                print(f"Cross-validation scores: {cv_scores}")
                print(f"CV Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
                mlflow.log_metric("cv_mean_score", cv_mean)
                mlflow.log_metric("cv_std_score", cv_std)

            print("Training model...")
            model.fit(X, y)

            y_pred = model.predict(X)
            train_accuracy = accuracy_score(y, y_pred)
            train_f1 = f1_score(y, y_pred, average="weighted")
            train_precision = precision_score(y, y_pred, average="weighted")
            train_recall = recall_score(y, y_pred, average="weighted")

            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Train F1: {train_f1:.4f}")

            metrics = {
                "train_accuracy": train_accuracy,
                "train_f1": train_f1,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "model_name": model_name,
                "model_type": cfg.model.type,
            }

            mlflow.log_metrics(
                {
                    "train_accuracy": train_accuracy,
                    "train_f1": train_f1,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                }
            )

            Path(model_file).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_file)
            print(f"Model saved to {model_file}")

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {metrics_file}")

            # Register model if configured
            if cfg.mlflow.register_model:
                run = mlflow.active_run()
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, cfg.mlflow.model_name)
                print(f"Model registered as {cfg.mlflow.model_name}")


if __name__ == "__main__":
    # Snakemake variables (snakemake object is injected by Snakemake at runtime)
    train_model_with_hydra(
        train_data=snakemake.input.train_data,  # type: ignore  # noqa: F821
        metadata_file=snakemake.input.metadata,  # type: ignore  # noqa: F821
        model_file=snakemake.output.model_file,  # type: ignore  # noqa: F821
        metrics_file=snakemake.output.metrics,  # type: ignore  # noqa: F821
        model_name=snakemake.params.model_name,  # type: ignore  # noqa: F821
    )
