"""
Make predictions using trained models from MLflow.
"""

from pathlib import Path
from typing import Any, Optional

import click
import mlflow
import numpy as np
import pandas as pd


def load_model_from_mlflow(
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    stage: str = "Production",
) -> tuple:
    """
    Load model from MLflow.

    Args:
        run_id: Specific run ID to load model from
        model_name: Registered model name
        stage: Model stage (Production, Staging, etc.)

    Returns:
        Tuple of (model, run_id)
    """
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        click.echo(f"Loading model from run: {run_id}")
    elif model_name:
        model_uri = f"models:/{model_name}/{stage}"
        click.echo(f"Loading model: {model_name} (stage: {stage})")
    else:
        raise ValueError("Must provide either run_id or model_name")

    model = mlflow.sklearn.load_model(model_uri)
    click.echo("Model loaded successfully")

    return model, run_id


def load_input_data(data_path: str, target_col: Optional[str] = None) -> tuple:
    """
    Load input data for prediction.

    Args:
        data_path: Path to input data CSV
        target_col: If provided, separate this column as ground truth

    Returns:
        Tuple of (X, y) where y is None if target_col not provided
    """
    df = pd.read_csv(data_path)
    click.echo(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
    else:
        X = df.values
        y = None

    return X, y


def make_predictions(
    model: Any, X: np.ndarray, probability: bool = False
) -> np.ndarray:
    """
    Make predictions using the model.

    Args:
        model: Trained model
        X: Input features
        probability: If True, return probability scores

    Returns:
        Predictions array
    """
    if probability:
        predictions = model.predict_proba(X)
        click.echo(f"Generated probability predictions for {len(predictions)} samples")
    else:
        predictions = model.predict(X)
        click.echo(f"Generated class predictions for {len(predictions)} samples")

    return predictions


def save_predictions(
    predictions: np.ndarray,
    output_path: str,
    input_data: Optional[pd.DataFrame] = None,
    probability: bool = False,
) -> None:
    """
    Save predictions to CSV file.

    Args:
        predictions: Array of predictions
        output_path: Path to save predictions
        input_data: Original input data to include
        probability: Whether predictions are probabilities
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if probability and predictions.ndim > 1:
        # Multi-class probabilities
        pred_df = pd.DataFrame(
            predictions,
            columns=[f"prob_class_{i}" for i in range(predictions.shape[1])],
        )
        pred_df["predicted_class"] = predictions.argmax(axis=1)
    else:
        # Binary classification or regular predictions
        if probability:
            pred_df = pd.DataFrame(
                {
                    "prob_class_0": 1 - predictions,
                    "prob_class_1": predictions,
                    "predicted_class": (predictions > 0.5).astype(int),
                }
            )
        else:
            pred_df = pd.DataFrame({"prediction": predictions})

    # Add input features if provided
    if input_data is not None:
        pred_df = pd.concat([input_data.reset_index(drop=True), pred_df], axis=1)

    pred_df.to_csv(output_path, index=False)
    click.echo(f"Predictions saved to: {output_path}")


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to input data CSV",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="MLflow run ID to load model from",
)
@click.option(
    "--model-name",
    type=str,
    default=None,
    help="Registered model name to load",
)
@click.option(
    "--stage",
    type=str,
    default="Production",
    help="Model stage (Production, Staging, etc.)",
)
@click.option(
    "--output",
    type=click.Path(),
    default="data/predictions.csv",
    help="Path to save predictions",
)
@click.option(
    "--probability",
    is_flag=True,
    help="Output probability scores instead of class labels",
)
@click.option(
    "--target-col",
    type=str,
    default=None,
    help="Column name of target variable (for evaluation)",
)
def main(
    data_path: str,
    run_id: str,
    model_name: str,
    stage: str,
    output: str,
    probability: bool,
    target_col: str,
) -> None:
    """Make predictions using MLflow model."""

    # Load model
    model, _ = load_model_from_mlflow(run_id=run_id, model_name=model_name, stage=stage)

    # Load data
    X, y = load_input_data(data_path, target_col=target_col)

    # Make predictions
    predictions = make_predictions(model, X, probability=probability)

    # Evaluate if ground truth available
    if y is not None:
        from sklearn.metrics import accuracy_score, classification_report

        if probability:
            y_pred = (
                (predictions > 0.5).astype(int)
                if predictions.ndim == 1
                else predictions.argmax(axis=1)
            )
        else:
            y_pred = predictions

        accuracy = accuracy_score(y, y_pred)
        click.echo(f"\nAccuracy: {accuracy:.4f}")
        click.echo("\nClassification Report:")
        click.echo(classification_report(y, y_pred))

    # Load original dataframe for saving
    df = pd.read_csv(data_path)
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Save predictions
    save_predictions(predictions, output, input_data=df, probability=probability)


if __name__ == "__main__":
    main()
