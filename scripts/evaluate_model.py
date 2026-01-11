"""
Evaluate trained model on test data.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    test_data: str,
    model_file: str,
    metadata_file: str,
    report_file: str,
    plots_dir: str,
    model_name: str,
) -> None:
    """Evaluate model on test data."""
    print(f"Evaluating model: {model_name}")

    # Load model
    model = joblib.load(model_file)
    print(f"Loaded model from {model_file}")

    # Load test data
    df = pd.read_csv(test_data)

    with open(metadata_file) as f:
        metadata = json.load(f)

    target_col = metadata["target_col"]
    X_test = df.drop(columns=[target_col]).values
    y_test = df[target_col].values

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Generate classification report
    class_report = classification_report(y_test, y_pred)
    print(f"\nClassification Report:\n{class_report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create plots directory
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_file = plots_path / "confusion_matrix.png"
    plt.savefig(cm_file)
    plt.close()
    print(f"Saved confusion matrix to {cm_file}")

    # Save report to file
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        f.write(f"Evaluation Report: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
        f.write("\n")

    print(f"Saved evaluation report to {report_file}")


if __name__ == "__main__":
    # Snakemake variables (snakemake object is injected by Snakemake at runtime)
    evaluate_model(
        test_data=snakemake.input.test_data,  # type: ignore  # noqa: F821
        model_file=snakemake.input.model_file,  # type: ignore  # noqa: F821
        metadata_file=snakemake.input.metadata,  # type: ignore  # noqa: F821
        report_file=snakemake.output.report,  # type: ignore  # noqa: F821
        plots_dir=snakemake.output.plots,  # type: ignore  # noqa: F821
        model_name=snakemake.params.model_name,  # type: ignore  # noqa: F821
    )
