"""
Compare all trained models.
"""

import json
from pathlib import Path

import pandas as pd


def compare_models(
    metrics_files: list[str],
    report_file: str,
    summary_file: str,
) -> None:
    """Compare all trained models."""
    print("Comparing all models...")

    # Load all metrics
    all_metrics = []
    for metrics_file in metrics_files:
        with open(metrics_file) as f:
            metrics = json.load(f)
            all_metrics.append(metrics)

    # Create comparison DataFrame
    df = pd.DataFrame(all_metrics)
    df = df.sort_values("train_accuracy", ascending=False)

    print("\nModel Comparison:")
    print(df.to_string(index=False))

    # Save summary CSV
    Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_file, index=False)
    print(f"\nSaved comparison to {summary_file}")

    # Generate report
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        f.write("Models Comparison Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total models trained: {len(all_metrics)}\n\n")

        f.write("Ranking by Training Accuracy:\n")
        f.write("-" * 60 + "\n")
        for _, row in df.iterrows():
            f.write(f"{row['model_name']:20s} ")
            f.write(f"Accuracy: {row['train_accuracy']:.4f} ")
            f.write(f"F1: {row['train_f1']:.4f}\n")

        f.write("\n")
        f.write(f"Best Model: {df.iloc[0]['model_name']}\n")
        f.write(f"Best Accuracy: {df.iloc[0]['train_accuracy']:.4f}\n")

    print(f"Saved comparison report to {report_file}")
    best_model = df.iloc[0]["model_name"]
    best_accuracy = df.iloc[0]["train_accuracy"]
    print(f"\nBest model: {best_model} with accuracy {best_accuracy:.4f}")


if __name__ == "__main__":
    # Snakemake variables (snakemake object is injected by Snakemake at runtime)
    compare_models(
        metrics_files=snakemake.input.metrics,  # type: ignore  # noqa: F821
        report_file=snakemake.output.report,  # type: ignore  # noqa: F821
        summary_file=snakemake.output.summary,  # type: ignore  # noqa: F821
    )
