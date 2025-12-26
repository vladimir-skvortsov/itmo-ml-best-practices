from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns


def get_experiment_runs(experiment_name: str) -> pd.DataFrame:
    """Fetch all runs from an experiment."""
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Get all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_accuracy DESC"],
    )

    return runs


def create_comparison_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar plot comparing model performances."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract model names and accuracy
    models = df["params.config_name"].values[:10]  # Top 10
    accuracy = df["metrics.test_accuracy"].values[:10]

    # Create bar plot
    bars = ax.barh(models, accuracy)

    # Color bars by performance
    colors = plt.cm.RdYlGn([a for a in accuracy])
    for bar, color in zip(bars, colors, strict=False):
        bar.set_color(color)

    ax.set_xlabel("Test Accuracy")
    ax.set_title("Model Comparison - Top 10 Models")
    ax.set_xlim(0, 1)

    # Add values on bars
    for i, (_, acc) in enumerate(zip(models, accuracy, strict=False)):
        ax.text(acc + 0.01, i, f"{acc:.4f}", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_overfit_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Create scatter plot showing overfitting."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract train and test accuracy
    train_acc = df["metrics.train_accuracy"]
    test_acc = df["metrics.test_accuracy"]
    models = df["params.algorithm"]

    # Scatter plot
    scatter = ax.scatter(
        train_acc, test_acc, c=range(len(df)), cmap="viridis", s=100, alpha=0.6
    )

    # Diagonal line (perfect fit)
    ax.plot([0, 1], [0, 1], "r--", label="No Overfit")

    # Add labels for top models
    for i in range(min(5, len(df))):
        ax.annotate(
            models.iloc[i],
            (train_acc.iloc[i], test_acc.iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Train Accuracy")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Overfitting Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label="Run Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_algorithm_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create box plot comparing algorithms."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by algorithm
    algorithms = df.groupby("params.algorithm")["metrics.test_accuracy"].apply(list)

    # Box plot
    ax.boxplot(
        [algorithms[alg] for alg in algorithms.index],
        labels=algorithms.index,
        patch_artist=True,
    )

    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Algorithm")
    ax.set_title("Algorithm Performance Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_metrics_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Create heatmap of metrics across models."""
    # Select top 10 models
    top_models = df.head(10)

    # Extract metrics
    metrics_cols = [col for col in df.columns if col.startswith("metrics.")]
    metrics_data = top_models[metrics_cols].values
    model_names = top_models["params.config_name"].values

    # Remove 'metrics.' prefix for labels
    metric_labels = [col.replace("metrics.", "") for col in metrics_cols]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        metrics_data,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=metric_labels,
        yticklabels=model_names,
        ax=ax,
    )

    ax.set_title("Metrics Heatmap - Top 10 Models")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_markdown_report(
    df: pd.DataFrame, experiment_name: str, output_path: Path, figures_dir: Path
) -> None:
    """Generate Markdown report with tables and figures."""
    report_lines = [
        f"# Experiment Report: {experiment_name}",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Runs:** {len(df)}",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|--------|------|-----|-----|-----|",
    ]

    # Add summary statistics
    metrics = ["test_accuracy", "train_accuracy", "overfit_gap"]
    for metric in metrics:
        col = f"metrics.{metric}"
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            report_lines.append(
                f"| {metric} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |"
            )

    report_lines.extend(
        [
            "",
            "## Top 10 Models",
            "",
            "| Rank | Model | Algorithm | Test Acc | Train Acc | Overfit |",
            "|------|-------|-----------|----------|-----------|---------|",
        ]
    )

    # Add top models table
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        model_name = row.get("params.config_name", "N/A")
        algorithm = row.get("params.algorithm", "N/A")
        test_acc = row.get("metrics.test_accuracy", 0)
        train_acc = row.get("metrics.train_accuracy", 0)
        overfit = row.get("metrics.overfit_gap", 0)

        report_lines.append(
            f"| {i} | {model_name} | {algorithm} | "
            f"{test_acc:.4f} | {train_acc:.4f} | {overfit:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "### Model Comparison",
            "",
            f"![Model Comparison]({figures_dir.name}/comparison.png)",
            "",
            "### Overfitting Analysis",
            "",
            f"![Overfitting]({figures_dir.name}/overfitting.png)",
            "",
            "### Algorithm Performance",
            "",
            f"![Algorithms]({figures_dir.name}/algorithms.png)",
            "",
            "### Metrics Heatmap",
            "",
            f"![Metrics]({figures_dir.name}/metrics_heatmap.png)",
            "",
            "## Best Model Details",
            "",
        ]
    )

    # Add best model details
    best_run = df.iloc[0]
    report_lines.extend(
        [
            f"**Model:** {best_run.get('params.config_name', 'N/A')}",
            f"**Algorithm:** {best_run.get('params.algorithm', 'N/A')}",
            f"**Test Accuracy:** {best_run.get('metrics.test_accuracy', 0):.4f}",
            f"**Train Accuracy:** {best_run.get('metrics.train_accuracy', 0):.4f}",
            f"**Overfit Gap:** {best_run.get('metrics.overfit_gap', 0):.4f}",
            "",
            "### Parameters",
            "",
        ]
    )

    # Add parameters
    param_cols = [
        col
        for col in df.columns
        if col.startswith("params.")
        and col not in ["params.config_name", "params.algorithm"]
    ]

    for col in param_cols:
        param_name = col.replace("params.", "")
        param_value = best_run.get(col, "N/A")
        report_lines.append(f"- **{param_name}:** {param_value}")

    report_lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )

    # Add recommendations based on analysis
    best_algorithm = (
        df.groupby("params.algorithm")["metrics.test_accuracy"].mean().idxmax()
    )
    report_lines.append(f"1. **Best performing algorithm overall:** {best_algorithm}")

    avg_overfit = df["metrics.overfit_gap"].mean()
    if avg_overfit > 0.05:
        report_lines.append(
            "2. **High overfitting detected** - consider regularization or more data"
        )
    else:
        report_lines.append("2. **Low overfitting** - models generalize well")

    top_accuracy = df["metrics.test_accuracy"].iloc[0]
    if top_accuracy > 0.95:
        report_lines.append("3. **Excellent performance** - ready for production")
    elif top_accuracy > 0.90:
        report_lines.append("3. **Good performance** - consider fine-tuning")
    else:
        report_lines.append(
            "3. **Room for improvement** - "
            "try different algorithms or feature engineering"
        )

    # Write report
    output_path.write_text("\n".join(report_lines))


def main() -> None:
    """Generate comprehensive experiment report."""
    # Configuration
    experiment_name = os.getenv("EXPERIMENT_NAME", "iris_experiments")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:3000")
    output_dir = Path("docs/experiments")
    figures_dir = output_dir / "figures"

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_uri)

    print(f"Fetching runs from experiment: {experiment_name}")
    df = get_experiment_runs(experiment_name)

    if len(df) == 0:
        print("No runs found in experiment")
        return

    print(f"Found {len(df)} runs")

    # Generate visualizations
    print("Creating comparison plot...")
    create_comparison_plot(df, figures_dir / "comparison.png")

    print("Creating overfitting plot...")
    create_overfit_plot(df, figures_dir / "overfitting.png")

    print("Creating algorithm comparison...")
    create_algorithm_comparison(df, figures_dir / "algorithms.png")

    print("Creating metrics heatmap...")
    create_metrics_heatmap(df, figures_dir / "metrics_heatmap.png")

    # Generate markdown report
    print("Generating markdown report...")
    generate_markdown_report(
        df, experiment_name, output_dir / "comparison.md", figures_dir
    )

    # Export CSV
    print("Exporting CSV...")
    df.to_csv(output_dir / "experiments.csv", index=False)

    print("\nReport generated successfully!")
    print(f"- Report: {output_dir / 'comparison.md'}")
    print(f"- Figures: {figures_dir}/")
    print(f"- CSV: {output_dir / 'experiments.csv'}")


if __name__ == "__main__":
    main()
