"""
Compare multiple MLflow model runs and versions.
"""

from pathlib import Path
from typing import List, Optional

import click
import mlflow
import pandas as pd
from tabulate import tabulate


def compare_runs_table(
    experiment_name: str, metric_cols: Optional[List[str]] = None, max_runs: int = 10
) -> pd.DataFrame:
    """
    Compare runs in an experiment.

    Args:
        experiment_name: Name of the experiment
        metric_cols: List of metrics to include in comparison
        max_runs: Maximum number of runs to display

    Returns:
        DataFrame with comparison results
    """
    if metric_cols is None:
        metric_cols = ["accuracy", "f1", "roc_auc"]

    # Search runs
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_runs,
        order_by=["start_time DESC"],
    )

    if len(runs) == 0:
        click.echo("No runs found in experiment")
        return pd.DataFrame()

    # Select relevant columns
    cols_to_show = ["run_id", "start_time"]

    # Add parameter columns
    param_cols = [col for col in runs.columns if col.startswith("params.")]
    cols_to_show.extend(param_cols)

    # Add metric columns
    for metric in metric_cols:
        metric_col = f"metrics.{metric}"
        if metric_col in runs.columns:
            cols_to_show.append(metric_col)

    # Add tags
    if "tags.model_type" in runs.columns:
        cols_to_show.append("tags.model_type")

    comparison_df = runs[cols_to_show].copy()

    # Shorten run_id for display
    comparison_df["run_id"] = comparison_df["run_id"].str[:8]

    # Format timestamp
    comparison_df["start_time"] = pd.to_datetime(
        comparison_df["start_time"]
    ).dt.strftime("%Y-%m-%d %H:%M")

    return comparison_df


def find_best_model(
    experiment_name: str, metric: str = "f1", ascending: bool = False
) -> pd.Series:
    """
    Find the best model based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to use for comparison
        ascending: If True, lower is better

    Returns:
        Series with best run information
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )
    if len(runs) == 0:
        raise ValueError("No runs found in experiment")

    return runs.iloc[0]


def compare_registered_models(model_name: str) -> pd.DataFrame:
    """
    Compare versions of a registered model.

    Args:
        model_name: Name of the registered model

    Returns:
        DataFrame with version comparison
    """
    client = mlflow.tracking.MlflowClient()

    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        click.echo(f"Error: {e}")
        return pd.DataFrame()

    if not model_versions:
        click.echo(f"No versions found for model '{model_name}'")
        return pd.DataFrame()

    versions_data = []
    for mv in model_versions:
        run = client.get_run(mv.run_id)
        versions_data.append(
            {
                "version": mv.version,
                "stage": mv.current_stage,
                "run_id": mv.run_id[:8],
                "created": pd.to_datetime(mv.creation_timestamp, unit="ms").strftime(
                    "%Y-%m-%d %H:%M"
                ),
                **{k: v for k, v in run.data.metrics.items()},
            }
        )

    return pd.DataFrame(versions_data)


@click.command()
@click.option(
    "--experiment-name",
    type=str,
    default="ml-best-practices",
    help="Name of the MLflow experiment",
)
@click.option(
    "--metric",
    type=str,
    default="f1",
    help="Metric to use for comparison",
)
@click.option(
    "--max-runs",
    type=int,
    default=10,
    help="Maximum number of runs to display",
)
@click.option(
    "--registered-model",
    type=str,
    default=None,
    help="Compare versions of a registered model",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Save comparison to CSV file",
)
def main(
    experiment_name: str,
    metric: str,
    max_runs: int,
    registered_model: str,
    output: str,
) -> None:
    """Compare MLflow model runs and versions."""

    if registered_model:
        # Compare registered model versions
        click.echo(f"\n=== Comparing versions of model '{registered_model}' ===\n")
        comparison_df = compare_registered_models(registered_model)
    else:
        # Compare experiment runs
        click.echo(f"\n=== Comparing runs in experiment '{experiment_name}' ===\n")
        comparison_df = compare_runs_table(
            experiment_name=experiment_name, max_runs=max_runs
        )

    if comparison_df.empty:
        return

    # Display table
    click.echo(
        tabulate(comparison_df, headers="keys", tablefmt="grid", showindex=False)
    )

    # Find best model
    if not registered_model:
        click.echo(f"\n=== Best model by {metric} ===\n")
        try:
            best_run = find_best_model(experiment_name, metric=metric)
            click.echo(f"Run ID: {best_run['run_id']}")
            click.echo(f"Start Time: {best_run['start_time']}")

            # Display metrics
            metric_cols = [col for col in best_run.index if col.startswith("metrics.")]
            if metric_cols:
                click.echo("\nMetrics:")
                for col in metric_cols:
                    metric_name = col.replace("metrics.", "")
                    click.echo(f"  {metric_name}: {best_run[col]:.4f}")

            # Display parameters
            param_cols = [col for col in best_run.index if col.startswith("params.")]
            if param_cols:
                click.echo("\nParameters:")
                for col in param_cols:
                    param_name = col.replace("params.", "")
                    click.echo(f"  {param_name}: {best_run[col]}")
        except ValueError as e:
            click.echo(f"Error: {e}")

    # Save to file
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        click.echo(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
