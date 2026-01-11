"""
Utility functions for working with MLflow experiments.
"""

from typing import Any

import mlflow
import pandas as pd
from tabulate import tabulate


def search_experiments(
    experiment_name: str,
    filter_string: str | None = None,
    order_by: list[str] | None = None,
    max_results: int = 100,
) -> pd.DataFrame:
    """
    Search and filter MLflow experiments.

    Args:
        experiment_name: Name of the experiment
        filter_string: MLflow filter string (e.g., "metrics.accuracy > 0.9")
        order_by: list of columns to order by (e.g., ["metrics.accuracy DESC"])
        max_results: Maximum number of results

    Returns:
        DataFrame with filtered runs

    Example:
        # Find runs with high accuracy
        runs = search_experiments(
            "ml-experiments",
            filter_string="metrics.test_accuracy > 0.9",
            order_by=["metrics.test_accuracy DESC"]
        )
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=order_by,
        max_results=max_results,
    )

    return runs


def compare_runs(
    experiment_name: str,
    run_ids: list[str] | None = None,
    metric_cols: list[str] | None = None,
    param_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare specific runs or all runs in an experiment.

    Args:
        experiment_name: Name of the experiment
        run_ids: list of run IDs to compare (if None, compare all)
        metric_cols: Metrics to include in comparison
        param_cols: Parameters to include in comparison

    Returns:
        DataFrame with comparison
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    if run_ids:
        filter_str = " or ".join([f"run_id = '{rid}'" for rid in run_ids])
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], filter_string=filter_str
        )
    else:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Select columns
    cols = ["run_id", "start_time"]

    if param_cols:
        cols.extend(
            [f"params.{p}" for p in param_cols if f"params.{p}" in runs.columns]
        )
    else:
        param_cols_available = [c for c in runs.columns if c.startswith("params.")]
        cols.extend(param_cols_available)

    if metric_cols:
        cols.extend(
            [f"metrics.{m}" for m in metric_cols if f"metrics.{m}" in runs.columns]
        )
    else:
        metric_cols_available = [c for c in runs.columns if c.startswith("metrics.")]
        cols.extend(metric_cols_available)

    comparison = runs[cols].copy()
    comparison["run_id"] = comparison["run_id"].str[:8]

    return comparison


def get_best_run(
    experiment_name: str, metric: str, ascending: bool = False
) -> dict[str, Any]:
    """
    Get the best run based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric name to optimize
        ascending: If True, lower is better

    Returns:
        dictionary with best run information
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
        raise ValueError("No runs found")

    best_run = runs.iloc[0]

    return {
        "run_id": best_run["run_id"],
        "run_name": best_run.get("tags.mlflow.runName", ""),
        "metric_value": best_run.get(f"metrics.{metric}"),
        "params": {
            k.replace("params.", ""): v
            for k, v in best_run.items()
            if k.startswith("params.")
        },
        "metrics": {
            k.replace("metrics.", ""): v
            for k, v in best_run.items()
            if k.startswith("metrics.")
        },
    }


def delete_runs(experiment_name: str, filter_string: str) -> None:
    """
    Delete runs matching a filter.

    Args:
        experiment_name: Name of the experiment
        filter_string: MLflow filter string

    Example:
        # Delete failed runs
        delete_runs("ml-experiments", "tags.status = 'failed'")
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=filter_string
    )

    client = mlflow.tracking.MlflowClient()

    for _, run in runs.iterrows():
        client.delete_run(run["run_id"])
        print(f"Deleted run: {run['run_id']}")

    print(f"Deleted {len(runs)} runs")


def print_run_summary(run_id: str) -> None:
    """
    Print detailed summary of a run.

    Args:
        run_id: MLflow run ID
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    print(f"\n{'=' * 60}")
    print(f"Run ID: {run_id}")
    print(f"Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"Status: {run.info.status}")
    print(f"Start Time: {run.info.start_time}")
    print(f"Duration: {(run.info.end_time - run.info.start_time) / 1000:.2f}s")
    print(f"{'=' * 60}")

    if run.data.params:
        print("\nParameters:")
        params_table = [[k, v] for k, v in sorted(run.data.params.items())]
        print(tabulate(params_table, headers=["Parameter", "Value"], tablefmt="simple"))

    if run.data.metrics:
        print("\nMetrics:")
        metrics_table = [[k, f"{v:.4f}"] for k, v in sorted(run.data.metrics.items())]
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="simple"))

    if run.data.tags:
        print("\nTags:")
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
        if tags:
            tags_table = [[k, v] for k, v in sorted(tags.items())]
            print(tabulate(tags_table, headers=["Tag", "Value"], tablefmt="simple"))

    print()


def export_experiments(experiment_name: str, output_path: str) -> None:
    """
    Export experiment runs to CSV.

    Args:
        experiment_name: Name of the experiment
        output_path: Path to save CSV file
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    runs.to_csv(output_path, index=False)

    print(f"Exported {len(runs)} runs to {output_path}")
