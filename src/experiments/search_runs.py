"""
CLI tool for searching and filtering MLflow runs.
"""

import click
from tabulate import tabulate

from src.experiments.utils import get_best_run, print_run_summary, search_experiments


@click.group()
def cli() -> None:
    """MLflow experiments search and filter tool."""
    pass


@cli.command()
@click.option("--experiment", default="ml-experiments", help="Experiment name")
@click.option(
    "--filter",
    "filter_str",
    default=None,
    help="Filter string (e.g., 'metrics.accuracy > 0.9')",
)
@click.option(
    "--order-by", default=None, help="Order by (e.g., 'metrics.accuracy DESC')"
)
@click.option("--max-results", default=20, help="Maximum results")
def search(
    experiment: str, filter_str: str | None, order_by: str | None, max_results: int
) -> None:
    """Search runs with filters."""
    order_list = [order_by] if order_by else None

    runs = search_experiments(
        experiment,
        filter_string=filter_str,
        order_by=order_list,
        max_results=max_results,
    )

    if len(runs) == 0:
        click.echo("No runs found")
        return

    # Display results
    display_cols = ["run_id"]

    # Add run name if available
    if "tags.mlflow.runName" in runs.columns:
        display_cols.append("tags.mlflow.runName")

    # Add key metrics
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    display_cols.extend(metric_cols[:5])  # Show first 5 metrics

    # Shorten run_id
    runs["run_id"] = runs["run_id"].str[:8]

    result = runs[display_cols].head(max_results)

    click.echo(f"\nFound {len(runs)} runs:\n")
    click.echo(tabulate(result, headers="keys", tablefmt="grid", showindex=False))


@cli.command()
@click.option("--experiment", default="ml-experiments", help="Experiment name")
@click.option("--metric", default="test_accuracy", help="Metric to optimize")
@click.option("--ascending", is_flag=True, help="Lower is better")
def best(experiment: str, metric: str, ascending: bool) -> None:
    """Find the best run by metric."""
    try:
        best_run = get_best_run(experiment, metric, ascending)

        click.echo(f"\n{'=' * 60}")
        click.echo(f"Best run by {metric} ({'min' if ascending else 'max'})")
        click.echo(f"{'=' * 60}")
        click.echo(f"Run ID: {best_run['run_id']}")
        click.echo(f"Run Name: {best_run['run_name']}")
        click.echo(f"{metric}: {best_run['metric_value']:.4f}")

        click.echo("\nKey metrics:")
        for k, v in list(best_run["metrics"].items())[:10]:
            if v is not None:
                click.echo(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        click.echo("\nParameters:")
        for k, v in best_run["params"].items():
            click.echo(f"  {k}: {v}")

        click.echo()

    except ValueError as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.argument("run_id")
def show(run_id: str) -> None:
    """Show detailed information about a run."""
    try:
        print_run_summary(run_id)
    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.option("--experiment", default="ml-experiments", help="Experiment name")
@click.option("--metric", required=True, help="Metric to compare")
@click.option("--top", default=5, help="Number of top runs to show")
def leaderboard(experiment: str, metric: str, top: int) -> None:
    """Show leaderboard for a metric."""
    runs = search_experiments(
        experiment, order_by=[f"metrics.{metric} DESC"], max_results=top
    )

    if len(runs) == 0:
        click.echo("No runs found")
        return

    click.echo(f"\nTop {top} runs by {metric}:\n")

    table_data = []
    for i, (_, run) in enumerate(runs.iterrows(), 1):
        metric_val = run.get(f"metrics.{metric}", "N/A")
        run_name = run.get("tags.mlflow.runName", run["run_id"][:8])

        table_data.append(
            [
                i,
                run_name,
                (
                    f"{metric_val:.4f}"
                    if isinstance(metric_val, float)
                    else str(metric_val)
                ),
                run["run_id"][:8],
            ]
        )

    click.echo(
        tabulate(
            table_data, headers=["Rank", "Run Name", metric, "Run ID"], tablefmt="grid"
        )
    )
    click.echo()


@cli.command()
@click.option("--experiment", default="ml-experiments", help="Experiment name")
@click.option("--output", default="experiments_export.csv", help="Output CSV file")
def export(experiment: str, output: str) -> None:
    """Export experiment runs to CSV."""
    from src.experiments.utils import export_experiments

    export_experiments(experiment, output)
    click.echo(f"Exported to {output}")


if __name__ == "__main__":
    cli()
