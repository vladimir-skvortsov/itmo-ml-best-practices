"""
MLflow configuration and utilities for model versioning.
"""
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient


class MLflowConfig:
    """MLflow configuration manager."""

    def __init__(
        self,
        experiment_name: str = "ml-best-practices",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow configuration.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: Location to store artifacts (models, plots, etc.)
        """
        self.experiment_name = experiment_name

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local directory
            mlruns_dir = Path("mlruns").absolute()
            mlruns_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlruns_dir}")

        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
        except Exception:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        self.experiment_id = experiment_id
        mlflow.set_experiment(experiment_name)

        self.client = MlflowClient()

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Additional tags to attach to the run

        Returns:
            Active MLflow run context
        """
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run."""
        mlflow.log_params(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """
        Log a local file or directory as an artifact.

        Args:
            local_path: Path to local file or directory
            artifact_path: Optional artifact path within the run's artifact URI
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a model to MLflow.

        Args:
            model: Model object to log
            artifact_path: Path within the run's artifact URI
            registered_model_name: If provided, register model with this name
            **kwargs: Additional arguments passed to mlflow.sklearn.log_model
        """
        mlflow.sklearn.log_model(
            model, artifact_path, registered_model_name=registered_model_name, **kwargs
        )

    def compare_runs(self, run_ids: list[str]) -> Dict[str, Any]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Dictionary with comparison data
        """
        runs_data = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            runs_data.append(
                {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                }
            )
        return {"runs": runs_data}

    def get_best_run(self, metric: str, ascending: bool = False) -> Any:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to use for comparison
            ascending: If True, lower is better

        Returns:
            Best run object
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )
        if len(runs) == 0:
            raise ValueError("No runs found in experiment")
        return runs.iloc[0]

    def load_model(self, run_id: str, artifact_path: str = "model") -> Any:
        """
        Load a model from a specific run.

        Args:
            run_id: Run ID to load model from
            artifact_path: Path to model artifact within run

        Returns:
            Loaded model object
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.sklearn.load_model(model_uri)

    def register_model(
        self, run_id: str, model_name: str, artifact_path: str = "model"
    ) -> Any:
        """
        Register a model from a run to the Model Registry.

        Args:
            run_id: Run ID containing the model
            model_name: Name to register the model under
            artifact_path: Path to model artifact within run

        Returns:
            Registered model version
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.register_model(model_uri, model_name)

    def transition_model_stage(
        self, model_name: str, version: int, stage: str, archive_existing: bool = True
    ) -> None:
        """
        Transition a model version to a new stage.

        Args:
            model_name: Registered model name
            version: Model version number
            stage: Target stage ('Staging', 'Production', 'Archived')
            archive_existing: Whether to archive existing versions in target stage
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )

    def get_production_model(self, model_name: str) -> Any:
        """
        Load the production version of a registered model.

        Args:
            model_name: Registered model name

        Returns:
            Production model object
        """
        model_uri = f"models:/{model_name}/Production"
        return mlflow.sklearn.load_model(model_uri)
