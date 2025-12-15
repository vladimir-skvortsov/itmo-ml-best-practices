"""
Decorators for automatic MLflow logging.
"""

import functools
import time
from typing import Any, Callable

import mlflow


def mlflow_experiment(experiment_name: str, run_name: str | None = None) -> Callable:
    """
    Decorator to automatically wrap function in MLflow run.

    Usage:
        @mlflow_experiment("my-experiment", "my-run")
        def train_model(data, params):
            model = fit(data, params)
            return model
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            mlflow.set_experiment(experiment_name)

            actual_run_name = run_name or func.__name__

            with mlflow.start_run(run_name=actual_run_name):
                mlflow.log_param("function", func.__name__)

                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                mlflow.log_metric("execution_time_seconds", elapsed)

                return result

        return wrapper

    return decorator


def log_params(func: Callable) -> Callable:
    """
    Decorator to automatically log function parameters to MLflow.

    Usage:
        @log_params
        def train_model(learning_rate=0.01, n_estimators=100):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Log all kwargs as parameters
        for key, value in kwargs.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)

        return func(*args, **kwargs)

    return wrapper


def log_metrics(metric_names: list[str]) -> Callable:
    """
    Decorator to automatically log return values as metrics.

    Usage:
        @log_metrics(["accuracy", "f1_score"])
        def evaluate(model, X, y):
            acc = accuracy_score(y, model.predict(X))
            f1 = f1_score(y, model.predict(X))
            return {"accuracy": acc, "f1_score": f1}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            if isinstance(result, dict):
                for key in metric_names:
                    if key in result:
                        mlflow.log_metric(key, result[key])

            return result

        return wrapper

    return decorator


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.

    Usage:
        @log_execution_time
        def train_model():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        mlflow.log_metric(f"{func.__name__}_time", elapsed)

        return result

    return wrapper


def log_artifacts(*artifact_paths: str) -> Callable:
    """
    Decorator to automatically log specified files as artifacts.

    Usage:
        @log_artifacts("model.pkl", "plot.png")
        def train_and_save():
            # Save files...
            return model
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            for path in artifact_paths:
                try:
                    mlflow.log_artifact(path)
                except Exception as e:
                    print(f"Failed to log artifact {path}: {e}")

            return result

        return wrapper

    return decorator


class MLflowContext:
    """
    Context manager for MLflow runs.

    Usage:
        with MLflowContext("experiment-name", "run-name") as mlf:
            mlf.log_param("lr", 0.01)
            model.fit(X, y)
            mlf.log_metric("accuracy", 0.95)
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict | None = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.run = None

    def __enter__(self) -> "MLflowContext":
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            mlflow.log_param("error", str(exc_val))
            mlflow.set_tag("status", "failed")
        else:
            mlflow.set_tag("status", "success")

        mlflow.end_run()

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter."""
        mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric."""
        mlflow.log_metric(key, value, step=step)

    def log_params(self, params: dict) -> None:
        """Log multiple parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log multiple metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        """Log an artifact."""
        mlflow.log_artifact(path)
