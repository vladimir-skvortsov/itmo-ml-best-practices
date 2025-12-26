# Experiments

## Quick Start

```bash
make docker-up
make experiments  # 17 experiments
```

View: http://localhost:3000

## Search

```bash
make leaderboard
make compare
```

```bash
python -m src.experiments.search_runs search \
    --filter "metrics.test_accuracy > 0.9"
```

## Decorators

```python
from src.experiments.decorators import mlflow_experiment, log_params, log_metrics

@mlflow_experiment(experiment_name="my_exp")
@log_params
@log_metrics(["accuracy"])
def train_model(n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return {"accuracy": model.score(X_test, y_test)}
```

## Model Registry

```python
# Register
mlflow.sklearn.log_model(model, "model", registered_model_name="IrisClassifier")

# Load
model = mlflow.sklearn.load_model("models:/IrisClassifier/Production")
```
