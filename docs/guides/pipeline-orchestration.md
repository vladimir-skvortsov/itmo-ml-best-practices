# Pipeline

## Commands

```bash
make pipeline      # Full run
make pipeline-viz  # Visualize DAG
```

## Snakemake

```python
MODELS = ["logistic_regression", "random_forest", "gradient_boosting", "svm"]

rule all:
    input: expand("reports/{model}/evaluation.txt", model=MODELS)

rule train_model:
    input: "data/processed/X_train.csv"
    output: "models/{model}/model.pkl"
    script: "scripts/train_with_hydra.py"
```

## Hydra Config

```yaml
# config/models/random_forest.yaml
model:
  type: random_forest
  parameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42
```

## Parallel

```bash
snakemake --cores 4
```
