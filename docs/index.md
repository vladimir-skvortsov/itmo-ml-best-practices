# ITMO ML Best Practices

MLOps проект для воспроизводимого машинного обучения на датасете Iris.

## Features

- Data versioning с Git LFS + S3
- Experiment tracking с MLflow
- Pipeline orchestration с Snakemake
- Configuration management с Hydra
- MLOps platform с ClearML
- Docker контейнеры
- Auto-generated docs

## Quick Start

```bash
git clone https://github.com/vladimirskvortsov/itmo-ml-best-practices.git
cd itmo-ml-best-practices
make requirements

# Start MLflow
make docker-up

# Run experiments
make experiments

# View results: http://localhost:3000
```

## Architecture

```
data/ → prepare → train (parallel) → evaluate → compare
  ↓        ↓         ↓                   ↓         ↓
 LFS    Snakemake  MLflow            Metrics   Reports
```

## Stack

- Python 3.10+, scikit-learn
- MLflow, ClearML
- Snakemake, Hydra
- Docker, PostgreSQL
- AWS S3, Git LFS
- MkDocs Material

## Next Steps

1. [Installation](getting-started/installation.md) - setup окружения
2. [Quick Start](getting-started/quick-start.md) - первые шаги
3. [Guides](guides/model-training.md) - подробные инструкции
