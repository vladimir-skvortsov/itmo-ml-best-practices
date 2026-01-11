# Installation

## Requirements

- Python 3.10+
- Docker & Docker Compose

## Setup

```bash
git clone https://github.com/vladimirskvortsov/itmo-ml-best-practices.git
cd itmo-ml-best-practices

# Via pip
pip install -r requirements.txt

# Via Poetry
poetry install --with dev
poetry shell
```

## Git LFS

```bash
git lfs install
```

## Services

```bash
make docker-up  # MLflow на http://localhost:3000
```

## Verify

```bash
python test_environment.py
```

## Env Variables

```bash
export MLFLOW_TRACKING_URI=http://localhost:3000
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

Or just use `make` commands.
