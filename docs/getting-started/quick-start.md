# Quick Start

## Install

```bash
git clone https://github.com/vladimirskvortsov/itmo-ml-best-practices.git
cd itmo-ml-best-practices
pip install -r requirements.txt
```

## Train

```bash
make docker-up  # Start MLflow
make train      # Train model
# View: http://localhost:3000
```

## Experiments

```bash
make experiments  # 17 experiments
make leaderboard  # Top models
```

## Pipeline

```bash
make pipeline     # Full workflow
```
