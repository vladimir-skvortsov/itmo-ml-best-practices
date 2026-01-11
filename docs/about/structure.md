# Project Structure

```
itmo-ml-best-practices/
├── config/                # Hydra configs
├── data/
│   ├── raw/              # Original data
│   └── processed/        # Train/test splits
├── docs/                 # Documentation
├── models/               # Trained models
├── scripts/              # Pipeline scripts
├── src/
│   ├── data/            # Data processing
│   ├── models/          # Training
│   ├── experiments/     # MLflow utils
│   ├── clearml_experiments/
│   └── clearml_pipelines/
├── docker-compose.yml
├── Makefile
├── mkdocs.yml
├── pyproject.toml
├── requirements.txt
└── Snakefile
```

## Commands

```bash
# Training
make train experiments compare leaderboard

# Pipeline
make pipeline pipeline-viz

# Docker
make docker-up docker-down

# Docs
make docs-serve
```
