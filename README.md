# itmo-ml-best-practices (iris)

A machine learning project based on the classical Iris dataset. The project includes data processing, basic analysis, classification model construction, and results evaluation.

## Requirements

- Python >= 3.10
- Git с Git LFS
- Poetry >= 2.0.0 (recommended) or pip
- Docker (optional, for containerized development)

## Installation

### Using Poetry

1. Clone the repository:

```bash
git clone https://github.com/vladimir-skvortsov/itmo-ml-best-practices
cd itmo-ml-best-practices
```

2. Install Poetry if you haven't already: https://python-poetry.org

3. Install dependencies:

```bash
poetry install
```

4. Activate the virtual environment:

```bash
poetry shell
```

### Using Docker

1. Build the Docker image:

```bash
docker build -t itmo-ml-best-practices .
```

2. Run a container:

```bash
docker run -it -v $(pwd):/app itmo-ml-best-practices
```

## Usage

### Basic Workflow

1. **Prepare the data**:

```bash
make data
```

Or directly:

```bash
python src/data/make_dataset.py data/raw data/processed
```

2. **Train a model**:

```bash
python src/models/train_model.py
```

3. **Make predictions**:

```bash
python src/models/predict_model.py
```

4. **Generate visualizations**:

```bash
python src/visualization/visualize.py
```

### Available Make Commands

Run `make help` to see all available commands:

- `make requirements` - Install Python dependencies
- `make data` - Generate processed dataset from raw data
- `make clean` - Remove all compiled Python files
- `make lint` - Run code linting with flake8
- `make test_environment` - Test that Python environment is set up correctly
- `make create_environment` - Set up Python interpreter environment (conda or virtualenv)

## Development

### Setting up Pre-commit Hooks

```bash
poetry install --with dev
pre-commit install
```

### Running Code Quality Checks

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
ruff check src/

# Type checking
mypy src/

# Security check
bandit -r src/
```

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── Dockerfile         <- Docker configuration for containerized development
    ├── pyproject.toml     <- Poetry configuration and project metadata
    ├── poetry.lock        <- Locked dependency versions
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    ├── tox.ini            <- tox file with settings for running tox
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

## Environment Variables

The project uses `.env` files for configuration. Create a `.env` file in the project root for local development settings.

## Testing

Test that your environment is set up correctly:

```bash
make test_environment
```

Or directly:

```bash
python test_environment.py
```

## Branches system

Main branch: `main`
Branch for development: `develop`

Each homework has its own branch in the format `hw{number}`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
