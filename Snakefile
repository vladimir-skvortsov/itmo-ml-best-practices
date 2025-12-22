"""
Snakemake workflow for ML pipeline with MLflow tracking.
"""

import os
from pathlib import Path

# Configuration
configfile: "config/pipeline_config.yaml"

# Variables
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Get all model configs
MODELS = config.get("models", ["logistic_regression", "random_forest"])

# Rules
rule all:
    input:
        expand(REPORTS_DIR + "/{model}/evaluation_report.txt", model=MODELS),
        REPORTS_DIR + "/comparison_report.txt"

rule prepare_data:
    """Prepare and split data for training."""
    input:
        raw_data = DATA_RAW + "/iris.csv"
    output:
        train_data = DATA_PROCESSED + "/train.csv",
        test_data = DATA_PROCESSED + "/test.csv",
        metadata = DATA_PROCESSED + "/metadata.json"
    params:
        test_size = config.get("data", {}).get("test_size", 0.2),
        random_state = config.get("data", {}).get("random_state", 42),
        target_col = config.get("data", {}).get("target_col", "Species")
    log:
        REPORTS_DIR + "/logs/prepare_data.log"
    script:
        "scripts/prepare_data.py"

rule train_model:
    """Train a specific model."""
    input:
        train_data = DATA_PROCESSED + "/train.csv",
        metadata = DATA_PROCESSED + "/metadata.json"
    output:
        model_file = MODELS_DIR + "/{model}/model.pkl",
        metrics = MODELS_DIR + "/{model}/metrics.json"
    params:
        model_name = "{model}",
        config_path = lambda wildcards: os.path.abspath(f"config/models/{wildcards.model}.yaml")
    log:
        REPORTS_DIR + "/logs/train_{model}.log"
    resources:
        mem_mb = 2000,
        cpus = 2
    script:
        "scripts/train_with_hydra.py"

rule evaluate_model:
    """Evaluate trained model on test data."""
    input:
        test_data = DATA_PROCESSED + "/test.csv",
        model_file = MODELS_DIR + "/{model}/model.pkl",
        metadata = DATA_PROCESSED + "/metadata.json"
    output:
        report = REPORTS_DIR + "/{model}/evaluation_report.txt",
        plots = directory(REPORTS_DIR + "/{model}/plots")
    params:
        model_name = "{model}"
    log:
        REPORTS_DIR + "/logs/evaluate_{model}.log"
    script:
        "scripts/evaluate_model.py"

rule compare_models:
    """Compare all trained models."""
    input:
        metrics = expand(MODELS_DIR + "/{model}/metrics.json", model=MODELS),
        reports = expand(REPORTS_DIR + "/{model}/evaluation_report.txt", model=MODELS)
    output:
        report = REPORTS_DIR + "/comparison_report.txt",
        summary = REPORTS_DIR + "/models_comparison.csv"
    log:
        REPORTS_DIR + "/logs/compare_models.log"
    script:
        "scripts/compare_all_models.py"

rule clean:
    """Clean generated files."""
    shell:
        """
        rm -rf {DATA_PROCESSED}/*
        rm -rf {MODELS_DIR}/*
        rm -rf {REPORTS_DIR}/*
        echo "Cleaned processed data, models, and reports"
        """

# Optional: visualize DAG
rule visualize:
    """Generate workflow visualization."""
    output:
        "workflow_dag.png"
    shell:
        "snakemake --dag | dot -Tpng > {output}"

