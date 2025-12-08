#!/bin/bash
# Test reproducibility by training the same model twice and comparing results

set -e

echo "=== Reproducibility Test ==="
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Configuration
DATA_PATH="data/raw/iris.csv"
MODEL_TYPE="random_forest"
RANDOM_STATE=42
N_ESTIMATORS=50
MAX_DEPTH=5

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found at $DATA_PATH"
    echo "Please ensure iris.csv is in data/raw/ directory"
    exit 1
fi

# Train first model
echo "Training first model..."
python src/models/train_model.py \
    --data-path "$DATA_PATH" \
    --model-type "$MODEL_TYPE" \
    --random-state $RANDOM_STATE \
    --n-estimators $N_ESTIMATORS \
    --max-depth $MAX_DEPTH \
    --run-name "reproducibility-test-1" \
    > /tmp/train1.log 2>&1

RUN1_ID=$(grep "MLflow Run ID:" /tmp/train1.log | awk '{print $NF}')
echo "First run ID: $RUN1_ID"
echo ""

# Extract metrics from first run
ACCURACY1=$(grep "accuracy:" /tmp/train1.log | awk '{print $NF}')
F1_1=$(grep "f1:" /tmp/train1.log | awk '{print $NF}')

echo "First run metrics:"
echo "  Accuracy: $ACCURACY1"
echo "  F1: $F1_1"
echo ""

# Wait a bit
sleep 2

# Train second model with same parameters
echo "Training second model with identical parameters..."
python src/models/train_model.py \
    --data-path "$DATA_PATH" \
    --model-type "$MODEL_TYPE" \
    --random-state $RANDOM_STATE \
    --n-estimators $N_ESTIMATORS \
    --max-depth $MAX_DEPTH \
    --run-name "reproducibility-test-2" \
    > /tmp/train2.log 2>&1

RUN2_ID=$(grep "MLflow Run ID:" /tmp/train2.log | awk '{print $NF}')
echo "Second run ID: $RUN2_ID"
echo ""

# Extract metrics from second run
ACCURACY2=$(grep "accuracy:" /tmp/train2.log | awk '{print $NF}')
F1_2=$(grep "f1:" /tmp/train2.log | awk '{print $NF}')

echo "Second run metrics:"
echo "  Accuracy: $ACCURACY2"
echo "  F1: $F1_2"
echo ""

# Compare results
echo "=== Comparison ==="
if [ "$ACCURACY1" = "$ACCURACY2" ] && [ "$F1_1" = "$F1_2" ]; then
    echo "✓ SUCCESS: Results are identical!"
    echo "  Both runs have accuracy: $ACCURACY1"
    echo "  Both runs have F1: $F1_1"
    echo ""
    echo "Reproducibility verified ✓"
    exit 0
else
    echo "✗ FAILURE: Results differ!"
    echo "  Run 1 - Accuracy: $ACCURACY1, F1: $F1_1"
    echo "  Run 2 - Accuracy: $ACCURACY2, F1: $F1_2"
    echo ""
    echo "Possible causes:"
    echo "  - Non-deterministic algorithm behavior"
    echo "  - Different random state"
    echo "  - Different library versions"
    exit 1
fi

