#!/bin/bash

# Test the latest trained model on capital questions
# Usage: ./scripts/test_latest_model.sh [base_model] [test_dir]

set -e

# Default values
TEST_DIR="${2:-training}"

# Auto-detect base model from most recent training run if not specified
if [ -z "$1" ]; then
    echo "🔍 Auto-detecting base model from most recent training run..."

    # Find the most recent training run directory
    LATEST_RUN=$(find "$TEST_DIR" -name "run-*" -type d | sort -V | tail -1)

    if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/training_log.json" ]; then
        # Extract model_id from training log
        BASE_MODEL=$(python3 -c "
import json
try:
    with open('$LATEST_RUN/training_log.json', 'r') as f:
        data = json.load(f)
        print(data.get('model_id', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'))
except:
    print('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        ")
        echo "✅ Detected model: $BASE_MODEL from $LATEST_RUN"
    else
        BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        echo "⚠️  No training log found, using default: $BASE_MODEL"
    fi
else
    BASE_MODEL="$1"
    echo "📝 Using specified model: $BASE_MODEL"
fi

echo "🧪 Testing Latest Trained Model"
echo "================================"
echo "Base Model: $BASE_MODEL"
echo "Test Directory: $TEST_DIR"
echo ""

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="src:$PYTHONPATH"

# Run the generation test (limited to 10 questions for quick testing)
python scripts/test_model_generation.py \
    --base-model "$BASE_MODEL" \
    --test-dir "$TEST_DIR" \
    --output "latest_model_test_results.json" \
    --limit 10

echo ""
echo "✅ Test completed! Check latest_model_test_results.json for detailed results."
