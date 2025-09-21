#!/bin/bash

# Test the latest trained model on capital questions
# Usage: ./scripts/test_latest_model.sh [base_model] [test_dir]

set -e

# Default values
BASE_MODEL="${1:-microsoft/DialoGPT-small}"
TEST_DIR="${2:-training}"

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

# Run the generation test
python scripts/test_model_generation.py \
    --base-model "$BASE_MODEL" \
    --test-dir "$TEST_DIR" \
    --output "latest_model_test_results.json"

echo ""
echo "✅ Test completed! Check latest_model_test_results.json for detailed results."