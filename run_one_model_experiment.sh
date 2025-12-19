#!/bin/bash

# Script to run benchmark for one model 4 times
# Usage: ./run_one_model_experiment.sh <model_name> [tasks_type] [concurrency] [num_servers]

set -e  # Exit on error

# Check if model_name is provided
if [ -z "$1" ]; then
    echo "Error: Model name is required"
    echo "Usage: $0 <model_name> [tasks_type] [concurrency] [num_servers]"
    echo ""
    echo "Arguments:"
    echo "  model_name   : Model name to test (required, e.g., gpt-4o-mini, gemini-2.5-flash)"
    echo "  tasks_type   : Type of tasks (optional, default: general_test)"
    echo "                 Options: day, pro, general_test"
    echo "  concurrency  : Number of concurrent requests (optional, default: from config.json)"
    echo "  num_servers  : Number of servers for agent construction (optional, default: from config.json)"
    echo ""
    echo "Examples:"
    echo "  $0 gpt-4o-mini"
    echo "  $0 gemini-2.5-flash general_test"
    echo "  $0 gpt-4o-mini general_test 5 20"
    exit 1
fi

MODEL_NAME="$1"
TASKS_TYPE="${2:-general_test}"
CONCURRENCY="${3:-}"
NUM_SERVERS="${4:-}"

# Validate tasks_type
if [[ ! "$TASKS_TYPE" =~ ^(day|pro|general_test)$ ]]; then
    echo "Error: Invalid tasks_type: $TASKS_TYPE"
    echo "Valid options: day, pro, general_test"
    exit 1
fi

# Check if model exists in llm_config.json
if [ ! -f "configs/llm_config.json" ]; then
    echo "Error: configs/llm_config.json not found"
    exit 1
fi

# Check if model exists in config (optional check, just for validation)
MODEL_EXISTS=$(python3 -c "
import json
import sys
try:
    with open('configs/llm_config.json', 'r') as f:
        config = json.load(f)
    models = [item.get('model', '') for item in config]
    if '$MODEL_NAME' in models:
        print('found')
    else:
        print('not_found')
except Exception as e:
    print('error')
    sys.exit(1)
")

if [ "$MODEL_EXISTS" = "not_found" ]; then
    echo "Warning: Model '$MODEL_NAME' not found in configs/llm_config.json"
    echo "Available models:"
    python3 -c "
import json
with open('configs/llm_config.json', 'r') as f:
    config = json.load(f)
    for item in config:
        print(f\"  - {item.get('model', 'N/A')}\")
" || true
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create safe model name for file naming (replace / with _)
SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's/\//_/g')

# Prepare base command
BASE_CMD="python runbenchmark.py --model \"$MODEL_NAME\" --tasks_type \"$TASKS_TYPE\""

# Add optional parameters
if [ -n "$CONCURRENCY" ]; then
    BASE_CMD="$BASE_CMD --concurrency $CONCURRENCY"
fi

if [ -n "$NUM_SERVERS" ]; then
    BASE_CMD="$BASE_CMD --num_servers $NUM_SERVERS"
fi

# Print configuration
echo "=========================================="
echo "Running Benchmark for Model: $MODEL_NAME"
echo "=========================================="
echo "Tasks Type: $TASKS_TYPE"
echo "Concurrency: ${CONCURRENCY:-from config.json}"
echo "Num Servers: ${NUM_SERVERS:-from config.json}"
echo "Output Directory: results/"
echo "=========================================="
echo ""

# Run 4 experiments
for i in {1..4}; do
    OUTPUT_NAME="${SAFE_MODEL_NAME}_${TASKS_TYPE}_run${i}"
    
    echo ""
    echo "=========================================="
    echo "Run $i/4: $OUTPUT_NAME"
    echo "=========================================="
    
    # Build command with output_name
    CMD="$BASE_CMD --output_name \"$OUTPUT_NAME\""
    
    # Execute command
    echo "Command: $CMD"
    echo ""
    
    if eval "$CMD"; then
        echo ""
        echo "✅ Run $i/4 completed successfully"
        echo "   Results saved to: results/${OUTPUT_NAME}_results.json"
        echo "   Logs saved to: logs/${OUTPUT_NAME}_response.json"
    else
        echo ""
        echo "❌ Run $i/4 failed!"
        echo "   Check the error messages above"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "✅ All 4 runs completed successfully!"
echo "=========================================="
echo ""
echo "Result files:"
for i in {1..4}; do
    OUTPUT_NAME="${SAFE_MODEL_NAME}_${TASKS_TYPE}_run${i}"
    if [ -f "results/${OUTPUT_NAME}_results.json" ]; then
        echo "  - results/${OUTPUT_NAME}_results.json"
    fi
done
echo ""
echo "You can now calculate pass@4 and avg@4 using these result files."

