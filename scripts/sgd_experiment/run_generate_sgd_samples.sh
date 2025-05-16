#!/bin/bash

# Default values
NUM_SAMPLES=32
NUM_WORKERS=32
INTERNAL_RESULT_PATH="/scratch/bell/wei402/fdp-estimation/results"
MODEL_TYPE="CNN"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --internal_result_path)
            INTERNAL_RESULT_PATH="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "Running generate_sgd_samples.py with:"
echo "Number of samples: $NUM_SAMPLES"
echo "Number of workers: $NUM_WORKERS"
echo "Internal result path: $INTERNAL_RESULT_PATH"
echo "Model type: $MODEL_TYPE"
# Run the script
python scripts/sgd_experiment/generate_sgd_samples.py \
    --num_samples $NUM_SAMPLES \
    --num_workers $NUM_WORKERS \
    --internal_result_path $INTERNAL_RESULT_PATH \
    --model_type $MODEL_TYPE