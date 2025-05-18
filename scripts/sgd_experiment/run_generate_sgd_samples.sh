#!/bin/bash

# Default values
NUM_SAMPLES=32
NUM_WORKERS=32
INTERNAL_RESULT_PATH="/scratch/bell/wei402/fdp-estimation/results"
MODEL_NAME="convnet_balanced"
NUM_TRAIN_SAMPLES=1000
NUM_TEST_SAMPLES=1000
DATABASE_SIZE=1000
EPOCHS=20
AUDITING_APPROACH="1d_cross_entropy"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_train_samples)
            NUM_TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --num_test_samples)
            NUM_TEST_SAMPLES="$2"
            shift 2
            ;;
        --database_size)
            DATABASE_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
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
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --auditing_approach)
            AUDITING_APPROACH="$2"
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
echo "Number of training samples: $NUM_TRAIN_SAMPLES"
echo "Number of test samples: $NUM_TEST_SAMPLES"
echo "Database size: $DATABASE_SIZE"
echo "Number of epochs: $EPOCHS"
echo "Number of workers: $NUM_WORKERS"
echo "Internal result path: $INTERNAL_RESULT_PATH"
echo "Model name: $MODEL_NAME"
echo "Auditing approach: $AUDITING_APPROACH"

# Run the script
python scripts/sgd_experiment/generate_sgd_samples.py \
        --num_train_samples $NUM_TRAIN_SAMPLES \
        --num_test_samples $NUM_TEST_SAMPLES \
        --database_size $DATABASE_SIZE \
        --epochs $EPOCHS \
        --num_workers $NUM_WORKERS \
        --internal_result_path $INTERNAL_RESULT_PATH \
        --model_name $MODEL_NAME \
        --auditing_approach $AUDITING_APPROACH