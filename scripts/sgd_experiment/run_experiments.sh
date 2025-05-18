#!/bin/bash

# Default values
NUM_WORKERS=20
EPOCHS_LIST="1,5,9,13,17"
DATABASE_SIZE=1000
DATABASE_NAME="white_cifar10"
MODEL_NAME="convnet_balanced"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --epochs_list)
            EPOCHS_LIST="$2"
            shift 2
            ;;
        --database_size)
            DATABASE_SIZE="$2"
            shift 2
            ;;
        --database_name)
            DATABASE_NAME="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "Running experiments with the following configuration:"
echo "Number of workers: $NUM_WORKERS"
echo "Epochs list: $EPOCHS_LIST"
echo "Database size: $DATABASE_SIZE"
echo "Database name: $DATABASE_NAME"
echo "Model name: $MODEL_NAME"

# Run the experiment
python -m mech.dpsgd_algs.run_parallel_experiments \
    --num_workers $NUM_WORKERS \
    --epochs_list $EPOCHS_LIST \
    --database_size $DATABASE_SIZE \
    --database_name $DATABASE_NAME \
    --model_name $MODEL_NAME 