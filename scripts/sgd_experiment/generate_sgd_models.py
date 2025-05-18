"""
This script generates models from two distributions P and Q using DPSGD (Differentially Private Stochastic Gradient Descent).
It trains CNN models on two different datasets (x0 and x1).
The script uses parallel processing to speed up model training and handles both loading existing models
and generating new ones as needed.
"""

import os
import sys
import argparse
import time

# Navigate to the parent directory of the project structure
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_dir = os.path.join(project_dir, 'src')
fig_dir = os.path.join(project_dir, 'fig')
data_dir = os.path.join(project_dir, 'data')
os.makedirs(fig_dir, exist_ok=True)

# Add the src directory to sys.path
if src_dir not in sys.path:
    sys.path.append(src_dir)

import mech.full_DPSGD as DPSGDModule
from mech.full_DPSGD import parallel_train_models
from mech.model_architecture import MODEL_MAPPING

def main():
    parser = argparse.ArgumentParser(description='Generate samples from DPSGD distributions')
    parser.add_argument('--num_train_samples', type=int, default=1000,
                      help='Number of samples to generate for training (default: 1000)')
    parser.add_argument('--num_test_samples', type=int, default=1000,
                      help='Number of samples to generate for testing (default: 1000)')
    parser.add_argument('--internal_result_path', type=str, 
                      default="/scratch/bell/wei402/fdp-estimation/results",
                      help='Path to store internal results')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers to use for parallel processing (default: 1)')
    parser.add_argument('--model_name', type=str, default="convnet_balanced",
                      help='Type of model to use (default: convnet_balanced)')
    parser.add_argument('--database_size', type=int, default=1000,
                      help='Size of the database to use for training')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of epochs to train for')

    args = parser.parse_args()
    internal_result_path_prefix = os.path.join(args.internal_result_path, args.model_name+'_'+str(args.database_size)+'_'+str(args.epochs))

    data_args = {
        "method": "default",
        "data_dir": data_dir,
        "internal_result_path": os.path.join(internal_result_path_prefix, 'train')
    }

    log_dir = os.path.join(project_dir, 'log', args.model_name)
    os.makedirs(log_dir, exist_ok=True)

    start_time = time.time()
    sampler_args = DPSGDModule.generate_params(data_args=data_args, log_dir=log_dir, model_name=args.model_name, database_size=args.database_size)
    parallel_train_models(sampler_args, args.num_train_samples, num_workers=args.num_workers)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Training models for {args.num_train_samples} samples done in {elapsed_time:.2f} seconds")

    data_args = {
        "method": "default",
        "data_dir": data_dir,
        "internal_result_path": os.path.join(internal_result_path_prefix, 'test')
    }

    start_time = time.time()
    sampler_args = DPSGDModule.generate_params(data_args=data_args, log_dir=log_dir, model_name=args.model_name, database_size=args.database_size)
    parallel_train_models(sampler_args, args.num_test_samples, num_workers=args.num_workers)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Training models for {args.num_test_samples} samples done in {elapsed_time:.2f} seconds")

    

if __name__ == "__main__":
    main()