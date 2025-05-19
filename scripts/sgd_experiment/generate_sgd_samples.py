"""
This script generates samples from two distributions P and Q using DPSGD (Differentially Private Stochastic Gradient Descent).
It trains CNN models on two different datasets (x0 and x1) and projects their parameters using a defined method.
The script uses parallel processing to speed up model training and handles both loading existing models
and generating new ones as needed.
"""

import os
import sys
import argparse
import numpy as np
import ast
# Navigate to the parent directory of the project structure
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_dir = os.path.join(project_dir, 'src')
fig_dir = os.path.join(project_dir, 'fig')
data_dir = os.path.join(project_dir, 'data')
log_dir = os.path.join(project_dir, 'log')

# Add the src directory to sys.path
if src_dir not in sys.path:
    sys.path.append(src_dir)

import mech.full_DPSGD as DPSGDModule

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
    parser.add_argument('--auditing_approach', type=str, default="1d_KLDivLoss",
                      help='Auditing approach to use (default: 1d_KLDivLoss)')
    parser.add_argument('--intermediate_epoch_list', type=str, default="[1, 5]",
                      help='List of intermediate epochs to store results from (format: "[1, 5, 10]")')

    args = parser.parse_args()

    # Convert string representation of list to actual list# Convert string representation of list to actual list
    try:
        args.intermediate_epoch_list = ast.literal_eval(args.intermediate_epoch_list)
        if not isinstance(args.intermediate_epoch_list, list):
            raise ValueError("intermediate_epoch_list must be a list")
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing intermediate_epoch_list: {e}")
        print(f"Received value: {args.intermediate_epoch_list}")
        sys.exit(1)

    data_args = {
        "method": "default",
        "data_dir": data_dir,
        "internal_result_path": os.path.join(args.internal_result_path, 'train')
    }

    log_dir = os.path.join(project_dir, 'log', args.model_name)
    os.makedirs(log_dir, exist_ok=True)
    sampler_args = DPSGDModule.generate_params(data_args=data_args, log_dir=log_dir, model_name=args.model_name, database_size=args.database_size, epochs=args.epochs, auditing_approach=args.auditing_approach, intermediate_epoch_list=args.intermediate_epoch_list)
    
    sampler = DPSGDModule.DPSGDSampler(sampler_args)

    sampler.preprocess(num_samples=args.num_train_samples, num_workers=args.num_workers)

    data_args = {
        "method": "default",
        "data_dir": data_dir,
        "internal_result_path": os.path.join(args.internal_result_path, 'test')
    }
    sampler_args = DPSGDModule.generate_params(data_args=data_args, log_dir=log_dir, model_name=args.model_name, database_size=args.database_size, epochs=args.epochs, auditing_approach=args.auditing_approach, intermediate_epoch_list=args.intermediate_epoch_list)
    sampler = DPSGDModule.DPSGDSampler(sampler_args)

    sampler.preprocess(num_samples=args.num_test_samples, num_workers=args.num_workers)



if __name__ == "__main__":
    main()