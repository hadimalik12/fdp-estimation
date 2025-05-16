"""
This script generates models from two distributions P and Q using DPSGD (Differentially Private Stochastic Gradient Descent).
It trains CNN models on two different datasets (x0 and x1).
The script uses parallel processing to speed up model training and handles both loading existing models
and generating new ones as needed.
"""

import os
import sys
import argparse
import numpy as np

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

def main():
    parser = argparse.ArgumentParser(description='Generate samples from DPSGD distributions')
    parser.add_argument('--num_samples', type=int, default=1,
                      help='Number of samples to generate (default: 1)')
    parser.add_argument('--internal_result_path', type=str, 
                      default="/scratch/bell/wei402/fdp-estimation/results",
                      help='Path to store internal results')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers to use for parallel processing (default: 1)')
    parser.add_argument('--model_type', type=str, default="CNN",
                      help='Type of model to use (default: CNN)')
    args = parser.parse_args()

    data_args = {
        "method": "default",
        "data_dir": data_dir,
        "internal_result_path": args.internal_result_path
    }

    log_dir = os.path.join(project_dir, 'log', args.model_type)
    os.makedirs(log_dir, exist_ok=True)
    sampler_args = DPSGDModule.generate_params(data_args=data_args, log_dir=log_dir, model_type=args.model_type)

    parallel_train_models(sampler_args, args.num_samples, num_workers=args.num_workers)

if __name__ == "__main__":
    main()