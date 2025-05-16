"""
This script generates samples from two distributions P and Q using DPSGD (Differentially Private Stochastic Gradient Descent).
It trains CNN models on two different datasets (x0 and x1) and projects their parameters to 1D for comparison.
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
log_dir = os.path.join(project_dir, 'log')
os.makedirs(fig_dir, exist_ok=True)

# Add the src directory to sys.path
if src_dir not in sys.path:
    sys.path.append(src_dir)

import mech.full_DPSGD as DPSGDModule

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

    sampler_args = DPSGDModule.generate_params(data_args=data_args, log_dir=log_dir, model_type=args.model_type)
    sampler = DPSGDModule.DPSGDSampler(sampler_args)

    samples_P, samples_Q = sampler.preprocess(num_samples=args.num_samples, num_workers=args.num_workers)
    
    # Save samples to CSV files
    np.savetxt(os.path.join(data_dir, 'prediction_d.csv'), samples_P, delimiter=',')
    np.savetxt(os.path.join(data_dir, 'prediction_dprime.csv'), samples_Q, delimiter=',')

if __name__ == "__main__":
    main()