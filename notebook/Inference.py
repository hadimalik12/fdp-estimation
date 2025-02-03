import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
# Define Gaussian curve function
import sys
import time
import logging

# Navigate to the parent directory of the project structure
project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
src_dir = os.path.join(project_dir, 'src')

# Add the src directory to sys.path
sys.path.append(src_dir)

import mech.GaussianDist as GaussianModule
import mech.LapDist as LaplaceModule
import mech.toy_DPSGD as DP_SGDModule
import mech.Subsampling as SubsamplingModule
from utils.inference_processing import deviation_matrix, compute_eta_values

# Now you can use the functions
subfolder_path = os.path.expanduser("~/Documents/R/f-DP/results/results_Gaussian/results_N10000")

deviation_mat = deviation_matrix(subfolder_path)

eta_max = 15
eta_values = compute_eta_values(eta_max, subfolder_path)


#Function to process each eta in parallel
def process_eta(eta, estimator):
    eta_array = np.array([eta])
    output = estimator.build(eta=eta_array)
    beta_estimate = output["beta"]
    return beta_estimate

# Fixed parameters
train_sample_size = 10**4
test_sample_size = 10**4

# Generate parameters
kwargs = GaussianModule.generate_params(
    num_train_samples=train_sample_size, 
    num_test_samples=test_sample_size
)

# Initialize the estimator
estimator = GaussianModule.GaussianDistEstimator(kwargs)

# Use a specific number of workers
num_workers = 2  # Replace with desired number of workers
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    beta_estimates = list(executor.map(process_eta, eta_values, [estimator] * len(eta_values)))

print(beta_estimates)
# Convert beta_estimates to a NumPy array for further analysis
beta_estimates = np.ravel(beta_estimates)  # Converts multi-dimensional array to 1D

# Save results to a DataFrame
results_df = pd.DataFrame({
    'eta_values': eta_values,
    'beta_estimates': beta_estimates  # Flattening if beta_estimates is multi-dimensional
})

# Save to a CSV file
results_df.to_csv("beta_eta_results.csv", index=False)
print("Beta estimates and eta values saved to 'beta_eta_results.csv'")
