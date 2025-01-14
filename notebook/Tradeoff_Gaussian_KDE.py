import os
import sys
import numpy as np
import pandas as pd
import os
import sys

# Dynamically add the 'src' directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(project_root)

# Import KDE_Estimator
from estimator.estimator_kde import KDE_Estimator

def Sum_Gauss(x, sigma=1):
    s = np.sum(x) + np.random.normal(0, sigma)
    return s

def main(N, h, eta_max, iteration, sigma=1, results_dir="results"):
    """
    Main function to run the KDE estimator and save the results.

    Parameters:
    - N: Number of iterations.
    - h: Bandwidth parameter for KDE.
    - eta_max: Maximum eta value for KDE.
    - iteration: Current iteration index.
    - sigma: Standard deviation for the Gaussian mechanism (default: 1).
    - results_dir: Directory where results will be saved (default: 'results').
    """
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Define parameters
    x1 = np.array([1] + [0] * 9)
    x2 = np.zeros(10)

    # Run KDE_Estimator
    output_df = KDE_Estimator(eta_max=eta_max, Mechanism=lambda x: Sum_Gauss(x, sigma=sigma), x1=x1, x2=x2, N=N, h=h)

    # Generate dynamic output filename
    script_name = os.path.basename(__file__).replace(".py", "")
    output_file = os.path.join(results_dir, f"{script_name}_N{N}_h{h}_eta{eta_max}_iter{iteration}.csv")

    # Save the output
    try:
        output_df = pd.DataFrame(output_df)  # Convert list of dictionaries to a DataFrame
        output_df.to_csv(output_file, index=False)
        print(f"Iteration {iteration}: Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run KDE Estimation with dynamic output file naming.")
    parser.add_argument("--N", type=int, default=100, help="Number of iterations (default: 100).")
    parser.add_argument("--h", type=float, default=0.1, help="Bandwidth parameter for KDE (default: 0.1).")
    parser.add_argument("--eta_max", type=int, default=15, help="Maximum eta value for KDE (default: 15).")
    parser.add_argument("--iteration", type=int, required=True, help="Current iteration index.")
    parser.add_argument("--sigma", type=float, default=1, help="Standard deviation for Gaussian mechanism (default: 1).")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results (default: 'results').")

    args = parser.parse_args()
    main(N=args.N, h=args.h, eta_max=args.eta_max, iteration=args.iteration, sigma=args.sigma, results_dir=args.results_dir)
