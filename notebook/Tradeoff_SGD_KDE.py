import numpy as np
import pandas as pd
import os
import sys
import rpy2.robjects as robjects

# Set R library path for all R sessions
robjects.r('.libPaths("/home/snek/R/x86_64-pc-linux-gnu-library/4.1")')
os.environ['R_LIBS_SITE'] = "/home/snek/R/x86_64-pc-linux-gnu-library/4.1"

# Debugging: Print current R library paths
print("Current R library paths:", robjects.r('.libPaths()'))

# Add the path to the `src/estimator/` directory
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
sys.path.append(project_root)

# Import KDE_Estimator from estimator_kde
from estimator.estimator_kde import KDE_Estimator

# Define Noisy_SGD
def Noisy_SGD(x, theta_0, T_, m, eta_learn, sigma):
    theta = np.float64(theta_0)
    for _ in range(T_):
        x_subsample = np.random.choice(x, size=m, replace=False)
        theta -= eta_learn * (np.sum(theta - x_subsample, dtype=np.float64) / m + np.random.normal(0, sigma))
    return theta

def main(N, h, eta_max, iteration, sigma, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    theta_0 = np.float64(0)
    m = 5
    eta_learn = 0.2
    T_ = 10
    n = 10
    x1 = np.zeros(n, dtype=np.float64)
    x2 = np.array([1] + [0] * (n - 1), dtype=np.float64)

    Mechanism = lambda x: Noisy_SGD(x, theta_0, T_, m, eta_learn, sigma)

    try:
        output_df = KDE_Estimator(eta_max=eta_max, Mechanism=Mechanism, x1=x1, x2=x2, N=N, h=h)
    except Exception as e:
        print(f"Error during KDE Estimation: {e}")
        return

    script_name = os.path.basename(__file__).replace(".py", "")
    output_file = os.path.join(results_dir, f"{script_name}_N{N}_h{h}_eta{eta_max}_iter{iteration}.csv")

    try:
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(output_file, index=False)
        print(f"Iteration {iteration}: Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving file to {output_file}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run KDE Estimation with dynamic output file naming.")
    parser.add_argument("--N", type=int, default=10000, help="Number of iterations (default: 10000).")
    parser.add_argument("--h", type=lambda x: np.float64(x), default=np.float64(0.1), help="Bandwidth parameter for KDE (default: 0.1).")
    parser.add_argument("--eta_max", type=int, default=15, help="Maximum eta value for KDE (default: 15).")
    parser.add_argument("--iteration", type=int, required=True, help="Current iteration index.")
    parser.add_argument("--sigma", type=lambda x: np.float64(x), default=np.float64(0.2), help="Standard deviation for Gaussian mechanism (default: 0.2).")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results (default: 'results').")

    args = parser.parse_args()
    main(
        N=args.N,
        h=args.h,
        eta_max=args.eta_max,
        iteration=args.iteration,
        sigma=args.sigma,
        results_dir=args.results_dir
    )
