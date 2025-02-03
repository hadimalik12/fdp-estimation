import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad

import os
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

# # Set R library path
# robjects.r('.libPaths("/home/snek/R/x86_64-pc-linux-gnu-library/4.1")')
# os.environ['R_LIBS_SITE'] = "/home/snek/R/x86_64-pc-linux-gnu-library/4.1"

# Import KernSmooth
KernSmooth = importr("KernSmooth")


def inner_int(x, eta, grid_width, p, p_q_ratio):
    p_vector = np.copy(p)
    p_vector[p_q_ratio <= eta + x] = 0
    int_value = np.sum(p_vector * grid_width)
    return int_value

def alpha_value(eta, x_vector, h, grid_width, hat_p, hat_p_q_ratio):
    alpha = 1 - np.sum([
        inner_int(x, eta, grid_width, hat_p, hat_p_q_ratio) / h * (x_vector[1] - x_vector[0])
        for x in x_vector
    ])
    return alpha

def beta_value(eta, x_vector, h, grid_width, hat_q, hat_p_q_ratio):
    beta = np.sum([
        inner_int(x, eta, grid_width, hat_q, hat_p_q_ratio) / h * (x_vector[1] - x_vector[0])
        for x in x_vector
    ])
    return beta

def KDE_Estimator(eta_max, Mechanism, x1, x2, N, h):
    p_sample = np.array([Mechanism(x1) for _ in range(N)])
    q_sample = np.array([Mechanism(x2) for _ in range(N)])
    
    t_min = min(np.min(p_sample), np.min(q_sample))
    t_max = max(np.max(p_sample), np.max(q_sample))
    
    range_x = FloatVector([t_min, t_max])  # R-compatible vector

    # Convert samples to R-compatible vectors
    p_sample_r = FloatVector(p_sample)
    q_sample_r = FloatVector(q_sample)

    # Compute bandwidths using dpik
    bw_p = KernSmooth.dpik(p_sample_r, kernel='normal', gridsize=1000, range_x=range_x)
    bw_q = KernSmooth.dpik(q_sample_r, kernel='normal', gridsize=1000, range_x=range_x)

    # Apply Gaussian KDE with computed bandwidths
    kde_p = gaussian_kde(p_sample, bw_method=float(bw_p[0]))
    kde_q = gaussian_kde(q_sample, bw_method=float(bw_q[0]))
    
    x_grid = np.linspace(t_min, t_max, 1000)
    hat_p = kde_p(x_grid)
    hat_q = kde_q(x_grid)
    grid_width = x_grid[1] - x_grid[0]
    
    hat_p_q_ratio = np.divide(hat_p, np.maximum(hat_q, 1e-10), out=np.zeros_like(hat_p))

    hat_p_q_ratio[np.isinf(hat_p_q_ratio)] = 0
    hat_p_q_ratio[np.isnan(hat_p_q_ratio)] = 0
    
    x_vector = np.linspace(-h / 2, h / 2, 1000)
    eta_vector = np.linspace(0, eta_max, 1000)
    
    alpha = np.array([alpha_value(eta, x_vector, h, grid_width, hat_p, hat_p_q_ratio) for eta in eta_vector])
    beta = np.array([beta_value(eta, x_vector, h, grid_width, hat_q, hat_p_q_ratio) for eta in eta_vector])
    
    output_df = {"alpha": alpha, "beta": beta}
    return output_df
