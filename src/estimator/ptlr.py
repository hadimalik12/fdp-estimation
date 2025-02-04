import numpy as np
import logging
import time
from scipy.stats import gaussian_kde
from scipy.integrate import quad

import os
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

# Import KernSmooth from R
KernSmooth = importr("KernSmooth")


class _PTLREstimator:
    def __init__(self, kwargs):
        self.sampler = None
        self.num_samples = kwargs["num_samples"]
        self.h = kwargs["h"] # ptlr algorithm's parameter

        self.output_ = None

    def check_sampler(self):
        assert self.sampler is not None, "ERR: you need to use a super class or to define the sampler first" 

    def build(self, eta_max):
        self.check_sampler()
        
        # Generate samples
        logging.info('Generating samples...')
        tic = time.perf_counter()
        (p_sample, q_sample) = self.sampler.preprocess(self.num_samples)
        p_sample = p_sample.ravel()
        q_sample = q_sample.ravel()
        logging.info(f"Generated {self.num_samples} samples in {time.perf_counter() - tic:0.4f} seconds")
        
        # ptlr estimation
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
        
        x_vector = np.linspace(-self.h / 2, self.h / 2, 1000)
        eta_array = np.linspace(0, eta_max, 1000)
        
        alpha_array = np.array([alpha_value(eta, x_vector, self.h, grid_width, hat_p, hat_p_q_ratio) for eta in eta_array])
        beta_array = np.array([beta_value(eta, x_vector, self.h, grid_width, hat_q, hat_p_q_ratio) for eta in eta_array])
        
        logging.info(f"Estimation is completed in {time.perf_counter() - tic:0.4f}s")

        # Output
        self.output_ = {
            "eta": eta_array, "alpha": alpha_array, "beta": beta_array,
            "sample_size": self.num_samples, 
        }

        return self.output_


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