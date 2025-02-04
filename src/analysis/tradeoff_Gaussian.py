import numpy as np
from scipy.special import erf
from scipy.stats import norm

def Gaussian_compute_tradeoff_curve(eta):
    """
    Parameters:
    eta (float): Input parameter (should be > 0).

    Returns:
    float: Computed (alpha, beta) value for Guassian pair with Mean 0, Variance 1; Mean 1, Variance 1 
    """
    alpha = 0.5 - 0.5 * erf((0.5 + np.log(1 / eta)) / np.sqrt(2))
    beta = 0.5 + 0.5 * erf((-0.5 + np.log(1 / eta)) / np.sqrt(2))
    return (alpha, beta)

def Gaussian_curve(alpha):
    """
    Parameters:
    alpha (float): Input parameter (should be > 0).

    Returns:
    float: Computed (alpha, beta) value for Guassian pair with Mean 0, Variance 1; Mean 1, Variance 1
    """
    return norm.cdf(norm.ppf(1 - alpha) - 1)