import numpy as np
from scipy.special import erf
from scipy.stats import norm
from scipy.optimize import brentq

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

def Gaussian_curve(alpha, mean_difference = 1):
    """
    Parameters:
    alpha (float): Input parameter (should be > 0).

    Returns:
    float: Computed (alpha, beta) value for Guassian pair with Mean mu1, Variance 1; Mean mu2, Variance 1, mean_difference = mu2 - mu1
    """
    return norm.cdf(norm.ppf(1 - alpha) - mean_difference)

def find_valid_indices(array):
    """
    Find indices of non-NaN real values from an array.
    
    Args:
        array: Input numpy array
        
    Returns:
        numpy.ndarray: Array of indices where values are non-NaN and real
    """
    # Check for NaN values
    non_nan_mask = ~np.isnan(array)
    # Check for real values (not complex)
    real_mask = np.isreal(array)
    # Combine both conditions
    valid_mask = non_nan_mask & real_mask
    # Get indices where both conditions are True
    valid_indices = np.where(valid_mask)[0]
    return valid_indices

def find_optimal_mu(alpha_values, beta_values, tolerance=10**(-3), mu_range=(-1, 10)):
    """
    Find the optimal mu-Gaussian curve that minimizes the error between Gaussian curve and estimated tradeoff curve
    using binary search.
    
    Args:
        alpha_values: Array of alpha values
        beta_values: Array of beta values
        tolerance: Tolerance for mu (default: 10**(-3))
        mu_range: Tuple of (min_mu, max_mu) for search range (default: (-10, 10))
        
    Returns:
        float: Optimal mu value with specified precision
    """
    def calculate_error(mu):
        return np.sum(np.abs(Gaussian_curve(alpha_values, mean_difference=mu) - beta_values))
    
    indices = find_valid_indices(Gaussian_curve(alpha_values, mean_difference = 1))
    alpha_values = alpha_values[indices]
    beta_values = beta_values[indices]  
    
    left, right = mu_range
    best_mu = None
    best_error = float('inf')
    
    # Binary search with specified precision
    while right - left > tolerance:
        mid = (left + right) / 2
        mid_error = calculate_error(mid)
        
        # Check slightly larger and smaller values
        mid_plus = mid + tolerance
        mid_minus = mid - tolerance
        error_plus = calculate_error(mid_plus)
        error_minus = calculate_error(mid_minus)
        
        # Update best result if we found a better solution
        if mid_error < best_error:
            best_error = mid_error
            best_mu = mid
        if error_plus < best_error:
            best_error = error_plus
            best_mu = mid_plus
        if error_minus < best_error:
            best_error = error_minus
            best_mu = mid_minus
        
        # Update search range
        if error_plus < error_minus:
            left = mid
        else:
            right = mid
    
    return best_mu

def delta_eps(eps, mu):
    """Forward δ(ε) function for μ-GDP."""
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)

def find_eps(mu, delta, eps_lower=0.0, eps_upper=20.0):
    """
    Finds ε such that δ(ε) equals the given δ for a μ-GDP mechanism.

    Parameters:
        mu (float): μ value of the GDP mechanism
        delta (float): target δ
        eps_lower (float): lower bound for ε search
        eps_upper (float): upper bound for ε search

    Returns:
        float: ε such that δ(ε) = delta
    """
    f = lambda eps: delta_eps(eps, mu) - delta
    return brentq(f, eps_lower, eps_upper)