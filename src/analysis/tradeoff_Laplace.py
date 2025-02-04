import numpy as np
from scipy.special import erf

def Laplace_compute_tradeoff_curve(eta):
    """
    Compute (alpha_eta, beta_eta) based on the piecewise-defined function.

    Parameters:
    eta (float or numpy array): Input value(s) for eta.

    Returns:
    tuple: Two numpy arrays (alpha_eta, beta_eta).
    """
    eta = np.array(eta)  # Ensure eta is a numpy array for vectorized operations
    
    # Initialize alpha and beta arrays
    alpha = np.zeros_like(eta)
    beta = np.zeros_like(eta)
    
    # Define intervals
    e = np.e
    conditions = [
        (eta > 0) & (eta <= e**-1),               # First case
        (eta > e**-1) & (eta <= e**(-1/2)),      # Second case
        (eta > e**(-1/2)) & (eta <= e**(1/2)),   # Third case
        (eta > e**(1/2)) & (eta <= e),           # Fourth case
        (eta > e),                               # Fifth case
    ]
    
    # Apply the piecewise definitions
    alpha[conditions[0]] = 0
    beta[conditions[0]] = 1

    alpha[conditions[1]] = (np.exp(-1) / 2)
    beta[conditions[1]] = 1 / 2

    alpha[conditions[2]] = eta[conditions[2]] * np.exp(-1/2) / 2
    beta[conditions[2]] = np.exp(-1/2) / (2 * eta[conditions[2]])

    alpha[conditions[3]] = 1 / 2
    beta[conditions[3]] = np.exp(-1)/2

    alpha[conditions[4]] = 1
    beta[conditions[4]] = 0
    
    return alpha, beta


# Define the Laplace curve function
def Laplace_curve(alpha):
    mu_2 = 1
    if alpha < np.exp(-mu_2) / 2:
        return 1 - np.exp(mu_2) * alpha
    elif np.exp(-mu_2) / 2 <= alpha <= 1 / 2:
        return np.exp(-mu_2) / (4 * alpha)
    elif alpha > 1 / 2:
        return np.exp(-mu_2) * (1 - alpha)
    else:
        return 0