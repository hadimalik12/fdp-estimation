import numpy as np
from scipy.stats import norm
from itertools import combinations

def calc_mu(x, m, eta_learn, T_):
    """
    Calculate mu based on the given parameters.
    
    Parameters: 
    x: database, the input of the SGD algorithm
    m: size of the subset of database in each interation
    eta_learn: learning rate of each step
    T_: the number of iteration
    """
    mu = np.sum(eta_learn * (1 - eta_learn)**(T_ - np.array(x)) / m)
    return mu

def toyDPSGD_compute_tradeoff_curve(alpha, kwargs):
    """
    Compute beta of corresponding alpha for the mechanism toyDPSGD. The neighboring input databases is fixed to one with all zero and one with all zero except one 1  
    

    Parameters:
    alpha (float or numpy array): Input value(s) for alpha.
    kwargs: parameters of the SGD algorithms 

    Returns:
    One numpy arrays beta_eta.
    """
    T = kwargs["sgd_alg"]["T"]
    m = kwargs["sgd_alg"]["m"]
    eta_learn = kwargs["sgd_alg"]["eta"]
    sigma = kwargs["sgd_alg"]["sigma"]
    
    alpha_list = np.array(alpha)
    beta_list = []
        
    # Initialize mu_vector
    mu_vector = []
    for k in range(1, T + 1):
        k_combinations = list(combinations(range(1, T + 1), k))
        mu_values = [calc_mu(x, m=m, eta_learn=eta_learn, T_=T) for x in k_combinations]
        mu_vector.extend(mu_values)
        
    # Calculate sigma_tilde
    sigma_tilde = eta_learn * sigma * np.sqrt((1 - (1 - eta_learn)**(2 * T)) / (1 - (1 - eta_learn)**2))
    tmp_array = np.array(mu_vector) / sigma_tilde
    T_hat = 2**T
    
    for alpha_ in alpha_list:
        beta_list.append(np.sum(norm.cdf(norm.ppf(1 - alpha_) - tmp_array) / T_hat))
    
    return np.array(beta_list)