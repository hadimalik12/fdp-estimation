import logging
import numpy as np

from utils.utils import extract_order_of_magnitude


def acc_evaluate(thm_tradeoff_func, estimator_cls, estimator_params, eta_array, num_train_samples, num_test_samples = 100000, classifier_args=None, logfile_path = "/../log/tradeoff-kNN-Gaussian.log", nworkers=1):
    # File handler attached for this function
    logger = logging.getLogger("func_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = True
    
    # Function logic here...   
    estimator = estimator_cls(estimator_params)
    output = estimator.build(eta = eta_array, nworkers=nworkers)
    alpha_values_estimate = output["alpha"]
    beta_values_estimate = output["beta"]
    
    alpha_values_thm, beta_values_thm = thm_tradeoff_func(eta_array)
    beta_values_thm - beta_values_estimate
    
    error_estimate = np.sum(np.abs((beta_values_thm - beta_values_estimate)))/len(beta_values_estimate)
    logger.info(f"For {estimator_cls.__name__}, estimation is {error_estimate} at error order {extract_order_of_magnitude(error_estimate)} with {num_train_samples} samples")
    
    # Detach the file handler after function execution
    logger.removeHandler(file_handler)
    file_handler.close()


# This is corresponding to the theoretical guarantee of Baybox kNN estimator; given in Thm 5.2 in our paper
def knn_baybox_acc_bound_1d(n, gamma):
    c_d = 3.8637  # Given value of c_d
    result = 12 * np.sqrt((2 * c_d ** 2 / n) * np.log(4 / gamma))
    return result