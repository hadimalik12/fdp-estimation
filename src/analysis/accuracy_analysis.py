import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from utils.utils import extract_order_of_magnitude
from analysis.tradeoff_Gaussian import Gaussian_curve


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


def create_plot(omega, alpha_estimate, beta_estimate, fine_points, alpha, beta, the_alpha, the_beta, filename="plot.png", claimed_curve=Gaussian_curve):
    """
    Create a plot with a Gaussian curve, KDE scatter, and a highlighted critical region.

    Parameters:
    - omega: Width/height of the shaded square (float).
    - alpha_estimate: The x-coordinate of the critical region center (float).
    - beta_estimate: The y-coordinate of the critical region center (float).
    - fine_points: Array of x-values for the Gaussian curve (array-like).
    - Gaussian_curve: Function to compute the Gaussian curve (function).
    - alpha: Array of x-coordinates for KDE scatter points (array-like).
    - beta: Array of y-coordinates for KDE scatter points (array-like).
    - the_alpha: The x-coordinate of the critical point (float).
    - the_beta: The y-coordinate of the second critical point (float).
    - filename: Name of the file to save the plot (str).
    """
    plt.figure(figsize=(8, 8))  # Adjust size for better presentation

    # points = np.array([claimed_curve(point) for point in fine_points])
    points = claimed_curve(fine_points)

    plt.plot(fine_points, points, color='blue', label=r'Claimed Curve $T^{(0)}$', linewidth=2)

    alpha = np.concatenate(([0], alpha, [1]))
    beta  = np.concatenate(([1],  beta, [0]))

    # Plot the curve
    plt.plot(alpha, beta, color="darkorange", label="KDE $\hat{T}_h$", linewidth=2)
    
    # Add a shaded rectangle around the critical region
    rectangle = patches.Rectangle(
        (alpha_estimate - omega, beta_estimate - omega),  # Bottom-left corner
        2 * omega,                     # Width
        2 * omega,                     # Height
        linewidth=3,                          # Line width
        edgecolor='purple',                   # Edge color
        facecolor='lavender',                 # Fill color
        alpha=0.3                             # Transparency of the fill
    )
    plt.gca().add_patch(rectangle)

    # Add a dummy plot for the legend
    #plt.plot([], [], color='purple', label="Confidence Region")

    # Keep aspect ratio equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Mark the critical points
    plt.plot(alpha_estimate, beta_estimate, 'o', markersize=8, color='purple',
             label=r'$(\tilde{\alpha}(\hat{\eta}^*), \tilde{\beta}(\hat{\eta}^*))$')  # Critical point 1
    #plt.plot(the_alpha, the_beta, 's', markersize=8, color='green',
    #         label=r'$(\hat{\alpha}(\hat{\eta}^*), \hat{\beta}(\hat{\eta}^*))$')  # Critical point 2
    plt.axvline(x=the_alpha, color='green', linestyle='--', linewidth=1.5)
    # Adjust labels, title, and legend for better readability
    plt.xlabel('Alpha', fontsize=24, labelpad=10)
    plt.ylabel('Beta', fontsize=24, labelpad=10)
    plt.legend(fontsize=24, loc='upper right', frameon=True, shadow=True)

    # Improve ticks and grid
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(alpha=0.4, linestyle='--')

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure with higher resolution
    plt.savefig(filename, dpi=300)

    # Display the plot
    plt.show()