import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

def deviation_matrix(subfolder_path):
    """
    Process a subfolder containing CSV files to compute the deviation matrix.
    
    Args:
        subfolder_path (str): Path to the subfolder containing the CSV files.
    
    Returns:
        np.ndarray: Deviation matrix with dimensions (num_files, 500) if files are processed successfully.
        None: If no valid data is processed in the subfolder.
    """
    # Expand user path
    subfolder = os.path.expanduser(subfolder_path)
    
    if not os.path.exists(subfolder):
        print(f"Subfolder '{subfolder}' not found.")
        return None
    
    print(f"Processing subfolder: {subfolder}")

    # Find all CSV files in the specific subfolder
    csv_files = glob.glob(os.path.join(subfolder, "*.csv"))

    # Placeholder for storing subfolder-specific results
    all_results = []

    # Generate 500 equidistant points between 0 and 1
    target_points = np.linspace(0, 1, 500)

    # Compute the Gaussian curve (true curve) values at target points
    def Gaussian_curve(alpha):
        mu_2 = 1  # Assuming mu_1 = 0 and mu_2 = 1 as constants
        return norm.cdf(norm.ppf(1 - alpha) - mu_2)

    gausspoints = Gaussian_curve(target_points)

    for file in csv_files:
        # Read the CSV file
        data = pd.read_csv(file)

        # Ensure the CSV contains 'alpha' and 'beta' columns
        if 'alpha' not in data.columns or 'beta' not in data.columns:
            print(f"Skipping {file}: Missing required columns 'alpha' and 'beta'.")
            continue

        # Filter the data to only include rows where 0 <= alpha <= 1
        filtered_data = data[(data['alpha'] >= 0) & (data['alpha'] <= 1)]
        filtered_data = filtered_data.drop_duplicates(subset=['alpha'], keep='first')
        
        # Skip if no valid rows remain
        if filtered_data.empty:
            print(f"Skipping {file}: No valid alpha values between 0 and 1.")
            continue

        # Extract alpha and beta values
        alpha_values = filtered_data['alpha'].values
        beta_values = filtered_data['beta'].values

        # Perform interpolation
        try:
            interpolation_function = interp1d(alpha_values, beta_values, kind='linear', fill_value="extrapolate")
        except ValueError as e:
            print(f"Skipping {file}: Interpolation error - {e}")
            continue

        # Evaluate interpolated values on the 500 equidistant points
        interpolated_beta_values = interpolation_function(target_points)

        # Store results for this file
        all_results.append(interpolated_beta_values)

    if all_results:
        # Convert list of interpolated results into a NumPy array for easy computation
        all_results_array = np.array(all_results)  # Shape: (num_files, 500)

        # Compute the deviation matrix: (num_files x 500 points)
        deviation_matrix = all_results_array - gausspoints

        # Output the shape of the deviation matrix
        print(f"Deviation matrix shape: {deviation_matrix.shape}")

        return deviation_matrix
    else:
        print("No valid data processed in the subfolder.")
        return None
def compute_eta_values(eta_max, results_path):
    """
    Compute eta_values based on the deviation matrix.

    Args:
        eta_max (float): Maximum value of eta.
        results_path (str): Path to the folder containing result CSV files.

    Returns:
        np.ndarray: Array of eta_values corresponding to the minimum deviations.
    """
    # Step 1: Create eta_vector
    eta_vector = np.linspace(0, eta_max, 1000)

    # Step 2: Compute the deviation matrix using the provided results path
    deviation_mat = deviation_matrix(results_path)  # Assumes `deviation_matrix` is already defined
    
    if deviation_mat is None:
        print("Error: Deviation matrix could not be computed.")
        return None

    # Step 3: Trim the deviation matrix to exclude the first and last columns
    trimmed_matrix = deviation_mat[:, 1:-1]  # Slicing to exclude first and last columns

    # Step 4: Find the minimum deviation indices
    min_locations = np.argmin(trimmed_matrix, axis=1)

    # Step 5: Compute eta_values based on the indices
    eta_values = eta_vector[(min_locations + 1) * 2]  # Adjust indices as needed

    return eta_values