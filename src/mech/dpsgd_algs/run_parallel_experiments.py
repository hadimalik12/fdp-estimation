import os
import sys

# Navigate to the parent directory of the project structure
project_dir = os.path.abspath(os.getcwd())
src_dir = os.path.join(project_dir, 'src')
fig_dir = os.path.join(project_dir, 'fig')
data_dir = os.path.join(project_dir, 'data')
log_dir = os.path.join(project_dir, 'log')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Add the src directory to sys.path
sys.path.append(src_dir)

import multiprocessing as mp
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mech.dpsgd_algs.cnn1 import compute_accuracy_privacy_point as compute_accuracy_privacy_point_cnn1



def run_single_experiment(epochs: int) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """
    Run a single experiment with specified number of epochs.
    
    Args:
        epochs: Number of epochs to train for
        
    Returns:
        Tuple containing (final_loss, (white_image_loss, black_image_loss), (epsilon, delta))
    """
    return compute_accuracy_privacy_point_cnn1(epochs=epochs)

def run_parallel_experiments(epoch_values: List[int] = [1, 2, 3, 4], num_workers: int = 4) -> dict:
    """
    Run experiments in parallel for different epoch values.
    
    Args:
        epoch_values: List of epoch values to experiment with
        num_workers: Number of parallel workers to use
        
    Returns:
        Dictionary containing results for each metric
    """
    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:        
        # Run experiments in parallel with progress bar
        results = list(tqdm(
            pool.imap(run_single_experiment, epoch_values),
            total=len(epoch_values),
            desc="Running experiments"
        ))
    
    # Organize results
    final_losses = []
    white_image_losses = []
    black_image_losses = []
    epsilons = []
    deltas = []
    
    for result in results:
        final_loss, (white_loss, black_loss), (epsilon, delta) = result
        final_losses.append(final_loss)
        white_image_losses.append(white_loss)
        black_image_losses.append(black_loss)
        epsilons.append(epsilon)
        deltas.append(delta)
    
    return {
        'epochs': epoch_values,
        'final_losses': final_losses,
        'white_image_losses': white_image_losses,
        'black_image_losses': black_image_losses,
        'epsilons': epsilons,
        'deltas': deltas
    }

def save_results(results: dict, filename: str = os.path.join(data_dir, 'accuracy_privacy_curve_cnn1.npz')):
    """Save experiment results to a numpy file."""
    np.savez(
        filename,
        epochs=np.array(results['epochs']),
        final_losses=np.array(results['final_losses']),
        white_image_losses=np.array(results['white_image_losses']),
        black_image_losses=np.array(results['black_image_losses']),
        epsilons=np.array(results['epsilons']),
        deltas=np.array(results['deltas'])
    )

    

def print_results(results: dict):
    """Print experiment results in a formatted way."""
    print("\nExperiment Results:")
    print("-" * 80)
    print(f"{'Epochs':>8} | {'Final Loss':>12} | {'White Loss':>12} | {'Black Loss':>12} | {'Epsilon':>10}")
    print("-" * 80)
    
    for i in range(len(results['epochs'])):
        print(f"{results['epochs'][i]:>8} | "
              f"{results['final_losses'][i]:>12.4f} | "
              f"{results['white_image_losses'][i]:>12.4f} | "
              f"{results['black_image_losses'][i]:>12.4f} | "
              f"{results['epsilons'][i]:>10.4f}")

def plot_epsilon_loss_curve(results: dict, save_path: str = os.path.join(fig_dir, 'epsilon_loss_curve_cnn1.png')):
    """
    Plot epsilon vs final loss curve and save the figure.
    
    Args:
        results: Dictionary containing experiment results
        save_path: Path to save the figure. If None, will save to fig_dir/epsilon_loss_curve.png
    """
    plt.figure(figsize=(10, 6))
    
    # Plot epsilon vs final loss
    plt.plot(results['epsilons'], results['final_losses'], 'bo-', label='Final Loss')
    
    # Add labels and title
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Privacy-Loss Trade-off Curve for CNN1', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Privacy-Loss Trade-off Curve for CNN1 plot saved to {save_path}")

if __name__ == "__main__":
    # Run experiments with different epoch values
    results = run_parallel_experiments()
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results)
    print(f"\nResults saved to experiment_results.npz")
    
    # Plot and save the epsilon-loss curve
    plot_epsilon_loss_curve(results) 