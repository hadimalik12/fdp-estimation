import argparse
import os
import sys

# Navigate to the parent directory of the project structure
project_dir = os.path.abspath(os.getcwd())
src_dir = os.path.join(project_dir, 'src')
fig_dir = os.path.join(project_dir, 'fig')
data_dir = os.path.join(project_dir, 'data', 'dpsgd_algs')
log_dir = os.path.join(project_dir, 'log')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Add the src directory to sys.path
sys.path.append(src_dir)

import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mech.dpsgd_algs.train_neural_network import compute_accuracy_privacy_point
from mech.model_architecture import MODEL_MAPPING

def parse_args():
    parser = argparse.ArgumentParser(description='Run parallel experiments with configurable parameters')
    
    parser.add_argument('--num_workers', type=int, default=20,
                      help='Number of parallel workers to use')
    parser.add_argument('--epochs_list', type=str, default='1,5,9,13,17',
                      help='Comma-separated list of epochs to train for')
    parser.add_argument('--database_size', type=int, default=1000,
                      help='Size of the database to use for training')
    parser.add_argument('--database_name', type=str, default='black_cifar10',
                      choices=['black_cifar10', 'white_cifar10'],
                      help='Name of the database to use')
    parser.add_argument('--model_name', type=str, default='convnet_balanced',
                      choices=list(MODEL_MAPPING.keys()),
                      help='Name of the model architecture to use')
    
    args = parser.parse_args()
    
    # Convert epochs_list from string to list of integers
    args.epochs_list = [int(x) for x in args.epochs_list.split(',')]
    
    return args

def run_single_experiment(args):
    """
    Run a single experiment with specified number of epochs.
    
    Args:
        args: Tuple containing (epochs_list, database_size)
        
    Returns:
        Dictionary containing results for each metric
    """
    epochs_list, database_size, database_name, model_name = args
    model_class = MODEL_MAPPING[model_name]

    final_losses, white_image_losses, black_image_losses, epsilons, deltas = compute_accuracy_privacy_point(epoch_list=epochs_list, database_size=database_size, database_name=database_name, model_class=model_class)

    return {
        'epochs': epochs_list,
        'final_losses': np.array(final_losses),
        'white_image_losses': np.array(white_image_losses),
        'black_image_losses': np.array(black_image_losses),
        'epsilons': np.array(epsilons),
        'deltas': np.array(deltas)
    }

def run_parallel_experiments(epochs_list = [1, 2, 3, 4], num_workers = 4, database_size = None, database_name = "white_cifar10", model_name = "convnet"):
    """
    Run experiments in parallel where each worker processes the entire epochs_list.
    
    Args:
        epochs_list: List of epoch values to train for
        num_workers: Number of parallel workers to use
        database_size: Size of the database to use for training
        
    Returns:
        Dictionary containing results for each metric
    """

    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:        
        # Run experiments in parallel with progress bar
        # Each worker will process the entire epochs_list
        results = list(tqdm(
            pool.imap(run_single_experiment, [(epochs_list, database_size, database_name, model_name)] * num_workers),
            total=num_workers,
            desc="Running experiments"
        ))

    # Organize results
    final_losses = np.zeros(len(epochs_list))
    white_image_losses = np.zeros(len(epochs_list))
    black_image_losses = np.zeros(len(epochs_list))
    epsilons = np.zeros(len(epochs_list))
    deltas = np.zeros(len(epochs_list))

    for result in results:
        final_losses = result['final_losses'] + final_losses
        white_image_losses = result['white_image_losses'] + white_image_losses
        black_image_losses = result['black_image_losses'] + black_image_losses
        epsilons = result['epsilons'] + epsilons
        deltas = result['deltas'] + deltas

    final_losses = final_losses / num_workers
    white_image_losses = white_image_losses / num_workers
    black_image_losses = black_image_losses / num_workers
    epsilons = epsilons / num_workers
    deltas = deltas / num_workers

    return {
        'database_size': database_size,
        'database_name': database_name,
        'model_name': model_name,
        'epochs': epochs_list,
        'final_losses': final_losses,
        'white_image_losses': white_image_losses,
        'black_image_losses': black_image_losses,
        'epsilons': epsilons,
        'deltas': deltas
    }

def save_results(results, data_dir = data_dir):
    """Save experiment results to a numpy file."""
    if results["database_size"] is None:
        results["database_size"] = "full"
    filename = os.path.join(data_dir, f"accuracy_privacy_curve_{results['model_name']}_{results['database_name']}_{results['database_size']}.npz")
    np.savez(
        filename,
        epochs=np.array(results['epochs']),
        final_losses=np.array(results['final_losses']),
        white_image_losses=np.array(results['white_image_losses']),
        black_image_losses=np.array(results['black_image_losses']),
        epsilons=np.array(results['epsilons']),
        deltas=np.array(results['deltas'])
    )

    return filename

# Save results
def load_results(data_dir = data_dir, database_name = "white_cifar10", database_size = None, model_name = "convnet"):
    """
    Load experiment results from a .npz file.
    
    Args:
        file_path: Path to the .npz file containing experiment results
        
    Returns:
        dict: Dictionary containing the loaded experiment results
    """
    if database_size is None:
        database_size = "full"
    try:
        filename = os.path.join(data_dir, f"accuracy_privacy_curve_{model_name}_{database_name}_{database_size}.npz")
        data = np.load(filename)
        results = {
            'database_size': database_size,
            'database_name': database_name,
            'model_name': model_name,
            'epochs': data['epochs'],
            'final_losses': data['final_losses'],
            'white_image_losses': data['white_image_losses'],
            'black_image_losses': data['black_image_losses'],
            'epsilons': data['epsilons']
        }
        return results
    except FileNotFoundError:
        print(f"Error: Results file {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None

    

def print_results(results: dict):
    """Print experiment results in a formatted way."""
    print(f"\nExperiment Results for {results['model_name']} on {results['database_name']} with database size {results['database_size']}:")
    print("-" * 80)
    print(f"{'Epochs':>8} | {'Final Loss':>12} | {'White Loss':>12} | {'Black Loss':>12} | {'Epsilon':>10}")
    print("-" * 80)
    
    for i in range(len(results['epochs'])):
        print(f"{results['epochs'][i]:>8} | "
              f"{results['final_losses'][i]:>12.4f} | "
              f"{results['white_image_losses'][i]:>12.4f} | "
              f"{results['black_image_losses'][i]:>12.4f} | "
              f"{results['epsilons'][i]:>10.4f}")

def plot_epsilon_loss_curve(results, fig_dir = fig_dir, model_name = "convnet", database_name = "white_cifar10", database_size = None):
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
    plt.title(f'Privacy-Loss (Total) Trade-off Curve for {model_name} on {database_name} with database size {database_size}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(fig_dir, f"epsilon_loss_curve_{model_name}_{database_name}_{database_size}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Privacy-Loss (Total) Trade-off Curve for {model_name} on {database_name} with database size {database_size} plot saved to {filename}")

def plot_epsilon_loss_curve_images(results, fig_dir = fig_dir, model_name = "convnet", database_name = "white_cifar10", database_size = None):
    """
    Plot epsilon vs final loss curve and save the figure.
    
    Args:
        results: Dictionary containing experiment results
        save_path: Path to save the figure. If None, will save to fig_dir/epsilon_loss_curve.png
    """
    plt.figure(figsize=(10, 6))
    
    # Plot epsilon vs final loss
    plt.plot(results['epsilons'], results['white_image_losses'], 'bo-', label='White Image Loss')
    plt.plot(results['epsilons'], results['black_image_losses'], 'ro-', label='Black Image Loss')
    
    # Add labels and title
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Privacy-Loss (Images) Trade-off Curve for {model_name} on {database_name} with database size {database_size}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(fig_dir, f"epsilon_loss_curve_images_{model_name}_{database_name}_{database_size}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Privacy-Loss (Images) Trade-off Curve for {model_name} on {database_name} with database size {database_size} plot saved to {filename}")

    return filename


def main():
    # Parse command line arguments
    args = parse_args()

    num_workers = args.num_workers
    epochs_list = args.epochs_list
    database_size = args.database_size
    database_name = args.database_name
    model_name = args.model_name
    
    # Run experiments with different epoch values
    results = run_parallel_experiments(epochs_list=epochs_list, num_workers=num_workers, database_size=database_size, database_name = database_name, model_name = model_name)
    
    # Print results
    print_results(results)

    # Save current results
    results_path = save_results(results, data_dir = data_dir)

    # Plot and save the epsilon-loss curve
    plot_epsilon_loss_curve(results, fig_dir = fig_dir, model_name = model_name, database_name = database_name, database_size = database_size) 
    plot_epsilon_loss_curve_images(results, fig_dir = fig_dir, model_name = model_name, database_name = database_name, database_size = database_size)

    # Load results
    results = load_results(data_dir = data_dir, database_name = database_name, database_size = database_size, model_name = model_name)
    print(f"Results loaded from {results_path}")
    print_results(results)


if __name__ == "__main__":
    main()