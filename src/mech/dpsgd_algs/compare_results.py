import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Navigate to the parent directory of the project structure
project_dir = os.path.abspath(os.getcwd())
src_dir = os.path.join(project_dir, 'src')
fig_dir = os.path.join(project_dir, 'fig')
data_dir = os.path.join(project_dir, 'data', 'dpsgd_algs')

# Add the src directory to sys.path
sys.path.append(src_dir)

from mech.dpsgd_algs.run_parallel_experiments import load_results

def parse_args():
    parser = argparse.ArgumentParser(description='Compare two experiment results')
    
    # First result parameters
    parser.add_argument('--model1', type=str, required=True,
                      choices=['convnet', 'convnet_balanced'],
                      help='Name of the first model architecture')
    parser.add_argument('--db1', type=str, required=True,
                      choices=['black_cifar10', 'white_cifar10'],
                      help='Name of the first database')
    parser.add_argument('--size1', type=int, required=True,
                      help='Size of the first database')
    
    # Second result parameters
    parser.add_argument('--model2', type=str, required=True,
                      choices=['convnet', 'convnet_balanced'],
                      help='Name of the second model architecture')
    parser.add_argument('--db2', type=str, required=True,
                      choices=['black_cifar10', 'white_cifar10'],
                      help='Name of the second database')
    parser.add_argument('--size2', type=int, required=True,
                      help='Size of the second database')
    
    return parser.parse_args()

def plot_comparison(results1, results2, args):
    """
    Plot comparison of two results.
    
    Args:
        results1: First experiment results
        results2: Second experiment results
        args: Command line arguments
    """
    plt.figure(figsize=(12, 8))
    
    # Plot white image losses
    plt.plot(results1['epsilons'], results1['white_image_losses'], 'bo-', 
             label=f'{args.model1} ({args.db1}) White Image')
    plt.plot(results2['epsilons'], results2['white_image_losses'], 'ro-', 
             label=f'{args.model2} ({args.db2}) White Image')
    
    # Plot black image losses
    plt.plot(results1['epsilons'], results1['black_image_losses'], 'b^--', 
             label=f'{args.model1} ({args.db1}) Black Image')
    plt.plot(results2['epsilons'], results2['black_image_losses'], 'r^--', 
             label=f'{args.model2} ({args.db2}) Black Image')
    
    # Add labels and title
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Comparison of Privacy-Loss Trade-off Curves', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(fig_dir, 
                          f"comparison_{args.model1}_{args.db1}_{args.size1}_vs_{args.model2}_{args.db2}_{args.size2}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {filename}")

def main():
    args = parse_args()
    
    # Load first results
    results1 = load_results(
        data_dir=data_dir,
        database_name=args.db1,
        database_size=args.size1,
        model_name=args.model1
    )
    
    if results1 is None:
        print(f"Error: Could not load results for {args.model1} on {args.db1} with size {args.size1}")
        return
    
    # Load second results
    results2 = load_results(
        data_dir=data_dir,
        database_name=args.db2,
        database_size=args.size2,
        model_name=args.model2
    )
    
    if results2 is None:
        print(f"Error: Could not load results for {args.model2} on {args.db2} with size {args.size2}")
        return
    
    # Plot comparison
    plot_comparison(results1, results2, args)

if __name__ == "__main__":
    main() 


# python compare_results.py --model1 convnet_balanced --db1 black_cifar10 --size1 1000 --model2 convnet_balanced --db2 white_cifar10 --size2 1000