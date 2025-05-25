import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_ppo_log(log_path):
    """Parse PPO log file and extract iteration and entropy values"""
    entropy_values = []
    iterations = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match lines containing entropy information
            match = re.search(r'PPO Iter: (\d+)\s+.*\s+Entropy: (\d+\.\d+)', line)
            if match:
                iteration = int(match.group(1))
                entropy = float(match.group(2))
                iterations.append(iteration)
                entropy_values.append(entropy)
    
    return iterations, entropy_values

def main():
    # Directory containing all PPO experiment results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Dictionary to store entropy data for each entropy coefficient
    entropy_data = defaultdict(list)
    
    # Process each PPO directory
    for dir_name in os.listdir(data_dir):
        if dir_name.startswith('ppo-'):
            # Extract entropy coefficient from directory name
            entropy_coeff = float(dir_name.split('-')[1])
            
            # Parse the log file
            log_path = os.path.join(data_dir, dir_name, 'ppo.log')
            if os.path.exists(log_path):
                iterations, entropy_values = parse_ppo_log(log_path)
                entropy_data[entropy_coeff] = (iterations, entropy_values)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # # Plot each entropy coefficient's data
    # for entropy_coeff, (iterations, entropy_values) in sorted(entropy_data.items()):
    #     plt.plot(iterations, entropy_values, label=f'Entropy Coeff: {entropy_coeff}')
    
    # Add rolling average for better visualization
    for entropy_coeff, (iterations, entropy_values) in sorted(entropy_data.items()):
        if len(entropy_values) > 10:
            rolling_avg = np.convolve(entropy_values, np.ones(10)/10, mode='valid')
            plt.plot(iterations[5:-4], rolling_avg, alpha=0.7, label=f'Entropy Coeff: {entropy_coeff} (Rolling Avg)')
    
    plt.xlabel('PPO Iteration')
    plt.ylabel('Policy Entropy')
    plt.title('Policy Entropy Evolution with Different Entropy Coefficients')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save and show plot
    save_path = os.path.join(data_dir, 'entropy_comparison_smooth.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
