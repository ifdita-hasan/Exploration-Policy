import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_ppo_log(log_path):
    """Parse PPO log file and extract iteration and reward values"""
    rewards = []
    iterations = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match lines containing reward information
            match = re.search(r'PPO Iter: (\d+)\s+.*\s+Avg Reward \(Batch\): ([-+]?\d*\.\d+)', line)
            if match:
                iteration = int(match.group(1))
                reward = float(match.group(2))
                iterations.append(iteration)
                rewards.append(reward)
    
    return iterations, rewards

def main():
    # Directory containing all PPO experiment results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Dictionary to store reward data for each entropy coefficient
    reward_data = defaultdict(list)
    
    # Process each PPO directory
    for dir_name in os.listdir(data_dir):
        if dir_name.startswith('ppo-'):
            # Extract entropy coefficient from directory name
            entropy_coeff = float(dir_name.split('-')[1])
            
            # Parse the log file
            log_path = os.path.join(data_dir, dir_name, 'ppo.log')
            if os.path.exists(log_path):
                iterations, reward_values = parse_ppo_log(log_path)
                reward_data[entropy_coeff] = (iterations, reward_values)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot rolling averages for each entropy coefficient
    for entropy_coeff, (iterations, reward_values) in sorted(reward_data.items()):
        if len(reward_values) > 10:  # Only plot if we have enough data points
            # Calculate rolling average with window size 10
            rolling_avg = np.convolve(reward_values, np.ones(10)/10, mode='valid')
            # Adjust iterations to match the length of rolling average
            adjusted_iterations = iterations[5:-4]
            plt.plot(adjusted_iterations, rolling_avg, label=f'Entropy Coeff: {entropy_coeff}')
    
    plt.xlabel('PPO Iteration')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Evolution with Different Entropy Coefficients')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save and show plot
    save_path = os.path.join(data_dir, 'reward_comparison_smooth.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
