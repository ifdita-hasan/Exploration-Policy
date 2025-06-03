#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_ppo_log_batch(log_path):
    """
    Parse PPO log file and extract iteration and raw batch reward values.
    Returns:
        iterations: np.ndarray of iteration numbers
        batch_rewards: np.ndarray of Avg Reward (Batch)
    """
    iterations = []
    batch_rewards = []
    pattern = re.compile(
        r"PPO Iter:\s*(\d+).*?Avg Reward \(Batch\):\s*([0-9]*\.?[0-9]+)"
    )
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            iterations.append(int(m.group(1)))
            batch_rewards.append(float(m.group(2)))
    return np.array(iterations), np.array(batch_rewards)


def main():
    # locate data directory (the 'data' subdirectory within the current script's directory)
    script_parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_parent_dir, 'data')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # collect raw batch reward data per entropy coefficient
    reward_data = {}
    for d in os.listdir(data_dir):
        if not d.startswith('ppo-'):
            continue
        try:
            coeff = float(d.split('-', 1)[1])
        except ValueError:
            continue
        log_path = os.path.join(data_dir, d, 'ppo.log')
        if not os.path.isfile(log_path):
            continue
        iters, batch = parse_ppo_log_batch(log_path)
        if iters.size > 0:
            reward_data[coeff] = (iters, batch)

    if not reward_data:
        print("No PPO experiment logs found under data/ppo-*")
        return

    # Sort coefficients and assign distinct colors
    coeffs = sorted(reward_data.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(coeffs)))

    # smoothing window parameters (centered rolling average)
    window = 10
    half = window // 2

    # Plot smoothed batch reward vs iteration
    plt.figure(figsize=(12, 8))
    for idx, coeff in enumerate(coeffs):
        iters, batch = reward_data[coeff]
        n = batch.size
        smoothed = np.array([
            batch[max(0, i-half):min(n, i+half+1)].mean()
            for i in range(n)
        ])
        plt.plot(
            iters,
            smoothed,
            linestyle='-',  # solid line
            color=colors[idx],
            linewidth=2,
            label=f'$c_2={coeff}$'
        )

    plt.xlabel('PPO Iteration')
    plt.ylabel('Smoothed Avg Reward (Batch)')
    plt.title('Smoothed Avg Reward per Batch vs. Iteration')
    plt.legend(title='$c_2$')
    plt.grid(True)
    plt.tight_layout()

    # save and show
    out_file = os.path.join(data_dir, 'reward_comparison_smoothed.png')
    plt.savefig(out_file)
    plt.show()
    print(f"Saved smoothed reward plot to: {out_file}")


if __name__ == '__main__':
    main()