#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_ppo_log(log_path):
    """Parse PPO log file and extract iteration and entropy values"""
    iterations = []
    entropies = []
    pattern = re.compile(r'PPO Iter:\s*(\d+)\s+.*Entropy:\s*([0-9]*\.?[0-9]+)')
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            iterations.append(int(m.group(1)))
            entropies.append(float(m.group(2)))
    return np.array(iterations), np.array(entropies)


def main():
    # locate data directory (the 'data' subdirectory within the current script's directory)
    script_parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_parent_dir, 'data')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Collect raw entropy data per entropy coefficient
    entropy_data = {}
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
        iters, ents = parse_ppo_log(log_path)
        if iters.size > 0:
            entropy_data[coeff] = (iters, ents)

    if not entropy_data:
        print("No PPO experiment logs found under data/ppo-*")
        return

    # Sort coefficients and assign colors
    coeffs = sorted(entropy_data.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(coeffs)))

    # Smoothing window (centered rolling average)
    window = 10
    half = window // 2

    plt.figure(figsize=(12, 8))
    for idx, coeff in enumerate(coeffs):
        iters, ents = entropy_data[coeff]
        # compute centered rolling average without edge artifacts
        n = ents.size
        smoothed = np.array([
            ents[max(0, i-half):min(n, i+half+1)].mean()
            for i in range(n)
        ])
        plt.plot(
            iters,
            smoothed,
            linestyle='-',  # solid lines
            color=colors[idx],
            linewidth=1.8,
            label=f'$c_2={coeff}$'
        )

    plt.xlabel('PPO Iteration')
    plt.ylabel('Smoothed Policy Entropy')
    plt.title('Smoothed Policy Entropy vs. Iteration')
    plt.legend(title='$c_2$')
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    out_file = os.path.join(data_dir, 'entropy_comparison_smoothed.png')
    plt.savefig(out_file)
    plt.show()
    print(f"Saved smoothed plot to: {out_file}")


if __name__ == '__main__':
    main()