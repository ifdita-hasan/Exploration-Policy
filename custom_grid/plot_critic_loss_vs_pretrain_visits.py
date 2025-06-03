import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
counts_path = os.path.join(DATA_DIR, 'state_action_counts.pkl')

# Load state-action counts and aggregate by state (sum over actions)
with open(counts_path, 'rb') as f:
    state_action_counts = pickle.load(f)
state_counts = dict()
for (s, a), count in state_action_counts.items():
    state_counts[s] = state_counts.get(s, 0) + count

import re
# Find all critic loss snapshot files
snapshot_files = [f for f in os.listdir(DATA_DIR) if f.startswith('critic_loss_at_iter_') and f.endswith('_entropy_0.5.pkl')]
def extract_iter_num(filename):
    match = re.search(r'critic_loss_at_iter_(\d+)_entropy_', filename)
    if match:
        return int(match.group(1))
    else:
        return -1
snapshot_iters = [extract_iter_num(f) for f in snapshot_files]
# Sort by iteration
snapshot_files = [f for _, f in sorted(zip(snapshot_iters, snapshot_files))]
snapshot_iters = sorted([i for i in snapshot_iters if i != -1])

for snapshot_file, iter_num in zip(snapshot_files, snapshot_iters):
    # Extract entropy value from filename
    entropy_match = re.search(r'_entropy_([\d\.eE+-]+)\.pkl$', snapshot_file)
    entropy_str = entropy_match.group(1) if entropy_match else 'unknown'
    # Prepare output directory
    entropy_dir = os.path.join(DATA_DIR, f'entropy_{entropy_str}')
    os.makedirs(entropy_dir, exist_ok=True)
    with open(os.path.join(DATA_DIR, snapshot_file), 'rb') as f:
        critic_loss_dict = pickle.load(f)
    # Only plot states that appear in state_counts
    states = list(state_counts.keys())
    visit_counts = np.array([state_counts[s] for s in states])
    # Unpack (critic_loss, value, target) tuple or assign np.nan if missing
    critic_losses = np.array([
        critic_loss_dict[s][0] if (s in critic_loss_dict and isinstance(critic_loss_dict[s], (tuple, list))) else
        (critic_loss_dict[s] if s in critic_loss_dict else np.nan)
        for s in states
    ])
    # Sort states by visit count (descending)
    sort_idx = np.argsort(-visit_counts)
    sorted_states = [states[i] for i in sort_idx]
    sorted_visits = visit_counts[sort_idx]
    sorted_losses = critic_losses[sort_idx]
    fig, ax1 = plt.subplots(figsize=(14, 7))
    bar = ax1.bar(
        range(len(sorted_states)), sorted_visits, width=1.0, color='lightgray', alpha=0.65, label='Pretraining Visit Count'
    )
    ax1.set_ylabel('Pretraining Visit Count', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    ax1.set_xlabel('State index (ordered by pretraining visit count, descending)')
    ax2 = ax1.twinx()
    line = ax2.plot(
        range(len(sorted_states)), sorted_losses, color='tab:blue', marker='o', linestyle='-', linewidth=2, markersize=4, label='Critic Loss'
    )
    ax2.set_ylabel('Critic Loss', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)
    # Legend handling
    lines_labels = [bar, line[0]]
    labels = ['Pretraining Visit Count', f'Critic Loss at iter {iter_num}']
    ax2.legend(lines_labels, labels, loc='upper right')
    plt.title(f'Critic Loss vs Pretraining State Visit Count\nIteration {iter_num} (States ordered by visit count)')
    plt.tight_layout()
    out_path = os.path.join(entropy_dir, f'critic_loss_vs_pretrain_visits_iter_{iter_num}.png')
    plt.savefig(out_path)
    print(f'Saved plot: {out_path}')
    plt.close()
