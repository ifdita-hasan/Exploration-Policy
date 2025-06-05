import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re

# --- Standard Setup (ensure this matches your environment) ---
try:
    SCRIPT_DIR = os.path.dirname(__file__)
except NameError:
    # If __file__ is not defined (e.g., in an interactive notebook),
    # set SCRIPT_DIR to your current working directory or the correct path.
    SCRIPT_DIR = '.'
    print(f"Warning: __file__ not defined. Assuming SCRIPT_DIR='{SCRIPT_DIR}'")

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
counts_path = os.path.join(DATA_DIR, 'state_action_counts.pkl')

# Basic checks for paths
if not os.path.exists(DATA_DIR):
    print(f"ERROR: DATA_DIR does not exist: {DATA_DIR}")
    print("Please ensure the 'data' directory is in the same location as the script, or adjust DATA_DIR.")
    exit()
if not os.path.exists(counts_path):
    print(f"ERROR: counts_path does not exist: {counts_path}")
    print(f"Please ensure '{os.path.basename(counts_path)}' is in the '{DATA_DIR}' directory.")
    exit()

# Load state-action counts and aggregate by state (sum over actions)
with open(counts_path, 'rb') as f:
    state_action_counts = pickle.load(f)

state_counts = {}
for (s, a), count in state_action_counts.items():
    state_counts[s] = state_counts.get(s, 0) + count

# --- RLFT state visitation counts will now be loaded per-entropy below ---

# Find all critic loss snapshot files
snapshot_files = [f for f in os.listdir(DATA_DIR)
                  if f.startswith('critic_loss_at_iter_') and f.endswith('_entropy_0.9.pkl')]

def extract_iter_num(filename):
    match = re.search(r'critic_loss_at_iter_(\d+)_entropy_', filename)
    return int(match.group(1)) if match else -1

# Sort files by iteration number
snapshot_iters = [extract_iter_num(f) for f in snapshot_files]
sorted_files = sorted(zip(snapshot_iters, snapshot_files))
filtered = [(i, f) for i, f in sorted_files if i != -1]
if filtered:
    snapshot_iters, snapshot_files = zip(*filtered)
else:
    snapshot_iters, snapshot_files = [], []
# --- End of Standard Setup ---

for snapshot_file, iter_num in zip(snapshot_files, snapshot_iters):
    entropy_match = re.search(r'_entropy_([\d\.eE+-]+)\.pkl$', snapshot_file)
    entropy_str = entropy_match.group(1) if entropy_match else 'unknown'
    entropy_dir = os.path.join(DATA_DIR, f'entropy_{entropy_str}')
    os.makedirs(entropy_dir, exist_ok=True)

    with open(os.path.join(DATA_DIR, snapshot_file), 'rb') as f:
        critic_loss_dict = pickle.load(f)

    states = list(state_counts.keys())
    visit_counts = np.array([state_counts.get(s, 0) for s in states])

    # --- Load RLFT state visitation counts for this entropy ---
    rlft_state_counts = None
    # Try to match entropy string to RLFT file robustly (handle floats, sci notation, etc)
    entropy_variants = set()
    try:
        entropy_val = float(entropy_str)
        entropy_variants.add(str(entropy_val))
        entropy_variants.add(f"{entropy_val:.8f}".rstrip('0').rstrip('.'))
        entropy_variants.add(f"{entropy_val:.16f}".rstrip('0').rstrip('.'))
        entropy_variants.add(f"{entropy_val:.1e}")
        entropy_variants.add(entropy_str)
    except Exception:
        entropy_variants.add(entropy_str)
    loaded_rlft_path = None
    for ent_str in entropy_variants:
        candidate = os.path.join(DATA_DIR, f'rlft_state_counts_entropy_{ent_str}.pkl')
        if os.path.exists(candidate):
            with open(candidate, 'rb') as f:
                rlft_state_counts = pickle.load(f)
            loaded_rlft_path = candidate
            print(f"Loaded RLFT state visitation counts from {candidate}")
            break
    if rlft_state_counts is None:
        rlft_counts_path = os.path.join(DATA_DIR, 'rlft_state_counts.pkl')
        if os.path.exists(rlft_counts_path):
            with open(rlft_counts_path, 'rb') as f:
                rlft_state_counts = pickle.load(f)
            loaded_rlft_path = rlft_counts_path
            print(f"Loaded RLFT state visitation counts from {rlft_counts_path}")
    if rlft_state_counts is None:
        print(f"WARNING: RLFT state visitation file not found for entropy {entropy_str}. No RLFT filtering will be applied.")

    # --- Filter states by RLFT visitation (>0) if available ---
    print(f"Iteration {iter_num}: {len(states)} states before RLFT filtering.")
    if rlft_state_counts is not None:
        mask_rlft = np.array([rlft_state_counts.get(s, 0) > 0 for s in states])
        print(f"Iteration {iter_num}: {np.sum(mask_rlft)} states after RLFT filtering.")
        if not np.any(mask_rlft):
            print(f"Iteration {iter_num}: No states with RLFT visitation > 0. Skipping.")
            continue
        states = [s for i, s in enumerate(states) if mask_rlft[i]]
        visit_counts = visit_counts[mask_rlft]
        print(f"Iteration {iter_num}: Unique pretraining visit counts after RLFT filtering: {np.unique(visit_counts)}")

    # --- Compute product of pretraining and RLFT visit counts for each state ---
    rlft_counts_for_states = np.array([rlft_state_counts.get(s, 0) if rlft_state_counts is not None else 0 for s in states])
    pretrain_counts_for_states = np.array([state_counts.get(s, 0) for s in states])
    product_counts = pretrain_counts_for_states * rlft_counts_for_states

    # Only keep states where both counts are > 0 (product > 0)
    mask_product = product_counts > 0
    if not np.any(mask_product):
        print(f"Iteration {iter_num}: No states with product(pretrain, RLFT) > 0. Skipping.")
        continue
    states = [s for i, s in enumerate(states) if mask_product[i]]
    product_counts = product_counts[mask_product]

    individual_critic_losses = []
    for s in states:
        val = critic_loss_dict.get(s, np.nan)
        if isinstance(val, (tuple, list, np.ndarray)): val = val[0]
        individual_critic_losses.append(val)
    individual_critic_losses = np.array(individual_critic_losses)

    # 1. Normalize individual critic losses globally
    valid_individual_losses = individual_critic_losses[~np.isnan(individual_critic_losses)]
    norm_individual_critic_losses = np.full_like(individual_critic_losses, np.nan, dtype=float)

    if len(valid_individual_losses) > 0:
        global_min_indiv = np.min(valid_individual_losses)
        global_max_indiv = np.max(valid_individual_losses)
        non_nan_mask_orig = ~np.isnan(individual_critic_losses) # Mask for original non-NaNs

        if global_max_indiv > global_min_indiv:
            norm_individual_critic_losses[non_nan_mask_orig] = \
                (individual_critic_losses[non_nan_mask_orig] - global_min_indiv) / (global_max_indiv - global_min_indiv)
        elif np.any(non_nan_mask_orig): # All valid values are the same
            norm_individual_critic_losses[non_nan_mask_orig] = 0.0

    # 2. Prepare data for violin plots using buckets of RLFT visit counts
    import numpy as np
    rlft_bucket_edges = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf]
    rlft_bucket_labels = []
    for i in range(len(rlft_bucket_edges) - 1):
        left = int(rlft_bucket_edges[i])
        right = rlft_bucket_edges[i+1]
        if np.isinf(right):
            label = f"{left}-inf"
        else:
            label = f"{left}-{int(right)}"
        rlft_bucket_labels.append(label)
    rlft_bucket_indices = np.digitize(rlft_counts_for_states, rlft_bucket_edges) - 1

    data_for_violins = []
    means_for_each_bucket = []
    bucket_centers = []

    for i in range(len(rlft_bucket_edges)-1):
        mask = (rlft_bucket_indices == i)
        current_losses = norm_individual_critic_losses[mask]
        current_losses_clean = current_losses[~np.isnan(current_losses)]
        if len(current_losses_clean) > 0:
            data_for_violins.append(current_losses_clean)
            means_for_each_bucket.append(np.mean(current_losses_clean))
            # Use geometric mean of bucket edges for center (except for last bucket)
            if np.isinf(rlft_bucket_edges[i+1]):
                bucket_centers.append(rlft_bucket_edges[i]*2 if rlft_bucket_edges[i]>0 else 1)
            else:
                bucket_centers.append(np.sqrt(rlft_bucket_edges[i]*rlft_bucket_edges[i+1]) if rlft_bucket_edges[i]>0 else rlft_bucket_edges[i+1]/2)
        else:
            data_for_violins.append([])
            means_for_each_bucket.append(np.nan)
            bucket_centers.append(np.sqrt(rlft_bucket_edges[i]*rlft_bucket_edges[i+1]) if rlft_bucket_edges[i]>0 else rlft_bucket_edges[i+1]/2)

    # Remove empty buckets for plotting
    non_empty = [len(d)>0 for d in data_for_violins]
    data_for_violins = [d for d, keep in zip(data_for_violins, non_empty) if keep]
    means_for_each_bucket = [m for m, keep in zip(means_for_each_bucket, non_empty) if keep]
    rlft_bucket_labels = [l for l, keep in zip(rlft_bucket_labels, non_empty) if keep]
    bucket_centers = [c for c, keep in zip(bucket_centers, non_empty) if keep]

    # 3. Generate Violin Plot (RLFT Visit Count Buckets vs. Normalized Critic Loss)
    fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    parts = ax.violinplot(data_for_violins, positions=range(len(rlft_bucket_labels)), showmeans=True, showmedians=True)
    ax.set_xticks(range(len(rlft_bucket_labels)))
    ax.set_xticklabels(rlft_bucket_labels, rotation=30, ha='right')
    ax.set_xlabel('RLFT Visit Count (buckets)')
    ax.set_ylabel('Normalized Critic Loss')
    ax.set_title(f'Violin Plot: Critic Loss vs RLFT Visit Buckets\nIteration {iter_num}, Entropy {entropy_str}')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    # Optionally plot mean trend
    ax.plot(range(len(means_for_each_bucket)), means_for_each_bucket, marker='o', color='red', label='Mean Critic Loss')
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(entropy_dir, f'norm_critic_loss_violin_buckets_iter_{iter_num}.png')
    plt.savefig(out_path)
    print(f'Saved violin plot: {out_path}')
    plt.close()
    plt.close(fig)

    # 3. Generate Scatter Plot (RLFT Visit Count vs. Normalized Critic Loss)
    fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    scatter = ax.scatter(
        rlft_counts_for_states, norm_individual_critic_losses,
        c=norm_individual_critic_losses, cmap='viridis', alpha=0.5, edgecolor='k', s=30, zorder=3
    )

    ax.set_xscale('log')
    ax.set_xlabel('RLFT Visit Count')
    ax.set_ylabel('Normalized Critic Loss')
    ax.set_title(f'Scatter: Critic Loss vs RLFT Visit Count\nIteration {iter_num}, Entropy {entropy_str}')
    ax.grid(True, linestyle='--', alpha=0.5, axis='both')
    plt.colorbar(scatter, ax=ax, label='Normalized Critic Loss')
    plt.tight_layout()
    out_path = os.path.join(entropy_dir, f'norm_critic_loss_scatter_rlft_iter_{iter_num}.png')
    plt.savefig(out_path)
    print(f'Saved scatter plot: {out_path}')
    plt.close()
    plt.close(fig)