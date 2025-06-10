import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from scipy.interpolate import make_interp_spline

def extract_expert_actions(pkl_path):
    """
    Extracts unique actions from the expert data file.
    Expert data structure is a list of trajectories, where each trajectory
    is a list of transition dictionaries.
    """
    actions = set()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    for trajectory in data:
        for transition in trajectory:
            if 'action' in transition:
                raw_action = transition['action']
                if isinstance(raw_action, np.ndarray):
                    action = raw_action.item() if raw_action.size == 1 else tuple(raw_action.flatten())
                else:
                    action = raw_action
                actions.add(action)
    return actions

def extract_expert_action_counts(pkl_path):
    """Counts actions from the expert data file."""
    action_counts = {}
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    for trajectory in data:
        for transition in trajectory:
            if 'action' in transition:
                raw_action = transition['action']
                if isinstance(raw_action, np.ndarray):
                    action = raw_action.item() if raw_action.size == 1 else tuple(raw_action.flatten())
                else:
                    action = raw_action
                action_counts[action] = action_counts.get(action, 0) + 1
    return action_counts

def extract_finetuned_actions(pkl_path):
    """
    Extracts unique actions from a fine-tuned counts file.
    Fine-tuned data is a dict of {(state, action): count}.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    actions = set(action for (_, action) in data.keys())
    return actions

def extract_finetuned_action_counts(pkl_path):
    """Counts actions from a fine-tuned counts file."""
    action_counts = {}
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    for (_, action), count in data.items():
        action_counts[action] = action_counts.get(action, 0) + count
    return action_counts

def main():
    
    # ENV_ID = 'GravitarNoFrameskip-v4'

    ENV_ID = 'BerzerkNoFrameskip-v4'
    # ENV_ID = 'PrivateEyeNoFrameskip-v4'
    ENV = 'Berzerk'
    entropy_value = 0.0
    expert_path = f'/iliad/u/jubayer/Exploration-Policy/atari/atari/data/{ENV_ID}/expert_data_{ENV_ID}.pkl'
    finetuned_paths = [ f'/iliad/u/jubayer/Exploration-Policy/atari/atari/data/{ENV_ID}/entropy_{entropy_value}/discretized_state_action_counts_{ts}.pkl' for ts in [250000, 500000, 750000, 1000000] ]
    output_dir = f'atari/data/{ENV}-finetune/entropy_{entropy_value}'
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(expert_path):
        print(f"Error: Expert policy file not found: {expert_path}")
        return
    print(f"Processing expert policy: {expert_path}")
    expert_actions = extract_expert_actions(expert_path)
    expert_counts = extract_expert_action_counts(expert_path)
    print(f"Expert policy: {len(expert_actions)} unique actions: {sorted(list(expert_actions))}")

    for ft_path in finetuned_paths:
        if not os.path.exists(ft_path):
            print(f"\nWarning: File not found, skipping: {ft_path}")
            continue

        print("-" * 50)
        print(f"Processing finetuned policy: {os.path.basename(ft_path)}")
        ft_actions = extract_finetuned_actions(ft_path)
        ft_counts = extract_finetuned_action_counts(ft_path)
        print(f"Finetuned policy: {len(ft_actions)} unique actions: {sorted(list(ft_actions))}")
        
        # --- Data preparation for plots ---
        all_actions = sorted(list(expert_actions | ft_actions))
        expert_vals = np.array([expert_counts.get(a, 0) for a in all_actions])
        ft_vals = np.array([ft_counts.get(a, 0) for a in all_actions])
        x = np.array(range(len(all_actions)))
        total_expert = expert_vals.sum()
        total_ft = ft_vals.sum()
        expert_probs = expert_vals / total_expert if total_expert > 0 else expert_vals
        ft_probs = ft_vals / total_ft if total_ft > 0 else ft_vals
        expert_smooth, ft_smooth, x_smooth = (None, None, None)
        if len(x) > 2:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl_expert = make_interp_spline(x, expert_probs, k=3)
            expert_smooth = spl_expert(x_smooth)
            spl_ft = make_interp_spline(x, ft_probs, k=3)
            ft_smooth = spl_ft(x_smooth)

        # --- Plot 1: Scaled Distributions to Emphasize Modes ---
        fig_scaled, ax_scaled = plt.subplots(figsize=(max(10, len(all_actions) * 0.5), 5))
        
        if expert_smooth is not None and ft_smooth is not None:
            # Scale each distribution so its max value is 1.0
            # This makes the shape and modes of each distribution easy to compare relatively.
            scaled_expert = (expert_smooth - np.min(expert_smooth)) / (np.max(expert_smooth) - np.min(expert_smooth))
            scaled_ft = (ft_smooth - np.min(ft_smooth)) / (np.max(ft_smooth) - np.min(ft_smooth))

            ax_scaled.plot(x_smooth, scaled_expert, color='mediumblue', linestyle='-', lw=2, label='Expert (Scaled)')
            ax_scaled.plot(x_smooth, scaled_ft, color='saddlebrown', linestyle='-', lw=2, label='Finetuned (Scaled)')

        ax_scaled.set_xlabel('Action')
        ax_scaled.set_ylabel('Scaled Height (Normalized to Max)')
        ax_scaled.set_title(f'Scaled Action Distributions to Highlight Modes\n({os.path.basename(ft_path)})')
        ax_scaled.set_xticks(x)
        ax_scaled.set_xticklabels(all_actions)
        ax_scaled.legend()
        ax_scaled.grid(axis='y', linestyle='--', alpha=0.7)
        fig_scaled.tight_layout()
        save_path_scaled = os.path.join(output_dir, f'action_distribution_scaled_{os.path.basename(ft_path).replace(".pkl", ".png")}')
        plt.savefig(save_path_scaled)
        plt.show()
        plt.close()

if __name__ == '__main__':
    main()

