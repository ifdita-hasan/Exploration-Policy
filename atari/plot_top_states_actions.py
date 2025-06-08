import pickle
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os
from ppo_atari import CNNActor, ENV_ID, FRAME_STACK, obs_to_np
# Define the total number of possible actions for the environment
# For GravitarNoFrameskip-v4, the action space is Discrete(18)
TOTAL_ACTIONS = 18
all_possible_actions = list(range(TOTAL_ACTIONS))
# Load the state-action counts
# Make sure this path is correct for your system
file_path = '/iliad/u/jubayer/Exploration-Policy/atari/atari/data/GravitarNoFrameskip-v4/discretized_state_action_counts_expert_data_GravitarNoFrameskip-v4.pkl'
# Check if the data file exists
if not os.path.exists(file_path):
    print(f"Error: Data file not found at {file_path}")
    exit()
with open(file_path, 'rb') as f:
    data = pickle.load(f)
# --- Step 1: Aggregate total counts for each state ---
state_counts = defaultdict(int)
state_action_counter = defaultdict(lambda: Counter())
for (state, action), count in data.items():
    state_counts[state] += count
    state_action_counter[state][action] += count
# --- Step 2: Find the 15 most frequently visited states ---
most_common_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:15]
# --- Step 3: For each state, plot the action counts with a consistent x-axis ---
output_dir = f'atari/data/state_action_barplots_{ENV_ID}'
os.makedirs(output_dir, exist_ok=True)
for idx, (state, total) in enumerate(most_common_states):
    action_counts = state_action_counter[state]
    # Get counts for all possible actions, defaulting to 0 if an action was not taken
    counts = [action_counts.get(a, 0) for a in all_possible_actions]
    plt.figure(figsize=(10, 6)) # Increased figure size for better readability
    bars = plt.bar(all_possible_actions, counts)
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title(f'State {idx+1}: Action Counts (Total Visits: {total})')
    # Set the x-axis to be the same for all plots
    plt.xticks(all_possible_actions)
    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0: # Only label bars with non-zero counts
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}',
                     ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'state_{idx+1}_action_counts.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Plot saved: {plot_path}')
print(f'\nDone! Plots for top 15 states saved in "{output_dir}"')