import pickle
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os

ENV_ID = 'BreakoutNoFrameskip-v4'
# Load the state-action counts
with open('atari/pretrained_breakout_discretized_sa_counts.pkl', 'rb') as f:
    data = pickle.load(f)

# Step 1: Aggregate total counts for each state
state_counts = defaultdict(int)
state_action_counter = defaultdict(lambda: Counter())

for (state, action), count in data.items():
    state_counts[state] += count
    state_action_counter[state][action] += count

# Step 2: Find the 5 most frequently visited states
most_common_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:15]

# Step 3: For each state, plot the action counts
output_dir = f'atari/data/state_action_barplots_{ENV_ID}'
os.makedirs(output_dir, exist_ok=True)

for idx, (state, total) in enumerate(most_common_states):
    action_counts = state_action_counter[state]
    actions = list(action_counts.keys())
    counts = [action_counts[a] for a in actions]

    plt.figure()
    bars = plt.bar(actions, counts)
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title(f'State {idx+1}: Most frequent actions (Total visits: {total})')
    plt.xticks(actions)
    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                 ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'state_{idx+1}_action_counts.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Plot saved: {plot_path}')

print('Done! Plots for top 15 states saved in', output_dir)
