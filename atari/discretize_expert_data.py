#!/usr/bin/env python3
import os
import sys
import pickle
from collections import defaultdict

# ------------------------------------------------------------------------------
# Edit these two paths as needed (no command-line args required):
# ------------------------------------------------------------------------------
INPUT_PATH = "/lfs/skampere1/0/iddah/explore_data/BreakoutNoFrameskip-v4/expert_data_BreakoutNoFrameskip-v4.pkl"
OUTPUT_PATH = "/lfs/skampere1/0/iddah/Exploration-Policy/atari/data/BreakoutNoFrameskip-v4/discretized_state_action_counts.pkl"
# ------------------------------------------------------------------------------

# Ensure the folder for OUTPUT_PATH exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Make sure "atari/utils.py" can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import discretize_frame


def main():
    # 1) Load the saved expert data (list of episodes)
    with open(INPUT_PATH, "rb") as f:
        expert_data = pickle.load(f)

    # 2) Build a (discretized_state, action) → count dictionary
    state_action_counts = defaultdict(int)

    for episode in expert_data:
        for step in episode:
            obs_np = step["obs"]      # raw array shape (4,84,84)
            action = step["action"]   # int
            
            # Discretize the 4×84×84 frame-stack into a 7×7→11-bin tuple
            bucket_key = discretize_frame(obs_np)
            
            # Increment count for this (bucket_key, action)
            state_action_counts[(bucket_key, action)] += 1

    # 3) Save the dictionary to OUTPUT_PATH
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(dict(state_action_counts), f)

    print(f"Loaded expert data from:\n  {INPUT_PATH}")
    print(f"Saved discretized (state, action) counts to:\n  {OUTPUT_PATH}")
    print(f"Unique (state,action) pairs: {len(state_action_counts)}")


if __name__ == "__main__":
    main()
