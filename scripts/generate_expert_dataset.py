import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.grid_environment import *
from core.suboptimal_expert import suboptimal_expert_policy
# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Configure logging to file and console
log_file = os.path.join(DATA_DIR, 'generate_expert_dataset.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)


def generate_imitation_learning_dataset(policy, num_trajectories, initial_state=INITIAL_STATE, goal_state=GOAL_STATE, max_steps_per_trajectory=200):
    """
    Generates a dataset of (state, action) pairs by running the given policy.

    Args:
        policy (callable): The policy function mapping state -> action.
        num_trajectories (int): The number of trajectories to simulate.
        initial_state (tuple): The starting state for each trajectory.
        goal_state (tuple): The goal state.
        max_steps_per_trajectory (int): Maximum steps before terminating a trajectory.

    Returns:
        list: A list of (state, action) tuples.
    """
    dataset = [] # List to store (state, action) pairs
    print(f"Generating dataset using {num_trajectories} trajectories...")

    for i in range(num_trajectories):
        state = initial_state
        steps = 0
        while not is_terminal(state, goal_state):
            if steps >= max_steps_per_trajectory:
                break
            action = policy(state)
            is_valid_action = (0 <= action < NUM_ACTIONS)
            if not is_valid_action:
                print(f"LOG: Policy returned action {action} which is outside the valid range [0, {NUM_ACTIONS-1}) for state {state}.")
                action = 1 # Correcting to action 1 (LEFT)
                print(f"LOG: Corrected action to {action} ({ACTION_NAMES[action]}).")
            dataset.append((state, action))
            try:
                next_state = get_next_state(state, action)
                state = next_state
                steps += 1
            except ValueError as e:
                print(f"ERROR: get_next_state failed for state={state}, action={action}. Error: {e}. Terminating trajectory {i+1}.")
                break
            if state == goal_state:
                break
    print(f"\nDataset generation complete. Total state-action pairs collected: {len(dataset)}")
    return dataset

if __name__ == "__main__":
    num_expert_trajectories = 1000
    expert_dataset = generate_imitation_learning_dataset(
        suboptimal_expert_policy,
        num_trajectories=num_expert_trajectories
    )

    print(f"\nDataset contains {len(expert_dataset)} state-action pairs.")
    if expert_dataset:
        print("First 5 entries:")
        for i in range(min(5, len(expert_dataset))):
            state, action_idx = expert_dataset[i]
            action_name = ACTION_NAMES[action_idx] if 0 <= action_idx < NUM_ACTIONS else "INVALID_ACTION_INDEX"
            print(f"  State: {state}, Action: {action_idx} ({action_name})")
        action_counts = {i: 0 for i in range(NUM_ACTIONS)}
        invalid_action_count = 0
        for _, action in expert_dataset:
            if 0 <= action < NUM_ACTIONS:
                action_counts[action] += 1
            else:
                invalid_action_count += 1
        print("\nAction distribution in the dataset:")
        for action_idx, count in action_counts.items():
            print(f"  Action {action_idx} ({ACTION_NAMES[action_idx]}): {count} times")
        if invalid_action_count > 0:
            print(f"  Invalid actions recorded in dataset (should be 0): {invalid_action_count}")
    # Now expert_dataset can be used for imitation learning.
