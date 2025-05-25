import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

from core.grid_environment import (
    NUM_ACTIONS, INITIAL_STATE, GOAL_STATE, X_BOUNDS, Y_BOUNDS, DANGER_ZONE_COORDS,
    get_next_state, is_terminal, in_danger_zone, _ACTION_EFFECTS
)

# You may need to import these from grid_environment if they are defined there
from core.grid_environment import upper_danger_coords, bottom_danger_coords, right_danger_coords

# Block definitions
TOP_RIGHT_BLOCK_X_MIN = 0
TOP_RIGHT_BLOCK_X_MAX = 18
TOP_RIGHT_BLOCK_Y_MIN = 10
TOP_RIGHT_BLOCK_Y_MAX = 20

TOP_LEFT_BLOCK_X_MIN = -17
TOP_LEFT_BLOCK_X_MAX = 0
TOP_LEFT_BLOCK_Y_MIN = 10
TOP_LEFT_BLOCK_Y_MAX = 20

BOTTOM_RIGHT_BLOCK_X_MIN = 0
BOTTOM_RIGHT_BLOCK_X_MAX = 18
BOTTOM_RIGHT_BLOCK_Y_MIN = 0
BOTTOM_RIGHT_BLOCK_Y_MAX = 10

BOTTOM_LEFT_BLOCK_X_MIN = -17
BOTTOM_LEFT_BLOCK_X_MAX = 0
BOTTOM_LEFT_BLOCK_Y_MIN = 0
BOTTOM_LEFT_BLOCK_Y_MAX = 10

def suboptimal_expert_policy(state, p=0.8):
    x, y = state
    initial_x, initial_y = INITIAL_STATE
    action = None

    if state == INITIAL_STATE:
        return random.choice([0, 1, 2]) # UP, LEFT, DOWN with equal probability
    
    # Check if the agent is at both "upper-of-danger" and "bottom-of-danger" coordinate
    is_in_upper = any(state in coords for coords in upper_danger_coords)
    is_in_bottom = any(state in coords for coords in bottom_danger_coords)
    if is_in_upper and is_in_bottom:
        action = 1  # Move LEFT
        return action

    # Check if the agent is at a "upper-of-danger" coordinate
    for boundary_coords in upper_danger_coords:
        if state in boundary_coords:
            if random.random() < 0.5:
                action = 1  # Move LEFT
            else:
                action = 0  # Move UP
            return action  # Return immediately after choosing UP or LEFT

    # Check if the agent is at a "bottom-of-danger" coordinate
    for boundary_coords in bottom_danger_coords:
        if state in boundary_coords:
            if random.random() < 0.5:
                action = 1  # Move LEFT
            else:
                action = 2  # Move DOWN
            return action

    # Check if the agent is at a "right-of-danger" coordinate
    for boundary_coords in right_danger_coords:
        if state in boundary_coords:
            if random.random() < 0.5:
                action = 0  # Move UP
            else:
                action = 2  # Move DOWN
            return action

    # If not at a "right-of-danger", "upper-of-danger", or "bottom-of-danger" coordinate or danger_zone_coordinate,
    # follow the  policy, with increased exploration
    if not in_danger_zone(state, DANGER_ZONE_COORDS):
        if (BOTTOM_LEFT_BLOCK_X_MIN <= x <= BOTTOM_LEFT_BLOCK_X_MAX and  # x_min=-17, x_max=0
              BOTTOM_LEFT_BLOCK_Y_MIN <= y <= BOTTOM_LEFT_BLOCK_Y_MAX):   # y_min=0, y_max=10
            # For BOTTOM_LEFT_BLOCK, potential actions are typically LEFT or UP.
            possible_actions = []
            # x and y are the current coordinates of the state

            # Check if moving LEFT is valid
            next_x_if_left = x + _ACTION_EFFECTS[1][0] # dx for LEFT
            can_move_left = (X_BOUNDS[0] <= next_x_if_left <= X_BOUNDS[1])

            # Check if moving UP is valid
            next_y_if_up = y + _ACTION_EFFECTS[0][1] # dy for UP
            can_move_up = (Y_BOUNDS[0] <= next_y_if_up <= Y_BOUNDS[1])

            if can_move_left:
                possible_actions.append(1)  # Add action for LEFT (index 1)
            if can_move_up:
                possible_actions.append(0)  # Add action for UP (index 0)

            if possible_actions:
                action = random.choice(possible_actions)  # Choose with equal probability
                return action

        elif (BOTTOM_RIGHT_BLOCK_X_MIN <= x <= BOTTOM_RIGHT_BLOCK_X_MAX and  # x_min=0, x_max=18
              BOTTOM_RIGHT_BLOCK_Y_MIN <= y <= BOTTOM_RIGHT_BLOCK_Y_MAX):   # y_min=0, y_max=10
            # For BOTTOM_RIGHT_BLOCK, potential actions are typically LEFT or DOWN.
            possible_actions = []
            # x and y are the current coordinates of the state

            # Check if moving LEFT is valid
            next_x_if_left = x + _ACTION_EFFECTS[1][0] # dx for LEFT
            can_move_left = (X_BOUNDS[0] <= next_x_if_left <= X_BOUNDS[1])

            # Check if moving DOWN is valid
            next_y_if_down = y + _ACTION_EFFECTS[2][1] # dy for DOWN
            can_move_down = (Y_BOUNDS[0] <= next_y_if_down <= Y_BOUNDS[1])

            if can_move_left:
                possible_actions.append(1)  # Add action for LEFT (index 1)
            if can_move_down:
                possible_actions.append(2)  # Add action for DOWN (index 2)

            if possible_actions:
                action = random.choice(possible_actions)  # Choose with equal probability
                return action
            # If no actions are possible from this block's logic, continue to next conditions or fallback
            
        elif (TOP_RIGHT_BLOCK_X_MIN <= x <= TOP_RIGHT_BLOCK_X_MAX and      # x_min=0, x_max=18
              TOP_RIGHT_BLOCK_Y_MIN <= y <= TOP_RIGHT_BLOCK_Y_MAX):       # y_min=10, y_max=20
            # For TOP_RIGHT_BLOCK, potential actions are typically LEFT or UP.
            possible_actions = []
            # x and y are the current coordinates of the state

            # Check if moving LEFT is valid
            next_x_if_left = x + _ACTION_EFFECTS[1][0] # dx for LEFT
            can_move_left = (X_BOUNDS[0] <= next_x_if_left <= X_BOUNDS[1])

            # Check if moving UP is valid
            next_y_if_up = y + _ACTION_EFFECTS[0][1] # dy for UP
            can_move_up = (Y_BOUNDS[0] <= next_y_if_up <= Y_BOUNDS[1])

            if can_move_left:
                possible_actions.append(1)  # Add action for LEFT (index 1)
            if can_move_up:
                possible_actions.append(0)  # Add action for UP (index 0)

            if possible_actions:
                action = random.choice(possible_actions)  # Choose with equal probability
                return action
                
        elif (TOP_LEFT_BLOCK_X_MIN <= x <= TOP_LEFT_BLOCK_X_MAX and        # x_min=-17, x_max=0
              TOP_LEFT_BLOCK_Y_MIN <= y <= TOP_LEFT_BLOCK_Y_MAX):         # y_min=10, y_max=20
            # For TOP_LEFT_BLOCK, potential actions are typically LEFT or DOWN.
            possible_actions = []
            # x and y are the current coordinates of the state

            # Check if moving LEFT is valid
            next_x_if_left = x + _ACTION_EFFECTS[1][0] # dx for LEFT
            can_move_left = (X_BOUNDS[0] <= next_x_if_left <= X_BOUNDS[1])

            # Check if moving DOWN is valid
            next_y_if_down = y + _ACTION_EFFECTS[2][1] # dy for DOWN
            can_move_down = (Y_BOUNDS[0] <= next_y_if_down <= Y_BOUNDS[1])

            if can_move_left:
                possible_actions.append(1)  # Add action for LEFT (index 1)
            if can_move_down:
                possible_actions.append(2)  # Add action for DOWN (index 2)

            if possible_actions:
                action = random.choice(possible_actions)  # Choose with equal probability
                return action

        elif x == -18 and y < 10:
            action = 0 # UP towards goal
            return action
        elif x == -18 and y > 10:
            action = 2 # DOWN towards goal
            return action
        elif y == 10 and x > -18:
            # equal probability to go left, down or bottom
            possible_actions = [0, 1, 2]  # UP, LEFT, DOWN
            action = random.choice(possible_actions)
            return action


def generate_trajectory(policy, initial_state=INITIAL_STATE, goal_state=GOAL_STATE, max_steps=300):
    """
    Generates a trajectory following the given policy.
    Stops when terminal or max_steps exceeded.
    """
    state = initial_state
    traj = [state]
    steps = 0

    while not is_terminal(state, goal_state) and steps < max_steps:
        a = policy(state)
        if a == None:
            print("Returns none at state:", state)
        elif 0 <= a < NUM_ACTIONS:
            state = get_next_state(state, a)
            traj.append(state)
        else:
            print(f"Invalid action {a} at state {state}; stopping.")
            break
        steps += 1

    return traj


def visualize_trajectory(trajectory, title=None, save_path=None):
    """
    Visualizes path, danger-zones, start & goal.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(X_BOUNDS[0]-1, X_BOUNDS[1]+1)
    ax.set_ylim(Y_BOUNDS[0]-1, Y_BOUNDS[1]+1)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, linestyle='--', alpha=0.6)

    # environment boundary
    env_rect = patches.Rectangle((X_BOUNDS[0], Y_BOUNDS[0]),
                                 X_BOUNDS[1]-X_BOUNDS[0],
                                 Y_BOUNDS[1]-Y_BOUNDS[0],
                                 edgecolor='black', facecolor='none')
    ax.add_patch(env_rect)

    # danger zones
    for i, (xmin, xmax, ymin, ymax) in enumerate(DANGER_ZONE_COORDS):
        rect = patches.Rectangle((xmin, ymin),
                                 xmax-xmin, ymax-ymin,
                                 facecolor='red', alpha=0.5,
                                 label='Danger Zone' if i==0 else None)
        ax.add_patch(rect)

    # start & goal
    ax.plot(INITIAL_STATE[0], INITIAL_STATE[1], 'bo', markersize=10, label='Initial State')
    ax.plot(GOAL_STATE[0], GOAL_STATE[1], 'g*', markersize=15, label='Goal')

    # trajectory line
    xs, ys = zip(*trajectory)
    ax.plot(xs, ys, 'k-', linewidth=2, label='Trajectory')

    ax.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Example usage and visualization:
if __name__ == "__main__":
    num_trajectories = 60
    for i in range(num_trajectories):
        trajectory = generate_trajectory(suboptimal_expert_policy)
        visualize_trajectory(trajectory, title=f"Suboptimal Expert Trajectory {i+1}")
