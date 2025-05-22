import math

ACTION_NAMES = ["UP", "LEFT", "DOWN", "RIGHT"]
_ACTION_EFFECTS = [(0, 1), (-1, 0), (0, -1), (1, 0)]
NUM_ACTIONS = len(_ACTION_EFFECTS)
X_BOUNDS = (-20, 20)
Y_BOUNDS = (0, 20)
GOAL_STATE = (-18, 10)
INITIAL_STATE = (18, 10)

# Define the desired width and height for all danger zones
OBSTACLE_WIDTH = 4
OBSTACLE_HEIGHT = 3

# Define the bottom-left corner (x_min, y_min) for each obstacle,
# ensuring they don't overlap the goal and are more towards the bottom.
OBSTACLE_POSITIONS = [
    (0, 15),     # Obstacle 1 (Top: y_min = 15 > 10)
    (-3, 9),    # Obstacle 2 (Middle: y is around 10)
    (-10, 6),    # Obstacle 3 (Bottom: y_max = 2 + 3 = 5 < 10)
    (-1, 4),      # Obstacle 4 (Bottom: y_max = 1 + 3 = 4 < 10)
    (7, 5),     # Obstacle 5 (Bottom: y_max = 5 + 3 = 8 < 10)
]

DANGER_ZONE_COORDS = []
for x_min, y_min in OBSTACLE_POSITIONS:
    x_max = x_min + OBSTACLE_WIDTH
    y_max = y_min + OBSTACLE_HEIGHT
    # Ensure no overlap with the goal state
    if not (GOAL_STATE[0] >= x_min and GOAL_STATE[0] <= x_max and GOAL_STATE[1] >= y_min and GOAL_STATE[1] <= y_max):
        DANGER_ZONE_COORDS.append((x_min, x_max, y_min, y_max))
    else:
        print(f"Warning: Obstacle at ({x_min}, {y_min}) overlaps the goal. Skipping.")
        
upper_danger_coords = []
for x_min, y_min in OBSTACLE_POSITIONS:
    upper_boundary_coords = []
    for x in range(x_min, x_min + OBSTACLE_WIDTH + 1):
        upper_boundary_coords.append((x, y_min + OBSTACLE_HEIGHT + 1))
    upper_danger_coords.append(tuple(upper_boundary_coords))
    
print(f"Upper-of-danger coordinates: {upper_danger_coords}")
    
right_danger_coords = []
for x_min, x_max, y_min, y_max in DANGER_ZONE_COORDS:
    right_boundary_coords = []
    for y in range(y_min, y_max + 1):
        right_boundary_coords.append((x_max + 1, y))
    right_danger_coords.append(tuple(right_boundary_coords))

print(f"Right-of-danger coordinates: {right_danger_coords}")

bottom_danger_coords = []
for x_min, y_min in OBSTACLE_POSITIONS:
    bottom_boundary_coords = []
    for x in range(x_min, x_min + OBSTACLE_WIDTH + 1):
        bottom_boundary_coords.append((x, y_min - 1))
    bottom_danger_coords.append(tuple(bottom_boundary_coords))

print(f"Bottom-of-danger coordinates: {bottom_danger_coords}")

def in_danger_zone(state, dz_coords=DANGER_ZONE_COORDS):
    """Check if a state is in any of the danger zones."""
    x, y = state
    for dz_x_min, dz_x_max, dz_y_min, dz_y_max in dz_coords:
        if (dz_x_min <= x <= dz_x_max) and (dz_y_min <= y <= dz_y_max):
            return True
    return False

def get_next_state(current_state, action_index, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS):
    """
    Assuming the environment is deterministic.
    Calculates the next state based on the current state and action.
    """
    if not (0 <= action_index < NUM_ACTIONS):
        raise ValueError(f"Invalid action_index: {action_index}")

    x, y = current_state
    dx, dy = _ACTION_EFFECTS[action_index]
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    next_x = x + dx
    next_y = y + dy

    return (next_x, next_y)

def get_reward(current_state, next_state, goal_state=GOAL_STATE, dz_coords=DANGER_ZONE_COORDS, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS):
    """Reward function for the new environment."""
    if next_state == goal_state:
        return 1000.0  # Positive reward for reaching the goal
    elif in_danger_zone(next_state, dz_coords):
        return -50.0  # Negative reward (penalty) for entering a danger zone

    # check if out of bounds
    if next_state[0] < x_bounds[0] or next_state[0] > x_bounds[1]:
        return -50.0
    if next_state[1] < y_bounds[0] or next_state[1] > y_bounds[1]:
        return -50.0

    default_reward = 0

    # Add reward for moving towards the goal
    goal_x, goal_y = goal_state
    next_x, next_y = next_state

    # Calculate distance to goal
    distance_to_goal = math.sqrt((goal_x - next_x) ** 2 + (goal_y - next_y) ** 2)
    # Calculate distance to current state
    current_x, current_y = current_state
    distance_from_current = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)
    # this might lead to greedy trajectories
    distance_reward = 0 # You can re-enable this if you want to reward moving closer: distance_from_current - distance_to_goal

    # Add time penalty
    time_penalty = -0.01

    total_reward = default_reward + distance_reward + time_penalty

    return total_reward

def is_terminal(state, goal_state=GOAL_STATE, dz_coords=DANGER_ZONE_COORDS, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS):
    """Checks if the current state is a terminal state (goal reached or in danger)."""
    if state == goal_state:
        return True
    if in_danger_zone(state, dz_coords):
        return True
    if state[0] < x_bounds[0] or state[0] > x_bounds[1]:
        return True
    if state[1] < y_bounds[0] or state[1] > y_bounds[1]:
        return True
    return False

if __name__ == "__main__":
    print(f"Danger Zone Coordinates (Integer Grid): {DANGER_ZONE_COORDS}")

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.ticker as ticker

    def visualize_environment(ax, x_bounds, y_bounds, goal_state, initial_state, danger_zones):
        """Visualizes the environment."""
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        # Set plot limits with a small buffer
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_aspect('equal', adjustable='box')  # Ensure square grid
        ax.set_title("20x20 Environment (Integer Grid)")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        # Set up integer ticks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(True, linestyle='--', alpha=0.6)  # Add grid lines

        # Draw boundaries
        rect = patches.Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min), linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        # Mark the goal state
        goal_x, goal_y = goal_state
        ax.plot(goal_x, goal_y, 'g*', markersize=15, label='Goal')

        # Mark the initial state
        initial_x, initial_y = initial_state
        ax.plot(initial_x, initial_y, 'bo', markersize=10, label='Initial State')

        # Draw danger zones
        for dz_x_min, dz_x_max, dz_y_min, dz_y_max in danger_zones:
            width = dz_x_max - dz_x_min
            height = dz_y_max - dz_y_min
            danger_rect = patches.Rectangle((dz_x_min, dz_y_min), width, height, facecolor='red', alpha=0.5, label='Danger Zone' if dz_x_min == danger_zones[0][0] else "")
            ax.add_patch(danger_rect)

        ax.legend()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Keep figure size square
    visualize_environment(ax, X_BOUNDS, Y_BOUNDS, GOAL_STATE, INITIAL_STATE, DANGER_ZONE_COORDS)
    plt.show()
