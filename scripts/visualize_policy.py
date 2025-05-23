import os
import torch
import logging

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Configure logging to file and console
log_file = os.path.join(DATA_DIR, 'visualize_policy.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

from core.policy import Policy
from core.suboptimal_expert import generate_trajectory, visualize_trajectory
from core.grid_environment import (
    NUM_ACTIONS, GOAL_STATE, INITIAL_STATE, DANGER_ZONE_COORDS, X_BOUNDS, Y_BOUNDS, is_terminal, in_danger_zone
)

# --- Configuration ---
NUM_VISUAL_TRAJECTORIES = 3 # How many trajectories to generate and plot
DETERMINISTIC_VISUALIZATION = False # Use deterministic actions for visualization?
MODEL_SUBDIR = "saved_models_exp" # The subdirectory within home used for saving
MODEL_FILENAME = "policy_il_trained.pth"

# --- Load the Trained Model ---
print("--- Loading Trained Policy ---")
trained_policy = None
model_loaded_successfully = False

try:
    # Get the user's home directory
    home_dir = os.path.expanduser("~")
    # Point at the Desktop
    desktop_dir = os.path.join(home_dir, "Documents/Github/Exploration-Policy/data/")
    # Use your MODEL_SUBDIR on the Desktop
    save_dir = os.path.join(desktop_dir, MODEL_SUBDIR)
    # Build the full path to the .pth file
    model_load_path = os.path.join(save_dir, MODEL_FILENAME)

    print(f"Attempting to load model from: {model_load_path}")

    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"Model file not found at: {model_load_path}")

    # Pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate your policy (must match the saved architecture)
    trained_policy = Policy(
        input_size=2,
        hidden_size=64,
        output_size=NUM_ACTIONS
    )

    # Load weights & set to eval
    trained_policy.load_state_dict(
        torch.load(model_load_path, map_location=device)
    )
    trained_policy.eval()

    print("Model loaded successfully.")
    model_loaded_successfully = True

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure:")
    print(f"  • You ran training and saved to '~/Desktop/{MODEL_SUBDIR}/{MODEL_FILENAME}'")
    print(f"  • That folder and filename are correct.")
except NameError as e:
    print(f"Error loading model: A required name is not defined ({e}).")
    print("Did you forget to define `Policy` or `NUM_ACTIONS`?")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")

# --- Generate and Visualize Trajectories ---
if model_loaded_successfully:
    print("\n--- Visualizing Trajectories ---")

    for i in range(NUM_VISUAL_TRAJECTORIES):
        print(f"\nGenerating trajectory {i+1}...")

        # Define a policy function that uses the trained model
        # The generate_trajectory function expects a callable: policy(state) -> action
        il_policy_func = lambda state: trained_policy.select_action(
            state, deterministic=DETERMINISTIC_VISUALIZATION
        )

        # Generate a trajectory using the TRAINED policy function
        try:
            # Make sure generate_trajectory exists and handles termination correctly.
            # Add max_steps for safety if your function supports it.
            il_trajectory = generate_trajectory(
                policy=il_policy_func,
                initial_state=INITIAL_STATE,
                goal_state=GOAL_STATE
                # Add max_steps if available in generate_trajectory, e.g.:
                # max_steps=200
            )

            # Visualize the trajectory using the visualize_trajectory function
            if il_trajectory: # Check if trajectory generation was successful
                title = f"Trained Policy (IL) Trajectory {i+1}"
                if DETERMINISTIC_VISUALIZATION:
                    title += " (Deterministic)"
                else:
                    title += " (Stochastic)"

                visualize_trajectory(
                    il_trajectory,
                    title=title
                )

                # Print trajectory summary
                print(f"  Trajectory length: {len(il_trajectory)} steps.")
                final_state = il_trajectory[-1]
                print(f"  Final state: {final_state}")

                # Check termination reason (assuming is_terminal covers all conditions)
                if is_terminal(final_state, GOAL_STATE):
                    print("  Termination Reason:")
                    if final_state == GOAL_STATE:
                        print("    -> Reached GOAL.")
                    elif in_danger_zone(final_state, DANGER_ZONE_COORDS):
                        print("    -> Entered DANGER ZONE.")
                    else:
                        # Check bounds if is_terminal includes them explicitly
                        x, y = final_state
                        x_min, x_max = X_BOUNDS
                        y_min, y_max = Y_BOUNDS
                        if not (x_min <= x <= x_max) or not (y_min <= y <= y_max):
                            print("    -> Went OUT OF BOUNDS.")
                        else:
                            print("    -> Unknown terminal state condition met.") # Should not happen if checks are exhaustive
                else:
                    # If not terminal, likely hit max steps if max_steps was used
                    print("  -> Did not reach a terminal state (possibly hit max steps).")
            else:
                print("  Trajectory generation failed or returned empty.")

        except NameError as e:
            print(f"\nError during trajectory generation/visualization: Required function/variable not defined ({e}).")
            print("Please ensure 'generate_trajectory', 'visualize_trajectory', 'is_terminal', 'in_danger_zone',")
            print("and all environment/policy constants/classes are defined and accessible.")
            break # Stop trying to visualize if prerequisites are missing
        except Exception as e:
            print(f"\nAn unexpected error occurred during trajectory generation/visualization: {e}")
            break # Stop further attempts
else:
    print("\nCannot visualize trajectories because the trained policy could not be loaded.")

print("\n--- Visualization Complete ---")
