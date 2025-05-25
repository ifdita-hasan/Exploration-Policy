#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker, patches

# ensure repo root on PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.policy import Policy
from core.suboptimal_expert import generate_trajectory, visualize_trajectory
from core.grid_environment import (
    NUM_ACTIONS,
    GOAL_STATE,
    INITIAL_STATE,
    DANGER_ZONE_COORDS,
    X_BOUNDS,
    Y_BOUNDS,
    is_terminal,
    in_danger_zone
)

def parse_args():
    p = argparse.ArgumentParser("Visualize IL or PPO policy trajectories")
    p.add_argument(
        "--ppo-entropy", type=float,
        help="If set, load PPO model from `data/ppo-{value}/ppo_final.pth`")
    p.add_argument(
        "--il-subdir", default="saved_models_exp",
        help="IL-model folder under `data/` (default: saved_models_exp)")
    p.add_argument(
        "--il-filename", default="policy_il_trained.pth",
        help="IL-model filename (default: policy_il_trained.pth)")
    p.add_argument(
        "--num-traj", type=int, default=10,
        help="Number of trajectories to visualize when stochastic")
    p.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic action selection and generate exactly one trajectory")
    return p.parse_args()


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "visualize_policy.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )


def main():
    args = parse_args()

    # Determine model directory and filename
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    if args.ppo_entropy is not None:
        subdir = f"ppo-{args.ppo_entropy}"
        model_file = "ppo_final.pth"
    else:
        subdir = args.il_subdir
        model_file = args.il_filename

    model_dir = os.path.join(data_root, subdir)
    setup_logging(model_dir)
    logging.info(f"Loading model from {model_dir}/{model_file}")

    model_path = os.path.join(model_dir, model_file)
    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Instantiate & load policy
    policy = Policy(input_size=2, hidden_size=64, output_size=NUM_ACTIONS).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    logging.info("Model loaded successfully.")

    # Decide how many trajectories to run
    runs = 1 if args.deterministic else args.num_traj
    mode = "Deterministic" if args.deterministic else "Stochastic"

    for i in range(1, runs + 1):
        logging.info(f"Generating {mode.lower()} trajectory {i}/{runs}")
        traj = generate_trajectory(
            policy=lambda s: policy.select_action(s, deterministic=args.deterministic),
            initial_state=INITIAL_STATE,
            goal_state=GOAL_STATE
        )

        title = f"{mode} Policy Trajectory"
        if not args.deterministic:
            title += f" #{i}"
        # build filename
        if args.deterministic:
            img_name = "deterministic_traj.png"
        else:
            img_name = f"stochastic_traj_{i}.png"
        save_png = os.path.join(model_dir, img_name)

        visualize_trajectory(traj, title=title, save_path=save_png)
        logging.info(f"Saved plot to {save_png}")

    logging.info("All done.")

if __name__ == "__main__":
    main()
