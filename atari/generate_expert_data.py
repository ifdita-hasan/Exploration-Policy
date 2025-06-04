import os
import torch
import numpy as np
from collections import defaultdict
import argparse
import pickle
import matplotlib.pyplot as plt
import sys

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atari.ppo_atari import CNNActor, make_atari_env, ENV_ID, FRAME_STACK

# --- Config ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# We will save expert data under a large‐quota directory:
EXPERT_DIR = '/lfs/skampere1/0/iddah/explore_data/' + ENV_ID
os.makedirs(EXPERT_DIR, exist_ok=True)

# Paths for saving expert episodes and bucket histogram
EXPERT_DATA_PATH = os.path.join(EXPERT_DIR, f"expert_data_{ENV_ID}.pkl")
HISTOGRAM_PATH  = os.path.join(EXPERT_DIR, f"bucket_histogram_{ENV_ID}.png")


def obs_to_np(obs):
    # Unpack if obs is a tuple and the first element is an array (Gym/Gymnasium (obs, info) style)
    if isinstance(obs, tuple) and len(obs) == 2 and hasattr(obs[0], 'shape'):
        obs = obs[0]
    # Handle LazyFrames
    if 'LazyFrames' in str(type(obs)):
        return np.array(obs)
    if hasattr(obs, 'shape') and isinstance(obs, np.ndarray):
        return obs
    if hasattr(obs, 'array'):
        return np.array(obs, copy=False)
    if isinstance(obs, (tuple, list)):
        arrs = [np.array(f) for f in obs if np.array(f).size > 0]
        if not arrs:
            raise ValueError("Empty observation sequence in obs_to_np.")
        shapes = [a.shape for a in arrs]
        if all(s == shapes[0] for s in shapes):
            return np.stack(arrs, axis=0)
        else:
            raise ValueError(f"Inconsistent frame shapes in obs: {shapes}")
    raise TypeError(f"Unrecognized observation type: {type(obs)}")



def discretize_frame(obs_np):
    """
    Convert a (4, 84, 84) uint8 frame-stack into a 7x7 grid of 11-level buckets.
    Steps:
      1. Average over the 4 stacked frames -> (84, 84) float array.
      2. Downsample by averaging each non-overlapping 12x12 block -> (7, 7).
      3. Quantize each block’s averaged value into one of 11 bins (0..10).
      4. Flatten to a tuple of length 7x7 to serve as a dictionary key.
    """
    # 1) Average over the 4 stacked frames → shape (84,84)
    gray = obs_np.astype(np.float32).mean(axis=0)  # → (84,84)

    # 2) Downsample 84×84 → 7×7 by averaging each non‐overlapping 12×12 block
    h, w = gray.shape
    assert h % 12 == 0 and w % 12 == 0, "Frame dimensions not divisible by 12 for 7x7 grid"
    downsampled = gray.reshape(7, 12, 7, 12).mean(axis=(1, 3))  # → shape (7,7)

    # 3) Quantize into 11 bins. Each downsampled pixel ∈ [0..255], so floor_divide by 24 → values 0..10
    # (Previous: floor_divide by 32 for 8 bins; floor_divide by 16 for 16 bins)
    quantized = np.floor_divide(downsampled, 24).astype(np.uint8) # Now 11 bins

    # 4) Flatten into a tuple of length 7×7 for use as a dictionary key
    return tuple(quantized.flatten().tolist())



def generate_expert_data(
    checkpoint_path,
    num_episodes=50,
    output_path=None,
    env_id=ENV_ID,
    frame_stack=FRAME_STACK,
    deterministic=True
):
    # Set up environment
    env = make_atari_env(env_id)
    num_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained policy
    actor = CNNActor(num_actions).to(device)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    actor.eval()

    expert_data   = []
    bucket_counts = defaultdict(int)

    for ep in range(num_episodes):
        obs   = env.reset()
        done  = False
        episode = []

        while not done:
            obs_np = obs_to_np(obs)  # shape (4, 84, 84)

            # Discretize and count this observation
            bucket_key = discretize_frame(obs_np)
            bucket_counts[bucket_key] += 1

            # Select action from policy
            action_out = actor.select_action(obs, deterministic=deterministic)
            if isinstance(action_out, tuple):
                action = action_out[0]
            else:
                action = action_out
            action = action.item() if hasattr(action, 'item') else int(action)

            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            episode.append({
                'obs':      obs_np,
                'action':   action,
                'reward':   reward,
                'done':     done,
                'next_obs': obs_to_np(next_obs)
            })
            obs = next_obs

        expert_data.append(episode)
        print(f"Episode {ep+1}/{num_episodes} collected, length: {len(episode)}")

    # Save expert data (to large‐quota directory)
    if output_path is None:
        output_path = EXPERT_DATA_PATH
    with open(output_path, 'wb') as f:
        pickle.dump(expert_data, f)
    print(f"Expert data saved to {output_path}")

    # Build a sorted array of visit counts (descending)
    counts = np.array(sorted(bucket_counts.values(), reverse=True))
    print(f"Total unique buckets (discretized states): {len(counts)}")

    # Only plot the first 100 buckets to avoid a long tail
    N = min(100, len(counts))
    top_counts = counts[:N]

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(N), top_counts, color='skyblue', edgecolor='k')
    plt.xlabel("Bucket rank (most‐visited → least‐visited)")
    plt.ylabel("Number of visits")
    plt.title(f"Top {N} discretized‐state visits ({env_id})")
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PATH)
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained policy checkpoint (.pth)'
    )
    parser.add_argument(
        '--num_episodes', type=int, default=500,
        help='Number of episodes to collect'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path for expert data (.pkl)'
    )
    parser.add_argument(
        '--env_id', type=str, default=ENV_ID,
        help='Gym environment ID'
    )
    parser.add_argument(
        '--deterministic', action='store_true',
        help='Use deterministic policy'
    )
    args = parser.parse_args()

    generate_expert_data(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        output_path=args.output,
        env_id=args.env_id,
        deterministic=args.deterministic
    )
