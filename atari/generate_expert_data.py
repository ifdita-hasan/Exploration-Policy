import os
import torch
import numpy as np
from collections import deque
import argparse
import sys

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atari.ppo_atari import CNNActor, make_atari_env, ENV_ID, FRAME_STACK

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


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
    # Load policy
    actor = CNNActor(num_actions)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    actor.eval()

    expert_data = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode = []
        while not done:
            obs_np = obs_to_np(obs)
            action = actor.select_action(obs, deterministic=deterministic)
            action = action.item() if hasattr(action, 'item') else int(action)
            next_obs, reward, done, info = env.step(action)
            episode.append({
                'obs': obs_np,
                'action': action,
                'reward': reward,
                'done': done,
                'next_obs': obs_to_np(next_obs)
            })
            obs = next_obs
        expert_data.append(episode)
        print(f"Episode {ep+1}/{num_episodes} collected, length: {len(episode)}")

    # Save expert data as .pkl
    import pickle
    if output_path is None:
        output_path = os.path.join(DATA_DIR, f"expert_data_{env_id}_{num_episodes}eps.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(expert_data, f)
    print(f"Expert data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained policy checkpoint (.pth)')
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of episodes to collect')
    parser.add_argument('--output', type=str, default=None, help='Output file path for expert data (.npz)')
    parser.add_argument('--env_id', type=str, default=ENV_ID, help='Gym environment ID')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    args = parser.parse_args()
    generate_expert_data(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        output_path=args.output,
        env_id=args.env_id,
        deterministic=args.deterministic
    )
