import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import gym
import cv2
from gym.wrappers import AtariPreprocessing, FrameStack
import logging
import argparse

# --- Logging Setup ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# --- Atari Environment Setup ---
ENV_ID = "BreakoutNoFrameskip-v4"
FRAME_STACK = 4
SEED = 42

def make_atari_env(env_id, seed=SEED):
    env = gym.make(env_id)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4, noop_max=30)
    env = FrameStack(env, num_stack=FRAME_STACK)
    env.seed(seed)
    env.action_space.seed(seed)
    return env

# --- CNN Actor & Critic for Atari ---
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

class CNNActor(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.to(self.device)
    def forward(self, obs):
        x = obs / 255.0  # normalize
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
    def select_action(self, obs, deterministic=False):
        self.eval()
        obs_arr = obs_to_np(obs)
        obs = torch.tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()


PRETRAINED_POLICY_PATH = os.path.join('data', 'pretrained_atari_policy.pth')
if os.path.exists(PRETRAINED_POLICY_PATH):
    print(f"Loading pretrained policy weights from {PRETRAINED_POLICY_PATH}")
    policy.load_state_dict(torch.load(PRETRAINED_POLICY_PATH, map_location=device))

class CNNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.to(self.device)
    def forward(self, obs):
        x = obs / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        value = self.fc(x)
        return value.squeeze(-1)

# --- PPOAgent (copied/adapted from scripts/ppo.py) ---
class PPOAgent:
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 ppo_epochs=4, ppo_mini_batch_size=64,
                 entropy_coef=0.01, value_loss_coef=0.5, device='cpu'):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.device = device
    def _compute_gae_and_returns(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns
    def update(self, transitions):
        if not transitions:
            return 0.0, 0.0, 0.0
        states = torch.stack([t['state'] for t in transitions]).to(self.device)
        actions = torch.tensor([t['action'] for t in transitions], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t['reward'] for t in transitions], dtype=torch.float32, device=self.device)
        next_states = torch.stack([t['next_state'] for t in transitions]).to(self.device)
        dones = torch.tensor([t['done'] for t in transitions], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in transitions], dtype=torch.float32, device=self.device)
        values = self.critic(states).detach()
        next_values = self.critic(next_states).detach()
        advantages, returns = self._compute_gae_and_returns(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_actor_loss, total_critic_loss, total_entropy = 0, 0, 0
        num_samples = len(transitions)
        for _ in range(self.ppo_epochs):
            idxs = np.arange(num_samples)
            np.random.shuffle(idxs)
            for start in range(0, num_samples, self.ppo_mini_batch_size):
                end = start + self.ppo_mini_batch_size
                mb_idx = idxs[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                logits = self.actor(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                values_pred = self.critic(mb_states)
                critic_loss = self.value_loss_coef * (mb_returns - values_pred).pow(2).mean()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        num_updates = self.ppo_epochs * (num_samples / self.ppo_mini_batch_size)
        return total_actor_loss / num_updates, total_critic_loss / num_updates, total_entropy / num_updates

# --- Trajectory Collection for Atari ---
def preprocess_obs(obs):
    # obs is LazyFrames (uint8), usually shape [84,84,4] (H,W,C) for FrameStack
    obs = np.array(obs)
    # If obs shape is (84, 84, 4), transpose to (4, 84, 84)
    if obs.ndim == 3 and obs.shape[-1] == 4:
        obs = np.transpose(obs, (2, 0, 1))
    # If already (4, 84, 84), do nothing
    return torch.tensor(obs, dtype=torch.float32)

def collect_trajectories(env, actor, num_steps, device):
    transitions = []
    obs = env.reset()
    # Handle tuple return from env.reset() (Gym >=0.26)
    if isinstance(obs, tuple):
        obs = obs[0]
    obs_tensor = preprocess_obs(obs)
    done = False
    episode_rewards = []
    ep_reward = 0
    for _ in range(num_steps):
        action, log_prob = actor.select_action(obs, deterministic=False)
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        next_obs_tensor = preprocess_obs(next_obs)
        transitions.append({
            'state': obs_tensor,
            'action': action,
            'reward': reward,
            'next_state': next_obs_tensor,
            'done': float(done),
            'log_prob': log_prob
        })
        ep_reward += reward
        obs = next_obs
        obs_tensor = next_obs_tensor
        if done:
            episode_rewards.append(ep_reward)
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs_tensor = preprocess_obs(obs)
            ep_reward = 0
            done = False
    if ep_reward > 0:
        episode_rewards.append(ep_reward)
    return transitions, episode_rewards

# --- Main Training Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient for PPO')
    args = parser.parse_args()
    entropy_coef = args.entropy_coef

    # Set up experiment-specific paths
    exp_name = f'entropy_{entropy_coef}'
    log_file = os.path.join(DATA_DIR, f'ppo_atari_{exp_name}.log')
    state_dict_save_path = os.path.join(DATA_DIR, f'ppo_atari_final_{exp_name}.pth')
    plot_save_path = os.path.join(DATA_DIR, f'ppo_atari_learning_curve_{exp_name}.png')

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_atari_env(ENV_ID, seed=SEED)
    num_actions = env.action_space.n
    actor = CNNActor(num_actions).to(device)
    critic = CNNCritic().to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=2.5e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=2.5e-4)
    agent = PPOAgent(actor, critic, actor_optimizer, critic_optimizer,
                     gamma=0.99, gae_lambda=0.95, clip_epsilon=0.1,
                     ppo_epochs=4, ppo_mini_batch_size=64,
                     entropy_coef=entropy_coef, value_loss_coef=0.5, device=device)
    total_steps = 1_000_000
    steps_per_update = 2048
    episode_rewards_deque = deque(maxlen=100)
    all_avg_rewards = []
    obs_shape = env.observation_space.shape
    logging.info(f"Observation shape: {obs_shape}, Num actions: {num_actions}")
    for step in range(0, total_steps, steps_per_update):
        transitions, episode_rewards = collect_trajectories(env, actor, steps_per_update, device)
        actor_loss, critic_loss, entropy = agent.update(transitions)
        for r in episode_rewards:
            episode_rewards_deque.append(r)
        avg_reward = np.mean(episode_rewards_deque) if episode_rewards_deque else 0.0
        all_avg_rewards.append(avg_reward)
        logging.info(f"Step {step+steps_per_update} | Avg Reward (100): {avg_reward:.2f} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | Entropy: {entropy:.4f}")
        if (step // steps_per_update) % 50 == 0:
            torch.save(actor.state_dict(), state_dict_save_path)
            logging.info(f"Saved model to {state_dict_save_path}")
    torch.save(actor.state_dict(), state_dict_save_path)
    logging.info(f"Final model saved to {state_dict_save_path}")
    # Plot learning curve
    try:
        import matplotlib.pyplot as plt
        plt.plot(all_avg_rewards)
        plt.xlabel('Update')
        plt.ylabel('Avg Reward (100)')
        plt.title(f'PPO Atari Breakout (entropy_coef={entropy_coef})')
        plt.savefig(plot_save_path)
        plt.show()
    except ImportError:
        logging.warning('matplotlib not installed, skipping plot.')

if __name__ == "__main__":
    main()
