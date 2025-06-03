import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt

from core.policy import Policy
from core.grid_environment import (
    NUM_ACTIONS, INITIAL_STATE, GOAL_STATE, X_BOUNDS, Y_BOUNDS, DANGER_ZONE_COORDS,
    get_next_state, is_terminal
)
from core.suboptimal_expert import visualize_trajectory

# --- Load pretraining state-action counts ---
import pickle
state_action_counts_path = os.path.join(DATA_DIR, 'state_action_counts.pkl')
if os.path.exists(state_action_counts_path):
    with open(state_action_counts_path, 'rb') as f:
        state_action_counts = pickle.load(f)
    print(f"Loaded state-action count dictionary from {state_action_counts_path}")
else:
    print(f"WARNING: state_action_counts.pkl not found at {state_action_counts_path}")
    state_action_counts = dict()

# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, input_size=2, hidden_size=128):
        super(Critic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.value_head = nn.Linear(hidden_size, 1)
        self.to(self.device)

    def forward(self, state_tensor):
        if not isinstance(state_tensor, torch.Tensor):
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        x = self.relu1(self.fc1(state_tensor))
        x = self.relu2(self.fc2(x))
        # x = self.relu3(self.fc3(x))  # Optionally add more depth
        value = self.value_head(x)
        return value

# --- PPO Data Collection ---
def get_reward_wrapper(state, next_state):
    # Use grid_environment's get_reward (signature: state, next_state, goal_state, dz_coords, x_bounds, y_bounds)
    from core.grid_environment import get_reward
    return get_reward(state, next_state, goal_state=GOAL_STATE, dz_coords=DANGER_ZONE_COORDS, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS)

def generate_ppo_trajectories(actor_policy, num_steps_to_collect, max_steps_per_episode, device):
    collected_transitions = []
    total_steps_collected_in_batch = 0
    episode_rewards_in_batch = []
    while total_steps_collected_in_batch < num_steps_to_collect:
        current_episode_transitions = []
        state = INITIAL_STATE
        current_episode_reward = 0
        for t in range(max_steps_per_episode):
            action = actor_policy.select_action(state, deterministic=False)
            log_prob_old = actor_policy.get_action_log_prob(state, action)
            next_state = get_next_state(state, action)
            reward = get_reward_wrapper(state, next_state)
            done = is_terminal(next_state)
            current_episode_transitions.append({
                'state': state, 'action': action, 'reward': reward,
                'next_state': next_state, 'done': done, 'log_prob_old': log_prob_old
            })
            current_episode_reward += reward
            total_steps_collected_in_batch += 1
            state = next_state
            if done or total_steps_collected_in_batch >= num_steps_to_collect:
                break
        collected_transitions.extend(current_episode_transitions)
        episode_rewards_in_batch.append(current_episode_reward)
        if total_steps_collected_in_batch >= num_steps_to_collect:
            break
    return collected_transitions, episode_rewards_in_batch

# --- PPO Agent Class ---
class PPOAgent:
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 ppo_epochs=10, ppo_mini_batch_size=64,
                 entropy_coef=0.2, value_loss_coef=0.5, device='cpu'):
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

    def _compute_gae_and_returns(self, rewards_tensor, values_tensor, next_values_tensor, dones_tensor):
        advantages = torch.zeros_like(rewards_tensor).to(self.device)
        gae_accumulator = 0
        num_steps = len(rewards_tensor)
        for t in reversed(range(num_steps)):
            delta = rewards_tensor[t] + self.gamma * next_values_tensor[t] * (1.0 - dones_tensor[t].float()) - values_tensor[t]
            gae_accumulator = delta + self.gamma * self.gae_lambda * (1.0 - dones_tensor[t].float()) * gae_accumulator
            advantages[t] = gae_accumulator
        returns_target = advantages + values_tensor
        return advantages, returns_target

    def update(self, transitions_batch):
        if not transitions_batch:
            return 0.0, 0.0, 0.0, None
        states = torch.tensor([t['state'] for t in transitions_batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t['action'] for t in transitions_batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t['reward'] for t in transitions_batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([t['next_state'] for t in transitions_batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t['done'] for t in transitions_batch], dtype=torch.bool, device=self.device)
        log_probs_old = torch.tensor([t['log_prob_old'] for t in transitions_batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values_current_states = self.critic(states).squeeze(-1)
            values_next_states_bootstrap = self.critic(next_states).squeeze(-1) * (1.0 - dones.float())
            advantages, returns_target = self._compute_gae_and_returns(rewards, values_current_states, values_next_states_bootstrap, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_actor_loss_epoch = 0
        total_critic_loss_epoch = 0
        total_entropy_epoch = 0
        total_kl_epoch = 0
        num_samples = len(states)
        indices = np.arange(num_samples)
        num_kl_updates = 0
        # Store old logits for all states BEFORE any updates
        with torch.no_grad():
            all_action_logits_old = self.actor.forward(states).detach()
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.ppo_mini_batch_size):
                end_idx = start_idx + self.ppo_mini_batch_size
                mini_batch_indices = indices[start_idx:end_idx]
                batch_states = states[mini_batch_indices]
                batch_actions = actions[mini_batch_indices]
                batch_log_probs_old = log_probs_old[mini_batch_indices]
                batch_advantages = advantages[mini_batch_indices]
                batch_returns_target = returns_target[mini_batch_indices]
                # Use the old logits saved before any update
                action_logits_old = all_action_logits_old[mini_batch_indices]
                dist_old = Categorical(logits=action_logits_old)
                action_logits_new = self.actor.forward(batch_states)
                dist_new = Categorical(logits=action_logits_new)
                log_probs_new = dist_new.log_prob(batch_actions)
                entropy = dist_new.entropy().mean()
                values_new_critic = self.critic(batch_states).squeeze(-1)
                ratio = torch.exp(log_probs_new - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values_new_critic, batch_returns_target)
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                # KL divergence calculation
                with torch.no_grad():
                    kl = torch.distributions.kl.kl_divergence(dist_old, dist_new).mean()
                    total_kl_epoch += kl.item()
                    num_kl_updates += 1
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                total_actor_loss_epoch += actor_loss.item()
                total_critic_loss_epoch += critic_loss.item()
                total_entropy_epoch += entropy.item()
        num_updates_in_epoch_run = self.ppo_epochs * (num_samples / self.ppo_mini_batch_size)
        avg_actor_loss = total_actor_loss_epoch / num_updates_in_epoch_run if num_updates_in_epoch_run > 0 else 0
        avg_critic_loss = total_critic_loss_epoch / num_updates_in_epoch_run if num_updates_in_epoch_run > 0 else 0
        avg_entropy = total_entropy_epoch / num_updates_in_epoch_run if num_updates_in_epoch_run > 0 else 0
        avg_kl = total_kl_epoch / num_kl_updates if num_kl_updates > 0 else None
        return avg_actor_loss, avg_critic_loss, avg_entropy, avg_kl

# --- PPO Training Configuration ---
LEARNING_RATE_ACTOR_PPO = 1e-5
LEARNING_RATE_CRITIC_PPO = 5e-5
GAMMA_PPO = 0.99
GAE_LAMBDA_PPO = 0.95
CLIP_EPSILON_PPO = 0.2
PPO_OPTIMIZATION_EPOCHS = 4
PPO_MINI_BATCH_SIZE = 64
ENTROPY_COEF_PPO = 0.9
VALUE_LOSS_COEF_PPO = 0.0
NUM_PPO_TRAIN_ITERATIONS = 1500
STEPS_PER_PPO_UPDATE = 2048
MAX_STEPS_PER_EPISODE_PPO = 400
PRINT_STATS_EVERY_N_ITERATIONS = 10
SAVE_MODEL_EVERY_N_ITERATIONS = 50
MODEL_SAVE_SUBDIR = f"ppo-{ENTROPY_COEF_PPO}"
# Match to your pretrain_policy.py and visualize_policy.py
IL_MODEL_FILENAME = "policy_il_trained_grid.pth"
IL_MODEL_PARENT_DIR = "saved_models_exp"
PPO_MODEL_FILENAME_FINAL = "ppo_final.pth"

# Configure logging to file and console
home_dir = os.path.expanduser("~")
# data_dir is now DATA_DIR from the top of the script, which is already corrected
    # ppo_model_save_dir = os.path.join(DATA_DIR, MODEL_SAVE_SUBDIR)
ppo_model_save_dir = os.path.join(DATA_DIR, MODEL_SAVE_SUBDIR)
if not os.path.exists(ppo_model_save_dir):
    os.makedirs(ppo_model_save_dir, exist_ok=True)
    print(f"Created PPO model save directory: {ppo_model_save_dir}")

log_file = os.path.join(ppo_model_save_dir, 'ppo.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
        # Match architecture and hidden_size to pretraining
    actor_ppo = Policy(input_size=2, hidden_size=64, output_size=NUM_ACTIONS).to(device)
    critic_ppo = Critic(input_size=2, hidden_size=64).to(device)
    home_dir = os.path.expanduser("~")
    # data_dir is now DATA_DIR from the top of the script, which is already corrected
    # ppo_model_save_dir = os.path.join(DATA_DIR, MODEL_SAVE_SUBDIR)
    il_model_load_path = os.path.join(os.path.dirname(__file__), IL_MODEL_PARENT_DIR, IL_MODEL_FILENAME)
    ppo_model_save_dir = os.path.join(DATA_DIR, MODEL_SAVE_SUBDIR)
    if not os.path.exists(ppo_model_save_dir):
        os.makedirs(ppo_model_save_dir, exist_ok=True)
        print(f"Created PPO model save directory: {ppo_model_save_dir}")
    try:
        logging.info(f"Attempting to load IL-trained actor from: {il_model_load_path}")
        if os.path.exists(il_model_load_path):
            actor_ppo.load_state_dict(torch.load(il_model_load_path, map_location=device))
            logging.info("Successfully loaded pre-trained IL actor.")
        else:
            logging.warning(f"IL model not found at {il_model_load_path}. Initializing PPO actor with random weights.")
    except Exception as e:
        logging.error(f"Error loading IL model: {e}. Initializing PPO actor with random weights.")
    actor_ppo.train()
    critic_ppo.train()
    actor_optimizer_ppo = optim.Adam(actor_ppo.parameters(), lr=LEARNING_RATE_ACTOR_PPO)
    critic_optimizer_ppo = optim.Adam(critic_ppo.parameters(), lr=LEARNING_RATE_CRITIC_PPO)
    ppo_agent = PPOAgent(
        actor_ppo, critic_ppo, actor_optimizer_ppo, critic_optimizer_ppo,
        gamma=GAMMA_PPO, gae_lambda=GAE_LAMBDA_PPO, clip_epsilon=CLIP_EPSILON_PPO,
        ppo_epochs=PPO_OPTIMIZATION_EPOCHS, ppo_mini_batch_size=PPO_MINI_BATCH_SIZE,
        entropy_coef=ENTROPY_COEF_PPO, value_loss_coef=VALUE_LOSS_COEF_PPO, device=device
    )
    logging.info("--- Starting PPO Fine-tuning ---")
    all_ppo_iteration_avg_rewards = []
    episode_rewards_deque = deque(maxlen=100)
    total_env_steps = 0
    # TensorBoard writer
    tensorboard_log_dir = os.path.join(DATA_DIR, "tensorboard", f"ppo_grid_{ENTROPY_COEF_PPO}")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logging.info(f"Logging experiment at: {tensorboard_log_dir}")
    for ppo_iter in range(1, NUM_PPO_TRAIN_ITERATIONS + 1):
        transitions, batch_episode_rewards = generate_ppo_trajectories(
            actor_ppo, STEPS_PER_PPO_UPDATE, MAX_STEPS_PER_EPISODE_PPO, device
        )
        if not transitions:
            logging.warning(f"PPO Iteration {ppo_iter}: No transitions collected, skipping update.")
            continue
        total_env_steps += len(transitions)
        for r in batch_episode_rewards:
            episode_rewards_deque.append(r)
        avg_actor_loss, avg_critic_loss, avg_entropy, kl_value = ppo_agent.update(transitions)
        avg_reward_this_iteration_batch = np.mean(batch_episode_rewards) if batch_episode_rewards else float('nan')
        all_ppo_iteration_avg_rewards.append(avg_reward_this_iteration_batch)
        # TensorBoard logging
        writer.add_scalar("Reward/BatchAvg", avg_reward_this_iteration_batch, global_step=ppo_iter)
        avg_rolling_score = np.mean(episode_rewards_deque) if episode_rewards_deque else float('nan')
        writer.add_scalar("Reward/Rolling100", avg_rolling_score, global_step=ppo_iter)
        writer.add_scalar("Loss/Actor", avg_actor_loss, global_step=ppo_iter)
        writer.add_scalar("Loss/Critic", avg_critic_loss, global_step=ppo_iter)
        writer.add_scalar("Entropy", avg_entropy, global_step=ppo_iter)
        if kl_value is not None:
            writer.add_scalar("KL_Divergence", kl_value, global_step=ppo_iter)


        if ppo_iter % PRINT_STATS_EVERY_N_ITERATIONS == 0:
            logging.info(
                f"PPO Iter: {ppo_iter}\tTotal Steps: {total_env_steps}\t"
                f"Avg Reward (Batch): {avg_reward_this_iteration_batch:.2f}\t"
                f"Avg Reward (Roll100): {avg_rolling_score:.2f}\t"
                f"Actor Loss: {avg_actor_loss:.4f}\tCritic Loss: {avg_critic_loss:.4f}\tEntropy: {avg_entropy:.4f}"
                + (f"\tKL Divergence: {kl_value:.6f}" if kl_value is not None else "")
            )
        if ppo_iter % SAVE_MODEL_EVERY_N_ITERATIONS == 0:
            periodic_save_path = os.path.join(ppo_model_save_dir, f"ppo_iter_{ppo_iter}.pth")
            torch.save(actor_ppo.state_dict(), periodic_save_path)
            logging.info(f"Saved PPO actor to {periodic_save_path}")
        # --- Save per-state critic loss every 250th iteration ---
        if ppo_iter % 250 == 0:
            import pickle
            # Collect all unique states in the current transitions
            states = [tuple(t['state']) for t in transitions]
            states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
            # Compute critic values
            values = critic_ppo(states_tensor).squeeze(-1).detach().cpu().numpy()
            # Compute targets
            rewards = torch.tensor([t['reward'] for t in transitions], dtype=torch.float32, device=device)
            next_states = torch.tensor([t['next_state'] for t in transitions], dtype=torch.float32, device=device)
            dones = torch.tensor([t['done'] for t in transitions], dtype=torch.bool, device=device)
            with torch.no_grad():
                values_current_states = critic_ppo(states_tensor).squeeze(-1)
                values_next_states_bootstrap = critic_ppo(next_states).squeeze(-1) * (1.0 - dones.float())
                advantages, returns_target = ppo_agent._compute_gae_and_returns(
                    rewards, values_current_states, values_next_states_bootstrap, dones
                )
                targets = returns_target.cpu().numpy()
            critic_losses = (values - targets) ** 2  # Per-state squared error
            # Build dictionary: {s: critic_loss}
            critic_loss_dict = {}
            for s, c_loss in zip(states, critic_losses):
                critic_loss_dict[s] = float(c_loss)
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(data_dir, exist_ok=True)
            snapshot_path = os.path.join(
                data_dir,
                f"critic_loss_at_iter_{ppo_iter}_entropy_{ENTROPY_COEF_PPO}.pkl"
            )
            with open(snapshot_path, 'wb') as f:
                pickle.dump(critic_loss_dict, f)
            logging.info(f"Saved critic loss snapshot to {snapshot_path}")
    logging.info("--- PPO Fine-tuning Complete ---")
    final_model_path = os.path.join(ppo_model_save_dir, PPO_MODEL_FILENAME_FINAL)
    torch.save(actor_ppo.state_dict(), final_model_path)
    logging.info(f"Final PPO-tuned actor saved to {final_model_path}")
    writer.close()

    def visualize_final_ppo_trajectory(actor, title="PPO Trained Policy"):
        logging.info(f"--- Visualizing Trajectory with: {title} ---")
        actor.eval()
        trajectory_states = []
        state = INITIAL_STATE
        trajectory_states.append(state)
        total_reward_viz = 0
        for _ in range(MAX_STEPS_PER_EPISODE_PPO + 100):
            action = actor.select_action(state, deterministic=True)
            next_state = get_next_state(state, action)
            reward = get_reward_wrapper(state, next_state)
            total_reward_viz += reward
            trajectory_states.append(next_state)
            if is_terminal(next_state):
                break
            state = next_state
        visualize_trajectory(trajectory_states, title=title)
        logging.info(f"Visualization: Trajectory length {len(trajectory_states)-1}, Final state: {trajectory_states[-1]}, Total Reward: {total_reward_viz:.2f}")
        if trajectory_states[-1] == GOAL_STATE:
            logging.info("SUCCESS: Reached Goal!")
    visualize_final_ppo_trajectory(actor_ppo, title="Final PPO-Tuned Policy (Deterministic)")
    logging.info("--- PPO Script Finished ---")
