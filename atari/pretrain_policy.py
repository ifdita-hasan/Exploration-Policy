import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ppo_atari import CNNActor, ENV_ID, FRAME_STACK, obs_to_np

# --- Config ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataGravitarNoFrameskip-v4/')
EXPERT_DATA_PATH = os.path.join(DATA_DIR, 'expert_data_GravitarNoFrameskip-v4.pkl')
PRETRAINED_POLICY_PATH = os.path.join(DATA_DIR, f'pretrained_{ENV_ID}_policy.pth')
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # For Iddah's path
# # Comment out if you are not Iddah
# EXPERT_DATA_PATH = 'data/'
# EXPERT_DATA_PATH = os.path.join(
#     EXPERT_DATA_PATH,
#     ENV_ID,
#     f"expert_data_{ENV_ID}.pkl"
# )

# --- Dataset ---
class AtariImitationDataset(Dataset):
    def __init__(self, expert_episodes):
        self.data = [ (step['obs'], step['action'])
                      for episode in expert_episodes
                      for step in episode ]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        obs, action = self.data[idx]
        obs = obs_to_np(obs)  # ensure correct format
        obs = torch.tensor(obs, dtype=torch.float32)  # (C, H, W)
        action = torch.tensor(action, dtype=torch.long)
        return obs, action

# --- Main Pretraining Routine ---
def pretrain_policy():
    # Load expert data
    with open(EXPERT_DATA_PATH, 'rb') as f:
        expert_episodes = pickle.load(f)
    dataset = AtariImitationDataset(expert_episodes)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Build model
    num_actions = 4  # Breakout has 4 actions; adjust if using a different game
    model = CNNActor(num_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss_history = []
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for obs, action in dataloader:
            obs = obs.to(DEVICE)
            action = action.to(DEVICE)
            logits = model(obs)
            loss = criterion(logits, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * obs.size(0)
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    # Plot and save loss curve
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    plot_path = os.path.join(data_dir, f'pretraining_loss_curve-{ENV_ID}.png')
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS+1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pretraining Loss Curve')
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Training history plot saved to: {plot_path}")
    plt.show()

    # Save pretrained policy
    torch.save(model.state_dict(), PRETRAINED_POLICY_PATH)
    print(f"Pretrained policy saved to {PRETRAINED_POLICY_PATH}")

if __name__ == "__main__":
    pretrain_policy()
