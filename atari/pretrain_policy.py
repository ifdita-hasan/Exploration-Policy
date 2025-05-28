import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from atari.ppo_atari import CNNActor, ENV_ID, FRAME_STACK, obs_to_np

# --- Config ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
EXPERT_DATA_PATH = os.path.join(DATA_DIR, 'expert_data_BreakoutNoFrameskip-v4_50eps.pkl')
PRETRAINED_POLICY_PATH = os.path.join(DATA_DIR, 'pretrained_atari_policy.pth')
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    # Save pretrained policy
    torch.save(model.state_dict(), PRETRAINED_POLICY_PATH)
    print(f"Pretrained policy saved to {PRETRAINED_POLICY_PATH}")

if __name__ == "__main__":
    pretrain_policy()
