import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.optim as optim
import logging

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Configure logging to file and console
log_file = os.path.join(DATA_DIR, 'pretrain_policy.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from core.policy import Policy
from scripts.generate_expert_dataset import generate_imitation_learning_dataset
from core.suboptimal_expert import suboptimal_expert_policy
from core.grid_environment import ACTION_NAMES, NUM_ACTIONS, INITIAL_STATE, GOAL_STATE

# --- Constants for Training ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 600
VAL_SPLIT = 0.01
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 1. Instantiate the policy model
policy = Policy(input_size=2, hidden_size=64, output_size=NUM_ACTIONS)

# 2. Generate or load the expert dataset
try:
    expert_dataset = generate_imitation_learning_dataset(
        suboptimal_expert_policy,
        num_trajectories=1000,
        initial_state=INITIAL_STATE,
        goal_state=GOAL_STATE
    )
except Exception as e:
    print(f"Could not generate expert dataset: {e}")
    dummy_state = (0,0)
    dummy_action = 1
    expert_dataset = [(dummy_state, dummy_action)] * 500

# --- PyTorch Dataset for Imitation Learning ---
class ImitationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.states = torch.tensor([item[0] for item in data], dtype=torch.float32)
        self.actions = torch.tensor([item[1] for item in data], dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# --- Training Function ---
def train_imitation_learning(policy_model, dataset, num_epochs, batch_size, learning_rate, val_split):
    if not dataset:
        print("Error: Expert dataset is empty. Cannot train.")
        return None, {}
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    if train_size == 0 or val_size == 0:
        print(f"Warning: Dataset too small (size {dataset_size}) for val_split={val_split}. Training on full dataset without validation.")
        train_dataset = ImitationDataset(dataset)
        val_dataset = None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
    else:
        train_subset, val_subset = random_split(dataset, [train_size, val_size])
        train_dataset = ImitationDataset(train_subset)
        val_dataset = ImitationDataset(val_subset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Training data size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation data size: {len(val_dataset)}")
    model = policy_model.to(policy_model.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_state = None
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (states, expert_actions) in enumerate(train_loader):
            states = states.to(model.device)
            expert_actions = expert_actions.to(model.device)
            optimizer.zero_grad()
            action_logits = model(states)
            loss = criterion(action_logits, expert_actions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * states.size(0)
            _, predicted_actions = torch.max(action_logits.data, 1)
            total_train += expert_actions.size(0)
            correct_train += (predicted_actions == expert_actions).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for states, expert_actions in val_loader:
                    states = states.to(model.device)
                    expert_actions = expert_actions.to(model.device)
                    action_logits = model(states)
                    loss = criterion(action_logits, expert_actions)
                    val_loss += loss.item() * states.size(0)
                    _, predicted_actions = torch.max(action_logits.data, 1)
                    total_val += expert_actions.size(0)
                    correct_val += (predicted_actions == expert_actions).sum().item()
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = correct_val / total_val
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = model.state_dict()
                print(f"  -> New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            history['val_loss'].append(None)
            history['val_acc'].append(None)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
            best_model_state = model.state_dict()
    print("\nTraining Finished.")
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state based on validation loss.")
    else:
        print("No validation set used or no best model state saved. Using final model state.")
    return model, history

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    if history['val_loss'][0] is not None:
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    if history['val_acc'][0] is not None:
        plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3. Train the model
trained_policy, training_history = train_imitation_learning(
    policy_model=policy,
    dataset=expert_dataset,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    val_split=VAL_SPLIT
)

# 4. Plot the training history
if training_history:
    plot_training_history(training_history)

# 5. Save the trained model (optional)
if trained_policy:
    home_dir = os.path.expanduser("~")
    desktop_dir = os.path.join(home_dir, "Documents/Github/Exploration-Policy/data/")
    save_dir = os.path.join(desktop_dir, "saved_models_exp")
    print(f"Target save directory: {save_dir}")
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {save_dir}: {e}")
        print("Please ensure your Desktop folder exists and is writable.")
        save_dir = None
    if save_dir:
        model_save_path = os.path.join(save_dir, "policy_il_trained.pth")
        print(f"Attempting to save trained policy to: {model_save_path}")
        try:
            torch.save(trained_policy.state_dict(), model_save_path)
            print("Trained policy saved successfully to your Desktop!")
        except Exception as e:
            print(f"Error saving model to {model_save_path}: {e}")
else:
    print("\nModel training did not complete successfully. Cannot save model.")
