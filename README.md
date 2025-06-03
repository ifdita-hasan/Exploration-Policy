# Exploration-Policy

This is a critical study of the entropy bonus in the PPO algorithm.

## Project Directory Structure

```
Exploration-Policy/
│
├── core/
│   ├── grid_environment.py
│   ├── policy.py
│   ├── suboptimal_expert.py
│   └── __init__.py
│
├── scripts/
│   ├── generate_expert_dataset.py
│   ├── ppo.py
│   ├── ppo_atari.py
│   ├── pretrain_policy.py
│   ├── visualize_policy.py
│   └── __init__.py
│
<<<<<<< HEAD
├── data/                    # Logs and experiment outputs
=======
├── data/                    ## Logging and Monitoring
>>>>>>> origin/main
├── saved_models_exp/        # Imitation learning models
├── PPO_Finetuned/           # PPO-finetuned models and plots
├── requirements.txt
└── README.md
```

## Environment Setup

### Using Conda (Recommended)

1. **Clone the repository** and navigate to the project directory:

   ```bash
   git clone <your-repo-url>
   cd Exploration-Policy
   ```

2. **Create a conda environment named `wander`**:

   ```bash
<<<<<<< HEAD
   conda create -n wander
=======
   conda create -n wander python=3.10
>>>>>>> origin/main
   conda activate wander
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Using pip and venv (Alternative)

1. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv wander
   source wander/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

<<<<<<< HEAD
- The requirements.txt lists all necessary Python packages except for system-level dependencies.

=======
>>>>>>> origin/main
## Running PPO Atari Experiments

This project supports running PPO on Atari environments (e.g., Breakout) with entropy coefficient sweeps using `scripts/ppo_atari.py`.

### 1. Install Dependencies

Make sure you have installed all required Python packages:

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
### 2. Install Atari ROMs

You must install the Atari ROMs for Gym's Atari environments:

```bash
AutoROM --accept-license
```

If `AutoROM` is not available, install it via:
```bash
pip install autorom[accept-rom-license]
```

=======
>>>>>>> origin/main
### 3. Run PPO Atari Experiments

You can run PPO on Breakout with a specified entropy coefficient using:

```bash
python scripts/ppo_atari.py --entropy_coef 0.01
```

- Logs, models, and plots will be saved under the `data/` directory, with filenames containing the entropy coefficient for easy experiment tracking.
- Adjust `total_steps` and `steps_per_update` in `scripts/ppo_atari.py` to control experiment duration.

### 4. Visualize Results

After training, learning curves will be saved as PNG files in `data/` (e.g., `ppo_atari_learning_curve_entropy_0.01.png`).

## Running the PPO Pipeline on custom grid environment

Follow these steps to run the full workflow:

1. **Generate Expert Dataset**

   ```bash
   python scripts/generate_expert_dataset.py
   ```

   This creates the expert dataset used for imitation learning.

2. **Pretrain Policy with Imitation Learning**

   ```bash
   python scripts/pretrain_policy.py
   ```

   This trains the policy to imitate the expert and saves the model to `saved_models_exp/policy_il_trained.pth`.

<<<<<<< HEAD
=======
   After pretraining, a plot of the training loss curve will be saved as `data/pretraining_loss_curve.png`. You can view this plot to inspect the convergence of the imitation learning process.

>>>>>>> origin/main
3. **Fine-tune Policy with PPO**

   ```bash
   python scripts/ppo.py
   ```

   This loads the imitation-learned policy and fine-tunes it with PPO. Results and logs are saved in `PPO_Finetuned/` and `data/`.

4. **Visualize the Trained Policy (Optional)**
   ```bash
   python -m scripts.visualize_policy
   ```
   This visualizes trajectories of the trained policy.

**Logs** for each step are saved in the `data/` folder (e.g., `data/ppo.log`).
<<<<<<< HEAD
=======

---

### Visualizing Training with TensorBoard

All major training metrics (reward, losses, entropy, KL divergence, etc.) are logged to TensorBoard for both Atari and Grid World experiments. To view live training curves and experiment metrics, launch TensorBoard from the root of the repository with:

```bash
tensorboard --logdir data/tensorboard
```

Open the provided URL in your browser to view interactive plots and experiment dashboards.
>>>>>>> origin/main
