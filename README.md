# Exploration-Policy

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
│   ├── pretrain_policy.py
│   ├── ppo.py
│   ├── visualize_policy.py
│   └── __init__.py
│
├── data/                    # Logs and experiment outputs
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

2. **Create a conda environment named `wander` with Python 3.10**:
   ```bash
   conda create -n wander python=3.10
   conda activate wander
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install roboschool** (if not installed by pip):
   - Roboschool may require installation from source. Visit the [roboschool GitHub page](https://github.com/openai/roboschool) for detailed instructions.
   - On macOS, you may need additional system dependencies (e.g., `brew install cmake boost`).

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

3. **Install roboschool** as above.

### Notes

- If you encounter issues installing roboschool, check their [issues page](https://github.com/openai/roboschool/issues) for platform-specific solutions.
- The requirements.txt lists all necessary Python packages except for system-level dependencies.


## Running the Pipeline

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

3. **Fine-tune Policy with PPO**
   ```bash
   python scripts/ppo.py
   ```
   This loads the imitation-learned policy and fine-tunes it with PPO. Results and logs are saved in `PPO_Finetuned/` and `data/`.

4. **Visualize the Trained Policy (Optional)**
   ```bash
   python scripts/visualize_policy.py
   ```
   This visualizes trajectories of the trained policy.

**Logs** for each step are saved in the `data/` folder (e.g., `data/ppo.log`).

This is a critical study of the entropy bonus in the PPO algorithm. 