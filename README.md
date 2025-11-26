# apple_deer_AROB

Multi-agent reinforcement learning experimental environment using the Apple Deer environment.

## Requirements

- Python 3.7 or higher (Python 3.8+ recommended)
- See `requirements.txt` for package dependencies

## Reproduction Steps

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/tzkwkblab/AROB2025_Inoue.git
cd AROB2025_Inoue
```

### 2. Install Dependencies

Install the required packages using the following command:

```bash
cd AROB2025_Inoue
pip install -r requirements.txt
```

#### Troubleshooting: Git LFS Error

If you encounter an error related to Git LFS (Large File Storage) when installing PantheonRL, such as:

```
Error downloading object: ... This repository exceeded its LFS budget.
```

This is a known issue with the PantheonRL repository's Git LFS storage quota. To work around this, install dependencies with the `GIT_LFS_SKIP_SMUDGE` environment variable set:

```bash
cd AROB2025_Inoue
GIT_LFS_SKIP_SMUDGE=1 pip install -r requirements.txt
```

This will skip downloading Git LFS files during installation, which is sufficient for most use cases. The missing LFS files are typically only needed for specific datasets that are not required for the Apple Deer environment.

### 3. Run Training

**Important:** Make sure you are in the `apple_deer_AROB` directory when running the training scripts.

#### Environment with Deer

Execute `train_AROB.py` from the `apple_deer_AROB` directory to start training in an environment that includes deer:

```bash
cd AROB2025_Inoue
python train_AROB.py
```

#### Environment without Deer

For training in an environment without deer, use `train_AROB_nodeer.py`:

```bash
cd AROB2025_Inoue
python train_AROB_nodeer.py
```

**Note:** If you get a `ModuleNotFoundError: No module named 'apple_deer_AROB'` error, make sure you are running the script from inside the `apple_deer_AROB` directory, not from the parent `AROB2025_Inoue` directory.

**Note:** The main difference is that `train_AROB_nodeer.py` uses the `apple_deer_nodeer` environment which does not include deer.

Training runs for 30,000,000 steps (default setting). During training, results are saved to the following directories (relative to the parent directory of `apple_deer_AROB`):

- Policy files: `policy/`
- Tensorboard logs: `tensorboard_log/`

### 4. Check Results

To view the training progress, start Tensorboard from the parent directory of `apple_deer_AROB`:

```bash
tensorboard --logdir=tensorboard_log/ --port=<PORT_NUMBER>
```

Replace `<PORT_NUMBER>` with an available port number (e.g., 6006). Access `http://localhost:<PORT_NUMBER>` in your browser to view the learning curves.

After training completes, policy files are saved in subdirectories under `policy/` (relative to the parent directory of `apple_deer_AROB`). The exact path depends on the configuration in `train_AROB.py`:

- `policy_ego`: Ego agent's policy
- `policy_partner0`: Partner agent's policy

## Detailed Configuration

### Customizing Training Scripts

You can modify `train_AROB.py` to change experimental settings.

#### Training Script Parameters

The following parameters can be configured in `train_AROB.py`:

**Environment Settings:**
- `x_size`, `y_size`: Environment size (e.g., 15x15)
- `max_cycles`: Maximum steps per episode (e.g., 300)
- `obs_range`: Observation range (e.g., 5)
- `tree_range`: Tree range (e.g., 1)

**Agent Settings:**
- `agents_dict`: Agent configuration (color, number, attack range, health, etc.)

**Environment Objects:**
- `deer_num`: Number of deer (e.g., 1)
- `nuts_num`: Number of nuts (e.g., 25)
- `apple_tree_num`: Number of apple trees (e.g., 1)
- `deer_health`: Deer health (e.g., 30)
- `apple_tree_health`: Apple tree health (e.g., 10)

**Reward Settings:**
- `apple_reward`: Reward for collecting apples (e.g., 100)
- `nut_reward`: Reward for collecting nuts (e.g., 1)
- `single_attack_reward`: Reward for single attack (e.g., 5)
- `double_attack_reward`: Reward for coordinated attack (e.g., 20)

**Other Settings:**
- `agent_respawn`: Enable agent respawn (e.g., True)
- `sequential_respawn`: Sequential respawn (e.g., True)
- `signal_visualization`: Signal visualization (e.g., False)

#### Training Results Storage

Training results are saved to the following directories (relative to the parent directory of `apple_deer_AROB`). The exact subdirectory paths depend on the configuration in `train_AROB.py`:

- Policies: `policy/<experiment_name>/`
  - `policy_ego`: Ego agent's policy
  - `policy_partner0`: Partner agent's policy
- Tensorboard logs: `tensorboard_log/<experiment_name>/`

#### Loading Saved Policies

After training completes, you can load and use policies as follows:

```python
from stable_baselines3 import PPO

# Load ego agent's policy (path is relative to parent directory of apple_deer_AROB)
# Replace <experiment_name> with the actual experiment directory name
ego = PPO.load("policy/<experiment_name>/policy_ego")

# Load partner agent's policy
partner0 = PPO.load("policy/<experiment_name>/policy_partner0")
```

### 5. Test Trained Policies

After training, you can test the trained policies and generate GIF animations using `test.py`:

```bash
cd apple_deer_AROB
python test.py
```

This script will:
- Load trained policies from `policy/` directory (relative to the parent directory of `apple_deer_AROB`)
- Run test episodes with the trained agents
- Generate GIF animations showing agent behavior
- Save GIFs to `GIF/` directory (relative to the parent directory of `apple_deer_AROB`)

#### Test Script Configuration

You can modify `test.py` to customize the test settings:

- `gif_num`: Number of GIFs to generate (default: 6)
- `num_steps`: Maximum number of steps per episode (default: 300)
- `policy_dir`: Directory containing saved policies (default: `policy/`)
- `gif_dir`: Directory to save GIF files (default: `GIF/`)
- Environment parameters: Same as training script (e.g., `obs_range`, `x_size`, `y_size`, etc.)

**Note:** Make sure the policy files (`policy_ego` and `policy_partner0`, `policy_partner1`, etc.) exist in the `policy/` directory before running the test script.

## Directory Structure

```
apple_deer_AROB/
├── apple_deer/          # Apple Deer environment implementation
│   ├── apple_deer.py    # Main environment class
│   ├── apple_deer_base.py
│   └── utils/           # Utility functions
├── apple_deer_v0.py     # Environment entry point
├── tensorboard_callback.py  # Tensorboard callback
├── train_AROB.py        # Training script (with deer)
├── train_AROB_nodeer.py # Training script (without deer)
├── test.py              # Test script for generating GIFs
├── requirements.txt     # Dependency list
└── README.md           # This file
```
