"""
Training script for Apple Deer environment without deer (nodeer) using AROB configuration.
This script sets up a multi-agent reinforcement learning environment without deer and trains agents using PPO.
"""

import os
import sys
from pathlib import Path
import importlib
import importlib.util
import types

# Setup module path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Setup apple_deer_AROB module
module_name = 'apple_deer_AROB'
try:
    importlib.import_module(module_name)
except (ImportError, ModuleNotFoundError):
    module = types.ModuleType(module_name)
    module.__path__ = [str(script_dir)]
    sys.modules[module_name] = module
    
    # Load apple_deer submodule
    if (script_dir / 'apple_deer' / '__init__.py').exists():
        try:
            module.apple_deer = importlib.import_module(f'{script_dir.name}.apple_deer')
        except (ImportError, ModuleNotFoundError):
            pass
    
    # Load apple_deer_v0 and tensorboard_callback
    for file_name, attr_name in [('apple_deer_v0.py', 'apple_deer_v0'),
                                  ('tensorboard_callback.py', 'tensorboard_callback')]:
        file_path = script_dir / file_name
        if file_path.exists():
            spec = importlib.util.spec_from_file_location(f'{module_name}.{attr_name}', file_path)
            if spec and spec.loader:
                submodule = importlib.util.module_from_spec(spec)
                submodule.__package__ = module_name
                spec.loader.exec_module(submodule)
                setattr(module, attr_name, submodule)

from apple_deer_AROB import apple_deer_v0 as e
from apple_deer_AROB.tensorboard_callback import TensorboardCallback as TbCa
from stable_baselines3 import PPO
import supersuit as ss
from supersuit.multiagent_wrappers import padding_wrappers as pw
from pantheonrl.common.agents import OnPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper

# Set paths relative to base directory
repo_base_dir = script_dir
policy_dir = str(repo_base_dir / "policy/")
tensorboard_dir = str(repo_base_dir / "tensorboard_log/")

# Environment configuration
# agents_dict: Defines agent properties (color, number, attack range, health, etc.)
agents_dict = {
    'Agent': {'color':'blue', 'num':2, 'attack_range':3, 'health':1, 'freeze':False, 'controller':None, 'move':True},
}

# Create environment with specified parameters (without deer)
# x_size, y_size: Environment dimensions (e.g., 15x15)
# max_cycles: Maximum steps per episode (e.g., 300)
# obs_range: Observation range (e.g., 5)
# tree_range: Tree range (e.g., 1)
# nuts_num: Number of nuts (e.g., 25)
# apple_tree_num: Number of apple trees (e.g., 1)
# apple_tree_health: Apple tree health (e.g., 10)
# apple_reward: Reward for collecting apples (e.g., 100)
# nut_reward: Reward for collecting nuts (e.g., 1)
# single_attack_reward: Reward for single attack (e.g., 5)
# double_attack_reward: Reward for coordinated attack (e.g., 20)
# agent_respawn: Enable agent respawn (e.g., True)
# sequential_respawn: Sequential respawn (e.g., True)
# Note: deer_num is not specified as this is a "nodeer" (no deer) environment
env = e.env(
    x_size=15,
    y_size=15,
    agents_dict=agents_dict, 
    max_cycles=300,
    obs_range=5,
    tree_range=1,
    nuts_num=25,
    apple_tree_num=1,
    apple_tree_health=10,
    apple_reward=100,
    nut_reward=1,
    single_attack_reward=5,
    double_attack_reward=20,
    
    #attack_bad_reward=2.0,

    agent_respawn=True,
    sequential_respawn=True,
    signal_visualization=False,
    guide_action_to_signal_system=False,
    correct_signal_action_system=False,
)

# Wrap environment for multi-agent RL
# pad_observations_v0: Pads observations to handle variable-sized observations
# pad_action_space_v0: Pads action space for consistency
# frame_stack_v1: Stacks frames for temporal information (3 frames)
env = pw.pad_observations_v0(env)
env = pw.pad_action_space_v0(env)
env = ss.frame_stack_v1(env, 3)

# Wrap with PettingZoo AEC wrapper for compatibility with PantheonRL
env = PettingZooAECWrapper(env)

# Create partner agents in the environment
# Partner agents learn alongside the ego agent
for i in range(env.n_players - 1):
    partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(i), verbose=1), tensorboard_log=tensorboard_dir, tb_log_name='partner'+str(i))
    env.add_partner_agent(partner, player_num=i + 1)

# Create ego agent and train
# total_timesteps: Total number of training steps (30,000,000)
# tb_log_name: Tensorboard log name for the ego agent
# callback: Tensorboard callback for logging
ego = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
ego.learn(total_timesteps=30000000, tb_log_name='ego', callback=TbCa(env))


# Save all agents' policies
# Policies are saved to policy_dir (relative to parent directory of apple_deer_AROB)
os.makedirs(policy_dir, exist_ok=True)
ego.save(os.path.join(policy_dir, 'policy_ego'))
for i, partner in enumerate(env.partners):
    partner[0].model.save(os.path.join(policy_dir, f'policy_partner{i}'))
