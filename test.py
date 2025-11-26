#!/usr/bin/env python
"""
Test script for Apple Deer environment with AROB configuration.
This script loads trained policies and generates GIF animations of agent behavior.
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
    
    # Load apple_deer_v0
    spec = importlib.util.spec_from_file_location(f'{module_name}.apple_deer_v0', 
                                                  script_dir / 'apple_deer_v0.py')
    if spec and spec.loader:
        submodule = importlib.util.module_from_spec(spec)
        submodule.__package__ = module_name
        spec.loader.exec_module(submodule)
        module.apple_deer_v0 = submodule

from apple_deer_AROB import apple_deer_v0 as e
from stable_baselines3 import PPO

repo_base_dir = script_dir  # Base directory for paths
import supersuit as ss
from supersuit.multiagent_wrappers import padding_wrappers as pw
from pantheonrl.common.agents import OnPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper

import numpy as np
from array2gif import write_gif

# Number of GIFs to generate
gif_num = 6  #10

for m in range(gif_num):
    # Set paths relative to base directory
    base_dir = repo_base_dir
    policy_dir = str(base_dir / "policy/")
    gif_dir = str(base_dir / "GIF/")

    num_steps = 300  # Maximum number of steps per episode

    # Environment configuration
    agents_dict = {
        'Agent': {'color':'blue', 'num':2, 'attack_range':3, 'health':1, 'freeze':False, 'controller':None, 'move':True},
    }

    # Create environment with specified parameters
    env = e.env(
        x_size=15,
        y_size=15,
        agents_dict=agents_dict, 
        max_cycles=300,
        obs_range=7,
        tree_range=1,
        deer_num=1,
        nuts_num=25,
        apple_tree_num=1,
        deer_health=30,
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
    env = pw.pad_observations_v0(env)
    env = pw.pad_action_space_v0(env)
    env = ss.frame_stack_v1(env, 3)
    env = PettingZooAECWrapper(env)

    # Load partner agents from saved policies
    for i in range(env.n_players - 1):
        partner = OnPolicyAgent(PPO.load(os.path.join(policy_dir, f'policy_partner{i}'), env=env.getDummyEnv(i)))
        env.add_partner_agent(partner, player_num=i+1)
    
    # Load ego agent from saved policy
    ego = PPO.load(os.path.join(policy_dir, 'policy_ego'), env=env)
    # Create output directory if it doesn't exist
    os.makedirs(gif_dir, exist_ok=True)

    # Prepare for test
    ego_obs = env.reset()

    obs_list = []
    total_reward = 0
    done = False

    # Run test episode
    for i in range(num_steps):
        ego_act = ego.predict(ego_obs, deterministic=True) if not done else None
        ego_obs, ego_rew, done, info = env.step(ego_act[0])
        total_reward += ego_rew
        obs_list.append(
            np.transpose(env.base_env.render(mode='rgb_array'), axes=(1,0,2))
        )
        if done:
            break
    env.close()

    # Output result and save GIF
    print(f'Episode {m}: Total reward = {total_reward}')
    gif_path = os.path.join(gif_dir, f'test{m}.gif')
    write_gif(obs_list, gif_path, fps=4)  # fps = 4