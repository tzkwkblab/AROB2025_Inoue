import os

from apple_deer_AROB import apple_deer_v0 as e
#import apple_deer_v0 as e
from apple_deer_AROB.tensorboard_callback import TensorboardCallback as TbCa
from stable_baselines3 import PPO
import supersuit as ss
from supersuit.multiagent_wrappers import padding_wrappers as pw
from pantheonrl.common.agents import OnPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#実験用
#policy_dir = "policy/XXq/policy_30m_all_br2.0/"
#tensorboard_dir = "tensorboard_log/XXq/policy_30m_all_br2.0/"

#デバッグ用
policy_dir = "policy/AROB/dh1/try7"
tensorboard_dir = "tensorboard_log/AROB/dh1/try7"

# Setting environment
agents_dict = {
    'Agent': {'color':'blue', 'num':2, 'attack_range':3, 'health':1, 'freeze':False, 'controller':None, 'move':True},
}

env = e.env(
    x_size=15,
    y_size=15,
    agents_dict=agents_dict, 
    max_cycles = 300,
    obs_range = 5,

	tree_range = 1,
    deer_num = 1,
    nuts_num = 25,
    apple_tree_num = 1,
    deer_health = 1,
    apple_tree_health = 10,
    apple_reward = 100,
    nut_reward = 1,
    single_attack_reward = 5,
    double_attack_reward = 20,
    
    #attack_bad_reward = 2.0,

    agent_respawn = True,
    sequential_respawn = True,
    signal_visualization = False,
    guide_action_to_signal_system = False,
    correct_signal_action_system = False,
)

env = pw.pad_observations_v0(env)
env = pw.pad_action_space_v0(env)
env = ss.frame_stack_v1(env, 3)

env = PettingZooAECWrapper(env)

# Create partner agents in env

for i in range(env.n_players - 1):
    partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(i), verbose=1), tensorboard_log=tensorboard_dir, tb_log_name='partner'+str(i))
    env.add_partner_agent(partner, player_num=i + 1)

# Create ego agent and learn
ego = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
ego.learn(total_timesteps=30000000, tb_log_name='ego', callback=TbCa(env))


# Save all agents' policies
os.makedirs(policy_dir, exist_ok=True)
ego.save(policy_dir+'policy_ego')
for i, partner in enumerate(env.partners):
    partner[0].model.save(policy_dir+'policy_partner'+str(i))
