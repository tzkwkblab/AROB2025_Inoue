import numpy as np
import pygame
from gym.utils import EzPickle
from collections import defaultdict

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .apple_deer_base import Plainworld as _env

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv, EzPickle):

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'plainworld_v0',
        'is_parallelizable': True,
        'render_fps': 5,
        'has_manual_policy': True,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        self.env = _env(*args, **kwargs)
        pygame.init()
        self.agents = self.env.move_agents
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        self.steps = 0
        self.closed = False

        self.count_total_reward_ego = 0
        self.count_nut_reward_ego = 0
        self.num_attack_on_tree_ego = 0
        self.count_apple_reward_ego = 0
        self.count_nut_reward_ego = 0
        self.count_single_attack_reward_ego = 0
        self.count_double_attack_reward_ego = 0
        self.num_get_nut_ego = 0
        self.num_single_attack_hit_ego = 0
        self.num_double_attack_hit_ego = 0
        self.num_apple_tree_now_ego = 0

        self.count_total_reward_partner = 0
        self.count_nut_reward_partner = 0
        self.num_attack_on_tree_partner = 0
        self.count_apple_reward_partner = 0
        self.count_single_attack_reward_partner = 0
        self.count_double_attack_reward_partner = 0
        self.num_get_nut_partner = 0
        self.num_single_attack_hit_partner = 0
        self.num_double_attack_hit_partner = 0
        self.num_apple_tree_now_partner = 0

        self.total_reward = 0
        self.total_nut_reward = 0
        self.total_num_attack_on_tree = 0
        self.total_apple_reward = 0
        self.total_nut_reward = 0
        self.total_single_attack_reward = 0
        self.total_double_attack_reward = 0
        self.total_num_get_nut = 0
        self.total_num_single_attack_hit = 0
        self.total_num_double_attack_hit = 0
        self.total_num_apple_tree_together = 0
        
        
        x = 0
        ac = []
        ob = []
        for i in range(len(self.env.move_agent_name)):
            x += self.env.agents_act_dims[self.env.move_agent_name[i]][0]
            for acc in  self.env.action_space(self.env.move_agent_name[i]):
                ac.append(acc)
            for obb in self.env.observation_space(self.env.move_agent_name[i]):
                ob.append(obb)
        ac = dict(zip(self.agents, ac))
        ob = dict(zip(self.agents, ob))
        self.n_act_agents = x
        self.action_spaces = ac
        self.observation_spaces = ob

    def seed(self, seed=None):
        self.env.seed(seed)

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents[:]
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.count_total_reward_ego = 0
        self.count_nut_reward_ego = 0
        self.num_attack_on_tree_ego = 0
        self.count_apple_reward_ego = 0
        self.count_nut_reward_ego = 0
        self.count_single_attack_reward_ego = 0
        self.count_double_attack_reward_ego = 0
        self.num_get_nut_ego = 0
        self.num_single_attack_hit_ego = 0
        self.num_double_attack_hit_ego = 0
        self.num_apple_tree_now_ego = 0

        self.count_total_reward_partner = 0
        self.count_nut_reward_partner = 0
        self.num_attack_on_tree_partner = 0
        self.count_apple_reward_partner = 0
        self.count_single_attack_reward_partner = 0
        self.count_double_attack_reward_partner = 0
        self.num_get_nut_partner = 0
        self.num_single_attack_hit_partner = 0
        self.num_double_attack_hit_partner = 0
        self.num_apple_tree_now_partner = 0

        self.total_reward = 0
        self.total_nut_reward = 0
        self.total_num_attack_on_tree = 0
        self.total_apple_reward = 0
        self.total_nut_reward = 0
        self.total_single_attack_reward = 0
        self.total_double_attack_reward = 0
        self.total_num_get_nut = 0
        self.total_num_single_attack_hit = 0
        self.total_num_double_attack_hit = 0
        self.total_num_apple_tree_together = 0

        self.env.reset()

    def close(self):
        if not self.closed:
            self.closed = True
            self.env.close()

    def render(self, mode='rgb_array'):
        if not self.closed:
            return self.env.render(mode)

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection
        agent_type = agent.split('_')[0]
        agent_id = int(agent.split('_')[1])-1
        self.env.step(
            action, agent_type, agent_id, self._agent_selector.is_last()
        )
        self.count_attack_on_tree(agent_type, agent_id, self.env)
        self.sum_reward(agent_type, agent_id, self.env)
        #self.sum_alter_reward(agent_type, agent_id, self.env)

        self.count_total_reward_ego += self.rewards['Agent_1']
        self.count_total_reward_partner += self.rewards['Agent_2']
        self.total_reward = self.count_total_reward_ego + self.count_total_reward_partner
        self.env.agents_death()
        for k in self.dones:
            if self.env.frames >= self.env.max_cycles:
                self.dones[k] = True
            else:
                self.dones[k] = self.env.is_terminal
        #for k in self.agents:
            #self.rewards[k] = self.env.latest_reward_state[self.agent_name_mapping[k]] 
        self.steps += 1

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def observe(self, agent):
        agent_type = agent.split('_')[0]
        ii = int(agent.split('_')[1]) - 1
        o = self.env.safely_observe(agent_type, ii)
        return np.swapaxes(o, 2, 0)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def sum_reward(self, agent_type, agent_id, env):
        hit_r = dict(zip(self.agents, [(0) for _ in self.agents]))
        apple_r = dict(zip(self.agents, [(0) for _ in self.agents]))
        nut_r = dict(zip(self.agents, [(0) for _ in self.agents]))
        
        hit_r = self.hit_reward(agent_type, agent_id, env)
        apple_r = self.apple_reward(agent_type, agent_id, env)
        nut_r = self.nut_reward(agent_type, agent_id, env)
        #print(hit_r)
        #print(apple_r)
        #print(nut_r)
        for key in self.agents:
            self.rewards[key] = hit_r[key] + apple_r[key] + nut_r[key]
        #print(self.rewards)
        #print("END")

    # Reward calculation for environment without nuts

    def sum_alter_reward(self, agent_type, agent_id, env):
        hit_r = dict(zip(self.agents, [(0) for _ in self.agents]))
        apple_r = dict(zip(self.agents, [(0) for _ in self.agents]))
        
        hit_r = self.hit_reward(agent_type, agent_id, env)
        apple_r = self.apple_reward(agent_type, agent_id, env)
        bad_r = self.bad_reward(agent_type, agent_id, env)
        #print(hit_r)
        #print(apple_r)
        #print(nut_r)
        for key in self.agents:
            self.rewards[key] = hit_r[key] + apple_r[key] - bad_r[key]
        #print(self.rewards)
        #print("END")


    def hit_reward(self, agent_type, agent_id, env):
        hit_r = dict(zip(self.agents, [(0) for _ in self.agents]))

        #ret_dict = {}
        if not agent_type=='Agent':
            return hit_r
        agent_layer = env.agents_layers[agent_type]
        if not agent_layer.allies[agent_id].attack_hit:
            return hit_r
        another_agent_hit = False
        for i_agent in range(agent_layer.nagents):
            if agent_id != i_agent:
                another_agent_hit = agent_layer.allies[i_agent].attack_hit
        #print(another_agent_hit)
        if another_agent_hit:
            self.total_num_double_attack_hit += 1
            #print(agent_id)
            if agent_id == 0:
                self.num_double_attack_hit_ego += 1
            elif agent_id == 1:
                self.num_double_attack_hit_partner += 1
                
            hit_r.update(Agent_1 =  env.double_attack_reward, Agent_2 = env.double_attack_reward)
            #print(hit_r)
            #new_reward = self.total_num_double_attack_hit * env.double_attack_reward
            #hit_r.update(Agent_1 = new_reward, Agent_2 = new_reward)
            #hit_r.update(Agent_1 = env.double_attack_reward, Agent_2 = env.double_attack_reward)
            
            self.count_double_attack_reward_ego += env.double_attack_reward
            self.count_double_attack_reward_partner += env.double_attack_reward
            self.total_double_attack_reward = self.count_double_attack_reward_ego + self.count_double_attack_reward_partner

            for i_deer in range(env.agents_layers["Deer"].nagents):
                env.agents_layers["Deer"].damage(i_deer)
                #env.agents_layers["Deer"].damage(i_deer)

            return hit_r

        else:
            if agent_id == 0:
                self.num_single_attack_hit_ego += 1
                #hit_r.update(Agent_1 = env.single_attack_reward)
                hit_r.update(Agent_1 = env.single_attack_reward)
                self.count_single_attack_reward_ego += env.single_attack_reward
            elif agent_id == 1:
                self.num_single_attack_hit_partner += 1
                #hit_r.update(Agent_2 = env.single_attack_reward)
                hit_r.update(Agent_2 = env.single_attack_reward)
                self.count_single_attack_reward_partner += env.single_attack_reward
            self.total_num_single_attack_hit = self.num_single_attack_hit_ego + self.num_single_attack_hit_partner
            self.total_single_attack_reward = self.count_single_attack_reward_ego + self.count_single_attack_reward_partner

            for i_deer in range(env.agents_layers["Deer"].nagents):
                env.agents_layers["Deer"].damage(i_deer)
            
            return hit_r

    def apple_reward(self, agent_type, agent_id, env):
        #ret_dict = {}
        apple_r = dict(zip(self.agents, [(0) for _ in self.agents]))
        if not agent_type=='Agent':
            return apple_r
        agent_layer = env.agents_layers[agent_type]
        tree_layer = env.agents_layers["Tree"]
        for i in range(agent_layer.nagents):
            if agent_id == i:
                current_agent_pos = agent_layer.get_position(i)
            else:
                another_agent_pos = agent_layer.get_position(i)
        for i_tree in range(tree_layer.nagents):
            tree_pos = tree_layer.get_position(i_tree)
            tree_ofst = env.tree_range / 2 - 1 / 2
            if ((tree_pos[0] - tree_ofst) <= current_agent_pos[0] <= (tree_pos[0] + tree_ofst)) and ((tree_pos[1] - tree_ofst) <= current_agent_pos[1] <= (tree_pos[1] + tree_ofst)):
                if agent_id == 0:
                    self.num_apple_tree_now_ego += 1
                elif agent_id == 1:
                    self.num_apple_tree_now_partner += 1
                if ((tree_pos[0] - tree_ofst) <= another_agent_pos[0] <= (tree_pos[0] + tree_ofst)) and ((tree_pos[1] - tree_ofst) <= another_agent_pos[1] <= (tree_pos[1] + tree_ofst)):
                    self.total_num_apple_tree_together += 1

                    apple_r.update(Agent_1 = env.apple_reward, Agent_2 = env.apple_reward)

                    self.count_apple_reward_ego += env.apple_reward
                    self.count_apple_reward_partner += env.apple_reward
                    self.total_apple_reward = self.count_apple_reward_ego + self.count_apple_reward_partner
                    tree_layer.damage(i_tree)
        return apple_r
    
    def nut_reward(self, agent_type, agent_id, env):
        nut_r = dict(zip(self.agents, [(0) for _ in self.agents]))
        if not agent_type=='Agent':
            return nut_r
        agent_pos = env.agents_layers[agent_type].get_position(agent_id)
        nuts_layer = env.agents_layers["Nut"]

        for i_nut in range(nuts_layer.nagents):
            nut_pos = nuts_layer.get_position(i_nut)
            if (nut_pos[0] == agent_pos[0]) and (nut_pos[1] == agent_pos[1]):
                if agent_id == 0:
                    self.num_get_nut_ego += 1
                    self.count_nut_reward_ego += env.nut_reward
                    nut_r.update(Agent_1 = env.nut_reward)
                elif agent_id == 1:
                    self.num_get_nut_partner += 1
                    self.count_nut_reward_partner += env.nut_reward
                    nut_r.update(Agent_2 = env.nut_reward)
                nuts_layer.damage(i_nut)
            self.total_num_get_nut = self.num_get_nut_ego + self.num_get_nut_partner
            self.total_nut_reward = self.count_nut_reward_ego + self.count_nut_reward_partner
        return nut_r

    def bad_reward(self, agent_type, agent_id, env):
        bad_r = dict(zip(self.agents, [(0) for _ in self.agents]))
        agent_layer = env.agents_layers[agent_type]
        if not agent_type=='Agent':
            return bad_r
        if not agent_layer.allies[agent_id].attack_and_signal:
            return bad_r

        if agent_id == 0:
            bad_r.update(Agent_1 = env.attack_bad_reward)
        elif agent_id == 1:
            bad_r.update(Agent_2 = env.attack_bad_reward)
        return bad_r

    def count_attack_on_tree(self, agent_type, agent_id, env):
        if not agent_type == "Agent":
            return
        agent_layer = env.agents_layers[agent_type]
        tree_layer = env.agents_layers["Tree"]
        if not agent_layer.allies[agent_id].attack_and_signal:
            return

        current_agent_pos = agent_layer.get_position(agent_id)
        for i_tree in range(tree_layer.nagents):
            tree_pos = tree_layer.get_position(i_tree)
            tree_ofst = env.tree_range / 2 - 1 / 2
            if ((tree_pos[0] - tree_ofst) <= current_agent_pos[0] <= (tree_pos[0] + tree_ofst)) and ((tree_pos[1] - tree_ofst) <= current_agent_pos[1] <= (tree_pos[1] + tree_ofst)):
                if agent_id == 0:
                    self.num_attack_on_tree_ego += 1
                elif agent_id == 1:
                    self.num_attack_on_tree_partner += 1
        self.total_num_attack_on_tree = self.num_attack_on_tree_ego + self.num_attack_on_tree_partner