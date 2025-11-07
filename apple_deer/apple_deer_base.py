# Import Libraries
from ast import Name
from collections import defaultdict
import math
import numpy as np
import pygame
from gym import spaces
from gym.utils import seeding
from .utils import agent_utils, two_d_maps
from .utils.agent_layer import AgentLayer
from .utils.controllers import RandomPolicy, SingleActionPolicy, DeerPolicy

class Plainworld:
    def __init__(
        self,
        x_size: int = 5,
        y_size: int = 5,
        max_cycles: int = 300,
        agents_dict = {'Agent': {'color':'blue', 'num':2, 'attack_range':3, 'health':1, 'freeze':False, 'controller':None, 'move':True}},
        obs_range: int = 5,
        tree_range: int = 1, 
        deer_num = 1,
        apple_tree_num = 1,
        nuts_num = 0,
        deer_health = 10,
        apple_tree_health = 3,
        single_attack_reward = 1,
        double_attack_reward = 5,

        #attack時のbadreward
        attack_bad_reward = 0,

        apple_reward = 10,
        nut_reward = 0,
        agent_respawn = True,
        sequential_respawn = True,
        signal_visualization = True,
        guide_action_to_signal_system = False,
        correct_signal_action_system = False,
    ):
        self.constraint_window = 1.0

        self.x_size = x_size
        self.y_size = y_size
        self.map_matrix = two_d_maps.plain_map(self.x_size, self.y_size)
        self.max_cycles = max_cycles
        self.single_attack_reward = single_attack_reward
        self.double_attack_reward = double_attack_reward

        #attack時のbadreward
        self.attack_bad_reward = attack_bad_reward

        self.apple_reward = apple_reward
        self.nut_reward = nut_reward
        self.agent_respawn = agent_respawn
        self.sequential_respawn = sequential_respawn
        self.seed()
        self.signal_visualization = signal_visualization
        self.guide_action_to_signal_system = guide_action_to_signal_system
        self.correct_signal_action_system = correct_signal_action_system
        
        deer_policy = DeerPolicy(self.np_random)
        self.agents_dict = {}
        self.agents_dict["Tree"] = {'color':'green',  'num':apple_tree_num, 'attack_range':0, 'health':apple_tree_health,    'freeze':True, 'controller':None, 'move':False}
        self.agents_dict["Nut"]  = {'color':'brown',  'num':nuts_num,       'attack_range':0, 'health':1,                    'freeze':True, 'controller':None, 'move':False}
        self.agents_dict["Deer"] = {'color':'red',    'num':deer_num,       'attack_range':0, 'health':deer_health,          'freeze':False, 'controller':deer_policy, 'move':False}
        self.agents_dict.update(agents_dict)
        move_agent_name = [key for key,value in agents_dict.items() if value['move']]
        move_agents = []
        for name in move_agent_name:
            for n in range(self.agents_dict[name]['num']):
                n_s = str(n+1)
                move_agents.append(f'{name}_{n_s}')
        self.move_agents = move_agents

        self.move_agent_name = move_agent_name
        self.freeze_agent_name = [key for key,value in self.agents_dict.items() if not value['move']]

        self.num_agents = len(self.move_agents)

        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.latest_obs = [None for _ in range(self.num_agents)]

        self.obs_range = obs_range
        self.obs_offset = int((self.obs_range - 1) / 2)
        self.tree_range = tree_range

        self.agents = {}
        self.agents_layers = {}
        self.n_act_agents = {}
        self.freeze_agents = {}
        for name in self.agents_dict:
            self.agents[name] = agent_utils.create_agents(self.agents_dict[name]['num'], 
                                                          self.map_matrix,
                                                          self.obs_range,
                                                          self.agents_dict[name]['attack_range'],
                                                          self.agents_dict[name]['health'],
                                                          self.np_random)
            self.agents_layers[name] = AgentLayer(self.x_size, self.y_size, self.agents[name],
                                            invincible=(name=='Agent'), agent_respawn=self.agent_respawn ,sequential_respawn=self.sequential_respawn)
            self.n_act_agents[name] = self.agents_layers[name].get_nactions(0)
            self.freeze_agents[name] = self.agents_dict[name]['freeze']

        self.agents_controllers = {}
        for name in self.agents_dict:
            if self.agents_dict[name]['freeze']:
                self.agents_controllers[name] = SingleActionPolicy(4)  # 単一アクション
            else:
                if self.agents_dict[name]['controller'] != None:
                    self.agents_controllers[name] = self.agents_dict[name]['controller']  # 指定したコントローラー
                else:
                    self.agents_controllers[name] = RandomPolicy(self.n_act_agents[name], self.np_random)  # ランダム

        self.current_agent_layer = np.zeros((self.x_size, self.y_size), dtype=np.int32)

        max_agents_overlap = max([self.agents_dict[name]['num'] for name in self.agents_dict])

        obs_space = spaces.Box(
            low=0,
            high=max_agents_overlap,
            shape=(self.obs_range, self.obs_range, 6),
            dtype=np.float32,
        )

        self.agents_action_spaces = {}
        self.agents_observation_spaces = {}
        self.agents_act_dims = {}
        self.agents_gone = {}
        for name in self.agents_dict:
            act_space = spaces.Discrete(self.n_act_agents[name])
            self.agents_action_spaces[name] = [act_space for _ in range(self.agents_dict[name]['num'])]
            self.agents_observation_spaces[name] = [obs_space for _ in range(self.agents_dict[name]['num'])]
            self.agents_act_dims[name] = [self.n_act_agents[name] for i in range(self.agents_dict[name]['num'])]
            self.agents_gone[name] = np.array([False for i in range(self.agents_dict[name]['num'])])
        
        self.model_state = np.zeros((5,) + self.map_matrix.shape, dtype=np.float32)
        self.renderOn = False
        self.pixel_scale = 30
        self.frames = 0

        self.reset()

        

    def observation_space(self, agent):
        return self.agents_observation_spaces[agent]

    def action_space(self, agent):
        return self.agents_action_spaces[agent]

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            pygame.quit()
            self.renderOn = False

    @property
    def agents_func(self):
        return self.agents

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        try:
            policies = list(self.agents_controllers.values())
            for policy in policies:
                try:
                    policy.set_rng(self.np_random)
                except AttributeError:
                    pass
        except AttributeError:
            pass

        return [seed_]

    def get_param_values(self):
        return self.__dict__

    def reset(self):

        for name in self.agents_gone:
            self.agents_gone[name].fill(False)
        
        x_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        y_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        xlb, xub = int(self.x_size * x_window_start), int(
            self.x_size * (x_window_start + self.constraint_window)
        )
        ylb, yub = int(self.y_size * y_window_start), int(
            self.y_size * (y_window_start + self.constraint_window)
        )
        constraints = [[xlb, xub], [ylb, yub]]

        for name in self.agents_dict:
            self.agents[name] = agent_utils.create_agents(self.agents_dict[name]['num'], 
                                                          self.map_matrix,
                                                          self.obs_range,
                                                          self.agents_dict[name]['attack_range'],
                                                          self.agents_dict[name]['health'],
                                                          self.np_random,
                                                          randinit=True,
                                                          constraints=constraints)
            self.agents_layers[name] = AgentLayer(self.x_size, self.y_size, self.agents[name],
                                            invincible=(name=='Agent'), agent_respawn=self.agent_respawn ,sequential_respawn=self.sequential_respawn)

        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.latest_obs = [None for _ in range(self.num_agents)]

        self.model_state[0] = self.map_matrix
        for i,name in enumerate(self.agents_dict):
            self.model_state[i+1] = self.agents_layers[name].get_state_matrix()

        self.frames = 0
        self.renderOn = False

    def step(self, action, agent_type, agent_id, is_last):
        agent_layer = self.agents_layers[agent_type]
        agent_layer.log_another_agent_attack_and_signal(agent_id)
        
        if self.correct_signal_action_system:
            action = self.correct_signal_action(agent_id, agent_layer, action)

        if self.guide_action_to_signal_system:
            if not self.check_agent_in_obs(agent_id, agent_layer):
                if agent_layer.check_another_agent_attack(agent_id):
                    action = self.move_signal_direction_to_pos(self.agents_layers["Agent"], agent_id, action)
        
        if self.correct_signal_action_system:
            action = self.correct_signal_action(agent_id, agent_layer, action)

        agent_layer.move_agent(agent_id, action)
        agent_layer.check_hit(agent_id, self.agents_layers["Deer"])

        self.model_state[1] = self.agents_layers[agent_type].get_state_matrix()

        if is_last:
            for name in self.freeze_agent_name:
                layer = self.agents_layers[name]
                controller = self.agents_controllers[name]
                for i in range(layer.n_agents()):
                    a = controller.act(self.model_state)
                    layer.move_agent(i, a)
            self.frames = self.frames + 1

        self.model_state[0] = self.map_matrix
        for i,name in enumerate(self.agents_dict):
            if name not in self.move_agent_name:
                self.model_state[i+1] = self.agents_layers[name].get_state_matrix()

    # 描画
    def draw_model_state(self):
        # -1 is building pixel flag
        x_len, y_len = self.model_state[0].shape
        for x in range(x_len):
            for y in range(y_len):
                pos = pygame.Rect(
                    self.pixel_scale * x,
                    self.pixel_scale * y,
                    self.pixel_scale,
                    self.pixel_scale,
                )
                col = (0, 0, 0)
                if self.model_state[0][x][y] == -1:
                    col = (255, 255, 255)
                pygame.draw.rect(self.screen, col, pos)

    def draw_agent_observations(self, agent):
        for i in range(self.agents_layers[agent].n_agents()):
            x, y = self.agents_layers[agent].get_position(i)
            patch = pygame.Surface(
                (self.pixel_scale * self.obs_range, self.pixel_scale * self.obs_range)
            )
            patch.set_alpha(128)
            patch.fill((255, 152, 72))
            ofst = self.obs_range / 2.0
            self.screen.blit(
                patch,
                (
                    self.pixel_scale * (x - ofst + 1 / 2),
                    self.pixel_scale * (y - ofst + 1 / 2),
                ),
            )

    def draw_agents(self):
        for name in self.agents_dict:
            for i in range(self.agents_layers[name].n_agents()):
                x, y = self.agents_layers[name].get_position(i)
                center = (
                    int(self.pixel_scale * x + self.pixel_scale / 2),
                    int(self.pixel_scale * y + self.pixel_scale / 2),
                )

                square_vortex = [
                    (int(self.pixel_scale * x), int(self.pixel_scale * y)),
                    (int(self.pixel_scale * x + self.pixel_scale), int(self.pixel_scale * y)),
                    (int(self.pixel_scale * x + self.pixel_scale), int(self.pixel_scale * y + self.pixel_scale)),
                    (int(self.pixel_scale * x), int(self.pixel_scale * y + self.pixel_scale))
                ]

                if self.agents_dict[name]['color'] == 'red':
                    col = (255, 0, 0)
                if self.agents_dict[name]['color'] == 'blue':
                    col = (0, 0, 255)
                if self.agents_dict[name]['color'] == 'brown':
                    col = (160, 87, 41)

                if name == "Deer":
                    pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))
                elif name == "Agent":
                    pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))
                elif name == "Nut":
                    pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 6))

    def draw_agent_counts(self):
        font = pygame.font.SysFont("Comic Sans MS", self.pixel_scale * 2 // 3)
        for name in self.agents_dict:
            positions = defaultdict(int)

            for i in range(self.agents_layers[name].n_agents()):
                x, y = self.agents_layers[name].get_position(i)
                positions[(x, y)] += 1

                for (x, y) in positions:
                    (pos_x, pos_y) = (
                        self.pixel_scale * x + self.pixel_scale // 2,
                        self.pixel_scale * y + self.pixel_scale // 2,
                    )

                agent_count = positions[(x, y)]
                count_text: str
                if agent_count < 1:
                    count_text = ""
                elif agent_count < 10:
                    count_text = str(agent_count)
                else:
                    count_text = "+"

                text = font.render(count_text, False, (220, 220, 220))
                self.screen.blit(text, (pos_x, pos_y))
    
    def draw_agent_attack(self, agent):
        for i in range(self.agents_layers[agent].n_agents()):
            if self.agents_layers[agent].allies[i].attack_and_signal:
                x, y = self.agents_layers[agent].get_position(i)
                attack_range = self.agents_layers[agent].allies[i].attack_range
                patch = pygame.Surface(
                    (self.pixel_scale * attack_range, self.pixel_scale * attack_range)
                )
                patch.set_alpha(128)
                patch.fill((255, 40, 40))
                ofst = attack_range / 2.0
                self.screen.blit(
                    patch,
                    (
                        self.pixel_scale * (x - ofst + 1 / 2),
                        self.pixel_scale * (y - ofst + 1 / 2),
                    ),
                )

    def draw_tree_range(self, agent):
        for i in range(self.agents_layers[agent].n_agents()):
            x, y = self.agents_layers[agent].get_position(i)
            tree_range = self.tree_range
            patch = pygame.Surface(
                (self.pixel_scale * tree_range, self.pixel_scale * tree_range)
            )
            patch.set_alpha(128)
            patch.fill((102, 255, 102))
            ofst = tree_range / 2.0
            self.screen.blit(
                patch,
                (
                    self.pixel_scale * (x - ofst + 1 / 2),
                    self.pixel_scale * (y - ofst + 1 / 2),
                ),
            )

    def draw_plot_signal(self, agent):
        for i in range(self.agents_layers[agent].n_agents()):
            if self.agents_layers[agent].allies[i].another_agent_attack_and_signal:
                x, y = self.agents_layers[agent].get_position(i)
                signal_direction = self.get_signal_direction_plot(agent, i)
                signal_x, signal_y = self.direction_to_pos(signal_direction)
                patch = pygame.Surface(
                    (self.pixel_scale * 1, self.pixel_scale * 1)
                )
                patch.set_alpha(128)
                patch.fill((255, 255, 255))
                self.screen.blit(
                    patch,
                    (
                        (self.pixel_scale * (x - (self.obs_range / 2)) + (self.pixel_scale // 2) + (self.pixel_scale * signal_x)),
                        (self.pixel_scale * (y - (self.obs_range / 2)) + (self.pixel_scale // 2) + (self.pixel_scale * signal_y))
                    ),
                )

    def render(self, mode='human'):
        if not self.renderOn:
            if mode == 'human':
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )
            else:
                self.screen = pygame.Surface(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )

            self.renderOn = True
        self.draw_model_state()

        self.draw_agent_observations('Agent')
        self.draw_tree_range('Tree')
        self.draw_agent_attack('Agent')
        if self.signal_visualization == True:
            self.draw_plot_signal('Agent')

        self.draw_agents()
        #self.draw_agent_counts()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if mode == "human":
            pygame.display.flip()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if mode == "rgb_array"
            else None )

    def save_image(self, file_name):
        self.render()
        capture = pygame.surfarray.array3d(self.screen)

        xl, xh = -self.obs_offset - 1, self.x_size + self.obs_offset + 1
        yl, yh = -self.obs_offset - 1, self.y_size + self.obs_offset + 1

        window = pygame.Rect(xl, yl, xh, yh)
        subcapture = capture.subsurface(window)

        pygame.image.save(subcapture, file_name)

    @property
    # ここ変えれる
    def is_terminal(self):
        #if self.agents_layers['evaders'].n_agents() == 0:
        #    return True
        return False

    def n_agents(self):
        n = 0 
        for name in self.move_agent_name:
            n += self.agents_layers[name].n_agents()
        return n
    #observe変更
    def safely_observe(self, agent_type, i):
        agent_layer = self.agents_layers[agent_type]
        size = self.agents_dict[agent_type]['num']
        obs = self.collect_obs(size, agent_layer, i)
        return obs

    def collect_obs(self, size, agent_layer, i):
        for j in range(size):
            if int(i) == int(j):
                return self.collect_obs_by_idx(agent_layer, int(i))
        assert False, "bad index"

    def collect_obs_by_idx(self, agent_layer, agent_idx):
        # returns a flattened array of all the observations
        obs = np.zeros((6, self.obs_range, self.obs_range), dtype=np.float32)
        obs[0].fill(1.0)  # border walls set to -0.1?
        xp, yp = agent_layer.get_position(agent_idx)
        obs_xlow, obs_xhi, obs_ylow, obs_yhi = self.plot_obs_clip(agent_layer, agent_idx)
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

        obs[0:5, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:5, xlo:xhi, ylo:yhi])
        obs[5] = self.plot_signal(agent_layer, agent_idx, obs_xlow, obs_xhi, obs_ylow, obs_yhi)
        return obs

    def obs_clip(self, x, y):
        xld = x - self.obs_offset
        xhd = x + self.obs_offset
        yld = y - self.obs_offset
        yhd = y + self.obs_offset
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_size - 1),
            np.clip(xhd, 0, self.x_size - 1),
            np.clip(yld, 0, self.y_size - 1),
            np.clip(yhd, 0, self.y_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.obs_offset, 0)), abs(
            np.clip(yld, -self.obs_offset, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1
    
    def plot_obs_clip(self, agent_layer, agent_idx):
        agent_pos = agent_layer.get_position(agent_idx)
        obs_ofst = int((self.obs_range - 1) / 2)
        obs_xlow = agent_pos[0] - obs_ofst
        obs_xhi = agent_pos[0] + obs_ofst
        obs_ylow = agent_pos[1] -obs_ofst
        obs_yhi = agent_pos[1] + obs_ofst

        return obs_xlow, obs_xhi, obs_ylow, obs_yhi
         

    def plot_signal(self, agent_layer, agent_idx, obs_xlow, obs_xhi, obs_ylow, obs_yhi):
        """
        According to another agent's direction, plot signal to obs matrix.
        If the agent sent signal, plot the signal, else, return np.zeros.
        If the agent in obs range, plot there, else, plot outermost.
        """
        obs_signal_layer = np.zeros((self.obs_range, self.obs_range), dtype=np.float32)
        if not agent_layer.check_another_agent_attack(agent_idx):
            return obs_signal_layer

        this_agent_pos, another_agent_pos = agent_layer.new_get_positions(agent_idx)
        if (obs_xlow <= another_agent_pos[0] <= obs_xhi) and (obs_ylow <= another_agent_pos[1] <= obs_yhi):
            obs_signal_layer[another_agent_pos[0] - obs_xlow][another_agent_pos[1] - obs_ylow] = 1
        else:
            x_distance = another_agent_pos[0] - this_agent_pos[0]
            y_distance = another_agent_pos[1] - this_agent_pos[1]
            signal_direction = math.atan2(y_distance, x_distance)
            plot_pos = self.direction_to_pos(signal_direction)
            obs_signal_layer[plot_pos[0]][plot_pos[1]] = 1
        return obs_signal_layer

    def direction_to_pos(self, signal_direction):
        if signal_direction < (-15/16 * math.pi):   
            x=0; y=2
        elif (-15/16 * math.pi) <= signal_direction < (-13/16 * math.pi):
            x=0; y=1
        elif (-13/16 * math.pi) <= signal_direction < (-11/16 * math.pi):
            x=0; y=0
        elif (-11/16 * math.pi) <= signal_direction < (-9/16 * math.pi): 
            x=1; y=0
        elif (-9/16 * math.pi) <= signal_direction < (-7/16 * math.pi):   
            x=2; y=0
        elif (-7/16 * math.pi) <= signal_direction < (-5/16 * math.pi):   
            x=3; y=0
        elif (-5/16 * math.pi) <= signal_direction < (-3/16 * math.pi):   
            x=4; y=0
        elif (-3/16 * math.pi) <= signal_direction < (-1/16 * math.pi):  
            x=4; y=1
        elif (-1/16 * math.pi) <= signal_direction < (1/16 * math.pi):  
            x=4; y=2
        elif (1/16 * math.pi) <= signal_direction < (3/16 * math.pi):   
            x=4; y=3
        elif (3/16 * math.pi) <= signal_direction < (5/16 * math.pi): 
            x=4; y=4
        elif (5/16 * math.pi) <= signal_direction < (7/16 * math.pi): 
            x=3; y=4
        elif (7/16 * math.pi) <= signal_direction < (9/16 * math.pi):   
            x=2; y=4
        elif (9/16 * math.pi) <= signal_direction < (11/16 * math.pi):   
            x=1; y=4
        elif (11/16 * math.pi) <= signal_direction < (13/16 * math.pi):  
            x=0; y=4
        elif (13/16 * math.pi) <= signal_direction < (15/16 * math.pi):  
            x=0; y=3
        elif (15/16 * math.pi) <= signal_direction:   
            x=0; y=2
        return [x,y]


    #視界内に相方エージェントがいるかどうかをチェック
    def check_agent_in_obs(self, agent_idx, agent_layer): 
        xp, yp = agent_layer.get_position(agent_idx)
        obs_xlow, obs_xhi, obs_ylow, obs_yhi = self.plot_obs_clip(agent_layer, agent_idx)
        another_agent_pos = agent_layer.get_another_agent_position(agent_idx)
        if (obs_xlow <= another_agent_pos[0] <= obs_xhi) and (obs_ylow <= another_agent_pos[1] <= obs_yhi):
            return True
        else:
            return False
    
    def get_signal_direction(self, agent_idx, agent_layer):
        this_agent_pos, another_agent_pos = agent_layer.new_get_positions(agent_idx)
        x_distance = another_agent_pos[0] - this_agent_pos[0]
        y_distance = another_agent_pos[1] - this_agent_pos[1]
        signal_direction = math.atan2(y_distance, x_distance)
        return signal_direction

    def get_signal_direction_plot(self, agent, i):
        this_agent_pos, another_agent_pos= self.agents_layers[agent].new_get_positions(i)
        x_distance = another_agent_pos[0] - this_agent_pos[0]
        y_distance = another_agent_pos[1] - this_agent_pos[1]
        signal_direction = math.atan2(y_distance, x_distance)
        return signal_direction



    def move_signal_direction_to_pos(self, agent_layer, agent_idx, action):
        signal_direction = self.get_signal_direction(agent_idx, agent_layer)

        if(-1 * math.pi) <= signal_direction < (-3/4 * math.pi):   # 左
            action = 0 #左に動かす
        elif(-3/4 * math.pi) <= signal_direction  < (-1/4 * math.pi):   # 下
            action = 3 #下に動かす
        elif (-1/4 * math.pi) <= signal_direction  < (1/4 * math.pi):   # 右
            action = 1 #右に動かす
        elif (1/4 * math.pi) <= signal_direction < (3/4 * math.pi):   # 上
            action = 2 #上に動かす
        elif (3/4 * math.pi) <= signal_direction  <=  (1 * math.pi):   # 左
            action = 0 #左に動かす
        return action

    def correct_signal_action(self, agent_id, agent_layer, action):
        if self.check_adjacent(agent_id, agent_layer, self.agents_layers["Deer"]):
            return action
        if self.check_adjacent(agent_id, agent_layer, self.agents_layers["Tree"]):
            return action
        if action == 5:
            action = 4
            return action
        else:
            return action

    def check_adjacent(self, agent_id, agent_layer, target_layer):
        this_agent_pos = agent_layer.get_position(agent_id)
        agent_xlow, agent_xhi, agent_ylow, agent_yhi= this_agent_pos[0] - 1, this_agent_pos[0] + 1, this_agent_pos[1] - 1, this_agent_pos[1] + 1
        target_pos_list = [i.current_position() for i in target_layer.allies]

        for target_pos in target_pos_list:
            if (agent_xlow <= target_pos[0] <= agent_xhi) and (agent_ylow <= target_pos[1] <= agent_yhi):
                return True
            else:
                return False

    def agents_death(self):
        for agents_layer in self.agents_layers.values():
            for i in range(agents_layer.n_agents()):
                if agents_layer.allies[i].health <= 0:
                    agents_layer.death_agent(i)