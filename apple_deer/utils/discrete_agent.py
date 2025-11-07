import numpy as np
from gym import spaces

from pettingzoo.sisl._utils import Agent

#################################################################
# Implements the Single 2D Agent Dynamics
#################################################################


class DiscreteAgent(Agent):

    # constructor
    def __init__(
        self,
        xs,
        ys,
        map_matrix,
        randomizer,
        obs_range=3,
        attack_range=0,
        health = 1,
        n_channels=4,
        seed=1,
        flatten=False,
    ):
        # map_matrix is the may of the environment (-1 are buildings)
        # n channels is the number of observation channels

        self.random_state = randomizer

        self.xs = xs
        self.ys = ys

        self.eactions = [
            0,  # move left
            1,  # move right
            2,  # move up
            3,  # move down
            4,  # stay
            5,  # attack and send signal
        ]

        self.motion_range = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0], [0, 0]]

        self.current_pos = np.zeros(2, dtype=np.int32)  # x and y position
        self.last_pos = np.zeros(2, dtype=np.int32)
        self.temp_pos = np.zeros(2, dtype=np.int32)

        self.map_matrix = map_matrix

        self.terminal = False

        self._obs_range = obs_range

        if flatten:
            self._obs_shape = (n_channels * obs_range**2 + 1,)
        else:
            self._obs_shape = (obs_range, obs_range, 6)
            # self._obs_shape = (4, obs_range, obs_range)
        
        self.attack_range=attack_range
        self.attack_and_signal = False
        self.another_agent_attack_and_signal = False
        self.attack_hit = False
        self.health = health
        self.max_health = health
        self.last_attacked = False

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=self._obs_shape)

    @property
    def action_space(self):
        return spaces.Discrete(len(self.eactions))

    # Dynamics Functions
    def step(self, a):
        cpos = self.current_pos
        lpos = self.last_pos
        # if dead or reached goal dont move
        if self.terminal:
            return cpos
        # if in building, dead, and stay there
        if self.inbuilding(cpos[0], cpos[1]):
            self.terminal = True
            return cpos
        # if agent was attacked, stay
        if self.last_attacked:
            self.last_attacked = False
            return cpos
        tpos = self.temp_pos
        tpos[0] = cpos[0]
        tpos[1] = cpos[1]

        # transition is deterministic
        tpos += self.motion_range[a]
        x = tpos[0]
        y = tpos[1]
        # check bounds
        if not self.inbounds(x, y):
            return cpos
        # if bumped into building, then stay
        if self.inbuilding(x, y):
            return cpos
        else:
            self.attack_and_signal = (a==5)
            lpos[0] = cpos[0]
            lpos[1] = cpos[1]
            cpos[0] = x
            cpos[1] = y
            return cpos

    def respawn(self):
        self.health = self.max_health
        while True:
            new_x = self.random_state.integers(0, self.xs)
            new_y = self.random_state.integers(0, self.ys)
            if (self.inbounds(new_x, new_y)) and (not self.inbuilding(new_x, new_y)):
                break
        self.set_position(new_x, new_y)

    def get_state(self):
        return self.current_pos

    # Helper Functions
    def inbounds(self, x, y):
        if 0 <= x < self.xs and 0 <= y < self.ys:
            return True
        return False

    def inbuilding(self, x, y):
        if self.map_matrix[x, y] == -1:
            return True
        return False

    def nactions(self):
        return len(self.eactions)

    def set_position(self, xs, ys):
        self.current_pos[0] = xs
        self.current_pos[1] = ys

    def current_position(self):
        return self.current_pos

    def last_position(self):
        return self.last_pos
