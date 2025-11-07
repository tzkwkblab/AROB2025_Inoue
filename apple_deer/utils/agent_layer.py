import numpy as np
from copy import deepcopy

from .discrete_agent import DiscreteAgent

#################################################################
# Implements a Cooperating Agent Layer for 2D problems
#################################################################


class AgentLayer:
    def __init__(self, xs, ys, allies, seed=1, invincible=True, agent_respawn=True, sequential_respawn=True):
        """
        xs: x size of map
        ys: y size of map
        allies: list of ally agents
        seed: seed

        Each ally agent must support:
        - move(action)
        - current_position()
        - nactions()
        - set_position(x, y)
        """

        self.allies = allies
        self.wait_respawn = []  # リスポーン待機列
        self.nagents = len(allies)
        self.global_state = np.zeros((xs, ys), dtype=np.int32)
        self.invincible = invincible    # エージェントが不死か否か
        self.agent_respawn = agent_respawn
        self.sequential_respawn = sequential_respawn    # 即時リスポーンするか否か

    def n_agents(self):
        return len(self.allies)

    def move_agent(self, agent_idx, action):

        return self.allies[agent_idx].step(action)

    def set_position(self, agent_idx, x, y):
        self.allies[agent_idx].set_position(x, y)

    def get_position(self, agent_idx):
        """
        Returns the position of the given agent
        """
        return self.allies[agent_idx].current_position()
    
    def get_positions(self, agent_idx):
        """
        Returns the position of the given agent, and the position of the another agent
        This method is usable if only 2 agents exist
        """
        for i in range(len(self.allies)):
            if i == agent_idx:
                given_agent_pos = self.current_position()
            else:
                another_agent_pos = self.current_position()
        return given_agent_pos, another_agent_pos

    
    def is_another_agent_sent_signal(self, agent_idx):
        for i in range(len(self.allies)):
            if i == agent_idx:
                continue
            return self.allies[i].attack_and_signal

    #自分以外のエージェントの位置を返す
    def get_another_agent_position(self, agent_idx):
        if agent_idx == 0:
            another_agent_pos = self.get_position(1)
        if agent_idx == 1:
            another_agent_pos = self.get_position(0)
        return another_agent_pos

    #自分と自分以外のエージェントの位置を特定して返す
    #現状エージェント二人まで
    def new_get_positions(self, agent_idx):
        if agent_idx == 0:
            given_agent_pos = self.get_position(0)
            another_agent_pos = self.get_position(1)
        if agent_idx == 1:
            given_agent_pos = self.get_position(1)
            another_agent_pos = self.get_position(0)
        return given_agent_pos, another_agent_pos

    def get_nactions(self, agent_idx):
        return self.allies[agent_idx].nactions()

    def death_agent(self, agent_idx):
        # 即時リスポーンならalliesに、全死後リスポーンならwait_respawnにリスポーン
        if self.sequential_respawn and self.agent_respawn:
            self.allies[agent_idx].respawn()
        else:
            if self.agent_respawn:
                self.wait_respawn.append(deepcopy(self.allies[agent_idx]))
                self.wait_respawn[-1].respawn()
            self.remove_agent(agent_idx)
        # 全死ならwait_respawnをalliesにしてwait_respawnを初期化
        if self.agent_respawn and (self.n_agents == 0):
            self.allies = deepcopy(self.wait_respawn)
            self.wait_respawn = []

    def remove_agent(self, agent_idx):
        # idx is between zero and nagents
        self.allies.pop(agent_idx)
        self.nagents -= 1

    def get_state_matrix(self):
        """
        Returns a matrix representing the positions of all allies
        Example: matrix contains the number of allies at give (x,y) position
        0 0 0 1 0 0 0
        0 2 0 2 0 0 0
        0 0 0 0 0 0 1
        1 0 0 0 0 0 5
        """
        gs = self.global_state
        gs.fill(0)
        for ally in self.allies:
            x, y = ally.current_position()
            gs[x, y] += 1
        return gs

    def log_another_agent_attack_and_signal(self, agent_idx):
        if self.check_another_agent_attack(agent_idx):
            self.allies[agent_idx].another_agent_attack_and_signal = True
        else:
            self.allies[agent_idx].another_agent_attack_and_signal = False

    def get_state(self):
        pos = np.zeros(2 * len(self.allies))
        idx = 0
        for ally in self.allies:
            pos[idx : (idx + 2)] = ally.get_state()
            idx += 2
        return pos


    def check_hit(self, agent_id, deer_layer):
        if not self.allies[agent_id].attack_and_signal: # そもそも攻撃していなければFalse
            self.allies[agent_id].attack_hit = False
            return
        deer_pos_list = [deer.current_position() for deer in deer_layer.allies]
        agent_pos = self.get_position(agent_id)
        for deer_pos in deer_pos_list:  # 注: 以下はdeerが1体であることが前提の処理
            # agentの攻撃範囲にdeerがいればattack_hitをTrueに
            # 注: agentの攻撃範囲は3*3のマジックナンバーで実装
            if (agent_pos[0]-1 <= deer_pos[0] <= agent_pos[0]+1) and (agent_pos[1]-1 <= deer_pos[1] <= agent_pos[1]+1):
                self.allies[agent_id].attack_hit = True
            else:
                self.allies[agent_id].attack_hit = False

    #自分以外のエージェントが攻撃を行ったかどうかを返す
    def check_another_agent_attack(self, agent_idx) :
        if agent_idx == 0:
            return self.allies[1].attack_and_signal
        if agent_idx == 1:
            return self.allies[0].attack_and_signal

    def damage(self, agent_id):
        if not self.invincible:
            self.allies[agent_id].health -= 1
            self.allies[agent_id].last_attacked = True