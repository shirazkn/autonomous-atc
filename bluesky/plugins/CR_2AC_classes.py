"""
-- NOT A VALID PLUGIN --
Class definitions for CR_2AC plugin
"""

import pdb
import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss, relu
from random import sample as random_sample
from mayavi import mlab
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# ----- Reinforcement Learning Specifications ----- #
SEPARATION_COST = 100.0  # Penalty assigned based on separation between aircrafts
DEVIATION_COST = 10.0  # Incentive to correct the orientation after conflict-resolution
# Incentive to correct orientation must be sufficiently lower than that to maintain separation

LEARNING_RATE = 0.005
TRAINING_RATIO = 0.2  # Fraction of the episode to learn from

# Training based on a decaying epsilon greedy policy
EPSILON_START = 0.3
EPSILON_MIN = 0.0
EPSILON_DECAY = 0.99999


class Exploration:
    def __init__(self):
        self.eps = EPSILON_START

    def decay(self):
        self.eps = max(EPSILON_DECAY * self.eps, EPSILON_MIN)


# PyTorch Neural Net
class ATC_Net(torch.nn.Module):

    def __init__(self, actions_enum):
        """
        :param actions_enum: Set of actions that correspond to outputs of neural net
        """
        super(ATC_Net, self).__init__()

        # Network input & output layers
        self.fcInp = torch.nn.Linear(5, 25)
        self.fcOut = torch.nn.Linear(25, len(actions_enum))

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0, amsgrad=False)

    def forward(self, x):
        x = self.fcOut(relu(self.fcInp(x)))
        return x

    def learn(self, buffer, training_set):
        """
        Learn Q-function using monte-carlo learning
        :param buffer: Buffer object
        :param training_set: <list> Indices of buffer.memory to train
        """
        n_transitions = len(buffer.memory)

        # Compute total return for each state-action
        returns = []
        total_return = 0.0
        for i in range(n_transitions):
            total_return += buffer.memory[-1-i][2]
            returns.append(total_return)
        returns = returns[::-1]

        # Update Q for each state-action pair in training_set
        for t in training_set:
            state = buffer.memory[t][0]
            action = buffer.memory[t][1]
            q_value = self.forward(state)[action]  # Our DQN's estimate of Q(s,a)
            mc_target = torch.tensor(returns[t])  # Target Q(s,a)
            loss = smooth_l1_loss(q_value, mc_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f"{buffer.ID} received {returns[0]} return.")

    def plot(self):
        # TODO
        # def show_policy():
        #     # My hdg = 0.0
        #     # my lat, lon = 0.0, 0.0
        #
        #     # Make mesh of points, x, y
        #     # t_angle =
        #
        #     return
        raise NotImplementedError


class Buffer:

    def __init__(self, _ID: str, actions_enum):
        self.ID = _ID
        self.REWARD_PENDING = False

        self.memory = []  # [ [S, A, R+] , [S+, A+, R++] ... [_, _, FINAL_REWARD] ]
        self.separation = 100.0  # Dummy starting value for separation between the aircrafts
        self.actions_enum = actions_enum

    def teach(self, n_net: ATC_Net):
        """
        Trains the neural net on a fraction of the episode
        """
        buffer_size = len(self.memory)
        n_samples = max(1, int(TRAINING_RATIO*buffer_size))
        training_set = self.get_training_set(n_samples)
        n_net.learn(self, training_set)

    def get_training_set(self, n_samples):
        """
        Returns a list of <n_samples> indices in buffer.memory, randomly ordered
        :return: list(<int>)
        """
        training_set = random_sample(range(len(self.memory)-1), n_samples)
        return training_set

    def empty(self):
        self.memory = []

    def set_separation(self, distance):
        self.separation = np.min([self.separation, distance])

    def add_state_action(self, state, action):
        """
        Add {state, action} pair to buffer memory
        """
        self.memory.append([state, action])
        self.REWARD_PENDING = True

    def assign_reward(self, reward):
        """
        Add reward for the last state-action in memory
        """
        self.memory[-1] = self.memory[-1][:2] + [reward]
        self.REWARD_PENDING = False
        self.separation = 100.0

    def get_reward_for_distance(self, critical_distance):
        """
        :return: <float> Negative reward based on separation between aircrafts
        """
        if self.separation < critical_distance:
            C1 = SEPARATION_COST * 0.5
            C2 = 1.0
            correction_term = C1 / (self.separation - (critical_distance + C2)) ** 2
            return -1 * SEPARATION_COST + correction_term

        else:
            return 0.0

    def get_reward_for_deviation(self, current_heading):
        """
        :return: <float> Most positive when heading towards destination
        """
        destination_qdr = self.memory[-1][0][3]
        deviation_factor = np.cos(np.deg2rad(destination_qdr - current_heading)) - 1.0

        return 0.5*DEVIATION_COST*deviation_factor

