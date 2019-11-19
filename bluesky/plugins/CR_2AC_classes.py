"""
-- NOT A VALID PLUGIN --
Class definitions for CR_2AC plugin
"""

import pdb
import numpy as np
from random import sample as random_sample
from bluesky.tools.geo import qdrdist

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import smooth_l1_loss, relu

import matplotlib.pyplot as plt
# import torchvision.utils
# from mayavi import mlab
# from mpl_toolkits import mplot3d

# ----- Reinforcement Learning Specifications ----- #
DISCOUNT = 0.9  # Discount factor / Gamma
N_STEPS = 3  # Number of steps for n-step TD implementation
SEPARATION_COST = 100.0  # Penalty assigned based on separation between aircrafts
TERMINAL_REWARD = 100.0  # Incentive to correct the orientation after conflict-resolution
# Incentive to correct orientation must be sufficiently lower than that to maintain separation

LEARNING_RATE = 0.005
TRAINING_RATIO = 0.15  # Fraction of the buffer memory to learn from
BUFFER_SIZE = 500

# Training based on a decaying epsilon greedy policy
EPSILON_START = 0.3
EPSILON_MIN = 0.0
EPSILON_DECAY = 0.99993

writer = SummaryWriter("saved_data/last_simulation")
# Use tensorboard --logdir=runs from terminal, then navigate to https://localhost:6006/


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
        writer.add_graph(self, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))

    def forward(self, x):
        x = self.fcOut(relu(self.fcInp(x)))
        return x

    def learn(self, buffer, training_set):
        """
        Learn Q-function using monte-carlo learning
        :param buffer: Buffer object
        :param training_set: <list> Indices of buffer.memory to train
        """
        # Update Q for each state-action pair in training_set

        for t in training_set:
            state = buffer.memory[t][0]
            action = buffer.memory[t][1]
            q_value = self.forward(state)[action]  # Our DQN's estimate of Q(s,a)
            target = buffer.memory[t][2]  # Target Q(s,a)
            loss = smooth_l1_loss(q_value, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if np.random.uniform(0, 1) < 0.02:
                print(f"UPDATED : {q_value} towards {target}. NEW VALUE : {self.forward(state)[action]}...")
            writer.add_scalar(f"Loss/Aircraft_{buffer.ID}", loss)

        # print(f"{buffer.ID} completed {buffer.updates} updates.")
        # print(f"EPISODE : {buffer.memory[-1 - buffer.episode_length:]}.")

    def plot(self):
        """
        Plots the optimal action (for various positions of the other aircraft)
        """
        fig = plt.figure(figsize=(6, 5))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        x_list = np.linspace(-0.4, 0.4)
        y_list = np.linspace(-0.4, 0.4)
        X, Y = np.meshgrid(x_list, y_list)
        Z = []
        for x in x_list:
            Z.append([])
            for y in y_list:
                qdr_ac, dist_ac = qdrdist(0.0, 0.0, y + 0.0001, x)
                state = torch.tensor([qdr_ac, dist_ac, qdr_ac, 0.0, 0.5])
                Z[-1].append(self.forward(state).max(0)[1])
        cp = plt.contourf(X, Y, Z, 2, colors=['g', 'b', 'r'])
        plt.colorbar(cp, ticks=[0, 1, 2])
        ax.set_title('Policy for an aircraft at 0, 0')
        ax.set_xlabel('Actions corresponding to indices of ["LEFT", "NO_ACTION", "RIGHT"]')
        plt.show()


class Buffer:

    def __init__(self, _ID: str, actions_enum):
        self.ID = _ID
        self.SIZE = BUFFER_SIZE

        self.memory = []  # [ [S, A, R+] , [S+, A+, R++] ... [_, _, FINAL_REWARD] ]
        self.separation = 100.0  # Dummy starting value for separation between the aircrafts
        self.actions_enum = actions_enum
        self.episode_length = 0
        self.updates = 0

    def teach(self, n_net: ATC_Net):
        """
        Trains the neural net on a fraction of the episode
        """
        self.updates += 1
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

    def check(self, n_net):
        if len(self.memory) > self.SIZE:
            self.teach(n_net)
            self.empty()

    def empty(self):
        self.memory = []

    def set_separation(self, distance):
        self.separation = np.min([self.separation, distance])

    def add_state_action(self, state, action):
        """
        Add {state, action} pair to buffer memory
        """
        self.episode_length += 1
        self.memory.append([state, action, 0.0])

    def update_targets(self, reward, atc_net):
        """
        Use current reward to update returns for previous transitions
        """
        add_rewards_to = np.min([N_STEPS + 1, self.episode_length])

        for i in range(add_rewards_to):
            # Earlier transitions get discounted rewards
            self.memory[-1 - i][2] += (reward * DISCOUNT**i)

        if self.episode_length > N_STEPS + 1:
            Q_max = atc_net(self.memory[-1][0]).max()
            self.memory[-1 - N_STEPS - 1][2] += Q_max * (DISCOUNT**(N_STEPS+1))

    def get_reward_for_distance(self, critical_distance):
        """
        :return: <float> Negative reward based on separation between aircrafts
        """
        reward = 0.0
        if self.separation < critical_distance:
            C1 = SEPARATION_COST * 0.5
            C2 = 1.0
            correction_term = C1 / (self.separation - (critical_distance + C2)) ** 2
            reward = -1 * SEPARATION_COST + correction_term

        self.separation = 100.0
        return reward

    def get_terminal_reward(self):
        """
        :return: <float> Most positive when heading towards destination
        """
        C1 = 0.3
        C2 = 0.5
        destination_dist = self.memory[-1][0][4]
        return 2*TERMINAL_REWARD*(np.exp(-C1*destination_dist) - C2)

    def terminate_episode(self):
        self.episode_length = 0

