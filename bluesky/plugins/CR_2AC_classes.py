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
ACTION_COST = 10.0  # <-1*ACTION_COST> reward for changing heading
SEPARATION_COST = 1000.0  # Penalty assigned the terminal state, based on minimum separation during episode
DESTINATION_REWARD = 2000.0  # Incentive to correct the orientation after conflict-resolution

LEARNING_RATE = 0.005
TRAINING_RATIO = 0.2  # Fraction of the episode to learn from

# Training based on a decaying epsilon greedy policy
EPSILON_START = 0.0
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
        self.fcInp = torch.nn.Linear(3, 20)
        self.fcOut = torch.nn.Linear(20, len(actions_enum))

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
        mlab.clf()
        x = np.linspace(-180, 180, 150)
        y = np.linspace(-180, 180, 150)
        X, Y = np.meshgrid(x, y)

        Z_Left = []
        Z_Right = []
        Z_Contour = []

        for _x in x:
            Z_Left.append([])
            Z_Right.append([])
            Z_Contour.append([])
            for _y in y:
                nn_input = torch.tensor([_x, 10.0, _y])
                nn_output = self.forward(nn_input)

                Z_Left[-1].append(float(nn_output[0]))
                Z_Right[-1].append(float(nn_output[2]))
                Z_Contour[-1].append(nn_output.max(0)[1])

        Z_Left = np.array(Z_Left)
        Z_Right = np.array(Z_Right)
        Z_Contour = np.array(Z_Contour)

        mlab.surf(x, y, Z_Left, colormap="Greens")
        mlab.surf(x, y, Z_Right, colormap="Reds")
        mlab.xlabel('qdr + self_hdg')
        mlab.ylabel('rel_hdg')
        mlab.title("Values for LEFT (g) and RIGHT (r)")
        mlab.show()

        # plt.contourf(X, Y, Z_Contour, levels=5, colors=('g', 'b', 'r'))
        # plt.show()

    # TODO
    # def show_policy():
    #     # My hdg = 0.0
    #     # my lat, lon = 0.0, 0.0
    #
    #     # Make mesh of points, x, y
    #     # t_angle =
    #
    #     return


class Buffer:

    def __init__(self, _ID: str, actions_enum, min_separation_allowed):
        self.ID = _ID
        self.REWARD_PENDING = False

        self.MIN_SEPARATION_ALLOWED = min_separation_allowed
        self.MIN_SEPARATION = 50.0  # Dummy starting value that's larger than simulation area

        self.memory = []  # [ [S, A, R+] , [S+, A+, R++] ... [_, _, FINAL_REWARD] ]
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
        self.MIN_SEPARATION = 50.0

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

    def get_reward_for_distance(self):
        """
        :return: <float> Negative reward if minimum separation condition was violated during episode
        """
        if self.MIN_SEPARATION < self.MIN_SEPARATION_ALLOWED:
            return -1*SEPARATION_COST
        else:
            return 0.0

    def get_reward_for_action(self):
        """
        :return: <float> Negative reward if heading-change (else 0)
        """
        last_action = self.actions_enum[self.memory[-1][1]]
        return -1 * ACTION_COST if last_action else 0.0

