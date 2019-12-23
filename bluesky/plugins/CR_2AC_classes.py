"""
-- NOT A VALID PLUGIN --
Class definitions for CR_2AC plugin
"""

import pdb
import numpy as np
from random import sample as random_sample
from random import shuffle as random_shuffle
from bluesky.tools.geo import qdrdist
from plugins.Sim_2AC import RADIUS_NM, RADIUS

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import smooth_l1_loss, leaky_relu

import matplotlib.pyplot as plt
# import torchvision.utils
# from mayavi import mlab
# from mpl_toolkits import mplot3d

# ----- Reinforcement Learning Specifications ----- #
DISCOUNT = 0.9  # Discount factor / Gamma
N_STEPS = 2  # Number of steps for n-step TD implementation
SEPARATION_COST = 10.0  # Penalty assigned based on separation between aircrafts
TERMINAL_REWARD = 2.0  # Incentive to correct the orientation after conflict-resolution
# Incentive to correct orientation must be sufficiently lower than that to maintain separation

LEARNING_RATE = 0.05
HIDDEN_NEURONS = 20
INITIALIZATION_POINTS = 500
TRAINING_RATIO = 0.3  # Fraction of the buffer memory to learn from
BUFFER_SIZE = 90

# Training based on a decaying epsilon greedy policy
EPSILON_START = 0.5
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

torch.manual_seed(0)
writer = SummaryWriter("saved_data/last_simulation")
# Use tensorboard --logdir=runs from terminal, then navigate to https://localhost:6006/


class Exploration:
    def __init__(self):
        # Probability of taking a random action instead of greedy one
        self.eps = EPSILON_START

    def decay(self):
        self.eps = max(EPSILON_DECAY * self.eps, EPSILON_MIN)


class ATC_Net(torch.nn.Module):
    """
    Neural network which acts as the air traffic controller
    """

    def __init__(self, actions_enum):
        """
        :param actions_enum: Set of actions that correspond to outputs of neural net
        """
        super(ATC_Net, self).__init__()

        # Network input & output layers
        self.Inp = torch.nn.Linear(5, HIDDEN_NEURONS)
        self.Out = torch.nn.Linear(HIDDEN_NEURONS, len(actions_enum))

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0, amsgrad=False)

        self.exploration = Exploration()
        self.lr_decay = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=EPSILON_DECAY)
        writer.add_graph(self, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))

    def forward(self, x):
        x = self.Out(leaky_relu(self.Inp(x)))
        return x

    def learn(self, buffer, training_set):
        """
        Learn Q-function using monte-carlo learning
        :param buffer: Buffer object
        :param training_set: <list> Indices of buffer.memory to train
        """
        # Update Q for each state-action pair in training_set
        self.optimizer.zero_grad()

        # Get Q-values and targets
        for t in training_set:
            state = buffer.memory[t][0]
            action = buffer.memory[t][1]
            q_value = self.forward(state)[action]  # Our DQN's estimate of Q(s,a)
            target = torch.tensor(buffer.memory[t][2]).detach_()  # Target Q(s,a)

            # Update Network parameters
            loss = smooth_l1_loss(q_value, target)
            loss.backward()

        for p in self.parameters():
            print(p.grad)

        self.optimizer.step()
        self.exploration.decay()
        self.lr_decay.step()

        print(f"Learning rate is {self.lr_decay.get_lr()} and exploration is {self.exploration.eps}.")
        writer.add_scalar(f"Loss/Aircraft_{buffer.ID}", loss)

    def initialize(self, action_values: dict, critical_distance=RADIUS_NM):
        """
        :param action_values: [Q(s,a1) , Q(s,a2) ... ]
        :param critical_distance: Minimum separation to maintain between aircrafts
        """
        n_grid = int(INITIALIZATION_POINTS**0.5)

        longitudes = np.linspace(-RADIUS, RADIUS, n_grid)
        latitudes = np.linspace(-RADIUS, RADIUS, n_grid)
        headings = np.linspace(-180, 180, 4)

        inputs = []
        outputs = []
        for heading in headings:
            for lon in longitudes:
                for lat in latitudes:
                    # Get input tuple
                    qdr_ac, dist_ac = _qdrdist(0.0, 0.0, lat, lon)
                    qdr_ac = make_angle_convex(qdr_ac)
                    rel_hdg = make_angle_convex(qdr_ac + heading)
                    state = [qdr_ac, dist_ac, rel_hdg, 0.0, 24.0]

                    # Get output tuple
                    if dist_ac < critical_distance:
                        if lon > 0.0:
                            output = [1.0, 0.5, 0.5]
                        else:
                            output = [0.5, 0.5, 1.0]

                    else:
                        output = [0.0, 0.5, 0.0]

                    inputs.append(torch.tensor(normalize(state)))
                    outputs.append(torch.tensor(output, requires_grad=False))

        training_set = list(zip(inputs, outputs))
        random_shuffle(training_set)

        losses = []
        for i in range(len(training_set)):
            self.optimizer.zero_grad()
            output = self.forward(training_set[i][0])
            loss = smooth_l1_loss(output, training_set[i][1])
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss))

        plt.plot(losses)
        plt.title("Losses during initialization")
        plt.ylabel("Loss")
        plt.xlabel("Data Points")
        self.plot()

    def plot(self):
        """
        Plots the optimal action (for various positions of the other aircraft)
        """
        fig = plt.figure(figsize=(6, 6))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        longitudes = np.linspace(-RADIUS, RADIUS, 80)
        latitudes = np.linspace(-RADIUS, RADIUS, 80)
        grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)

        policy = []
        for lat in latitudes:
            policy.append([])
            for lon in longitudes:
                qdr_ac, dist_ac = _qdrdist(0.0, 0.0, lat, lon)
                qdr_ac = make_angle_convex(qdr_ac)
                rel_hdg = make_angle_convex(qdr_ac + 180.0)
                state = [qdr_ac, dist_ac, rel_hdg, 0.0, 24.0]
                state = torch.tensor(normalize(state))
                policy[-1].append(self.forward(state).max(0)[1])

        cp = plt.contourf(grid_lon, grid_lat, policy, 2, colors=['g', 'b', 'r'])
        plt.colorbar(cp, ticks=[0, 1, 2])
        ax.set_title('Policy for an aircraft at (0, 0) heading North')
        ax.set_xlabel('Actions corresponding to indices of ["LEFT", "NO_ACTION", "RIGHT"]')
        plt.show()


class Buffer:
    """
    Stores training data and decides when and how to teach the neural net
    """

    def __init__(self, _ID: str, actions_enum):
        self.ID = _ID
        self.SIZE = BUFFER_SIZE

        self.memory = []  # [ [S, A, Return] , [S+, A+, Return+] ... [_, _, FINAL_REWARD] ]
        self.separation = 2*RADIUS_NM  # Dummy starting value for separation between the aircrafts
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
        """
        :param distance: in Nm
        """
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
        :param critical_distance: desired separation between aircrafts (in Nm)
        :return: <float> Negative reward based on separation between aircrafts
        """
        reward = 0.0
        if self.separation < critical_distance:
            C1 = SEPARATION_COST * 0.5
            C2 = 1.0
            correction_term = C1 / (self.separation - (critical_distance + C2)) ** 2
            reward = -1 * SEPARATION_COST + correction_term

        self.separation = 2*RADIUS_NM
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


# ----- Some helper functions ----- #

def make_angle_convex(angle):
    """
    :param angle: <float> angle in deg
    :return: <float> angle between -180 to 180
    """
    angle %= 360
    if angle > 180:
        return angle - 360
    return angle


def _qdrdist(lat1, lon1, lat2, lon2):
    """
    Prevents BlueSky's qdrdist function from returning singular value
    :return qdr (in deg), dist (in Nm)
    """
    return qdrdist(lat1, lon1, lat2 + 0.00001, lon2)


def normalize(state):
    """
    :param state: [angle, distance, angle, angle, distance]
    :return:
    """
    return [
        state[0]/180.0,
        state[1]/RADIUS_NM,
        state[2]/180.0,
        state[3]/180.0,
        state[4]/RADIUS_NM
    ]
