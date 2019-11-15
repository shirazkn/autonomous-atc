"""
Plugin for training a DQN for conflict-resolution, using monte-carlo learning
Author : Shiraz Khan
"""

from bluesky import traf, stack
from bluesky.tools.geo import qdrdist
from plugins.Sim_2AC import reset_aircrafts, RADIUS_NM, AIRCRAFTS

import torch
from torch.nn.functional import smooth_l1_loss, relu
import torch.optim as optim

from random import sample as random_sample
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import pdb
from mpl_toolkits import mplot3d

# ----- Problem Specifications ----- #
dHeading = 5  # Max heading allowed in one time-step
actions_enum = [-dHeading, 0,  dHeading]  # Once of these actions given as a heading-change input
RESOLVE_PERIOD = 300  # Every ___ steps of simulation, take conflict-resolution action
MIN_SEPARATION_ALLOWED = 4.0  # Negative reward is incurred if aircrafts come closer than this

# ----- Reinforcement Learning Specifications ----- #
ACTION_COST = 20.0  # <-1*ACTION_COST> reward for changing heading
SEPARATION_COST = 1000.0  # Penalty assigned the terminal state, based on minimum separation during episode
# ORIENTATION_COST = 20.0  # Incentive to correct the orientation after conflict-resolution

LEARNING_RATE = 0.005
TRAINING_RATIO = 0.1  # Fraction of the episode to learn from

EXPLORATION_START = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.9999

# ----- Debugging and Simulation Parameters ----- #
VIEW_SIMULATIONS = False  # When True, simulations will be updated even if there's not conflict
DEBUGGING = False

SAVED_STATE = None
# SAVED_STATE = "saved_data/atc-policy-1.pt"
EXPLORATION = EXPLORATION_START
COUNTER = 0
EPISODE_COUNT = 0


# PyTorch Neural Net
class ATC_Net(torch.nn.Module):

    def __init__(self):
        super(ATC_Net, self).__init__()

        # Network input & output layers
        self.fcInp = torch.nn.Linear(3, 20)
        self.fcOut = torch.nn.Linear(20, len(actions_enum))

    def forward(self, x):
        x = self.fcOut(relu(self.fcInp(x)))
        return x

    def learn(self, buffer, training_set):
        """
        Learn Q-function using monte-carlo learning
        :param buffer: Buffer object
        :param training_set: <list> Indices of buffer.memory to train
        """
        global DEBUGGING
        n_transitions = len(buffer.memory)

        # Compute total return for each state-action
        returns = []
        total_return = 0.0
        for i in range(n_transitions):
            total_return += buffer.memory[-1-i][2]
            returns.append(total_return)
        returns = returns[::-1]

        if DEBUGGING:
            DEBUGGING = False
            pdb.set_trace()

        # Update Q for each state-action pair in training_set
        for t in training_set:
            state = buffer.memory[t][0]
            action = buffer.memory[t][1]
            q_value = self.forward(state)[action]  # Our DQN's estimate of Q(s,a)
            mc_target = torch.tensor(returns[t])  # Target Q(s,a)
            loss = smooth_l1_loss(q_value, mc_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        global EXPLORATION, EPISODE_COUNT
        EPISODE_COUNT += 0.5
        EXPLORATION = max(EXPLORATION_DECAY*EXPLORATION, EXPLORATION_MIN)
        print(f"{EPISODE_COUNT} : {buffer.ID} received {returns[0]} return.")


class Buffer:

    def __init__(self, _ID: str):
        self.ID = _ID
        self.TEACHING = True
        self.REWARD_PENDING = False
        self.MIN_SEPARATION = 50.0
        self.memory = []  # [ [S, A, R+] , [S+, A+, R++] ... [_, _, FINAL_REWARD] ]

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

    def pause(self, option: bool):
        """
        Allows pausing and resuming of Q-Learning
        """
        self.empty()
        self.TEACHING = (not option)

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
        if self.MIN_SEPARATION < MIN_SEPARATION_ALLOWED:
            return -1*SEPARATION_COST
        else:
            return 0.0

    def get_reward_for_action(self):
        """
        :return: <float> Negative reward if heading-change (else 0)
        """
        last_action = actions_enum[self.memory[-1][1]]
        return -1 * ACTION_COST if last_action else 0.0


# Initialization of static variables
atc_net = ATC_Net()
if SAVED_STATE:
    atc_net.load_state_dict(torch.load(SAVED_STATE))

buffers = []
for ID in AIRCRAFTS:
    buffers.append(Buffer(ID))

optimizer = optim.Adam(atc_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


def init_plugin():

    # Configuration parameters
    config = {
        'plugin_name':     'CR_2AC',
        'plugin_type':     'sim',
        'update':          update,
        'preupdate':       preupdate
        }
    stackfunctions = {
        'VIEW_SIM': [
            'VIEW_SIM [0 or 1]',
            'txt',
            set_view,
            'Set to 0 for training, 1 for viewing simulations.'],
        'CHECK_NET': [
            'CHECK_NET',
            None,
            check_net,
            'Use this to PDB into code and check values.'],
        'SHOW_NET': [
            'SHOW_NET',
            None,
            show_net,
            'Displays the Q(s,a) values for actions LEFT and RIGHT.']
    }
    return config, stackfunctions


def preupdate():
    global COUNTER
    COUNTER += 1

    # Rate of conflict-resolution is lesser than rate of simulation
    if not COUNTER % RESOLVE_PERIOD:
        COUNTER = 0

        # Check if aircrafts have been created yet
        if traf.ntraf > 1:
            resolve()

        # Else, wait for BlueSky to process the aircraft creation
        else:
            return

        # Check if aircrafts are still in simulation area
        for ac in range(2):
            (lat, lon) = (traf.lat[ac], traf.lon[ac])
            _, distance_from_center = _qdrdist(lat, lon, 0.0, 0.0)
            if distance_from_center > RADIUS_NM + 0.1:
                for buffer in buffers:
                    buffer.assign_reward(buffer.get_reward_for_action() + buffer.get_reward_for_distance())
                    buffer.teach(atc_net)

                reset_all()
                return
                # TODO : Use _qdrdist(0.0, 0.0, LAT[0], LON[0])[0] in input vector to correct the trajectory


def reset_all():
    """
    Resets values after one batch of experience has been trained
    """
    for buffer in buffers:
        buffer.empty()

    reset_aircrafts()


def update():
    # Find the minimum separation maintained by the aircrafts in this episode
    if traf.ntraf > 1:
        for buffer in buffers:
            _, dist = _qdrdist(traf.lat[0], traf.lon[0], traf.lat[1], traf.lon[1])
            buffer.MIN_SEPARATION = np.min([buffer.MIN_SEPARATION, dist])


def resolve():
    # Assign reward for previous state-action
    for buffer in buffers:
        buffer.assign_reward(buffer.get_reward_for_action()) if buffer.REWARD_PENDING else None

        # Choose action for current time-step
        state = get_state(buffer.ID)
        q_values = atc_net.forward(state)
        action = get_action(q_values)

        # Store S, A in buffer (R will be observed later)
        buffer.add_state_action(state, action)

        # Execute action
        new_heading = traf.hdg[traf.id2idx(buffer.ID)] + float(actions_enum[action])
        stack.stack(f"HDG {buffer.ID} {new_heading}")


def make_angle_convex(angle):
    """
    :param angle: <float> angle in deg
    :return: <float> angle between -180 to 180
    """
    angle %= 360
    if angle > 180:
        return angle - 360
    return angle


def get_state(ac_id: str):
    """
    Get current state of simulation
    :param ac_id: Aircraft to be controlled
    :return: torch.Tensor (Input for the neural network, containing enough information to render the problem Markov)
    """
    self = traf.id2idx(ac_id)
    enemy = (self + 1) % 2
    self_hdg = traf.hdg[self]
    enemy_hdg = traf.hdg[enemy]
    rel_hdg = make_angle_convex(enemy_hdg - self_hdg)

    qdr, dist = _qdrdist(traf.lat[self], traf.lon[self], traf.lat[enemy], traf.lon[enemy])
    qdr = make_angle_convex(qdr)
    return torch.tensor([make_angle_convex(qdr + self_hdg), dist, rel_hdg])


def get_action(q_values):
    """
    :param q_values: <torch.Tensor> Output of the neural net
    :return: With probability EXPLORATION returns a random int, else gives the index of max(q_values)
    """
    sample = np.random.uniform(0, 1)
    if sample < EXPLORATION:
        return np.random.randint(len(actions_enum))

    else:
        return q_values.max(0)[1]


def _qdrdist(lat1, lon1, lat2, lon2):
    """
    Prevents BlueSky's qdrdist function from returning singular value
    """
    return qdrdist(lat1, lon1, lat2 + 0.000001, lon2)


def show_net():
    """
    Visualizes the current Q(s,a_1) and Q(s,a_2) values of neural net
    Use BlueSky command SHOW_NET
    """
    mlab.clf()
    x = np.linspace(-180, 180, 100)
    y = np.linspace(-180, 180, 100)
    X, Y = np.meshgrid(x, y)

    Z_Left = []
    Z_Right = []
    Z_Contour = []

    for _x in x:
        Z_Left.append([])
        Z_Right.append([])
        Z_Contour.append([])
        for _y in y:
            nn_input = torch.tensor([_x, 7.0, _y])
            nn_output = atc_net.forward(nn_input)

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


def check_net():
    """
    Use BlueSky command CHECK_NET to open pdb here
    """
    global DEBUGGING
    DEBUGGING = True
    pdb.set_trace()


def set_view(input_text):
    """
    Allows you to use the command VIEW_SIM 1 from BlueSky to simulate full episodes.
    """
    global VIEW_SIMULATIONS, EXPLORATION
    if int(input_text):
        VIEW_SIMULATIONS = True
        EXPLORATION = 0.0
    else:
        VIEW_SIMULATIONS = False
        EXPLORATION = EXPLORATION_START  # Note that exploration gets reset here
        [buffer.pause(False) for buffer in buffers]
