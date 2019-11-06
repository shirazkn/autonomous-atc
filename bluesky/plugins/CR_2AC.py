# Plugin for implementing ML-based conflict-resolution method
# Simulates continuing 2-aircraft episodes to learn CR policy
# CR is done using a net with Experience Replay

from numpy import random
from bluesky import traf, stack
from bluesky.traffic.asas import PluginBasedCR
from bluesky.tools.geo import qdrdist
from plugins.Sim_2AC import reset_aircrafts

from torch import tensor, nn
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss, relu
from random import sample as random_sample

dHeading = 10  # Max heading allowed in one time-step
actions_enum = [-dHeading, 0,  dHeading]  # Once of these actions given as a heading-change input
VIEW_SIMULATIONS = False  # When True, simulations will be updated even if there's not conflict

GAMMA = 1.0  # Discount factor
FINAL_REWARD = 0.0  # <FINAL_REWARD> reward assigned when conflict is resolved (terminal state)
SEPARATION_COST_FACTOR = 10  # <-R_M/(distance_of_separation)^2> reward every time-step
ACTION_COST = 10.0  # <-1*ACTION_COST> reward for changing heading

LEARNING_RATE = 0.005
TRAINING_RATIO = 0.3  # Fraction of the episode to learn from

EXPLORATION_START = 0.9
EXPLORATION_MIN = 0.2
EXPLORATION_DECAY = 0.9999
EXPLORATION = EXPLORATION_START

DEBUGGING = False
COUNTER = 0


# PyTorch Neural Net
class ATC_Net(nn.Module):

    def __init__(self):
        super(ATC_Net, self).__init__()

        # Network input & output layers
        self.fcInp = nn.Linear(3, 5)
        self.fcH1 = nn.Linear(5, 10)
        self.fcH2 = nn.Linear(10, 10)
        self.fcH3 = nn.Linear(10, 5)
        self.fcOut = nn.Linear(5, len(actions_enum))

    def forward(self, x):
        x = self.fcOut(self.fcH3(relu(self.fcH2(relu(self.fcH1(self.fcInp(x)))))))
        return x

    def learn(self, training_set):
        """
        Learn q-function using Monte-Carlo method
        :param transitions: <list> Indices of buffer.memory to train
        """
        # Learn from experience
        n_transitions = len(buffer.memory)

        returns = []
        total_return = 0.0

        for i in range(n_transitions):
            total_return += buffer.memory[-1-i][2]
            returns.append(total_return)

        returns = returns[::-1]

        for t in training_set:
            state = buffer.memory[t][0]
            action = buffer.memory[t][1]
            mc_target = tensor(returns[t])
            q_value = self.forward(state)[action]  # Value of chosen ('optimal' or explored) action
            loss = smooth_l1_loss(q_value, mc_target)
            loss.backward()
            optimizer.step()

        global EXPLORATION, COUNTER
        EXPLORATION = max(EXPLORATION_DECAY*EXPLORATION, EXPLORATION_MIN)
        if not COUNTER % 1000:
            print(f"Exploration is {EXPLORATION}, return was {returns[0]}.")


class Buffer:
    TEACHING = True
    REWARD_PENDING = False
    memory = []  # [ [S, A, R+] , [S+, A+, R++] ... [_, _, FINAL_REWARD] ]

    def teach(self, n_net: ATC_Net):
        buffer_size = len(buffer.memory)
        n_samples = max(1, int(TRAINING_RATIO*buffer_size))
        training_set = self.get_training_set(n_samples)
        n_net.learn(training_set)

    def get_training_set(self, n_samples):
        training_set = random_sample(range(len(self.memory)-1), n_samples)
        return training_set

    def empty(self):
        self.REWARD_PENDING = False
        self.memory = []

    def pause(self, option):
        self.empty()
        self.TEACHING = (not option)

    def add_state_action(self, state, action):
        if self.REWARD_PENDING:
            raise RuntimeError
        self.memory.append([state, action])
        self.REWARD_PENDING = True

    def assign_reward(self, reward):
        self.memory[-1] = self.memory[-1][:2] + [reward]
        self.REWARD_PENDING = False


def weights_init(m):
    classname = m.__class__.__name__
    # For every Linear layer in a model,
    if classname.find('Linear') != -1:
        # Use a uniform distribution for weights
        m.weight.data.uniform_(-0.001, 0.0)
        m.bias.data.fill_(0)


# Global variables for Neural Net
atc_net = ATC_Net()
atc_net.apply(weights_init)
buffer = Buffer()
optimizer = optim.SGD(atc_net.parameters(), lr=LEARNING_RATE, momentum=0.5)


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
            'Use this to PDB into code and check values.']
    }
    return config, stackfunctions


def preupdate():
    global COUNTER
    COUNTER += 1
    if not COUNTER % 1000:
        if traf.asas.confpairs:
            try:
                resolve()
            except:
                buffer.empty()

        elif traf.ntraf == 2 and buffer.REWARD_PENDING:
            buffer.assign_reward(FINAL_REWARD)
            buffer.teach(atc_net)
            buffer.empty()
            reset_aircrafts()

        else:
            buffer.empty()
            reset_aircrafts()


def update():
    pass


def resolve():
    """
    Called in place of built-in `resolve` method
    """
    if buffer.REWARD_PENDING:
        buffer.assign_reward(get_reward_for_distance() + get_reward_for_action())

    # Choose action
    state = get_state()
    q_values = atc_net.forward(state)
    action = get_action(q_values)

    # Store S, A in buffer (R will be observed later)
    buffer.add_state_action(state, action)

    # Execute action
    hdgval = traf.hdg[traf.id2idx('SELF')] + float(actions_enum[action])
    stack.stack(f"HDG SELF {hdgval}")


def set_view(input_text):
    """
    Allows you to use the command VIEW_SIM 1 from BlueSky to simulate full episodes.
    """
    global VIEW_SIMULATIONS, EXPLORATION
    if int(input_text):
        VIEW_SIMULATIONS = True
        EXPLORATION = 0.0
        buffer.pause(True)
    else:
        VIEW_SIMULATIONS = False
        EXPLORATION = EXPLORATION_START  # Note that exploration gets reset here
        buffer.pause(False)


def make_angle_convex(angle):
    """
    :param angle: <float> angle in deg
    :return: <float> angle between -180 to 180
    """
    angle %= 360
    if angle > 180:
        return angle - 360
    return angle


def get_state():
    """
    Get current state of simulation
    :return: torch.Tensor (Input for the neural network, containing enough information to render the problem Markov)
    """
    self = traf.id2idx('SELF')
    enemy = traf.id2idx('ENEMY')
    self_hdg = make_angle_convex(traf.hdg[self])
    enemy_hdg = make_angle_convex(traf.hdg[enemy])
    rel_hdg = enemy_hdg - self_hdg

    qdr, dist = _qdrdist(traf.lat[self], traf.lon[self], traf.lat[enemy], traf.lon[enemy])
    qdr = make_angle_convex(qdr)
    return tensor([qdr + self_hdg, dist, rel_hdg])


def get_action(q_values):
    """
    :param q_values: <torch.Tensor> Output of the neural net
    :return: With probability EXPLORATION returns random.choice([0, 1, 2]), else gives the index of max(q_values)
    """
    sample = random.uniform(0, 1)
    if sample < EXPLORATION:
        return random.randint(len(actions_enum))

    else:
        return q_values.max(0)[1]


def check_net():
    """
    Use BlueSky command CHECK_NET to open pdb here
    """
    global DEBUGGING
    DEBUGGING = True
    import pdb
    pdb.set_trace()


def get_reward_for_distance():
    """
    :return: Negative reward inversely proportional to distance of separation
    """
    _, dist = _qdrdist(traf.lat[0], traf.lon[0], traf.lat[1], traf.lon[1])
    return -1 * SEPARATION_COST_FACTOR/max(dist**2, 0.5)


def get_reward_for_action():
    """
    :return: <float> Negative reward if heading-change (else 0)
    """
    last_action = actions_enum[buffer.memory[-1][1]]
    return -1 * ACTION_COST if last_action else 0.0


def _qdrdist(lat1, lon1, lat2, lon2):
    """
    Prevents BlueSky's qdrdist function from returning singular value
    """
    return qdrdist(lat1, lon1, lat2 + 0.0001, lon2)
