# Plugin for implementing ML-based conflict-resolution method
# Simulates continuing 2-aircraft episodes to learn CR policy
# CR is done using a net with Experience Replay

from numpy import random, cos
from bluesky import traf
from bluesky.traffic.asas import PluginBasedCR
from bluesky.tools.geo import qdrdist
from plugins.Sim_2AC import reset_aircrafts

from torch import tensor, nn
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss, relu
from random import sample as random_sample


dHeading = 2  # Max heading allowed in one time-step
actions_enum = [-dHeading, 0.0, dHeading]  # Once of these actions given as a heading-change input
VIEW_SIMULATIONS = False  # When True, simulations will be updated even if there's not conflict
N_STEPS = 2  # Number of steps for TD implementation
FINAL_REWARD = 0.0
GAMMA = 1.0
EXPLORATION_START = 0.9
EXPLORATION_MIN = 0.3
EXPLORATION_DECAY = 0.999
EXPLORATION = EXPLORATION_START
REWARD_MULTIPLIER = 0.05

COUNTER = 1


# PyTorch Neural Net
class ATC_Net(nn.Module):

    def __init__(self):
        super(ATC_Net, self).__init__()

        # Network input & output layers
        self.fcInp = nn.Linear(3, 5)
        self.fcH1 = nn.Linear(5, 10)
        self.fcH2 = nn.Linear(10, 10)
        self.fcH3 = nn.Linear(10, 5)
        self.fcOut = nn.Linear(5, 3)

    def forward(self, x):
        x = self.fcOut(self.fcH3(relu(self.fcH2(relu(self.fcH1(self.fcInp(x)))))))
        return x

    def learn(self, transitions):
        """
        Learn q-function using n-step TD method
        :param transitions: [ [S, A, R+] , [S+, A+, R++] ... ],
                            Last entry is either [S, A, R] or [S, A, R, "TERMINATED"]
        """
        # Learn from experience
        td_target = tensor(0.0)  # R1 + Gamma*R2 + Gamma**2*Q(s,a)
        discount = 1.0
        is_terminated = False

        for t in transitions[:-1]:
            reward = t[2]
            td_target += discount*reward
            discount *= GAMMA
            if len(t) > 3:  # True if this is the terminal state
                is_terminated = True
                print(f"Terminating state. My td target is {td_target}")
                break

        if not is_terminated:
            td_target += discount*(self.forward(transitions[-1][0]).max())

        this_state = transitions[0][0]
        this_action = transitions[0][1]
        q_value = self.forward(this_state)[this_action]  # Value of chosen ('optimal' or explored) action

        global COUNTER
        COUNTER += 1
        print(f"My q_value is {q_value} and my td_target is {td_target}.") if not (COUNTER % 100) else None
        loss = smooth_l1_loss(q_value, td_target)
        loss.backward()
        optimizer.step()
        print(f"q_value updated to {self.forward(this_state)[this_action]}") if not (COUNTER % 100) else None

        global EXPLORATION
        EXPLORATION = max(EXPLORATION_DECAY*EXPLORATION, EXPLORATION_MIN)


class Buffer:
    TEACHING = True
    REWARD_PENDING = False
    memory = []
    size = 500
    n_samples = 20

    def check_if_full(self, n_net: ATC_Net):
        if (len(self.memory) + 1) > self.size:
            if not VIEW_SIMULATIONS:
                self.teach(n_net)
            self.empty()

    def teach(self, n_net: ATC_Net):
        transition_sets = self.get_samples()
        for transition_set in transition_sets:
            n_net.learn(transition_set)

    def get_samples(self):
        samples = random_sample(range(len(self.memory)-N_STEPS+1), self.n_samples)
        return [self.memory[s:s+N_STEPS + 1] for s in samples]

    def empty(self):
        self.memory = []

    def pause(self, option):
        self.empty()
        self.TEACHING = (not option)

    def assign_terminal_reward(self):
        if len(self.memory) > N_STEPS + 1:
            final_hdg = abs(traf.hdg[traf.id2idx('SELF')])

            final_reward = FINAL_REWARD*abs(cos(final_hdg * 0.5))
            self.assign_reward(final_reward)
            self.REWARD_PENDING = False
            self.memory[-1].append("TERMINATED")

        else:
            buffer.empty()

    def add_state_action(self, state, action):
        self.memory.append([state, action])
        self.REWARD_PENDING = True

    def assign_reward(self, reward):
        self.memory[-1] = self.memory[-1][:2] + [reward]
        self.REWARD_PENDING = False


# Global variables for Neural Net
atc_net = ATC_Net()
buffer = Buffer()
optimizer = optim.SGD(atc_net.parameters(), lr=0.0005, momentum=0.5)


def init_plugin():
    PluginBasedCR.start(None, resolve)

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


def update():
    if len(traf.asas.confpairs):
        if buffer.REWARD_PENDING:
            buffer.assign_reward(get_reward_from_distance())
    else:
        buffer.assign_terminal_reward()

    buffer.check_if_full(atc_net)


def preupdate():
    if len(traf.asas.confpairs) < 1:
        if not VIEW_SIMULATIONS:
            reset_aircrafts()


def resolve(asas, traf):
    """
    Called in place of built-in `resolve` method
    """
    # Choose action
    state = get_state()
    q_values = atc_net.forward(state)
    action = get_action(q_values)

    # Execute action
    traf.hdg[traf.id2idx('SELF')] += float(actions_enum[action])

    # Store in buffer
    buffer.add_state_action(state, action)


def set_view(input_text):
    """
    Allows you to use the command VIEW_SIM 1 from BlueSky to simulate full episodes (past conflict-resolution).
    """
    global VIEW_SIMULATIONS, EXPLORATION
    if int(input_text):
        VIEW_SIMULATIONS = True
        EXPLORATION = 0.0
        buffer.pause(True)
    else:
        VIEW_SIMULATIONS = False
        EXPLORATION = EXPLORATION_START
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
        return random.randint(3)

    else:
        return q_values.max(0)[1]


def check_net():
    import pdb
    pdb.set_trace()


def get_reward_from_distance():
    _, dist = _qdrdist(traf.lat[0], traf.lon[0], traf.lat[1], traf.lon[1])
    return - REWARD_MULTIPLIER/max(dist, 0.001)


def _qdrdist(lat1, lon1, lat2, lon2):
    """
    Prevents BlueSky's qdrdist function from returning singular value
    """
    return qdrdist(lat1, lon1, lat2 + 0.0001, lon2)
