"""
Plugin for training a DQN for conflict-resolution, using monte-carlo learning
Author : Shiraz Khan

Saved ATC Policies ~
atc-policy-1 : 90% Learnt policy for single-aircraft control (using costs 10.0 and 1000.0)
atc-policy-2 : 40% Learnt policy for two-aircraft control
"""

from bluesky import traf, stack
from bluesky.tools.geo import qdrdist
from plugins.Sim_2AC import reset_aircrafts, RADIUS_NM, AIRCRAFTS
from plugins.CR_2AC_classes import ATC_Net, Buffer, Exploration

import pdb
import torch
import numpy as np

# ----- Problem Specifications ----- #
dHeading = 5  # Max heading allowed in one time-step
actions_enum = [-dHeading, 0,  dHeading]  # Once of these actions given as a heading-change input
RESOLVE_PERIOD = 300  # Every ___ steps of simulation, take conflict-resolution action
MIN_SEPARATION_ALLOWED = 4.0  # Negative reward is incurred if aircrafts come closer than this
AIRCRAFTS["ONE"].ATC = False  # Aircraft ONE is not being controlled by ATC

# ----- Initialization ----- #
SAVED_STATE = "saved_data/atc-policy-1"
COUNTER = 0

atc_net = ATC_Net(actions_enum)
if SAVED_STATE:
    atc_net.load_state_dict(torch.load(SAVED_STATE))

buffers = []
for ID, aircraft in AIRCRAFTS.items():
    if aircraft.ATC:
        buffers.append(Buffer(ID, actions_enum, MIN_SEPARATION_ALLOWED))

exploration = Exploration()


def init_plugin():
    """
    Initialization of BlueSky plugin
    Called once during plugin import
    """

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
    """
    Called once before simulation is updated
    at every time-step
    """
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

    exploration.decay()
    reset_aircrafts()


def update():
    """
    Called once along with BlueSky simulation update
    at every time-step
    """
    # Find the minimum separation maintained by the aircrafts in this episode
    if traf.ntraf > 1:
        for buffer in buffers:
            _, dist = _qdrdist(traf.lat[0], traf.lon[0], traf.lat[1], traf.lon[1])
            buffer.MIN_SEPARATION = np.min([buffer.MIN_SEPARATION, dist])


def resolve():
    """
    Take conflict resolution action for every aircraft
    """
    # Assign reward for previous state-action
    for buffer in buffers:
        buffer.assign_reward(buffer.get_reward_for_action()) if buffer.REWARD_PENDING else None

        # Choose action for current time-step
        state = get_state(buffer.ID)
        q_values = atc_net.forward(state)
        action = get_action(q_values, epsilon=exploration.eps)

        # Store S, A in buffer (R will be observed later)
        buffer.add_state_action(state, action)

        # Execute action
        new_heading = traf.hdg[traf.id2idx(buffer.ID)] + float(actions_enum[action])
        stack.stack(f"HDG {buffer.ID} {new_heading}")


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


def get_action(q_values, epsilon):
    """
    Decaying epsilon-greedy action
    :param q_values: <torch.Tensor> Output of the neural net
    :param epsilon: <float>
    :return: With probability EXPLORATION returns a random int, else gives the index of max(q_values)
    """
    sample = np.random.uniform(0, 1)
    if sample < epsilon:
        return np.random.randint(len(actions_enum))

    else:
        return q_values.max(0)[1]


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
    """
    return qdrdist(lat1, lon1, lat2 + 0.000001, lon2)


def show_net():
    """
    Visualizes the current Q(s,a_1) and Q(s,a_2) values of neural net
    Use BlueSky command SHOW_NET
    """
    atc_net.plot()


def check_net():
    """
    Use BlueSky command CHECK_NET to open pdb here
    """
    pdb.set_trace()


def set_view(input_text):
    """
    Allows you to use the command VIEW_SIM 1 from BlueSky to simulate full episodes.
    """
    # TODO
    raise NotImplementedError

