"""
Plugin for training a DQN for conflict-resolution, using monte-carlo learning
Author : Shiraz Khan

Saved ATC Policies ~
atc-policy-1 : 90% Learnt policy for single-aircraft control (using costs 10.0 and 1000.0)
atc-policy-2 : 40% Learnt policy for two-aircraft control
"""

from bluesky import traf, stack
from bluesky.tools.geo import qdrdist

from plugins.Sim_2AC import reset_aircrafts, RADIUS_NM, AIRCRAFT_IDs
from plugins.CR_2AC_classes import ATC_Net, Buffer, Exploration
from plugins.CR_2AC_classes import make_angle_convex, _qdrdist, normalize

import pdb
import torch
from torch.optim.lr_scheduler import StepLR


import numpy as np

# ----- Problem Specifications ----- #
dHeading = 8  # Max heading allowed in one time-step
actions_enum = [-dHeading, 0,  dHeading]  # Once of these actions given as a heading-change input
critical_distance = 8.0  # Negative reward is incurred if aircrafts come closer than this
RESOLVE_PERIOD = 400  # Every ___ steps of simulation, take conflict-resolution action

"""
Notes : 
Setting reward period to 600 and simulation radius to 24Nm gives episode length of ~10
"""

# ----- Initialization ----- #
# SAVED_STATE = None
SAVED_STATE = "saved_data/last_simulation/atc-policy.pt"
aircrafts = None
COUNTER = 0

atc_net: ATC_Net = ATC_Net(actions_enum)

try:
    atc_net.load_state_dict(torch.load(SAVED_STATE))

except:
    print("No compatible saved state found. Using random initialization.")
    atc_net.initialize(action_values={"0": 200.0, "1": 100.0, "2": 200.0}, critical_distance=critical_distance)

buffers = {ID: Buffer(ID, actions_enum) for ID in AIRCRAFT_IDs}


# ----- BlueSky plugin functions ----- #
def init_plugin():
    """
    Initialization of BlueSky plugin
    Called once during plugin import
    """
    reset_all()

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
    return


def reset_all():
    """
    Resets values after one batch of experience has been trained
    """
    global aircrafts
    aircrafts = reset_aircrafts()
    for buffer in buffers.values():
        buffer.empty()


def update():
    """
    Called once along with BlueSky simulation update
    at every time-step
    """
    global COUNTER
    COUNTER += 1

    for buffer in buffers.values():
        # Get distance between aircrafts
        dist_ac = qdrdist(traf.lat[0], traf.lon[0], traf.lat[1], traf.lon[1])[1]
        buffer.set_separation(dist_ac)

    # Rate of conflict-resolution is lesser than rate of simulation
    if not COUNTER % RESOLVE_PERIOD:
        COUNTER = 0
        resolve()

        for ac in aircrafts.values():
            idx = traf.id2idx(ac.ID)
            (lat, lon) = (traf.lat[idx], traf.lon[idx])

            # Check if aircraft is still in simulation area
            _, distance_from_center = _qdrdist(lat, lon, 0.0, 0.0)
            if distance_from_center > RADIUS_NM + 0.1:

                # Reached terminal state
                terminal_reward = buffers[ac.ID].get_terminal_reward()
                buffers[ac.ID].update_targets(terminal_reward, atc_net)
                buffers[ac.ID].terminate_episode()

                # Reset everything (and maybe teach neural net)
                buffers[ac.ID].check(atc_net)

                # Move this aircraft to a boundary point that doesn't conflict with the other aircraft
                other_ac_idx = (idx + 1) % 2
                other_ac = {
                    "lat": traf.lat[other_ac_idx],
                    "lon": traf.lon[other_ac_idx],
                    "separation": critical_distance
                            }
                ac.soft_reset(other_ac=other_ac)


def resolve():
    """
    Take conflict resolution action for every aircraft
    """
    for buffer in buffers.values():
        # Assign reward for previous state-action
        current_heading = traf.hdg[traf.id2idx(buffer.ID)]
        if buffer.episode_length:
            reward = buffer.get_reward_for_distance(critical_distance)
            buffer.update_targets(reward, atc_net)

        # Choose action for current time-step
        state = get_state(buffer.ID)
        q_values = atc_net.forward(state)
        action = get_action(q_values, epsilon=atc_net.exploration.eps)

        # Store S, A in buffer (R will be observed later)
        buffer.add_state_action(state, action)

        # Execute action
        new_heading = current_heading + float(actions_enum[action])
        stack.stack(f"HDG {buffer.ID} {new_heading}")


def get_state(ac_id: str):
    """
    Get current state tuple of aircraft
    :param ac_id: Aircraft to be controlled
    :return: torch.Tensor (Input for the neural network, containing enough information to render the problem Markov)
    """
    self = traf.id2idx(ac_id)
    enemy = (self + 1) % 2

    # Relative heading of other aircraft
    self_hdg = traf.hdg[self]
    enemy_hdg = traf.hdg[enemy]
    rel_hdg = enemy_hdg - self_hdg

    # Relative position of other aircraft
    qdr_ac, dist_ac = _qdrdist(traf.lat[self], traf.lon[self], traf.lat[enemy], traf.lon[enemy])

    # Relative position of destination
    dest_lat, dest_lon = aircrafts[ac_id].dest
    qdr_dn, dist_dn = _qdrdist(traf.lat[self], traf.lon[self], dest_lat, dest_lon)

    state = [
        make_angle_convex(qdr_ac + self_hdg),
        dist_ac,
        make_angle_convex(rel_hdg),
        make_angle_convex(qdr_dn + self_hdg),
        dist_dn
    ]
    state = normalize(state)
    return torch.tensor(state)


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


def show_net():
    """
    Visualizes the current Q(s,a_1) and Q(s,a_3) values of neural net
    Use BlueSky command SHOW_NET
    """
    atc_net.plot()


def check_net():
    """
    Use BlueSky command CHECK_NET to open pdb here (And save state of Neural Network)
    """
    torch.save(atc_net.state_dict(), "saved_data/last_simulation/atc-policy.pt")
    # for p in atc_net.parameters():
    #     print(p.grad)
    pdb.set_trace()


def set_view(input_text):
    """
    Allows you to use the command VIEW_SIM 1 from BlueSky to simulate full episodes.
    """
    # TODO
    raise NotImplementedError

