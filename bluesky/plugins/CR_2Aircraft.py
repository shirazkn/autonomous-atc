# Plugin for implementing ML-based conflict-resolution method
# Simulates continuing 2-aircraft episodes to learn CR policy

from numpy import random, cos, sin, deg2rad
from bluesky import stack, traf
from bluesky.traffic.asas import PluginBasedCR
from bluesky.tools.geo import qdrdist
from plugins.ML import ATC_2AC
# Other BlueSky modules : settings, navdb, sim, scr, tools

from torch import tensor
import torch.optim as optim

# Radius of simulation area
RADIUS = 0.4  # in "Latitudes"
_, RADIUS_NM = qdrdist(-RADIUS, 0, 0, 0)  # in nautical miles

# AI-Based conflict resolution
atc_net = ATC_2AC.ATC_Net()
optimizer = optim.SGD(atc_net.parameters(), lr=0.01, momentum=0.5)


def init_plugin():
    reset()

    # Configuration parameters
    config = {
        'plugin_name':     'CR_2AIRCRAFT',
        'plugin_type':     'sim',
        'update':          update,
        'preupdate':       preupdate,
        'reset':           reset
        }
    stackfunctions = {}
    return config, stackfunctions


def reset():
    # Makes sure 'resolve' function defined in plugin is used by ASAS
    PluginBasedCR.start(None, resolve)

    # Sets area of interest, flights exiting this area are deleted
    stack.stack(f"CIRCLE SimCirc 0 0 {RADIUS_NM}")
    stack.stack("AREA SimCirc")


def update():
    """
    Called every `update_interval` seconds
    """
    pass


def preupdate():
    num_ac = traf.ntraf
    if num_ac < 2:
        create_self_ac()
        create_enemy_ac()


def create_self_ac():
    """
    Create aircraft going from South to North
    """
    pos_lat = -RADIUS
    pos_lon = 0
    hdg = 0
    stack.stack(f"CRE SELF B744 {pos_lat} {pos_lon} {hdg} FL200 400")


def create_enemy_ac():
    """
    Create aircraft coming inwards from a random point on left/right boundaries
    """
    hdg = random.uniform(0, 360.0)
    hdg_r = deg2rad(hdg)
    pos_lon = -1 * RADIUS * sin(hdg_r)
    pos_lat = -1 * RADIUS * cos(hdg_r)
    stack.stack(f"CRE ENEMY B744 {pos_lat} {pos_lon} {hdg} FL200 400")


def resolve(asas, traf):
    """
    Called in place of built-in `resolve` method
    """
    # TODO
    self = traf.id2idx('SELF')
    enemy = traf.id2idx('ENEMY')

    # Eventually needs to be done for each aircraft
    # for confpair in asas.confpairs:
    #     resolve
        #  > Take action based on output

    #  > Let simulation run for another step
    #  loss = ATC_2AC.get_loss(atc_out, next_asas_state)
    #  loss.backward()
    #  optimizer.step()
