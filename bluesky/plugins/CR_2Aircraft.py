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

loss = tensor(0.0)
output_tensor = tensor(0.0)
running_loss = 0.0


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
    global running_loss
    running_loss += ATC_2AC.accumulate_loss(traf.asas)


def preupdate():
    global loss
    num_ac = traf.ntraf
    if num_ac < 2:
        create_self_ac()
        create_enemy_ac()
        loss = tensor(0.0)


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
    self = traf.id2idx('SELF')
    enemy = traf.id2idx('ENEMY')
    global loss, output_tensor, running_loss
    loss = ATC_2AC.cost_of_resolution(output_tensor, running_loss)
    loss.backward()
    optimizer.step()
    qdr, dist = qdrdist(traf.lat[self], traf.lon[self], traf.lat[enemy], traf.lon[enemy])
    rel_hdg = traf.hdg[enemy] - traf.hdg[self]
    input_tensor = tensor([qdr, dist, rel_hdg])
    output_tensor = atc_net.forward(input_tensor)
    traf.hdg[self] += float(output_tensor)
