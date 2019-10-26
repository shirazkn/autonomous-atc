# Plugin for implementing ML-based conflict-resolution method
# Simulates continuing 2-aircraft episodes to learn CR policy

from numpy import random, cos, sin, deg2rad
from bluesky import stack, traf
from bluesky.tools.geo import qdrdist

# Radius (in "Latitudes")
RADIUS = 0.4

# Radius
_, RADIUS_NM = qdrdist(-RADIUS, 0, 0, 0)


def init_plugin():
    reset()

    # Configuration parameters
    config = {
        'plugin_name':     'Sim_2AC',
        'plugin_type':     'sim',
        'preupdate':       preupdate,
        'reset':           reset
        }
    stackfunctions = {}

    return config, stackfunctions


def reset():
    # Sets area of interest, flights exiting this area are deleted
    stack.stack(f"CIRCLE SimCirc 0 0 {RADIUS_NM}")
    stack.stack("AREA SimCirc")


def preupdate():
    if traf.ntraf < 2:
        reset_aircrafts()


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
    Create aircraft coming inwards from a random point on circumference
    """
    hdg = random.uniform(0, 360.0)
    hdg_r = deg2rad(hdg)
    pos_lon = -1 * RADIUS * sin(hdg_r)
    pos_lat = -1 * RADIUS * cos(hdg_r)
    stack.stack(f"CRE ENEMY B744 {pos_lat} {pos_lon} {hdg} FL200 400")


def reset_aircrafts():
    stack.stack(f"DEL SELF")
    stack.stack(f"DEL ENEMY")
    create_self_ac()
    create_enemy_ac()
