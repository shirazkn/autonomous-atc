"""
Plugin for simulating 2-aircraft conflict episodes, to train a conflict-resolution DQN
Author : Shiraz Khan
"""

from numpy import random, cos, sin, deg2rad
from bluesky import stack, traf
from bluesky.tools.geo import qdrdist

# Radius (in "Latitudes")
RADIUS = 0.4

# Radius (in nautical miles)
_, RADIUS_NM = qdrdist(-RADIUS, 0.0, 0.0, 0.0)

# Aircrafts in the simulation
AIRCRAFT_IDs = ["ONE", "TWO"]


class Aircraft:
    def __init__(self, _ID, destination=None, ATC=True):
        self.ID = _ID  # ID used for this a/c in BlueSky; redundant
        self.dest = destination  # Most desirable terminal state for this aircraft (Destination)
        self.ATC = ATC  # Whether ATC conflict-resolution is being applied to this aircraft


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
    # Visualize simulation area
    stack.stack(f"CIRCLE SimCirc 0 0 {RADIUS_NM}")
    # stack.stack("AREA SimCirc")  ~~ This is now being done explicitly in plugins.CR_2AC.preupdate()


def preupdate():
    pass


def create_ac(ID):
    """
    Create aircraft coming inwards from a random point on circumference
    """
    hdg = random.uniform(0, 360.0)
    hdg_r = deg2rad(hdg)
    pos_lon = -1 * RADIUS * sin(hdg_r)
    pos_lat = -1 * RADIUS * cos(hdg_r)
    stack.stack(f"CRE {ID} B744 {pos_lat} {pos_lon} {hdg} FL200 400")
    return Aircraft(ID, destination=[-1*pos_lat, -1*pos_lon])


def reset_aircrafts():
    """
    Creates both aircrafts.
    Note : CR_2AC relies on the order of creation
    """
    delete_aircrafts()
    return {ID: create_ac(ID) for ID in AIRCRAFT_IDs}


def delete_aircrafts():
    for ID in AIRCRAFT_IDs:
        stack.stack(f"DEL {ID}")
