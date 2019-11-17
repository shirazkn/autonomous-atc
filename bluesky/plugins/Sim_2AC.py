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
class Aircraft:
    def __init__(self, _ID):
        self.ID = _ID  # ID used for this a/c in BlueSky; redundant
        self.destination = None  # Most desirable terminal state for this aircraft
        self.RESOLUTION = False  # Whether ATC conflict-resolution is being applied to this aircraft


AIRCRAFT_IDs = ["ONE", "TWO"]
AIRCRAFTS = {_id: Aircraft(_id) for _id in AIRCRAFT_IDs}


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
    reset_aircrafts()


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
    AIRCRAFTS[ID].destination = [-1*pos_lat, -1*pos_lon]


def reset_aircrafts():
    """
    Creates both aircrafts.
    Note : CR_2AC relies on the order of creation
    """
    delete_aircrafts()
    for ID in AIRCRAFTS:
        create_ac(ID)


def delete_aircrafts():
    for ID in AIRCRAFTS:
        stack.stack(f"DEL {ID}")
