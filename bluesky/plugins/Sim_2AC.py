"""
Plugin for simulating 2-aircraft conflict episodes, to train a conflict-resolution DQN
Author : Shiraz Khan
"""

from numpy import random, cos, sin, deg2rad, abs
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

    def soft_reset(self, other_ac=None):
        """
        Moves aircraft to random point on circumference ( New episode )
        """
        pos_lat, pos_lon, hdg = get_circumference_point()

        # If there's another aircraft, avoid starting too close to it
        if other_ac:
            while qdrdist(pos_lat, pos_lon, other_ac["lat"], other_ac["lon"])[1] < 2.0*other_ac["separation"]:
                pos_lat, pos_lon, hdg = get_circumference_point()

        self.dest = [-1.2 * pos_lat, -1.2 * pos_lon]
        stack.stack(f"MOVE {self.ID} {pos_lat} {pos_lon} FL200 {hdg} 400")


def get_circumference_point():
    hdg = random.uniform(-180.0, 180.0)
    hdg_r = deg2rad(hdg)
    pos_lat = -1 * RADIUS * cos(hdg_r)
    pos_lon = -1 * RADIUS * sin(hdg_r)
    return pos_lat, pos_lon, hdg


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


def preupdate():
    pass


def create_ac(ID):
    """
    Create new aircraft
    """
    stack.stack(f"CRE {ID} B744 {0.0} {0.0} {0.0} FL200 400")
    aircraft = Aircraft(ID)
    aircraft.soft_reset()
    return aircraft


def reset_aircrafts():
    """
    Deletes everything and creates new aircrafts.
    Note : CR_2AC relies on the order of creation
    """
    delete_aircrafts()
    return {ID: create_ac(ID) for ID in AIRCRAFT_IDs}


def delete_aircrafts():
    for ID in AIRCRAFT_IDs:
        stack.stack(f"DEL {ID}")
