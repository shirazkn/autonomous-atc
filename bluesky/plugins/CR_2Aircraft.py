# Plugin for implementing ML-based conflict-resolution method
# Simulates continuing 2-aircraft episodes to learn CR policy

from numpy import random, pi
from bluesky import stack, traf
from bluesky.traffic.asas import PluginBasedCR
from math import atan
# Other import-able modules : settings, navdb, sim, scr, tools

# Box/Area of simulation
# (specified as [lower, upper])
LIMITS_LAT = [-2, 2]
LIMITS_LON = [-2, 2]


def init_plugin():
    reset()

    # Configuration parameters
    config = {
        'plugin_name':     'CR_2AIRCRAFT',
        'plugin_type':     'sim',
        'update':          update,
        'preupdate':       preupdate,
        'reset':         reset
        }

    stackfunctions = {
        'CHECK_PLUGIN': [
            'CHECK_PLUGIN',
            'txt',
            check,
            'Prints the values required to monitor performance of ML-based CR method.']
    }

    return config, stackfunctions


def reset():
    # Makes sure 'resolve' function defined in plugin is used by ASAS
    PluginBasedCR.start()

    # Sets area of interest, flights exiting this area are deleted
    limits_str = f"{LIMITS_LAT[0]} {LIMITS_LON[0]} {LIMITS_LAT[1]} {LIMITS_LON[1]}"
    stack.stack(f"BOX Box1 {limits_str}")
    stack.stack("AREA Box1")


def update():
    """
    Called every `update_interval` seconds
    """
    pass


def preupdate():
    # num_ac = traf.ntraf
    self_idx = traf.id2idx("SELF")
    if self_idx < 0:
        create_self_ac()

    enemy_idx = traf.id2idx("ENEMY")
    if enemy_idx < 0:
        create_enemy_ac()


def resolve(asas, traf):
    """
    Called in place of built-in `resolve` method
    """
    pass


def check(input_text):
    """
    Print plugin variables, for debugging/monitoring
    For now, it just spits the input back into the console
    """
    stack.stack(f"ECHO CR_2Aircraft received input : '{input_text}'...")


def get_hdg_limits(pos_lat, pos_lon):
    """
    Get range of Headings that point inwards from a point on the boundary of the Box
    :param pos_lat: Lat of point on boundary
    :param pos_lon: Lon of point on boundary
    :return: Heading min, Heading max
    """
    if pos_lat == LIMITS_LAT[0]:
        stack.stack("ECHO Unlikely value in get_hdg_limits!")
        return 0, 0

    # _, qdr = qdrdist(pos_lat, pos_lon, LIMITS_LAT[0], LIMITS_LON[0])
    # Unsure how the angle QDR is defined in BlueSky, so using a workaround...
    arg = atan((pos_lon - LIMITS_LON[0])/(pos_lat - LIMITS_LAT[0])) * 180 / pi

    if arg == 0.0:
        return 0.0, 180.0

    elif 0.0 < arg <= 45.0:
        return 90.0, 270.0

    elif 45.0 < arg < 90.0:
        return 180.0, 360.0

    elif arg == 90.0:
        return 270.0, 450.0

    else:
        # For debugging
        stack.stack("ECHO Wrong implementation of get_hdg_limits!")
        return 0.0, 0.0


def create_self_ac():
    """
    Create aircraft going from South to North
    """
    pos_lat = LIMITS_LAT[0]
    pos_lon = 0.5 * (LIMITS_LON[0] + LIMITS_LON[1])
    hdg = 0
    stack.stack(f"CRE SELF B744 {pos_lat} {pos_lon} {hdg} FL200 400")


def create_enemy_ac():
    """
    Create aircraft coming inwards from a random point on left/right boundaries
    """
    pos_lat = random.uniform(LIMITS_LAT[0], LIMITS_LAT[1])
    pos_lon = random.choice([LIMITS_LON[0], LIMITS_LON[1]])
    hdg_min, hdg_max = get_hdg_limits(pos_lat, pos_lon)
    hdg = random.uniform(hdg_min, hdg_max) % 360
    stack.stack(f"CRE ENEMY B744 {pos_lat} {pos_lon} {hdg} FL200 400")
