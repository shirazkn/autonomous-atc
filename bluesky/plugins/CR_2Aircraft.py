# Plugin for implementing ML-based conflict-resolution method
# Simulates continuing 2-aircraft episodes to learn CR policy

from numpy import random, cos, sin, deg2rad
from bluesky import stack, traf
from bluesky.traffic.asas import PluginBasedCR
from bluesky.tools.geo import qdrdist
# Other import-able modules : settings, navdb, sim, scr, tools

# Radius (in "Latitudes")
RADIUS = 0.4

# adius in nautical miles
_, RADIUS_NM = qdrdist(-RADIUS, 0, 0, 0)


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
    PluginBasedCR.start(None, resolve)

    # Sets area of interest, flights exiting this area are deleted
    stack.stack(f"CIRCLE SimCirc 0 0 {RADIUS_NM}")
    stack.stack("AREA SimCirc")


n_call_upd = 0


def update():
    """
    Called every `update_interval` seconds
    """
    global n_call_upd
    n_call_upd += 1
    if n_call_upd % 100 == 0:
        stack.stack(f"ECHO Update called {n_call_upd} times..")


def preupdate():
    num_ac = traf.ntraf
    if num_ac < 2:
        create_self_ac()
        create_enemy_ac()


n_call_res = 0


def resolve(asas, traf):
    """
    Called in place of built-in `resolve` method
    """
    global n_call_res
    n_call_res += 1
    if n_call_res % 10 == 0:
        stack.stack(f"ECHO Resolve called {n_call_res} times..")


def check(input_text):
    """
    Print plugin variables, for debugging/monitoring
    For now, it just spits the input back into the console
    """
    stack.stack(f"ECHO CR_2Aircraft received input : '{input_text}'...")


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
