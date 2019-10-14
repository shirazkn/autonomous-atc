# Resolve conflict using a plugin, to enable easy implementation of Reinforcement Learning methods
# Author : Shiraz Khan

from bluesky import stack
from bluesky.traffic.asas import DoNothing


# By default, do nothing
resolve_with_plugin = DoNothing.resolve


def start(asas, resolve_fn=None):
    """
    Call this function as part of your plugin's initialization
    See plugins/CR_2Aircraft_BOX.py
    """
    global resolve_with_plugin
    if resolve_fn:
        resolve_with_plugin = resolve_fn


def resolve(asas, traf):
    """
    This function is called for conflict-resolution in BlueSky,
    whenever at least one pairwise conflict is detected
    """
    resolve_with_plugin(asas, traf)
