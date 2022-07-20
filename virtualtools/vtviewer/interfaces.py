from ..world import VTWorld
from ..interfaces import VTInterface
from .movies import demonstrate_path
import pygame as pg
from typing import Dict, List, Tuple


def demonstrate_action(interface: VTInterface,
                       action: Dict,
                       hz: float=None):
    hz = hz or 1./interface.bts
    # Get the paths
    path, _, _ = interface.observe_full_path(action)
    # Get the world dictionary with the action added
    actworld = interface.place(action, interface.world)
    # Make the movie
    demonstrate_path(actworld.to_dict(), path, hz)
