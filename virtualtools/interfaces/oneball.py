"""
Interface for a level that includes dropping a 
"""

from typing import Tuple, Annotated, Dict
from .vtinterface import VTInterface, place_ball
from ..world import VTWorld
import json

__all__ = ['OneBall', 'load_one_ball']

class OneBall(VTInterface):
    def __init__(self,
                 gamedict: Dict,
                 basic_timestep: float=0.1,
                 maxtime: float=20.,
                 world_timestep: float=None):
        super().__init__(gamedict['world'], basic_timestep,
                         maxtime, world_timestep)
        self._ballsize = gamedict['ballsize']
        self._obdict = gamedict
        
    @property
    def action_keys(self):
        return ['position']
    
    @property
    def interface_type(self):
        return "OneBall"
    
    @property
    def ballsize(self):
        return self._ballsize
    
    def place(self,
              action: Dict,
              world: VTWorld=None) -> VTWorld:
        self._check_action(action)
        world = place_ball(world, self._ballsize, action['position'])
        return world
    
def load_one_ball(ob_json_file, basic_timestep=0.1):
    with open(ob_json_file, 'r') as jfl:
        return OneBall(json.load(jfl), basic_timestep)