from typing import Tuple, Annotated, Dict
from .vtinterface import VTInterface, place_object_by_polys
from ..world import VTWorld
import json, os, random, copy

__all__ = ['ToolPicker', 'load_tool_picker']


class ToolPicker(VTInterface):
    def __init__(self,
                 gamedict: Dict,
                 basic_timestep: float=0.1,
                 maxtime: float=20.,
                 world_timestep: float=None):
        super().__init__(gamedict['world'], basic_timestep,
                         maxtime, world_timestep)
        self._tools = gamedict['tools']
        self._tpdict = gamedict


    @property
    def action_keys(self):
        return ['tool', 'position']

    @property
    def interface_type(self):
        return "ToolPicker"

    def place(self,
              action: Dict,
              world: VTWorld=None) -> VTWorld:
        world = world or load_vt_from_dict(self._worlddict)
        self._check_action(action)
        tool = action['tool']
        pos = action['position']
        assert tool in self._tools.keys(), "Tool " + tool + " does not exist in tool set " + self.toolnames
        tool_polys = self._tools[tool]
        world = place_object_by_polys(world, tool_polys, pos)
        return world
    
    def to_dict(self):
        return self._tpdict

    @property
    def toolnames(self):
        return list(self._tools.keys())
    
    @property
    def tools(self):
        return copy.deepcopy(self._tools)

def load_tool_picker(tp_json_file, basic_timestep = 0.1):
    with open(tp_json_file, 'r') as jfl:
        return ToolPicker(json.load(jfl), basic_timestep)
