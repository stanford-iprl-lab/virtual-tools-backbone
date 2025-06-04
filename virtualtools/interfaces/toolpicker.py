from typing import Tuple, Annotated, Dict
from .vtinterface import VTInterface, place_object_by_polys
from ..world import VTWorld, load_vt_from_dict
import json, os, random
import numpy as np
from ..vtviewer import visualizePathSingleImageVT
from scipy.spatial.distance import cdist

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
        self.runs = {}


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
        if pos[0] < 0 or pos[1] < 0:
            return world
        assert tool in self._tools.keys(), "Tool " + tool + " does not exist in tool set " + self.toolnames
        tool_polys = self._tools[tool]
        world = place_object_by_polys(world, tool_polys, pos)
        return world
    
    def to_dict(self):
        return self._tpdict
    
    def get_objects(self):
        # get objects and their initial positions
        world = load_vt_from_dict(self._worlddict)
        return world.objects
    
    def get_global_min_dist(self, goal_pos):
        path_dict, success, t, wd = self.observe_placement_path(action={'tool':'obj1', 'position':(-10,-10)},maxtime=20.,return_world=True)
        # visualizePathSingleImageVT(wd, path_dict)
        min_dist = 600
        for obj_name, path in path_dict.items():
            if ("ball" not in obj_name.lower()) and ("goalblock" not in obj_name.lower()):
                continue
            distance_to_goal = cdist(np.array(path)[:,:2], np.array([goal_pos])).min()
            if distance_to_goal < min_dist:
                min_dist = distance_to_goal
        return min_dist
    
    def add_run(self, key, run):
        self.runs[key] = run

    def get_runs(self):
        return self.runs
    
    def tool_bbox(self, toolname):
        assert toolname in self.toolnames, "Tool not found: " + str(toolname)
        tool = self._tools[toolname]
        minx = 99999999
        miny = 99999999
        maxx = -99999999
        maxy = -99999999
        for p in tool:
            for v in p:
                if v[0] < minx:
                    minx = v[0]
                if v[0] > maxx:
                    maxx = v[0]
                if v[1] < miny:
                    miny = v[1]
                if v[1] > maxy:
                    maxy = v[1]
        return [[minx, miny], [maxx, maxy]]

    @property
    def toolnames(self):
        return list(self._tools.keys())

def load_tool_picker(tp_json_file, basic_timestep = 0.1):
    with open(tp_json_file, 'r') as jfl:
        return ToolPicker(json.load(jfl), basic_timestep)
