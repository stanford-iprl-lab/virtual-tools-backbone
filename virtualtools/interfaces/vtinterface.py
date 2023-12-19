from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import warnings
from ..world import VTWorld, noisify_world, load_vt_from_dict
from .running import (run_game, get_path, get_state_path, get_collisions, 
        get_geom_path, get_game_outcomes, CollisionError)
from geometry import ear_clip, lines_intersect, check_counterclockwise, gift_wrap
from ..helpers import any_line_intersections


"""
Determine whether a shape defined by a set of polygons will collide with anything currently existing in a VTWorld

Args:
    world (VTWorld): the VTWorld to check for collisions
    polys (List[List[Tuple[float, float]]]): a list of convex polygons making up an arbitrary shape. Polygons are defined as a set of CCW vertices. Vertices are defined relative to the placement point at (0, 0)
    position (Tuple[float, float]): the position where the putative shape will be placed

Returns:
    bool: returns True if the shape will collide with anything existing in the VTWorld; False otherwise

"""
def check_collision_by_polys(world: VTWorld,
                             polys: List[List[Tuple[float, float]]],
                             position: Tuple[float, float]
                             ) -> bool:
    for pverts in polys:
        if world.check_collision(position, pverts):
            return True
    return False


"""
Places a shape defined by a set of polygons into a VTWorld with the name "PLACED"

Args:
    world (VTWorld): the VTWorld to check for collisions
    polys (List[List[Tuple[float, float]]]): a list of convex polygons making up an arbitrary shape. Polygons are defined as a set of CCW vertices. Vertices are defined relative to the placement point at (0, 0)
    position (Tuple[float, float]): the position where the putative shape will be placed

Returns:
    VTWorld: a pointer to the same `world` input, with a new "PLACED" object
"""

def place_object_by_polys(world: VTWorld,
                          polys: List[List[Tuple[float, float]]],
                          position: Tuple[float, float]
                          ) -> VTWorld:
    if check_collision_by_polys(world, polys, position):
        raise CollisionError()
    placed = []
    for poly in polys:
        placed.append([(p[0]+position[0], p[1]+position[1]) for p in poly])
    world.add_placed_compound("PLACED", placed, (0,0,255))
    return world

"""
Places a shape defined by a set of polygons into a VTWorld with the name "PLACED"

Args:
    world (VTWorld): the VTWorld to check for collisions
    poly (List[Tuple[float, float]]): a list of vertices making a convex polygon. Polygons are defined as a set of CCW vertices. Vertices are defined relative to the placement point at (0, 0)
    position (Tuple[float, float]): the position where the putative shape will be placed

Returns:
    VTWorld: a pointer to the same `world` input, with a new "PLACED" object
"""

def place_object_by_single_poly(world: VTWorld,
                                poly: List[Tuple[float, float]],
                                position: Tuple[float, float]
                                ) -> VTWorld:
    if check_collision_by_polys(world, [poly], position):
        raise CollisionError()
    placed = [(p[0]+position[0], p[1]+position[1]) for p in poly]
    world.add_placed_poly("PLACED", placed, (0,0,255))
    return world

"""
Places a shape defined by a set of ordered vertices into a VTWorld with the name "PLACED"

Args:
    world (VTWorld): the VTWorld to check for collisions
    vertices (List[Tuple[float, float]]): a list of ordered vertices defining the concave hull of an object. The edges must not cross one another, and CCW winding is assumed. Vertices are defined relative to the placement point at (0, 0)
    position (Tuple[float, float]): the position where the putative shape will be placed

Returns:
    VTWorld: a pointer to the same `world` input, with a new "PLACED" object
"""

def place_object_by_vertex_list(
    world: VTWorld,
    vertices: List[Tuple[float, float]],
    position: Tuple[float, float]
) -> VTWorld:
    # Ensure this is a legal vertex set
    assert check_counterclockwise(vertices),\
        "Input to place_object_by_vertex_list must have CCW winding"
    assert not any_line_intersections(vertices),\
        "Segments defined by connecting vertices may not cross"
    # Check if this can be a simple convex polygon
    if len(vertices) == len(gift_wrap(vertices)):
        return place_object_by_single_poly(world, vertices, position)
    # Otherwise use ear clipping to make triangles from concave polys
    else:
        triangles = ear_clip(vertices)
        return place_object_by_polys(world, triangles, position)
    
"""
Places a ball shape into a VTWorld with the name "PLACED"

Args:
    world (VTWorld): the VTWorld to check for collisions
    radius (float): a number indicating the radius of the ball in pixels
    position (Tuple[float, float]): the position where the putative shape will be placed

Returns:
    VTWorld: a pointer to the same `world` input, with a new "PLACED" object
"""

def place_ball(world: VTWorld,
                                radius: float,
                                position: Tuple[float, float]
                                ) -> VTWorld:
    if world.check_circle_collision(position, radius):
        raise CollisionError()
    world.add_placed_circle("PLACED", position, radius, (0, 0, 255))
    return world


class VTInterface(ABC):
    def __init__(self,
                 vt_worlddict: Dict,
                 basic_timestep: float=0.1,
                 maxtime: float=20.,
                 world_timestep: float=None):
        self._worlddict = vt_worlddict
        self._maxtime = maxtime
        self.bts = basic_timestep
        world_timestep = world_timestep or self._worlddict['bts']
        if basic_timestep < world_timestep:
            warnings.warn("Cannot set smaller basic_timestep than world_timestep; setting both to basic_timestep")
            world_timestep = basic_timestep
        self._worlddict['bts'] = world_timestep

    # Required knowledge about interface: action definition + name
    @property
    @abstractmethod
    def action_keys(self):
        pass

    @property
    @abstractmethod
    def interface_type(self):
        pass

    def _check_action(self, action:Dict) -> None:
        missing_actions = [a for a in self.action_keys if a not in action.keys()]
        if len(missing_actions) != 0:
            raise VTActionError(missing_actions)
        return

    @abstractmethod
    def place(self, action: Dict, world: VTWorld) -> VTWorld:
        raise NotImplementedError("place method must be overwritten")

    @abstractmethod
    def to_dict(self):
        return {'world': self.worlddict}

    def noisy_placement(self,
                        action: Dict,
                        noise: Dict,
                        world: VTWorld) -> VTWorld:
        nworld = noisify_world(world, **noise)
        return self.place(action, nworld)
    
    @property
    def dict(self):
        return self.to_dict()

    def _setup_world(self,
                      action: Dict,
                      noise: Dict=None,
                      stop_on_goal: bool=True,
                      new_object_properties: Dict=None
                      ) -> VTWorld:
        # To keep running after the goal condition, strip the goal
        if stop_on_goal:
            wd = self._worlddict
        else:
            wd = strip_goal(self._worlddict)
        # Optional adjutment of object properties (for modeling)
        if new_object_properties:
            wd = update_object_properties(wd, new_object_properties)
        # Run the action, return [None, -1] as illegal action flag
        w = load_vt_from_dict(wd)
        if noise is not None:
            return self.noisy_placement(action, noise, w)
        else:
            return self.place(action, w)

    def run_placement(self,
                      action: Dict,
                      noise: Dict=None,
                      maxtime: float=None,
                      stop_on_goal: bool=True,
                      new_object_properties: Dict=None
                      ) -> Tuple[bool, float]:
        maxtime = maxtime or self._maxtime
        try:
            w = self._setup_world(action,
                                  noise,
                                  stop_on_goal,
                                  new_object_properties)
        except CollisionError:
            return [None, -1] # Error code for illegal action
        return run_game(w, maxtime, self.bts)

    def observe_placement_path(self,
                               action: Dict,
                               noise: Dict=None,
                               maxtime: float=None,
                               stop_on_goal: bool=True,
                               new_object_properties: Dict=None
                               ) -> Tuple[Dict, bool, float]:
        maxtime = maxtime or self._maxtime
        try:
            w = self._setup_world(action,
                                  noise,
                                  stop_on_goal,
                                  new_object_properties)
        except CollisionError:
            return [None, None, -1] # Error code for illegal action
        return get_path(w, maxtime, self.bts)

    def observe_full_path(self,
                          action: Dict,
                          noise: Dict=None,
                          maxtime: float=None,
                          stop_on_goal: bool=True,
                          new_object_properties: Dict=None
                          ) -> Tuple[Dict, bool, float]:
        maxtime = maxtime or self._maxtime
        try:
            w = self._setup_world(action,
                                  noise,
                                  stop_on_goal,
                                  new_object_properties)
        except CollisionError:
            return [None, None, -1] # Error code for illegal action
        return get_state_path(w, maxtime, self.bts)

    def observe_geom_path(self,
                          action: Dict,
                          noise: Dict=None,
                          maxtime: float=None,
                          stop_on_goal: bool=True,
                          new_object_properties: Dict=None
                          ) -> Tuple[Dict, bool, float]:
        maxtime = maxtime or self._maxtime
        try:
            w = self._setup_world(action,
                                  noise,
                                  stop_on_goal,
                                  new_object_properties)
        except CollisionError:
            return [None, None, -1] # Error code for illegal action
        return get_geom_path(w, maxtime, self.bts)
    
    def observe_game_path(self,
                          action: Dict,
                          noise: Dict=None,
                          maxtime: float=None,
                          stop_on_goal: bool=True,
                          new_object_properties: Dict=None
                          ) -> Tuple[Dict, List, bool, float]:
        maxtime = maxtime or self._maxtime
        try:
            w = self._setup_world(action,
                                  noise,
                                  stop_on_goal,
                                  new_object_properties)
        except CollisionError:
            return [None, None, None, -1] # Error code for illegal action
        return get_game_outcomes(w, maxtime, self.bts)

    def observe_collision_events():
        raise NotImplementedError("TO IMPLEMENT")

    @property
    def worlddict(self) -> Dict:
        return self._worlddict

    @worlddict.setter
    def worlddict(self, newdict=Dict):
        try:
            load_vt_from_dict(newdict)
            self._worlddict = newdict
        except:
            raise Exception("Set worlddict with a dictionary that cannot be interpreted as a VTWorld object")

    @property
    def basic_timestep(self) -> float:
        return self._bts

    @property
    def world_timestep(self) -> float:
        return

    @property
    def maxtime(self) -> float:
        return self._maxtime

    @maxtime.setter
    def maxtime(self, time: float):
        assert time > 0, "Cannot set non-positive maximum time default"
        self._maxtime = time

    @property
    def interface_type(self) -> str:
        return self._interface_type

    @property
    def world(self) -> VTWorld:
        return load_vt_from_dict(self.worlddict)



class VTActionError(Exception):
    def __init__(self, missing_keys, expected_actions):
        self._mk = missing_keys
        self._ea = expected_actions
        self.message = "Action requires: " + str(expected_actions) + '; missing: ' + str(missing_keys())
        super().__init__(self.message)

    def __str__(self):
        return self.message
