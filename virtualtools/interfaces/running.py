from ..world import VTWorld
import numpy as np
from typing import Tuple, Dict, List

"""
Various functions for running the gameworld forward

Typically take in three arguments:
    gameworld (virtualtools.VTWorld): the gameworld (with all objects added)
    maxtime (float): the total time to run the world until time out. Defaults to 20s
    step_size (float): the time between checking for solutions. Defaults to 0.1s
"""

"""
Simple game running

Returns tuple of:
    bool: whether the goal was accomplished
    float: the time at which the world was stopped (due to solution or timeout)
"""
def run_game(gameworld: VTWorld,
             maxtime: float=20.,
             step_size: float=0.1
             ) -> Tuple[bool, float, VTWorld]:
    running = True
    t = 0
    while running:
        gameworld.step(step_size)
        t += step_size
        if gameworld.check_end() or (t >= maxtime):
            running = False
    return gameworld.check_end(), t


"""
Run the game and keep track of object positions

Returns tuple of:
    Dict[(obj_name, List[(x, y)])]: a dictionary with keys being the names of each of the non-static objects in the world. Each entry is a list of positions of that object at each of the timesteps
    bool: whether the goal was accomplished
    float: the time at which the world was stopped (due to solution or timeout)
"""
def get_path(gameworld: VTWorld,
             maxtime: float=20.,
             step_size: float=0.1
             ) -> Tuple[Dict, bool, float]:
    running = True
    t = 0
    pathdict = dict()
    tracknames = []
    for onm, o in gameworld.objects.items():
        if not o.check_end():
            tracknames.append(onm)
            pathdict[onm] = [o.position]
    while running:
        gameworld.step(step_size)
        t += step_size
        for onm in tracknames:
            pathdict[onm].append(gameworld.objects[onm].position)
        if gameworld.check_end() or (t >= maxtime):
            running = False
    return pathdict, gameworld.check_end(), t

"""
Run the game and keep track of object states (position, rotation, velocity)

Returns tuple of:
    Dict[(obj_name, List[(x, y, rot, vx, vy)])]: a dictionary with keys being the names of each of the non-static objects in the world. Each entry is a list of 5-length Tuples that include: (position_x, position_y, rotation_angle, velocity_x, velocity_y)
    bool: whether the goal was accomplished
    float: the time at which the world was stopped (due to solution or timeout)
"""
def get_state_path(gameworld: VTWorld,
             maxtime: float=20.,
             step_size: float=0.1
             ) -> Tuple[Dict, bool, float]:
    running = True
    t = 0
    pathdict = dict()
    tracknames = []
    for onm, o in gameworld.objects.items():
        if not o.is_static():
            tracknames.append(onm)
            pathdict[onm] = [[o.position[0], o.position[1], o.rotation, o.velocity[0], o.velocity[1]]]
    while running:
        gameworld.step(step_size)
        t += step_size
        for onm in tracknames:
            pathdict[onm].append([gameworld.objects[onm].position[0], gameworld.objects[onm].position[1], gameworld.objects[onm].rotation, gameworld.objects[onm].velocity[0], gameworld.objects[onm].velocity[1]])
        if gameworld.check_end() or (t >= maxtime):
            running = False
    return pathdict, gameworld.check_end(), t


"""
Run the game and keep track of object vertices over time

Returns tuple of:
    Dict[(obj_name, List)]: a dictionary with keys being the names of each of the non-static objects in the world. Each entry is a list of lists, representing the updated vertices (in world coordinates) at each timestep)
    bool: whether the goal was accomplished
    float: the time at which the world was stopped (due to solution or timeout)
"""
def get_geom_path(gameworld: VTWorld,
             maxtime: float=20.,
             step_size: float=0.1
             ) -> Tuple[Dict, bool, float]:
    running = True
    t = 0
    pathdict = dict()
    tracknames = []
    def togeom(o):
        if o.type == 'Poly':
            return o.vertices
        elif o.type == 'Ball':
            return [o.position, o.radius]
        elif o.type == 'Container' or o.type == 'Compound':
            return o.polys
        else:
            raise Exception('Shape type "' + o.type + '" not found')
    for onm, o in gameworld.objects.items():
        if not o.is_static():
            tracknames.append(onm)
            pathdict[onm] = [[o.position, o.rotation, o.velocity]]
    while running:
        gameworld.step(step_size)
        t += step_size
        for onm in tracknames:
            o = gameworld.objects[onm]
            pathdict[onm].append([o.type, togeom(o)])
        if gameworld.check_end() or (t >= maxtime):
            running = False
    return pathdict, gameworld.check_end(), t



"""
Run the game and keep track of object positions

Returns tuple of:
    Dict[(obj_name, List[(x, y)])]: a dictionary with keys being the names of each of the non-static objects in the world. Each entry is a list of positions of that object at each of the timesteps
    List[TBD]: NEED TO LOOK INTO COLLISION EVENTS AGAIN
    bool: whether the goal was accomplished
    float: the time at which the world was stopped (due to solution or timeout)
"""
def get_collisions(gameworld: VTWorld,
             maxtime: float=20.,
             step_size: float=0.1,
             collision_slop: float=0.2001
             ) -> Tuple[Dict, List, bool, float]:
    running = True
    t = 0
    pathdict = dict()
    tracknames = []
    for onm, o in gameworld.objects.items():
        if not o.is_static():
            tracknames.append(onm)
            pathdict[onm] = [o.position]
    while running:
        gameworld.step(step_size)
        t += step_size
        for onm in tracknames:
            pathdict[onm].append(gameworld.objects[onm].position)
        if gameworld.checkEnd() or (t >= maxtime):
            running = False
    collisions = filter_collision_events(gameworld.collision_events,
                                         collision_slop)
    return pathdict, collisions, gameworld.check_end(), t

class CollisionError(Exception):

    def __init__(self):
        self.message = "Illegal object placement"
        super().__init__(self.message)

    def __str__(self):
        return self.message
