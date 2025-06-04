from typing import Tuple, Annotated, Dict, List
from ..world.abstracts import VTObject, VTCond_Base
import pymunk as pm
import numpy as np
import scipy.spatial as sps
import re
import operator
import copy

def _euclidDist(p1: Tuple[float, float],
               p2: Tuple[float, float]):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def object_bounding_box(object: VTObject) -> Tuple[Tuple[float, float]]:
    """Returns the bounding box of an object

    Args:
        object (VTObject): a Virtual Tools world VTObject to calculate the bounding box of

    Returns:
        Tuple[Tuple[float, float]]: A set of ((right, bottom), (left, top)) coordinates of the bounding box. Returns None if the object is not a Ball, Poly, Container, or Compound type
    """    
    bb = [0,0]
    if object.type == 'Ball':
        bb[0] = [object.position[0] - object.radius, object.position[1] - object.radius]
        bb[1] = [object.position[0] + object.radius, object.position[1] + object.radius]
    elif object.type == 'Poly' or object.type == 'Container':
        vert_x = [vert[0] for vert in object.vertices]
        vert_y = [vert[1] for vert in object.vertices]
        bb[0] = [min(vert_x), min(vert_y)]
        bb[1] = [max(vert_x), max(vert_y)]
    elif object.type == 'Compound':
        vert_x = [vert[0] for o in object.polys for vert in o ]
        vert_y = [vert[1] for o in object.polys for vert in o ]
        bb[0] = [min(vert_x), min(vert_y)]
        bb[1] = [max(vert_x), max(vert_y)]
    else:
        bb = None
    return bb


def distance_to_object(object: VTObject,
                       point: Tuple[float, float]) -> float:
    """Returns the minimum distance between a point and the center of an object. Except for containers... then the minimum distance between the point and the container opening
    
    However this appears to be broken and depends on a function that no longer exists. I really hope this isn't needed anywhere... but if it is, please fix the `line_to_point_dist` function call!

    Args:
        object (VTObject): a Virtual Tools VTObject
        point (Tuple[float, float]): the point to calculate the distance to

    Raises:
        NotImplementedError: always

    Returns:
        float: the distance between the object and point in world units
    """    
    raise NotImplementedError('distance_to_object seems to be broken; please do not call')
    
    if object.type != 'Container':
        return _euclidDist(object.position, point)
    else:
        wall_list = object.seglist
        wall_opening = wall_list[0]
        wall_closing = wall_list[-1]

        distance = line_to_point_dist(wall_opening, wall_closing, point)
        return distance
    
def filter_collision_events(eventlist, slop_time = .2):
    begin_list = {}
    last_list = {}
    col_list = {}
    col_list_beg = {}
    output_events = []

    for o1,o2,tp,tm,ci in eventlist:
        if o2 < o1:
            tmp = o2
            o2 = o1
            o1 = tmp
            # Also need to swap the normals
            ci[0] = -ci[0]
        comb = re.sub('__', '_', o1+"_"+o2).strip('_') # Filtering for objects that start with _
        if tp == 'begin':
            # We have already seen them disconnect
            if comb in last_list.keys():
                # Long break since last time they were connected
                if tm - last_list[comb] > slop_time:
                    try:
                        output_events.append([o1,o2,begin_list[comb],last_list[comb], col_list[comb]])
                    except:
                        output_events.append([o1, o2, 0.1, last_list[comb], col_list[comb]])

                    del last_list[comb]
                    del col_list[comb]
                    begin_list[comb] = tm
                    col_list_beg[comb] = ci
                # Short break since connection
                else:
                    del last_list[comb]
                    del col_list[comb]
            # We have not yet seen them disconnect -- so they have never been together
            else:
                begin_list[comb] = tm
                col_list_beg[comb] = ci
        elif tp == 'end':
            last_list[comb] = tm
            col_list[comb] = col_list_beg[comb]

    # Clear out disconnects that never reconnect
    for comb, tm in last_list.items():
        o1, o2 = comb.split('_')
        try:
            output_events.append([o1,o2,begin_list[comb], last_list[comb], col_list[comb]])
            del begin_list[comb]
            del col_list_beg[comb]
        except:
            # Sometimes beginning touch doesn't show up on Kelsey's computer...
            output_events.append([o1,o2,0.1, last_list[comb], col_list[comb]])


    # Add in the items still in contact
    for comb, tm in begin_list.items():
        o1, o2 = comb.split('_')
        output_events.append([o1,o2,tm,None,col_list_beg[comb]])

    return sorted(output_events, key=operator.itemgetter(2))

def strip_goal(worlddict):
    wd = copy.deepcopy(worlddict)
    wd['objects']['FAKE_GOAL_7621895'] = {
        "type": "Goal",
        "color": "green",
        "density": 0,
        "vertices": [[-10, -10], [-10, -5], [-5, -5], [-5, -10]]
    }
    wd['objects']['FAKE_BALL_213232'] = {
        "type": "Ball",
        "color": "red",
        "density": 1,
        "position": [-100, -10],
        "radius": 2
    }
    wd['gcond'] = {
        "type": "SpecificInGoal",
        "goal": 'FAKE_GOAL_7621895',
        "obj": 'FAKE_BALL_213232',
        "duration": 2000
    }
    return wd
