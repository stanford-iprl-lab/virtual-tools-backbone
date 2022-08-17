from typing import Tuple, Annotated, Dict, List
from ..world.abstracts import VTObject, VTCond_Base
import pymunk as pm
import numpy as np
import scipy.spatial as sps

def _euclidDist(p1: Tuple[float, float],
               p2: Tuple[float, float]):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Returns the bounding box
def object_bounding_box(object: VTObject):
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
                       point: Tuple[float, float]):
    if object.type != 'Container':
        return _euclidDist(object.position, point)
    else:
        wall_list = object.seglist
        wall_opening = wall_list[0]
        wall_closing = wall_list[-1]

        distance = line_to_point_dist(wall_opening, wall_closing, point)
        return distance
