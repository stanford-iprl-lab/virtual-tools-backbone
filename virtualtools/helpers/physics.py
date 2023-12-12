from typing import Tuple, Annotated, Dict, List
from ..world.abstracts import VTObject, VTCond_Base
import pymunk as pm
import numpy as np
import scipy.spatial as sps

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
