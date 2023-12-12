from typing import Tuple, Annotated, Dict, List
import pymunk as pm


__all__ = ['verts_to_vec2d', 'poly_to_vec2d']

def verts_to_vec2d(vertex: Annotated[Tuple[float], 2]) -> pm.Vec2d:
    """_Translates a vertex defined as (x,y) into a pymunk vector

    Args:
        vertex (Annotated[Tuple[float], 2]): An (x,y) point

    Returns:
        pm.Vec2d: The same point defined as a pymunk Vec2d object
    """
    return pm.Vec2d(vertex[0], vertex[1])

def poly_to_vec2d(vertices: List[Tuple[float, float]]) -> List[pm.Vec2d]:
    """Translates a list of (x,y) vertices to a list of pymunk vectors

    Args:
        vertices (List[Tuple[float, float]]): a list of (x,y) vertices to translate

    Returns:
        List[pm.Vec2d]: A list of pymunk Vec2d objects with the same points
    """    
    return [verts_to_vec2d(v) for v in vertices]
