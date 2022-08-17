from typing import Tuple, Annotated, Dict, List
import pymunk as pm


__all__ = ['verts_to_vec2d', 'poly_to_vec2d']

# Translates a vertex defined as (x,y) into a pymunk vector
def verts_to_vec2d(vertex: Annotated[Tuple[float], 2]) -> pm.Vec2d:
    return pm.Vec2d(vertex[0], vertex[1])

# Translates a list of (x,y) vertices to a list of pymunk vectors
def poly_to_vec2d(vertices: List[Tuple[float, float]]) -> List[pm.Vec2d]:
    return [verts_to_vec2d(v) for v in vertices]
