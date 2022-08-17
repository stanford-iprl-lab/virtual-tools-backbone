from typing import Tuple, Annotated, Dict
from .vtinterface import VTInterface, place_object_by_vertex_list
from ..world import VTWorld
from geometry import check_counterclockwise

__all__ = ['VertexDrawer']

class VertexDrawer(VTInterface):
    def __init__(self,
                 worlddict: Dict,
                 basic_timestep: float=0.1,
                 maxtime: float=20.,
                 world_timestep: float=None):
        super().__init__(worlddict, basic_timestep, maxtime, world_timestep)

    @property
    def action_keys(self):
        return ['vertexlist', 'position']

    @property
    def interface_type(self):
        return "VertexDrawer"

    def place(self,
              action: Dict,
              world: VTWorld=None) -> VTWorld:
        world = world or load_vt_from_dict(self._worlddict)
        self._check_action(action)
        vertices = action['vertexlist']
        pos = action['position']
        if not check_counterclockwise(vertices):
            vertices.reverse()
            assert not check_counterclockwise(vertices), "Bad vertex definition: no consistent winding"
        world = place_object_by_vertex_list(world, vertices, pos)
        return world
