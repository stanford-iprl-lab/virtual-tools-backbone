from typing import Tuple, Annotated, Dict, List
from .vtgenerator import VTGenerator
from ..interfaces import ToolPicker
from ..world import VTWorld
import random
import numpy as np

__all__ = ['ToolPickerGenerator', 'TOOL_COLLECTION', 'resize_tool']

def _cvert_from_idx(i):
    ang = ((30-i) / 30) * 2 * np.pi
    return [20*np.cos(ang), 20*np.sin(ang)]

TOOL_COLLECTION = {
    'circle': [[_cvert_from_idx(i) for i in range(30)]],
    'diamond': [[[-20,0],[0,20],[20,0],[0,-20]]],
    'short_push_r': [[[-30,-15],[-30,15],[30,15],[0,-15]]],
    'short_push_l': [[[-30, 15], [30, 15], [30, -15], [0, -15]]],
    'tall_push_r': [[[-30,-25],[-30,25],[30,25],[30,5],[0,-25]]],
    'tall_push_l': [[[-30, 5], [-30, 25], [30, 25], [30, -25], [0, -25]]],
    'horizontal': [[[-40,-5],[-40,5],[40,5],[40,-5]]],
    'vertical': [[[-5, -40], [-5, 40], [5, 40], [5, -40]]],
    'block': [[[-20, -20], [-20, 20], [20, 20], [20, -20]]],
    'wedge': [[[-20, 10], [20, 10], [0, -10]]],
    'pyramid_down': [[[-5,-35], [-40,35], [40,35], [5,-35]]],
    'pyramid_up': [[[-40,-35],[-5,35],[5,35],[40,-35]]],
    'hook': [[[25,10],[37,10],[37,0],[25,0]],
         [[25,0],[37,0],[37,-5],[25,-5]],
         [[-25,10],[25,10],[25,0],[-25,0]],
         [[-37,10],[-25,10],[-25,0],[-37,0]],
         [[-37,0],[-25,0],[-25,-30],[-37,-30]]],
    'diag_r': [[[-30,-40],[-40,-40],[30,40],[40,40]]],
    'cross': [[[-5, -5], [-5, 5], [5,5], [5, -5]],
          [[-20, -5], [-20, 5], [-5, 5], [-5, -5]],
          [[-5, 5], [-5, 20], [5, 20], [5, 5]],
          [[5, -5], [5, 5], [20, 5], [20, -5]],
          [[-5, -20], [-5, -5], [5, -5], [5, -20]]],
    'opener': [[[-30, 15], [-30, 30], [0, 30], [0, 15]],
           [[0, 15], [0, 30], [30, 30], [30, 15]],
           [[0, -30], [0, 15], [30, 15], [30, -30]]],
    'opener_side': [[[-30, -30], [-30, 0], [-15, 0], [-15, -30]],
               [[-30, 0], [-30, 30], [-15, 30], [-15, 0]],
               [[-15, 0], [-15, 30], [30, 30], [30, 0]]],
    'triangle_up': [[[-15, -5], [0, 10], [15, -5]]],
    'slope_l': [[[-20, -20], [20, 20], [20, -20]]],
    'slope_r': [[[-20, -20], [-20, 20], [20, -20]]]
}

def resize_tool(tool: List, scale_x: float = 1.0, scale_y: float = 1.0) -> List:
    newtool = []
    for pi in range(len(tool)):
        poly = tool[pi]
        newtool.append(
            [[poly[i][0] * scale_x, poly[i][1] * scale_y] for i in range(len(poly))]
        )
    return newtool

class ToolPickerGenerator(VTGenerator):
    
    def generate_random_placement(self, toolpicker: ToolPicker):
        rtool = random.select(toolpicker.toolnames)
        w = toolpicker.worlddict
        maxx = w['dims'][0]
        maxy = w['dims'][1]
        rpos = [random.randint(1, maxx - 1), radnom.randint(1, maxy - 1)]
        return {
            'tool': rtool,
            'position': rpos
        }