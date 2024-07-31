from ..world import VTWorld
from ..interfaces import VTInterface, ToolPicker
from .movies import demonstrate_path
from .visualization import draw_world, draw_tool
import pygame as pg
from typing import Dict, List, Tuple


def demonstrate_action(interface: VTInterface,
                       action: Dict,
                       hz: float=None):
    hz = hz or 1./interface.bts
    # Get the paths
    path, _, _ = interface.observe_full_path(action)
    # Get the world dictionary with the action added
    actworld = interface.place(action, interface.world)
    # Make the movie
    demonstrate_path(actworld.to_dict(), path, hz)


def draw_TP_actions(tp: ToolPicker, actionlist: List, connect_lines: bool=False, radius: int=5, tooldims: List[int] =[90, 90]):
    # Make the basic screen from the tp
    wdict = tp.worlddict
    dims = wdict['dims']
    world = tp.world
    wscreen = draw_world(world)
    
    TCOLORS = [
        (255, 255, 0, 255),
        (0, 255, 255, 255),
        (255, 0, 255, 255)
    ]
    
    # Make the tool screens
    tools = tp.tools
    tnames = tp.toolnames
    ntools = len(tnames)
    if ntools > 3:
        raise NotImplementedError("Cannot draw more than 3 tools currently; world has " + str(ntools))
    
    
    toolscreens = [draw_tool(tools[tnames[i]], size=tooldims, color=TCOLORS[i]) for i in range(ntools)]
    
    # Draw the placements on the world screen
    for a in actionlist:
        tp._check_action(a)
        pos = world._invert(a['position'])
        c = TCOLORS[tnames.index(a['tool'])]
        pg.draw.circle(wscreen, c, pos, radius)
    
    # Compose together
    swidth = dims[0] + tooldims[0] + 15 # Total width
    
    tot_toolsc_height = ntools * (tooldims[1] + 6) + (ntools - 1) * 20
    cur_ypos = int((dims[1] - tot_toolsc_height) / 2)
    tool_xpos = swidth - tooldims[0] - 6
    
    screen = pg.Surface((swidth, dims[1]))
    screen.blit(wscreen, [0,0])
    
    for i, ts in enumerate(toolscreens):
        pg.draw.rect(screen, TCOLORS[i], pg.Rect(tool_xpos, cur_ypos, swidth, tooldims[1] + 6))
        screen.blit(ts, [tool_xpos + 3, cur_ypos + 3])
        cur_ypos += tooldims[1] + 26
        
    return screen
    
    
    
    
    