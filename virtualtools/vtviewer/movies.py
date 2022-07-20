import pygame as pg
from pygame.constants import QUIT
from typing import Tuple, Dict, List
from ..world import VTWorld, load_vt_from_dict
from .visualization import draw_world

def demonstrate_world(world: VTWorld,
                      hz: float = 30.):
    pg.init()
    sc = pg.display.set_mode(world.dims)
    clk = pg.time.Clock()
    sc.blit(draw_world(world), (0,0))
    pg.display.flip()
    running = True
    tps = 1./hz
    clk.tick(hz)
    disp_finish = True
    while running:
        world.step(tps)
        sc.blit(draw_world(world), (0, 0))
        pg.display.flip()
        clk.tick(hz)
        for e in pg.event.get():
            if e.type == QUIT:
                running = False
        if disp_finish and world.check_end():
            print("Goal accomplished")
            disp_finish = False
    pg.quit()

def demonstrate_path(worlddict: Dict,
                   path: Dict,
                   hz: float=30.):
    world = load_vt_from_dict(worlddict)
    pg.init()
    sc = pg.display.set_mode(world.dims)
    clk = pg.time.Clock()
    sc.blit(draw_world(world), (0, 0))
    pg.display.flip()
    clk.tick(hz)
    if len(path[(list(path.keys())[0])]) == 2:
        nsteps = len(path[list(path.keys())[0]][0])
    else:
        nsteps = len(path[list(path.keys())[0]])

    for i in range(nsteps):
        for onm, o in world.objects.items():
            if not o.is_static():
                o.position = path[onm][i][0:2]
                o.rotation = path[onm][i][2]
        sc.blit(draw_world(world), (0,0))
        pg.display.flip()
        for e in pg.event.get():
            if e.type == QUIT:
                pg.quit()
                return
        clk.tick(hz)
    pg.quit()
