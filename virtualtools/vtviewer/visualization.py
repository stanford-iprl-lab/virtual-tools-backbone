import pymunk as pm
import pygame as pg
import numpy as np
from ..world import VTWorld, load_vt_from_dict
from ..world.constants import DEFAULT_COLOR, DEFAULT_GOAL_COLOR
from ..world.object import VTObject, VTPoly, VTBall, VTSeg, VTContainer, VTCompound, VTGoal, VTBlocker


WHITE = (255, 255, 255, 255)

def _lighten_rgb(rgba, amt=.2):
    assert 0 <= amt <= 1, "Lightening must be between 0 and 1"
    r = int(255- ((255-rgba[0]) * (1-amt)))
    g = int(255- ((255-rgba[1]) * (1-amt)))
    b = int(255- ((255-rgba[2]) * (1-amt)))
    if len(rgba) == 3:
        return (r, g, b)
    else:
        return (r, g, b, rgba[3])

def _draw_line_gradient(start, end, steps, rgba, surf):
    diffs = np.array(end) - np.array(start)
    dX = (end[0] - start[0]) / steps
    dY = (end[1] - start[1]) / steps

    points = np.array(start) + np.array([[dX,dY]])*np.array([range(0,steps),]*2).transpose()
    cols = [_lighten_rgb(rgba, amt=0.9*step/steps) for step in range(0, steps)]
    for i, point in enumerate(points[:-1]):
        pg.draw.line(surf, cols[i], point, points[i+1], 3)
    return surf

def _filter_unique(mylist):
    newlist = []
    for ml in mylist:
        if ml not in newlist:
            newlist.append(ml)
    return newlist

def _draw_obj(o, s, makept, lighten_amt=0):
    if o.type == 'Poly':
        vtxs = [makept(v) for v in o.vertices]
        col = _lighten_rgb(o.color, lighten_amt)
        pg.draw.polygon(s, col, vtxs)
    elif o.type == 'Ball':
        pos = makept(o.position)
        rad = int(o.radius)
        col = _lighten_rgb(o.color, lighten_amt)
        pg.draw.circle(s, col, pos, rad)
        # Draw small segment that adds a window
        rot = o.rotation
        mixcol = [int((3.*oc + 510.)/5.) for oc in o.color]
        mixcol = _lighten_rgb(mixcol, lighten_amt)
        for radj in range(5):
            ru = radj*np.pi / 2.5 + rot
            pts = [(.65*rad*np.sin(ru) + pos[0], .65*rad*np.cos(ru) + pos[1]),
                   (.7 * rad * np.sin(ru) + pos[0], .7 * rad * np.cos(ru) + pos[1]),
                   (.7 * rad * np.sin(ru+np.pi/20.) + pos[0], .7 * rad * np.cos(ru+np.pi/20.) + pos[1]),
                   (.65 * rad * np.sin(ru+np.pi/20.) + pos[0], .65 * rad * np.cos(ru+np.pi/20.) + pos[1])]
            pg.draw.polygon(s, mixcol, pts)
    elif o.type == 'Segment':
        pa, pb = [makept(p) for p in o.points]
        col = _lighten_rgb(o.color, lighten_amt)
        pg.draw.line(s, col, pa, pb, o.r)
    elif o.type == 'Container':
        for poly in o.polys:
            ocol = col = _lighten_rgb(o.outer_color, lighten_amt)
            vtxs = [makept(p) for p in poly]
            pg.draw.polygon(s, ocol, vtxs)
        garea = [makept(p) for p in o.vertices]
        if o.inner_color is not None:
            acolor = (o.inner_color[0], o.inner_color[1], o.inner_color[2], 128)
            acolor = _lighten_rgb(acolor, lighten_amt)
            pg.draw.polygon(s, acolor, garea)
    elif o.type == 'Compound':
        col = _lighten_rgb(o.color, lighten_amt)
        for poly in o.polys:
            vtxs = [makept(p) for p in poly]
            pg.draw.polygon(s, col, vtxs)
    elif o.type == 'Goal':
        if o.color is not None:
            col = _lighten_rgb(o.color, lighten_amt)
            vtxs = [makept(v) for v in o.vertices]
            pg.draw.polygon(s, col, vtxs)
    else:
        print ("Error: invalid object type for drawing:", o.type)

def draw_tool(toolverts, size=[90, 90], color=(0,0,0,255)):
    s = pg.Surface(size)
    s.fill(WHITE)
    def _shift_pt(p):
        rc_p_x = p[0] + size[0] / 2
        rc_p_y = size[1] / 2 - p[1]
        return [int(rc_p_x), int(rc_p_y)]
    
    for poly in toolverts:
        vtxs = [_shift_pt(p) for p in poly]
        pg.draw.polygon(s, color, vtxs)
    return s


def draw_world(world, background_only=False, lighten_placed=False):
    s = pg.Surface(world.dims)
    s.fill(world.bk_col)

    def makept(p):
        return [int(i) for i in world._invert(p)]

    for b in world.blockers.values():
        drawpts = [makept(p) for p in b.vertices]
        pg.draw.polygon(s, b.color, drawpts)

    for o in world.objects.values():
        if not background_only or o.is_static():
            if lighten_placed and o.name == 'PLACED':
                _draw_obj(o, s, makept, .5)
            else:
                _draw_obj(o, s, makept)
    return s





def makeImageArray(worlddict, path, sample_ratio=1):
    world = loadFromDict(worlddict)
    #pg.init()
    images = [drawWorld(world)]
    if len(path[(list(path.keys())[0])]) == 2:
        nsteps = len(path[list(path.keys())[0]][0])
    else:
        nsteps = len(path[list(path.keys())[0]])

    for i in range(1,nsteps,sample_ratio):
        for onm, o in world.objects.items():
            if not o.isStatic():
                if len(path[onm])==2:
                    o.setPos(path[onm][0][i])
                    o.setRot(path[onm][1][i])
                else:
                    o.setPos(path[onm][i][0:2])
                    o.setRot(path[onm][i][2])
        images.append(drawWorld(world))
    return images

def drawPathSingleImage(worlddict, path, pathSize=3, lighten_amt=.5):
    # set up the drawing
    world = loadFromDict(worlddict)
    #pg.init()
    #sc = pg.display.set_mode(world.dims)
    sc = drawWorld(world, backgroundOnly=True)
    def makept(p):
        return [int(i) for i in world._invert(p)]
    # draw the paths in the background
    for onm, o in world.objects.items():
        if not o.isStatic():
            if o.type == 'Container':
                col = o.outer_color
            else:
                col = o.color
            pthcol = _lighten_rgb(col, lighten_amt)
            if len(path[onm]) == 2:
                poss = path[onm][0]
            else:
                poss = [path[onm][i][0:2] for i in range(0, len(path[onm]))]
            #for p in poss:
            #    pg.draw.circle(sc, pthcol, makept(p), pathSize)
            pts = _filter_unique([makept(p) for p in poss])
            if len(pts) > 1:
                pg.draw.lines(sc, pthcol, False, pts, pathSize)
    # Draw the initial tools, lightened
    for onm, o in world.objects.items():
        if not o.isStatic():
            _draw_obj(o, sc, makept, lighten_amt=lighten_amt)
    # Draw the end tools
    for onm, o in world.objects.items():
        if not o.isStatic():
            if len(path[onm])==2:
                o.setPos(path[onm][0][-1])
                o.setRot(path[onm][1][-1])
            else:
                o.setPos(path[onm][-1][0:2])
            _draw_obj(o, sc, makept)
    #pg.display.flip()
    #pg.quit()
    return sc

def drawMultiPathSingleImage(worlddict, path_set, pathSize=3, lighten_amt=.5):
    # set up the drawing
    world = loadFromDict(worlddict)
    #pg.init()
    #sc = pg.display.set_mode(world.dims)
    sc = drawWorld(world, backgroundOnly=True)
    def makept(p):
        return [int(i) for i in world._invert(p)]
    # draw the paths in the background
    for path in path_set:
        for onm, o in world.objects.items():
            if not o.isStatic():
                if o.type == 'Container':
                    col = o.outer_color
                else:
                    col = o.color
                pthcol = _lighten_rgb(col, lighten_amt)
                if len(path[onm]) == 2:
                    poss = path[onm][0]
                else:
                    poss = [path[onm][i][0:2] for i in range(0, len(path[onm]))]
                #for p in poss:
                #    pg.draw.circle(sc, pthcol, makept(p), pathSize)
                pts = _filter_unique([makept(p) for p in poss])
                if len(pts) > 1:
                    pg.draw.lines(sc, pthcol, False, pts, pathSize)
    # Draw the initial tools, lightened
    for onm, o in world.objects.items():
        if not o.isStatic():
            _draw_obj(o, sc, makept, lighten_amt=lighten_amt)
    # Draw the end tools
    for path in path_set:
        for onm, o in world.objects.items():
            if not o.isStatic():
                if len(path[onm])==2:
                    o.setPos(path[onm][0][-1])
                    o.setRot(path[onm][1][-1])
                else:
                    o.setPos(path[onm][-1][0:2])
                _draw_obj(o, sc, makept)
    #pg.display.flip()
    #pg.quit()
    return sc
