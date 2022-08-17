from typing import Tuple, Annotated, Dict, List
import pymunk as pm
import numpy as np
from .pmhelp import verts_to_vec2d
from geometry import lines_intersect


# Returns the area of a segment defined by two endpoint vertices and a thickness (r)
def area_for_segment(a: Tuple[float, float],
                     b: Tuple[float, float],
                     r: float) -> float:
    va = verts_to_vec2d(a)
    vb = verts_to_vec2d(b)
    return r * (np.pi*r + 2*va.get_distance(vb))

def _isleft(spt, ept, testpt):
    seg1 = (ept[0]-spt[0], ept[1]-spt[1])
    seg2 = (testpt[0]-spt[0], testpt[1]-spt[1])
    cross = seg1[0]*seg2[1]-seg1[1]*seg2[0]
    return cross > 0

# Transforms a set of connected line segments with a width into a set of convex hulls
def segs_to_poly(seglist: Tuple[Tuple[float, float]],
                 r: float):
    vlist = [verts_to_vec2d(v) for v in seglist]
    #vlist = list(map(lambda p: pm.Vec2d(p), seglist))
    # Start by figuring out the initial edge (ensure ccw winding)
    iseg = vlist[1] - vlist[0]
    ipt = vlist[0]
    iang = iseg.angle
    if iang <= (-np.pi / 4.) and iang >= (-np.pi * 3. / 4.):
        # Going downwards
        prev1 = (ipt.x - r, ipt.y)
        prev2 = (ipt.x + r, ipt.y)
    elif iang >= (np.pi / 4.) and iang <= (np.pi * 3. / 4.):
        # Going upwards
        prev1 = (ipt.x + r, ipt.y)
        prev2 = (ipt.x - r, ipt.y)
    elif iang >= (-np.pi / 4.) and iang <= (np.pi / 4.):
        # Going rightwards
        prev1 = (ipt.x, ipt.y - r)
        prev2 = (ipt.x, ipt.y + r)
    else:
        # Going leftwards
        prev1 = (ipt.x, ipt.y + r)
        prev2 = (ipt.x, ipt.y - r)

    polylist = []
    for i in range(1, len(vlist)-1):
        pi = vlist[i]
        pim = vlist[i-1]
        pip = vlist[i+1]
        sm = pim - pi
        sp = pip - pi
        # Get the angle of intersetction between two lines
        angm = sm.angle
        angp = sp.angle
        angi = (angm - angp) % (2*np.pi)
        # Find the midpoint of this angle and turn it back into a unit vector
        angn = (angp + (angi / 2.)) % (2*np.pi)
        if angn < 0:
            angn += 2*np.pi
        unitn = pm.Vec2d(np.cos(angn), np.sin(angn))
        #unitn = pm.Vec2d.unit()
        #unitn.angle = angn
        xdiff = r if unitn.x >= 0 else -r
        ydiff = r if unitn.y >= 0 else -r
        next3 = (pi.x + xdiff, pi.y + ydiff)
        next4 = (pi.x - xdiff, pi.y - ydiff)
        # Ensure appropriate winding -- next3 should be on the left of next4
        if _isleft(prev2, next3, next4):
            tmp = next4
            next4 = next3
            next3 = tmp
        polylist.append((prev1, prev2, next3, next4))
        prev1 = next4
        prev2 = next3

    # Finish by figuring out the final edge
    fseg = vlist[-2] - vlist[-1]
    fpt = vlist[-1]
    fang = fseg.angle
    if fang <= (-np.pi / 4.) and fang >= (-np.pi * 3. / 4.):
        # Coming from downwards
        next3 = (fpt.x - r, fpt.y)
        next4 = (fpt.x + r, fpt.y)
    elif fang >= (np.pi / 4.) and fang <= (np.pi * 3. / 4.):
        # Coming from upwards
        next3 = (fpt.x + r, fpt.y)
        next4 = (fpt.x - r, fpt.y)
    elif fang >= (-np.pi / 4.) and fang <= (np.pi / 4.):
        # Coming from rightwards
        next3 = (fpt.x, fpt.y - r)
        next4 = (fpt.x, fpt.y + r)
    else:
        # Coming from leftwards
        next3 = (fpt.x, fpt.y + r)
        next4 = (fpt.x, fpt.y - r)
    polylist.append((prev1, prev2, next3, next4))
    return polylist


"""
Takes in a list of (x,y) vertices, checks if drawing segments between the vertices in
order will have any intersections
"""
def any_line_intersections(vertices: List[Tuple[float, float]]) -> bool:
    assert len(vertices) > 1, "Cannot find intersections with a single point"
    if len(vertices) <= 3:
        return False # There can be no intersections with two connected line segments
    cverts = copy.deepcopy(vertices)
    cverts.append(copy.copy(cverts[0]))

    for i in range(len(vertices) - 1):
        a = cverts[i]
        b = cverts[i+1]
        for j in range(i+1, len(vertices)):
            c = cverts[j]
            d = cverts[j+1]
            if lines_intersect(a,b,c,d):
                return True
    return False
