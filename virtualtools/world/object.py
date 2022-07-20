import pymunk as pm
import numpy as np
from .constants import *
from .abstracts import VTObject
from ..helpers import *
import pdb
import copy

__all__ = ['VTPoly','VTBall','VTSeg','VTContainer',
           'VTCompound','VTGoal','VTBlocker']

class VTPoly(VTObject):

    def __init__(self,name,space,vertices,density = DEFAULT_DENSITY,elasticity=DEFAULT_ELASTICITY,
                 friction=DEFAULT_FRICTION,color=DEFAULT_COLOR):
        VTObject.__init__(self, name, "Poly", space, color, density, friction, elasticity)

        #vertices = map(lambda v: map(float, v), vertices)
        vertices = [[float(vp) for vp in v] for v in vertices]
        loc = centroid_for_poly(vertices)
        self.area = area_for_poly(vertices)
        mass = density * self.area

        if mass == 0:
            self._cpShape = pm.Poly(space.static_body, vertices)
            self._cpShape.elasticity = elasticity
            self._cpShape.friction = friction
            self._cpShape.collision_type = COLTYPE_SOLID
            self._cpShape.name = name
            space.add(self._cpShape)
        else:
            recenter_poly(vertices)
            imom = pm.moment_for_poly(mass, vertices)
            self._cpBody = pm.Body(mass, imom)
            self._cpShape = pm.Poly(self._cpBody, vertices)
            self._cpShape.elasticity = elasticity
            self._cpShape.friction = friction
            self._cpShape.collision_type = COLTYPE_SOLID
            self._cpShape.name = name
            self._cpBody.position = loc
            space.add(self._cpBody, self._cpShape)

    def get_vertices(self):
        if self.is_static():
            verts = [np.array(v) for v in self._cpShape.get_vertices()]
            verts.reverse()
        else:
            verts = []
            pos = self.position
            rot = self.rotation
            for v in self._cpShape.get_vertices():
                vcp = v.rotated(rot) + pos
                verts = [np.array(vcp)] + verts
                #verts.append(np.array(vcp))
        return verts

    def get_area(self):
        return self.area

    # Overwrites for static polygons too
    def get_pos(self):
        if self.is_static():
            vertices = [[float(vp) for vp in v] for v in self.vertices]
            return centroid_for_poly(vertices)
        else:
            return np.array(self._cpBody.position)

    def distance_from_point(self, point):
        d, _ = self._cpShape.point_query(point)
        return d

    vertices = property(get_vertices)


class VTBall(VTObject):

    def __init__(self, name, space, position, radius, density = DEFAULT_DENSITY,
                 elasticity=DEFAULT_ELASTICITY, friction=DEFAULT_FRICTION,color=DEFAULT_COLOR):
        VTObject.__init__(self, name, "Ball", space, color, density, friction, elasticity)
        area = np.pi * radius * radius
        mass = density * area
        imom = pm.moment_for_circle(mass, 0, radius)
        if mass == 0:
            self._cpShape = pm.Circle(space.static_body, radius, position)
            self._cpShape.elasticity = elasticity
            self._cpShape.friction = friction
            self._cpShape.collision_type = COLTYPE_SOLID
            self._cpShape.name = name
            space.add(self._cpShape)
        else:
            self._cpBody = pm.Body(mass, imom)
            self._cpShape = pm.Circle(self._cpBody, radius, (0,0))
            self._cpShape.elasticity = elasticity
            self._cpShape.friction = friction
            self._cpShape.collision_type = COLTYPE_SOLID
            self._cpShape.name = name
            self._cpBody.position = position
            space.add(self._cpBody, self._cpShape)

    def get_radius(self):
        return self._cpShape.radius

    def get_area(self):
        r = self.get_radius()
        return np.pi * r * r

    # Overwrites for static circles too
    def get_pos(self):
        if self.is_static():
            return self._cpShape.offset
        else:
            return self._cpBody.position

    radius = property(get_radius)
    area = property(get_area)


class VTSeg(VTObject):

    def __init__(self, name, space, p1, p2, width, density = DEFAULT_DENSITY,
                 elasticity=DEFAULT_ELASTICITY, friction=DEFAULT_FRICTION,color=DEFAULT_COLOR):
        VTObject.__init__(self, name, "Segment", space, color, density, friction, elasticity)
        self.r = width / 2
        area = area_for_segment(p1, p2, self.r)
        self.area = area
        mass = density*area
        pos = pm.Vec2d((p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.)
        self.pos = pos
        if mass == 0:
            self._cpShape = pm.Segment(space.static_body, p1, p2, self.r)
            self._cpShape.elasticity = elasticity
            self._cpShape.friction = friction
            self._cpShape.collision_type = COLTYPE_SOLID
            self._cpShape.name = name
            space.add(self._cpShape)
        else:
            pos = pm.Vec2d((p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.)
            v1 = pm.Vec2d(p1) - pos
            v2 = pm.Vec2d(p2) - pos
            imom = pm.moment_for_segment(mass, v1, v2, 0)
            self._cpBody = pm.Body(mass, imom)
            self._cpShape = pm.Segment(self._cpBody, v1, v2, self.r)
            self._cpShape.elasticity = elasticity
            self._cpShape.friction = friction
            self._cpShape.collision_type = COLTYPE_SOLID
            self._cpShape.name = name
            self._cpBody.position = pos
            space.add(self._cpBody, self._cpShape)

    def get_points(self):
        v1 = self._cpShape.a
        v2 = self._cpShape.b
        if self.is_static():
            p1 = np.array(v1)
            p2 = np.array(v2)
        else:
            pos = self.get_pos()
            rot = self.get_rot()
            p1 = np.array(pos + v1.rotated(rot))
            p2 = np.array(pos + v2.rotated(rot))
        return p1,p2

    def get_pos(self):
        if self.is_static():
            return self.pos
        else:
            return self._cpBody.position

    points = property(get_points)

class VTContainer(VTObject):

    def __init__(self,name, space, ptlist, width, density = DEFAULT_DENSITY,
                 elasticity=DEFAULT_ELASTICITY, friction=DEFAULT_FRICTION,
                 inner_color=DEFAULT_GOAL_COLOR, outer_color=DEFAULT_COLOR):
        VTObject.__init__(self, name, "Container", space, outer_color, density, friction, elasticity)
        self.inner_color = inner_color
        self.outer_color = outer_color
        self.r = width / 2

        loc = centroid_for_poly(ptlist)
        self.pos = np.array([loc.x, loc.y])
        ptlist = copy.deepcopy(ptlist)
        if density != 0:
            ptlist = recenter_poly(ptlist)
        #self.seglist = map(lambda p: pm.Vec2d(p), ptlist)
        self.seglist = [verts_to_vec2d(p) for p in ptlist]

        self._area = np.pi * self.r * self.r
        imom = 0
        for i in range(len(self.seglist)-1):
            v1 = self.seglist[i]
            v2 = self.seglist[i+1]
            larea = 2*self.r* v1.get_distance(v2)
            self._area += larea
            imom += pm.moment_for_segment(larea*density, v1, v2, 0)

        mass = density * self._area
        if mass == 0:
            uBody = space.static_body
        else:
            self._cpBody = uBody = pm.Body(mass, imom)
            space.add(self._cpBody)

        self._cpPolyShapes = []
        self.polylist = segs_to_poly(ptlist, self.r)

        for pl in self.polylist:
            pshp = pm.Poly(uBody, pl)
            pshp.elasticity = elasticity
            pshp.friction = friction
            pshp.collision_type = COLTYPE_SOLID
            pshp.name = name
            self._cpPolyShapes.append(pshp)
            space.add(pshp)

        # Make sure we have ccw
        if not poly_validate(ptlist):
            ptlist.reverse()

        self._cpSensor = pm.Poly(uBody, ptlist)
        self._cpSensor.sensor = True
        self._cpSensor.collision_type = COLTYPE_SENSOR
        self._cpSensor.name = name
        space.add(self._cpSensor)
        if mass != 0:
            self._cpBody.position = loc


    def get_polys(self):
        if self.is_static():
            polys = self.polylist
        else:
            pos = self.position
            rot = self.rotation
            polys = []
            for i in range(len(self.polylist)):
                tpol = []
                for j in range(len(self.polylist[i])):
                    vj = pm.Vec2d(self.polylist[i][j])
                    tpol.append(np.array(pos + vj.rotated(rot)))
                polys.append(tpol)
        return polys

    def get_pos(self):
        if self.is_static():
            return self.pos
        else:
            p = self._cpBody.position
            return np.array([p.x, p.y])

    def get_vertices(self):
        if self.is_static():
            #return map(lambda s: np.array(s), self.seglist)
            return [np.array(s) for s in self.seglist]
        else:
            b = self._cpBody
            return [np.array(b.local_to_world(s)) for s in self.seglist]

    def point_in(self, p):
        v = pm.Vec2d(p[0], p[1])
        return self._cpSensor.point_query(v)

    def get_friction(self):
        return self._cpPolyShapes[0].friction

    def set_friction(self, val):
        assert val >= 0, "Friction must be greater than or equal to 0"
        for s in self._cpPolyShapes:
            s.friction = val

    def get_elasticity(self):
        return self._cpPolyShapes[0].elasticity

    def set_elasticity(self, val):
        assert val >= 0, "Elasticity must be greater than or equal to 0"
        for s in self._cpPolyShapes:
            s.elasticity = val

    def _expose_shapes(self):
        return self._cpPolyShapes

    def distance_from_point(self, point):
        d, _ = self._cpSensor.point_query(point)
        return d

    def distance_from_point_XY(self, point):
        d, info = self._cpSensor.point_query(point)
        return point - info.point

    def get_area(self):
        return self._area


    polys = property(get_polys)
    vertices = property(get_vertices)
    friction = property(get_friction, set_friction)
    elasticity = property(get_elasticity, set_elasticity)
    area = property(get_area)


class VTCompound(VTObject):

    def __init__(self, name, space, polygons, density = DEFAULT_DENSITY,
                 elasticity=DEFAULT_ELASTICITY, friction=DEFAULT_FRICTION,color=DEFAULT_COLOR):
        VTObject.__init__(self, name, "Compound", space, color, density, friction, elasticity)

        self._area = 0
        self.polylist = []
        self._cpShapes = []
        # If it's static, add polygons inplace
        if density == 0:
            polyCents = []
            areas = []
            for vertices in polygons:
                polyCents.append(centroid_for_poly(vertices))
                areas.append(area_for_poly(vertices))
                sh = pm.Poly(space.static_body, vertices)
                sh.elasticity = elasticity
                sh.friction = friction
                sh.collision_type = COLTYPE_SOLID
                sh.name = name
                space.add(sh)
                self._cpShapes.append(sh)
                self.polylist.append([pm.Vec2d(p) for p in vertices])

            gx = gy = 0
            for pc, a in zip(polyCents, areas):
                gx += pc[0] * a
                gy += pc[1] * a
                self._area += a
            gx /= self._area
            gy /= self._area
            loc = pm.Vec2d(gx, gy)
            self.pos = np.array([gx, gy])

        else:
            polyCents = []
            areas = []
            for i in range(len(polygons)):
                vertices = polygons[i]
                polyCents.append(centroid_for_poly(vertices))
                vertices = recenter_poly(vertices)
                polygons[i] = vertices
                areas.append(area_for_poly(vertices))
            gx = gy = 0
            for pc, a in zip(polyCents, areas):
                gx += pc[0] * a
                gy += pc[1] * a
                self._area += a
            gx /= self._area
            gy /= self._area
            loc = pm.Vec2d(gx, gy)
            imom = 0
            for pc, a, verts in zip(polyCents, areas, polygons):
                pos = pm.Vec2d(pc[0] - loc.x, pc[1] - loc.y)
                imom += pm.moment_for_poly(density*a, vertices, pos)
                rcverts = [pm.Vec2d(p[0]+pos.x, p[1]+pos.y) for p in verts]
                self._cpShapes.append(pm.Poly(None, rcverts))
                self.polylist.append(rcverts)
            mass = self._area*density
            self._cpBody = pm.Body(mass, imom)
            space.add(self._cpBody)
            for sh in self._cpShapes:
                sh.body = self._cpBody
                sh.elasticity = elasticity
                sh.friction = friction
                sh.collision_type = COLTYPE_SOLID
                sh.name = name
                space.add(sh)
            self._cpBody.position = loc

    def get_polys(self):
        if self.is_static():
            rpolys = []
            for poly in self.polylist:
                rpolys.append([np.array(p) for p in poly])
            return rpolys
        else:
            pos = self.position
            rot = self.rotation
            rpolys = []
            for poly in self.polylist:
                rpolys.append([np.array(p.rotated(rot) + pos) for p in poly])
            return rpolys

    def get_area(self):
        return self._area

    def get_friction(self):
        return self._cpShapes[0].friction

    def set_friction(self, val):
        assert val >= 0, "Friction must be greater than or equal to 0"
        for s in self._cpShapes:
            s.friction = val

    def get_elasticity(self):
        return self._cpShapes[0].elasticity

    def set_elasticity(self, val):
        assert val >= 0, "Elasticity must be greater than or equal to 0"
        for s in self._cpShapes:
            s.elasticity = val

    def _expose_shapes(self):
        return self._cpShapes

    def get_pos(self):
        if self.is_static():
            return self.pos
        else:
            p = self._cpBody.position
            return np.array([p.x, p.y])

    polys = property(get_polys)
    friction = property(get_friction, set_friction)
    elasticity = property(get_elasticity, set_elasticity)
    area = property(get_area)

    def distance_from_point(self, point):
        dists = [s.point_query(point-s.body.position)[0] for s in self._cpBody.shapes]
        return min(dists)

    def distance_from_point_XY(self, point):
        dists = [point - (s.point_query(point-s.body.position)[1].point+s.body.position) for s in self._cpBody.shapes]
        min_s = min(dists, key = lambda t: t[0])
        #pdb.set_trace()
        #min_s = list(self._cpBody.shapes)[idx]
        return min_s


class VTGoal(VTObject):

    def __init__(self, name, space, vertices, color):
        VTObject.__init__(self, name, "Goal", space, color, 0, 0, 0)
        self._cpShape = pm.Poly(space.static_body, vertices)
        self._cpShape.sensor = True
        self._cpShape.collision_type = COLTYPE_SENSOR
        self._cpShape.name = name
        space.add(self._cpShape)
        loc = centroid_for_poly(vertices)
        self.pos = np.array([loc.x, loc.y])

    def get_vertices(self):
        verts = [np.array(v) for v in self._cpShape.get_vertices()]
        verts.reverse()
        return verts

    def point_in(self,p):
        v = pm.Vec2d(p[0], p[1])
        return self._cpShape.point_query(v)

    def get_pos(self):
        return self.pos

    vertices = property(get_vertices)

class VTBlocker(VTObject):

    def __init__(self, name, space, vertices, color):
        VTObject.__init__(self, name, "Blocker", space, color, 0, 0, 0)
        self._cpShape = pm.Poly(space.static_body, vertices)
        self._cpShape.sensor = True
        self._cpShape.collision_type = COLTYPE_BLOCKED
        self._cpShape.name = name
        space.add(self._cpShape)
        loc = centroid_for_poly(vertices)
        self.pos = np.array([loc.x, loc.y])

    def get_vertices(self):
        verts = [np.array(v) for v in self._cpShape.get_vertices()]
        verts.reverse()
        return verts

    def point_in(self, p):
        v = pm.Vec2d(p[0], p[1])
        return self._cpShape.point_query(v)

    def get_pos(self):
        return self.pos

    vertices = property(get_vertices)
