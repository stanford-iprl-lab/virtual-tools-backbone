from abc import ABC, abstractmethod
import numpy as np
import pymunk as pm

# Abstract classes for VTObjects And VTCond_Base
# Used to describe arbitrary objects/conditions in other helpers

class VTObject(ABC):

    def __init__(self, name, otype, space, color, density, friction, elasticity):
        assert otype in ['Ball','Poly','Segment','Container', 'Compound','Goal','Blocker'], \
            "Illegal 'type' of object"
        from ..helpers.misc import word_to_color
        self.name = name
        self.type = otype
        self.space = space
        self.color = word_to_color(color)
        self.density = density
        self._cpBody = None
        self._cpShape = None

    def is_static(self):
        return self._cpBody is None

    def get_pos(self):
        assert not self.is_static(), "Static bodies do not have a position"
        p = self._cpBody.position
        return np.array([p.x, p.y])

    def set_pos(self, p):
        assert not self.is_static(), "Static bodies do not have a position"
        assert len(p) == 2, "Setting position requires vector of length 2"
        self._cpBody.position = pm.Vec2d(p[0], p[1])

    def get_vel(self):
        assert not self.is_static(), "Static bodies do not have a velocity"
        v = self._cpBody.velocity
        return np.array([v.x, v.y])

    def set_vel(self, v):
        assert not self.is_static(), "Static bodies do not have a velocity"
        assert len(v) == 2, "Setting position requires vector of length 2"
        self._cpBody.velocity = pm.Vec2d(v[0], v[1])

    def get_rot(self):
        assert not self.is_static(), "Static bodies do not have a rotation"
        return self._cpBody.angle

    def set_rot(self, a):
        assert not self.is_static(), "Static bodies do not have a rotation"
        self._cpBody.angle = a

    def get_mass(self):
        if self.is_static():
            return 0
        else:
            return self._cpBody.mass

    def _expose_shapes(self):
        return [self._cpShape]

    def check_contact(self, object):
        for myshapes in self._expose_shapes():
            for oshapes in object._expose_shapes():
                if len(myshapes.shapes_collide(oshapes).points) > 0:
                    return True
        return False

    def set_mass(self, val):
        assert val > 0, "Must set a positive mass value"
        if self.is_static():
            raise Exception("Cannot set the mass of a static object")
        else:
            self._cpBody.mass = val

    def get_friction(self):
        assert self._cpShape is not None, "Shape not yet set"
        return self._cpShape.friction

    def set_friction(self, val):
        assert self._cpShape is not None, "Shape not yet set"
        assert val >= 0, "Friction must be greater than or equal to 0"
        self._cpShape.friction = val

    def get_elasticity(self):
        assert self._cpShape is not None, "Shape not yet set"
        return self._cpShape.elasticity

    def set_elasticity(self, val):
        assert self._cpShape is not None, "Shape not yet set"
        assert val >= 0, "Elasticity must be greater than or equal to 0"
        self._cpShape.elasticity = val

    def to_geom(self):
        if (self.type == "Poly"):
            return self.get_vertices()
        elif (self.type == "Ball"):
            return [self.get_pos(), self.get_radius()]
        elif self.type == "Container" or self.type == "Compound":
            return self.get_polys()
        else:
            print('not a valid object type')
            return None

    def kick(self, impulse, position, unsafe = False):
        assert not self.is_static(), "Cannot kick a static object"
        if not unsafe:
            for s in self._expose_shapes():
                if not s.point_query(position):
                    raise AssertionError("Must kick an object within the object (or set as unsafe)")
        self._cpBody.apply_impulse_at_world_point(impulse, position)

    def distance_from_point(self, point):
        d, _ = self._cpShape.point_query(point)
        return d

    def distance_from_point_XY(self, point):
        d, info = self._cpShape.point_query(point)
        return point - info.point

    # Add pythonic decorators
    position = property(get_pos, set_pos)
    velocity = property(get_vel, set_vel)
    rotation = property(get_rot, set_rot)
    mass = property(get_mass, set_mass)
    friction = property(get_friction, set_friction)
    elasticity = property(get_elasticity, set_elasticity)


class VTCond_Base(ABC):

    def __init__(self):
        self.goal = self.obj = self.parent = self.dur = None

    def _get_time_in(self):
        return -1

    def remaining_time(self):
        ti = self._get_time_in()
        if ti == -1:
            return None
        curtime = self.parent.time - ti
        return max(self.dur - curtime, 0)

    def is_won(self):
        return self.remaining_time() == 0

    @abstractmethod
    def attach_hooks(self):
        raise NotImplementedError("Cannot attach hooks from base condition object")
