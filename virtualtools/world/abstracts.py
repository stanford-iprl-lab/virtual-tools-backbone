from typing import Tuple, List, Dict
from abc import ABC, abstractmethod
import numpy as np
import pymunk as pm

# Abstract classes for VTObjects And VTCond_Base
# Used to describe arbitrary objects/conditions in other helpers  


class VTObject(ABC):

    def __init__(self, name: str, otype: str, space: pm.Space, color: str, density: float, friction: float, elasticity: float):
        """Abstract object initialization. This should never be called directly, only from child classes

        Args:
            name (str): The name flag for the object
            otype (str): The object type definition
            space (pm.Space): The pymunk space the object will be added to
            color (str): A word describing the object color (['blue', 'red', 'green', 'black', 'white', 'grey', 'gray', 'lightgrey', 'none'])
            density (float): The density of the object. Set to 0 to make this static
            friction (float): The friction of the object. Must be greater than 0
            elasticity (float): The elasticity of the object. Must be greater than 0; should be less than 1 or non-physical stuff can happen
        
        Raises:
            AssertionError: if the otype argument is not one of the defined - ['Ball','Poly','Segment','Container', 'Compound','Goal','Blocker']
        """        
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

    def is_static(self) -> bool:
        """Returns true if the object is static (density==0)

        Returns:
            bool: a flag if the object is static
        """        
        return self._cpBody is None

    def get_pos(self):
        """Gets the position of an object

        Returns:
            numpy.array: An array containing the [x,y] position of the object
            
        Raises:
            AssertionError: if called on a static object
        """        
        assert not self.is_static(), "Static bodies do not have a position"
        p = self._cpBody.position
        return np.array([p.x, p.y])

    def set_pos(self, p: Tuple[float, float]):
        """Sets the position of an object. Throws an error if the object is static
        
        Args:
            p (Tuple[float, float]): A tuple or list of length 2 defining the [x,y] coordinates to move to
            
        Raises:
            AssertionError: if called on a static or if p has a length other than 2
        """
        assert not self.is_static(), "Static bodies do not have a position"
        assert len(p) == 2, "Setting position requires vector of length 2"
        self._cpBody.position = pm.Vec2d(p[0], p[1])

    def get_vel(self):
        """Gets the velocity of an object

        Returns:
            numpy.array: An array containing the [x,y] velocity of the object as world units/s
            
        Raises:
            AssertionError: if called on a static
        """              
        assert not self.is_static(), "Static bodies do not have a velocity"
        v = self._cpBody.velocity
        return np.array([v.x, v.y])

    def set_vel(self, v: Tuple[float, float]):
        """Sets the velocity of an object

        Raises:
            AssertionError: if called on a static or if v has a length other than 2

        Args:
            v (Tuple[float, float]): the (x,y) velocity in units/s to set
        """        
        assert not self.is_static(), "Static bodies do not have a velocity"
        assert len(v) == 2, "Setting position requires vector of length 2"
        self._cpBody.velocity = pm.Vec2d(v[0], v[1])

    def get_rot(self) -> float:
        """Returns the angle of rotation of the object

        Raises:
            AssertionError: if called on a static

        Returns:
            float: the angle of rotation in radians
        """        
        assert not self.is_static(), "Static bodies do not have a rotation"
        return self._cpBody.angle

    def set_rot(self, a: float):
        """Sets the angle of rotation of the object
        
        Raises:
            AssertionError: if called on a static

        Args:
            a (float): the angle of rotation in radians
        """        
        assert not self.is_static(), "Static bodies do not have a rotation"
        self._cpBody.angle = a

    def get_mass(self) -> float:
        """Returns the mass of the object

        Returns:
            float: the mass of the object. This is 0 if the object is static
        """        
        if self.is_static():
            return 0
        else:
            return self._cpBody.mass

    def _expose_shapes(self):
        return [self._cpShape]

    def check_contact(self, object) -> bool:
        """Checks for contact between this and another object

        Args:
            object (VTObject): the other object that might contact this one

        Returns:
            bool: returns true if the objects are overlapping, false otherwise
        """        
        for myshapes in self._expose_shapes():
            for oshapes in object._expose_shapes():
                if len(myshapes.shapes_collide(oshapes).points) > 0:
                    return True
        return False

    def set_mass(self, val: float):
        """Sets the mass of an object
        
        Raises:
            AssertionError: if called on a static, or if mass is set to a negative value

        Args:
            val (float): the mass of the object
        """        
        assert val > 0, "Must set a positive mass value"
        assert not self.is_static(), "Cannot set the mass of a static object"
        self._cpBody.mass = val

    def get_friction(self) -> float:
        """Returns the friction of the object

        Raises:
            AssertionError: if no shape has been assigned to the object

        Returns:
            float: the friction
        """        
        assert self._cpShape is not None, "Shape not yet set"
        return self._cpShape.friction

    def set_friction(self, val: float):
        """Sets the friction of the object

        Raises:
            AssertionError: if no shape has been assigned to the object, or friction is set to a negative number

        Args:
            val (float): the friction
        """        
        assert self._cpShape is not None, "Shape not yet set"
        assert val >= 0, "Friction must be greater than or equal to 0"
        self._cpShape.friction = val

    def get_elasticity(self) -> float:
        """Returns the elasticity of the object
        
        Raises:
            AssertionError: if no shape has been assigned to the object

        Returns:
            float: the elasticity
        """        
        assert self._cpShape is not None, "Shape not yet set"
        return self._cpShape.elasticity

    def set_elasticity(self, val: float):
        """Sets the elasticity of an object
        
        Raises:
            AssertionError: if no shape has been assigned to the object, or if elasticity is set to a negative number

        Args:
            val (float): the elasticity
        """        
        assert self._cpShape is not None, "Shape not yet set"
        assert val >= 0, "Elasticity must be greater than or equal to 0"
        self._cpShape.elasticity = val

    def to_geom(self):
        """Returns serializable descriptions of object geometry

        Returns:
            if self.type == "poly": List[Tuple[float, float]] - a list of the vertices (x,y) of the polygon
            if self.type == "ball": Tuple[Tuple[float, float], float] - the (x,y) position and the radius
            if self.type == "container" or "compound": List[List[Tuple[float, float]]] - a list of convex polygons, described as a list of (x,y) vertices
            otherwise: None
        """        
        if (self.type == "Poly"):
            return self.get_vertices()
        elif (self.type == "Ball"):
            return [self.get_pos(), self.get_radius()]
        elif self.type == "Container" or self.type == "Compound":
            return self.get_polys()
        else:
            print('not a valid object type')
            return None

    def kick(self, impulse: Tuple[float, float], position: Tuple[float, float], unsafe: bool = False):
        """Applies an impulse to an object at particular world coordinates. The point of impulse must be set within the object unless `unsafe` is flagged as True

        Args:
            impulse (Tuple[float, float]): the impulse (mass*velocity) transfer in vector form
            position (Tuple[float, float]): the position to apply this impulse at
            unsafe (bool, optional): whether to allow the position to lie outside the object. Defaults to False.

        Raises:
            AssertionError: if unsafe is set to False, this is raised if the position lies outside of this object; also if this object is static
        """        
        assert not self.is_static(), "Cannot kick a static object"
        if not unsafe:
            for s in self._expose_shapes():
                if not s.point_query(position):
                    raise AssertionError("Must kick an object within the object (or set as unsafe)")
        self._cpBody.apply_impulse_at_world_point(impulse, position)

    def distance_from_point(self, point: Tuple[float, float]) -> float:
        """Returns the shortest distance between this object and a point. Will be negative if the point lies in this object

        Args:
            point (Tuple[float, float]): The (x,y) coordinates of the point

        Returns:
            float: the distance in world units
        """        
        d, _ = self._cpShape.point_query(point)
        return d

    def distance_from_point_XY(self, point: Tuple[float, float]) -> pm.Vec2d:
        """Returns the vector between a point and the nearest point on this object

        Args:
            point (Tuple[float, float]): the (x,y) coordinates of the point

        Returns:
            pm.Vec2d: a pymunk Vec2d object describing the vector difference
        """        
        _, info = self._cpShape.point_query(point)
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
        """Abstract initialization of a win condition. Should never be called directly; only inherited
        """        
        self.goal = self.obj = self.parent = self.dur = None

    def _get_time_in(self):
        return -1

    def remaining_time(self) -> float:
        """Returns the time left until the victory condition is met. If the countdown hasn't yet started, this will be None

        Returns:
            float: the time remaining (or None)
        """        
        ti = self._get_time_in()
        if ti == -1:
            return None
        curtime = self.parent.time - ti
        return max(self.dur - curtime, 0)

    def is_won(self) -> bool:
        """Returns whether the victory condition has been met already

        Returns:
            bool: True if there is a win, False otherwise
        """        
        return self.remaining_time() == 0

    @abstractmethod
    def attach_hooks(self):
        """An abstract method that *must* be overwritten that hooks into the parent World object to set up when the Victory Condition should be called/checked

        Raises:
            NotImplementedError: always; must be inherited
        """        
        raise NotImplementedError("Cannot attach hooks from base condition object")
