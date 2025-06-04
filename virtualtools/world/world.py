from typing import Tuple, Callable, Annotated, Dict, List
import pymunk as pm
import numpy as np
from .constants import * # Add period here
from .object import VTPoly, VTBall, VTSeg, VTContainer, VTCompound, \
    VTGoal, VTBlocker, VTObject
from .conditions import *
from ..helpers import word_to_color, distance_to_object
from copy import deepcopy
import warnings
import pdb

__all__ = ["VTWorld", "load_vt_from_dict"]

def _empty_collision_handler(arb: pm.Arbiter, space: pm.Space):
    return True

def _empty_object_handler(o1: VTObject, o2: VTObject):
    return

def resolve_arbiter(arb: pm.Arbiter) -> Tuple[str, str]:
    """Returns the Virtual Tool names involved in a pymunk arbiter

    Args:
        arb (pm.Arbiter): the pymunk arbiter to query

    Returns:
        Tuple[str, str]: the two names of the objects involved
    """    
    shs = arb.shapes
    o1, o2 = shs
    return o1.name, o2.name

def pull_collision_information(arb: pm.Arbiter):
    norm = arb.contact_point_set.normal
    setpoints = []
    for cp in arb.contact_point_set.points:
        setpoints.append([list(cp.point_a), list(cp.point_b), cp.distance])
    restitution = arb.restitution
    return [norm, restitution, setpoints]

def _listify(l):
    if hasattr(l, "__iter__") and not isinstance(l, str):
        return [_listify(i) for i in l]
    else:
        return l

class VTWorld(object):

    def __init__(self,
                 dimensions: Tuple[float, float],
                 gravity: float,
                 closed_ends: Annotated[Tuple[bool], 4] = [True,True,True,True],
                 basic_timestep: float = 0.01,
                 def_density: float = DEFAULT_DENSITY,
                 def_elasticity: float = DEFAULT_ELASTICITY,
                 def_friction: float = DEFAULT_FRICTION,
                 bk_col: Annotated[Tuple[int], 3] = (255,255,255),
                 def_col: Annotated[Tuple[int], 3] = (0,0,0)):
        """Instantiates a virtual tools world

        Args:
            dimensions (Tuple[float, float]): the (x,y) limits of the world
            gravity (float): the strength of gravity pointing down in units/s^2
            closed_ends (Annotated[Tuple[bool], 4], optional): a set of flags to say whether a [left, bottom, right, top] wall should be added around the world. Defaults to [True,True,True,True].
            basic_timestep (float, optional): the amount of time to step forward when calculatign physics. Smaller values will slow down the simulation but be more precise. Larger values might cause "tunneling" or other physics instabilities. Defaults to 0.01.
            def_density (float, optional): the default density of all objects in the world, unless otherwise specified. Defaults to 1.
            def_elasticity (float, optional): the default elasticity of all objects in the world, unless otherwise specified. Defaults to 0.5.
            def_friction (float, optional): _description_. the default friction of all objects in the world, unless otherwise specified. Defaults to 0.5.
            bk_col (Annotated[Tuple[int], 3], optional): the RGB value of the background of the tools world. Defaults to white (255,255,255).
            def_col (Annotated[Tuple[int], 3], optional): the default RGB color of objects added to the world. Not guaranteed to catch. Defaults to black (0,0,0).
        """        

        self.def_density = def_density
        self.def_elasticity = def_elasticity
        self.def_friction = def_friction
        self.bk_col = bk_col
        self.def_col = def_col

        self.dims = dimensions
        self.bts = basic_timestep
        self.time = 0
        self.has_place_collision = False

        self._cpSpace = pm.Space()
        self._cpSpace.gravity = (0, -gravity)
        self._cpSpace.sleep_time_threshold = 5.

        self.objects = dict()
        self.blockers = dict()
        self.constraints = dict() # Not implemented yet

        self.goal_cond = None
        self.win_callback = None
        self._collision_events = []
        self._ssBegin = _empty_collision_handler
        self._ssPre = _empty_collision_handler
        self._ssPost = _empty_collision_handler
        self._ssEnd = _empty_collision_handler
        self._sgBegin = _empty_collision_handler
        self._sgEnd = _empty_collision_handler

        def do_solid_solid_begin(arb, space, data):
            return self._solid_solid_begin(arb, space, data)
        def do_solid_solid_pre(arb, space, data):
            return self._solid_solid_pre(arb, space, data)
        def do_solid_solid_post(arb, space, data):
            return self._solid_solid_post(arb, space, data)
        def do_solid_solid_end(arb, space, data):
            return self._solid_solid_end(arb, space, data)
        def do_solid_goal_begin(arb, space, data):
            return self._solid_goal_begin(arb, space, data)
        def do_solid_goal_end(arb, space, data):
            return self._solid_goal_end(arb, space, data)

        ssch = self._cpSpace.add_collision_handler(COLTYPE_SOLID, COLTYPE_SOLID)
        ssch.begin = do_solid_solid_begin
        ssch.pre_solve = do_solid_solid_pre
        ssch.post_solve = do_solid_solid_post
        ssch.separate = do_solid_solid_end

        psch = self._cpSpace.add_collision_handler(COLTYPE_PLACED, COLTYPE_SOLID)
        psch.begin = do_solid_solid_begin
        psch.pre_solve = do_solid_solid_pre
        psch.post_solve = do_solid_solid_post
        psch.separate = do_solid_solid_end

        ssench = self._cpSpace.add_collision_handler(COLTYPE_SOLID, COLTYPE_SENSOR)
        ssench.begin = do_solid_goal_begin
        ssench.separate = do_solid_goal_end

        psench = self._cpSpace.add_collision_handler(COLTYPE_PLACED, COLTYPE_SENSOR)
        psench.begin = do_solid_goal_begin
        psench.separate = do_solid_goal_end

        if closed_ends[0]:
            self.add_box("_LeftWall",[-1,-1,1,self.dims[1]+1], self.def_col, 0)
        if closed_ends[1]:
            self.add_box("_BottomWall", [-1,-1,self.dims[0]+1, 1], self.def_col, 0)
        if closed_ends[2]:
            self.add_box("_RightWall", [self.dims[0] - 1, -1, self.dims[0] + 1, self.dims[1] + 1], self.def_col, 0);
        if closed_ends[3]:
            self.add_box("_TopWall", [-1, self.dims[1] - 1, self.dims[0] + 1, self.dims[1] + 1], self.def_col, 0);

    def step(self, t: float):
        """Steps the world forward by t seconds. Not that this will not affect the precision of physics simulation; that is done via the `basic_timestep` value upon initialization

        Args:
            t (float): the amount of time (in seconds) to step forward
        """        
        nsteps = int(np.floor(t / self.bts))
        remtime = self.bts % t
        self.time += t
        for i in range(nsteps):
            self._cpSpace.step(self.bts)
            if self.check_end() and self.win_callback is not None:
                self.win_callback()
        if remtime / self.bts > .01:
            self._cpSpace.step(remtime)
        if self.check_end() and self.win_callback is not None:
            self.win_callback()

    def _invert(self, pt):
        return (pt[0], self.dims[1] - pt[1])

    def _yinvert(self, y):
        return self.dims[1] - y

    def check_end(self) -> bool:
        """Returns whether the victory condition of the world has been cleared

        Returns:
            bool: true if victory is achieved; false if not (or if no condition exists)
        """       
        if self.goal_cond is None:
            return False
        return self.goal_cond.is_won()

    def get_object(self, name: str) -> VTObject:
        """Returns an object that exists in the world

        Args:
            name (str): the Virtual Tools name flag of the object

        Returns:
            VTObject: the object within the world
        
        Raises:
            AssertionError: if the object doesn't exist
        """        
        assert name in self.objects.keys(), "No object by that name: " + name
        return self.objects[name]

    def get_gravity(self) -> float:
        """Returns the gravity in units/s^2 downwards

        Returns:
            float: gravity
        """        
        return -self._cpSpace.gravity.y

    def set_gravity(self, val: float):
        """Sets the gravity in units/s^2 downwards. Negative values make things fall "up"

        Args:
            val (float): gravity
        """        
        self._cpSpace.gravity = (0, -val)

    ########################################
    # Adding things to the world
    ########################################
    def add_poly(self,
                 name, vertices, color, density = None, elasticity = None, friction = None):
        assert name not in self.objects.keys(), "Name already taken: " + name
        if density is None:
            density = self.def_density
        if elasticity is None:
            elasticity = self.def_elasticity
        if friction is None:
            friction = self.def_friction

        this_obj = VTPoly(name, self._cpSpace, vertices, density, elasticity, friction, color)
        self.objects[name] = this_obj
        return this_obj

    def add_box(self, name, bounds, color, density = None, elasticity = None, friction = None):
        assert name not in self.objects.keys(), "Name already taken: " + name
        assert len(bounds) == 4, "Need four numbers for bounds [l,b,r,t]"
        if density is None:
            density = self.def_density
        if elasticity is None:
            elasticity = self.def_elasticity
        if friction is None:
            friction = self.def_friction

        l = bounds[0]
        b = bounds[1]
        r = bounds[2]
        t = bounds[3]
        vertices = [(l,b), (l,t), (r,t), (r,b)]

        this_obj = VTPoly(name, self._cpSpace, vertices, density, elasticity, friction, color)
        self.objects[name] = this_obj
        return this_obj

    def add_ball(self, name, position, radius, color, density = None, elasticity = None, friction = None):
        assert name not in self.objects.keys(), "Name already taken: " + name
        if density is None:
            density = self.def_density
        if elasticity is None:
            elasticity = self.def_elasticity
        if friction is None:
            friction = self.def_friction

        this_obj = VTBall(name, self._cpSpace, position, radius, density, elasticity, friction, color)
        self.objects[name] = this_obj
        return this_obj

    def add_segment(self, name, p1, p2, width, color, density = None, elasticity = None, friction = None):
        assert name not in self.objects.keys(), "Name already taken: " + name
        if density is None:
            density = self.def_density
        if elasticity is None:
            elasticity = self.def_elasticity
        if friction is None:
            friction = self.def_friction

        this_obj = VTSeg(name, self._cpSpace, p1, p2, width, density, elasticity, friction, color)
        self.objects[name] = this_obj
        return this_obj

    def add_container(self, name, ptlist, width, inner_color, outer_color, density = None, elasticity = None, friction = None):
        assert name not in self.objects.keys(), "Name already taken: " + name
        if density is None:
            density = self.def_density
        if elasticity is None:
            elasticity = self.def_elasticity
        if friction is None:
            friction = self.def_friction

        this_obj = VTContainer(name, self._cpSpace, ptlist, width, density, elasticity, friction, inner_color, outer_color)
        self.objects[name] = this_obj
        return this_obj

    def add_compound(self, name, polys, color, density = None, elasticity = None, friction = None):
        assert name not in self.objects.keys(), "Name already taken: " + name
        if density is None:
            density = self.def_density
        if elasticity is None:
            elasticity = self.def_elasticity
        if friction is None:
            friction = self.def_friction

        this_obj = VTCompound(name, self._cpSpace, polys, density, elasticity, friction, color)
        self.objects[name] = this_obj
        return this_obj

    def add_poly_goal(self, name, vertices, color):
        assert name not in self.objects.keys(), "Name already taken: " + name
        this_obj = VTGoal(name, self._cpSpace, vertices, color)
        self.objects[name] = this_obj
        return this_obj

    def add_box_goal(self, name, bounds, color):
        assert name not in self.objects.keys(), "Name already taken: " + name
        assert len(bounds) == 4, "Need four numbers for bounds [l,b,r,t]"
        l = bounds[0]
        b = bounds[1]
        r = bounds[2]
        t = bounds[3]
        vertices = [(l, b), (l, t), (r, t), (r, b)]
        this_obj = VTGoal(name, self._cpSpace, vertices, color)
        self.objects[name] = this_obj
        return this_obj

    def add_placed_poly(self, name, vertices, color, density = None, elasticity = None, friction = None):
        this_obj = self.add_poly(name, vertices, color, density, elasticity, friction)
        this_obj._cpShape.collision_type = COLTYPE_PLACED
        return this_obj

    def add_placed_compound(self, name, polys, color, density = None, elasticity = None, friction = None):
        this_obj = self.add_compound(name, polys, color, density, elasticity, friction)
        for cpsh in this_obj._cpShapes:
            cpsh.collision_type = COLTYPE_PLACED
        return this_obj
    
    def add_placed_circle(self, name, position, radius, color, density=None, elasticity=None, friction=None):
        this_obj = self.add_ball(name, position, radius, color, density, elasticity, friction)
        this_obj._cpShape.collision_type = COLTYPE_PLACED
        return this_obj

    def add_block(self, name, bounds, color):
        assert name not in self.blockers.keys(), "Name already taken: " + name
        assert len(bounds) == 4, "Need four numbers for bounds [l,b,r,t]"
        l = bounds[0]
        b = bounds[1]
        r = bounds[2]
        t = bounds[3]
        vertices = [(l, b), (l, t), (r, t), (r, b)]
        this_obj = VTBlocker(name, self._cpSpace, vertices, color)
        self.blockers[name] = this_obj
        return this_obj

    def add_poly_block(self, name, vertices, color):
        assert name not in self.blockers.keys(), "Name already taken: " + name
        this_obj = VTBlocker(name, self._cpSpace, vertices, color)
        self.blockers[name] = this_obj
        return this_obj

    ########################################
    # Callbacks
    ########################################
    def get_solid_collision_pre(self) -> Callable:
        return self._ssPre

    def set_solid_collision_pre(self, fnc: Callable = _empty_object_handler):
        assert callable(fnc), "Must pass legal function to callback setter"
        self._ssPre = fnc

    def get_solid_collision_post(self) -> Callable:
        return self._ssPost

    def set_solid_collision_post(self, fnc: Callable = _empty_object_handler):
        assert callable(fnc), "Must pass legal function to callback setter"
        self._ssPost = fnc

    def get_solid_collision_begin(self) -> Callable:
        return self._ssBegin

    def set_solid_collision_begin(self, fnc: Callable = _empty_object_handler):
        assert callable(fnc), "Must pass legal function to callback setter"
        self._ssBegin = fnc

    def get_solid_collision_end(self) -> Callable:
        return self._ssEnd

    def set_solid_collision_end(self, fnc: Callable = _empty_object_handler):
        assert callable(fnc), "Must pass legal function to callback setter"
        self._ssEnd = fnc

    def get_goal_collision_begin(self) -> Callable:
        return self._sgBegin

    def set_goal_collision_begin(self, fnc: Callable = _empty_object_handler):
        assert callable(fnc), "Must pass legal function to callback setter"
        self._sgBegin = fnc

    def get_goal_collision_end(self) -> Callable:
        return self._sgEnd

    def set_goal_collision_end(self, fnc: Callable = _empty_object_handler):
        assert callable(fnc), "Must pass legal function to callback setter"
        self._sgEnd = fnc

    def _solid_solid_pre(self, arb, space, data):
        onms = resolve_arbiter(arb)
        o1 = self.get_object(onms[0])
        o2 = self.get_object(onms[1])
        self._ssPre(o1,o2)
        return True

    def _solid_solid_post(self, arb, space, data):
        onms = resolve_arbiter(arb)
        o1 = self.get_object(onms[0])
        o2 = self.get_object(onms[1])
        self._ssPost(o1, o2)
        return True

    def _solid_solid_begin(self, arb, space, data):
        onms = resolve_arbiter(arb)
        o1 = self.get_object(onms[0])
        o2 = self.get_object(onms[1])
        # Add any non-static/static collisions to the events
        if not (o1.is_static() and o2.is_static()):
            collision_info = pull_collision_information(arb)
            self._collision_events.append([onms[0],onms[1], "begin",self.time, collision_info])
        self._ssBegin(o1, o2)
        return True

    def _solid_solid_end(self, arb, space, data):
        onms = resolve_arbiter(arb)
        o1 = self.get_object(onms[0])
        o2 = self.get_object(onms[1])
        # Add any non-static/static collisions to the events
        if not (o1.is_static() and o2.is_static()):
            collision_info = pull_collision_information(arb)
            self._collision_events.append([onms[0], onms[1], "end", self.time, collision_info])
        self._ssEnd(o1, o2)
        return True

    def _solid_goal_begin(self, arb, space, data):
        onms = resolve_arbiter(arb)
        o1 = self.get_object(onms[0])
        o2 = self.get_object(onms[1])
        self._sgBegin(o1, o2)
        return True

    def _solid_goal_end(self, arb, space, data):
        onms = resolve_arbiter(arb)
        o1 = self.get_object(onms[0])
        o2 = self.get_object(onms[1])
        self._sgEnd(o1, o2)
        return True

    ########################################
    # Victory conditions
    ########################################
    def _get_callback_on_win(self):
        return self.win_callback

    def _set_callback_on_win(self, fnc):
        assert callable(fnc), "Must pass legal function to callback setter"
        self.win_callback = fnc

    def attach_any_in_goal(self, goalname: str, duration: float, exclusions: List[str] = []):
        """Sets a victory condition in which any object can make it into the goal area (except those specified in exclusions)

        Args:
            goalname (str): the Virtual Tools name of the goal object
            duration (float): the amount of time (in seconds) an object must remain in the goal
            exclusions (List[str], optional): a list of Virtual Tools names of objects that will *not* trigger victory. Defaults to [].
        """        
        self.goal_cond = VTCond_AnyInGoal(goalname, duration, self, exclusions)
        self.goal_cond.attach_hooks()

    def attach_specific_in_goal(self, goalname: str, objname: str, duration: float):
        """Sets a victory condition in which a specific object must make it into a specific goal area

        Args:
            goalname (str): the Virtual Tools name of the goal object
            objname (str): the Virtual Tools name of the target object to go into the goal
            duration (float): the amount of time (in seconds) an object must remain in the goal
        """        
        self.goal_cond = VTCond_SpecificInGoal(goalname, objname, duration, self)
        self.goal_cond.attach_hooks()

    def attach_many_in_goal(self, goalname: str, objlist: List[str], duration: float):
        """Sets a victory condition in which any of a set of objects must make it into a specific goal area

        Args:
            goalname (str): the Virtual Tools name of the goal object
            objlist (List[str]): a list of Virtual Tools names of objects to go into the goal
            duration (float): the amount of time (in seconds) an object must remain in the goal
        """        
        self.goal_cond = VTCond_ManyInGoal(goalname, objlist, duration, self)
        self.goal_cond.attach_hooks()

    def attach_any_touch(self, objname: str, duration: float):
        """Sets a victory condition in which any dynamic object must touch a specific object

        Args:
            objname (str): the Virtual Tools name of the target object to be touched
            duration (float): the amount of time (in seconds) the objects must remain in contact
        """        
        self.goal_cond = VTCond_AnyTouch(objname, duration, self)
        self.goal_cond.attach_hooks()

    def attach_specific_touch(self, obj1: str, obj2: str, duration: float):
        """Sets a victory condition in which two objects must come into contact

        Args:
            obj1 (str): the Virtual Tools name of one of the target objects
            obj2 (str): the Virtual Tools name of the other target object
            duration (float): the amount of time (in seconds) the objects must remain in contact
        """        
        self.goal_cond = VTCond_SpecificTouch(obj1, obj2, duration, self)
        self.goal_cond.attach_hooks()

    def check_finishers(self) -> bool:
        """Makes sure there is a way to exit the world -- a victory condition and a win_callback that happens afterwards

        Returns:
            bool: true if goal_cond and win_callback exist, false otherwise
        """        
        return self.goal_cond is not None and self.win_callback is not None

    ########################################
    # Checking collisions
    ########################################

    def reset_collisions(self):
        """Clears out the collision events list
        """        
        self._collision_events = []

    def _get_collision_events(self):
        return self._collision_events

    ########################################
    # Misc
    ########################################
    def check_collision(self, pos: Tuple[float, float], verts: List[Tuple[float, float]]) -> bool:
        """Checks whether placing a convex polygon in the world would cause a collision

        Args:
            pos (Tuple[float, float]): the position of the hypothetical polygon (where all vertices are calculated wrt)
            verts (List[Tuple[float, float]]): a list of (x,y) vertices of the convex polygon, relative to the position

        Returns:
            bool: true if there would be a collision, false if not
        """        
        nvert = [(v[0]+pos[0], v[1]+pos[1]) for v in verts]
        tmpBody = pm.Body(1,1)
        placeShape = pm.Poly(tmpBody, nvert)
        placeShape.collision_type = COLTYPE_CHECKER
        placeShape.sensor = True
        self._cpSpace.step(.000001)

        self.has_place_collision = False
        squery = self._cpSpace.shape_query(placeShape)
        """ Code doesn't account for blockers (sensors)
        # Update 12/12/23: I don't see why this wouldn't but if you're getting incorrect output, check the blockers!
        if len(squery) == 0:
            return False
        else:
            for sq in squery:
                for p in sq.contact_point_set.points:
                    if p.distance > 0:
                        return True
            return False
        """
        return len(squery) > 0

    def check_circle_collision(self, pos: Tuple[float, float], rad: float) -> bool:
        """Checks if there would be a colision with an object if a circular object were placed in the world

        Args:
            pos (Tuple[float, float]): the center of the hypothetical circular object
            rad (float): the radius of the hypothetical circular object

        Returns:
            bool: true if there would be a colision, false if not
        """        
        tmpBody = pm.Body(1,1)
        placeShape = pm.Circle(tmpBody, rad, pos)
        placeShape.collision_type = COLTYPE_CHECKER
        placeShape.sensor = True
        self._cpSpace.step(.000001)

        self.has_place_collision = False
        squery = self._cpSpace.shape_query(placeShape)
        return len(squery) > 0

    def kick(self, objectname: str, impulse: Tuple[float, float], position: Tuple[float, float]):
        """Applies an impulse to an object at a particular point

        Args:
            objectname (str): the Virtual Tools name of the object in this world
            impulse (Tuple[float, float]): the impulse (momentum) vector
            position (Tuple[float, float]): the point to apply the impulse to in world coordinates. Note: this must be inside the object!
        """        
        o = self.get_object(objectname)
        o.kick(impulse, position)

    def distance_to_goal(self, point: Tuple[float, float]) -> float:
        """Returns the distance between the nearest object achieving victory and the goal area / other object
        
        WARNING: this seems like an old function and doesn't account for a number of VTCond types, plus the math looks off... so be careful!

        Args:
            point (Tuple[float, float]): _description_

        Returns:
            float: _description_
        
        Raises:
            AssertionError: if no goal condition is specified
        """        
        assert self.goal_cond, "Goal condition must be specified to get distance"
        warnings.warn("This function is old and looks wrong - be careful using it")
        # Special case... requires getting two distances
        if type(self.goal_cond) == VTCond_SpecificTouch:
            o1 = self.get_object(self.goal_cond.o1)
            o2 = self.get_object(self.goal_cond.o2)
            #in this case, we actually want the distance between these two objects...
            return np.abs(o1.distance_from_point([0,0]) - o2.distance_from_point([0,0])) #distance between these two objects is thing that matters
        else:
            gobj = self.get_object(self.goal_cond.goal)
            return max(gobj.distance_from_point(point), 0)

    def distance_to_goal_container(self, point: Tuple[float, float]):
        """Returns the distance between the nearest object achieving victory and the goal area / other object; specifies that for container objects, you want the distance to the top of the container
        
        WARNING: this seems like an old function and doesn't account for a number of VTCond types, plus the math looks off... so be careful!


        Args:
            point (Tuple[float, float]): _description_

        Returns:
            _type_: _description_
        """
        assert self.goal_cond, "Goal condition must be specified to get distance"
        warnings.warn("This function is old and looks wrong - be careful using it")
        try:
            # Special case... requires getting two distances
            if type(self.goal_cond) == VTCond_SpecificTouch:
                o1 = self.get_object(self.goal_cond.o1)
                o2 = self.get_object(self.goal_cond.o2)
                #in this case, we actually want the distance between these two objects...
                return np.abs(o1.distance_from_point([0,0]) - o2.distance_from_point([0,0])) #distance between these two objects is thing that matters
            else:
                gobj = self.get_object(self.goal_cond.goal)
                if gobj.type != 'Container':
                    return gobj.distance_from_point(point)
                else:
                    if self.distance_to_goal(point) == 0:
                        return 0
                    else:
                        return distance_to_object(gobj, point)
        except:
            pdb.set_trace()

    def get_dynamic_objects(self) -> List[VTObject]:
        """Returns a list of all dynamic (not static) objects in the world

        Returns:
            List[VTObject]: a List of all of the dynamics VTObjects
        """        
        return [self.objects[i] for i in self.objects.keys() if not self.objects[i].is_static()]

    def to_dict(self) -> Dict:
        """Outputs a JSON-serializable Dict describing the world

        Raises:
            Exception: if invalid object or goal types exist in the world. In theory this should never happen

        Returns:
            Dict: a JSON-serializable Dict
        """        
        wdict = dict()
        wdict['dims'] = tuple(self.dims)
        wdict['bts'] = self.bts
        wdict['gravity'] = self.gravity
        wdict['defaults'] = dict(density=self.def_density, friction=self.def_friction,
                                 elasticity=self.def_elasticity, color=self.def_col, bk_color=self.bk_col)

        wdict['objects'] = dict()
        for nm, o in self.objects.items():
            attrs = dict(type=o.type, color=list(o.color), density=o.density,
                         friction=o.friction, elasticity=o.elasticity)
            if o.type == 'Poly':
                attrs['vertices'] = _listify(o.vertices)
            elif o.type == 'Ball':
                attrs['position'] = list(o.position)
                attrs['radius'] = o.radius
            elif o.type == 'Segment':
                attrs['p1'], attrs['p2'] = _listify(o.points)
                attrs['width'] = o.r * 2
            elif o.type == 'Container':
                attrs['points'] = _listify(o.vertices)
                attrs['width'] = o.r * 2
                attrs['innerColor'] = o.inner_color
                attrs['outerColor'] = o.outer_color
            elif o.type == 'Goal':
                attrs['vertices'] = _listify(o.vertices)
            elif o.type == 'Compound':
                attrs['polys'] = _listify(o.polys)
            else:
                raise Exception('Invalid object type provided')
            wdict['objects'][nm] = attrs

        wdict['blocks'] = dict()
        for nm, b in self.blockers.items():
            attrs = {'color': list(b.color), 'vertices': _listify(b.vertices)}
            wdict['blocks'][nm] = attrs

        wdict['constraints'] = dict()

        if self.goal_cond is None:
            wdict['gcond'] = None
        else:
            gc = self.goal_cond
            if gc.type == 'AnyInGoal':
                wdict['gcond'] = {'type': gc.type, 'goal': gc.goal, 'obj': '-',
                                  'exclusions': gc.excl, 'duration': gc.dur}
            elif gc.type == 'SpecificInGoal':
                wdict['gcond'] = {'type': gc.type, 'goal': gc.goal, 'obj': gc.obj, 'duration': gc.dur}
            elif gc.type == 'ManyInGoal':
                wdict['gcond'] = {'type': gc.type, 'goal': gc.goal, 'objlist': gc.objlist, 'duration': gc.dur}
            elif gc.type == "AnyTouch":
                wdict['gcond'] = {'type': gc.type, 'goal': gc.goal, 'obj': '-', 'duration': gc.dur}
            elif gc.type == 'SpecificTouch':
                wdict['gcond'] = {'type': gc.type, 'goal': gc.o1, 'obj': gc.o2, 'duration': gc.dur}
            else:
                raise Exception('Invalid goal condition type provided')

        return wdict

    def copy(self):
        return load_vt_from_dict(self.to_dict())

    ########################################
    # Properties
    ########################################
    gravity = property(get_gravity, set_gravity)
    solid_collision_pre = property(get_solid_collision_pre,
                                    set_solid_collision_pre)
    solid_collision_post = property(get_solid_collision_post,
                                     set_solid_collision_post)
    solid_collision_begin = property(get_solid_collision_begin,
                                    set_solid_collision_begin)
    solid_collision_end = property(get_solid_collision_end,
                                  set_solid_collision_end)
    goal_collision_begin = property(get_goal_collision_begin,
                                   set_goal_collision_begin)
    goal_collision_end = property(get_goal_collision_end,
                                 set_goal_collision_end)
    callback_on_win = property(_get_callback_on_win, _set_callback_on_win)
    collision_events = property(_get_collision_events)


########################################
# Loading
########################################

def load_vt_from_dict(d: Dict) -> VTWorld:
    """Takes a serializable Dict describing the world, and turns it into a VTWorld object

    Args:
        d (Dict): a serializable vesion of the world (e.g., from VTWorld.to_dict())

    Raises:
        Exception: if invalid object or goal types are described in the Dict

    Returns:
        VTWorld: the world described in the Dict
    """    
    d = deepcopy(d)
    def_elast = float(d['defaults']['elasticity'])
    def_fric = float(d['defaults']['friction'])

    vtw = VTWorld(d['dims'], d['gravity'], [False, False, False, False], d['bts'],
                  float(d['defaults']['density']), def_elast, def_fric,
                  word_to_color(d['defaults']['bk_color']), word_to_color(d['defaults']['color']))

    for nm, o in d['objects'].items():
        elasticity = float(o.get('elasticity', def_elast))
        friction = float(o.get('friction', def_fric))
        density = float(o.get('density', d['defaults']['density']))

        if o['type'] == 'Poly':
            vtw.add_poly(nm, o['vertices'], word_to_color(o['color']), density, elasticity, friction)
        elif o['type'] == 'Ball':
            vtw.add_ball(nm, o['position'], o['radius'], word_to_color(o['color']), density, elasticity, friction)
        elif o['type'] == 'Segment':
            vtw.add_segment(nm, o['p1'], o['p2'], o['width'], word_to_color(o['color']), density, elasticity, friction)
        elif o['type'] == 'Container':
            if 'innerColor' not in o:
                if 'color' in o:
                    ic = word_to_color(o['color'])
                else:
                    ic = None
            else:
                ic = word_to_color(o['innerColor'])
            if 'outerColor' not in o:
                oc = DEFAULT_COLOR
            else:
                oc = word_to_color(o['outerColor'])
            vtw.add_container(nm, o['points'], o['width'], ic, oc, density, elasticity, friction)
        elif o['type'] == 'Goal':
            vtw.add_poly_goal(nm, o['vertices'], word_to_color(o['color']))
        elif o['type'] == 'Compound':
            vtw.add_compound(nm, o['polys'], word_to_color(o['color']), density, elasticity, friction)
        else:
            raise Exception("Invalid object type given: " + o['type'])

    for nm, b in d['blocks'].items():
        vtw.add_poly_block(nm, b['vertices'], word_to_color(b['color']))

    if d['gcond'] is not None:
        g = d['gcond']
        if g['type'] == 'AnyInGoal':
            excl = g.get('exclusions', [])
            vtw.attach_any_in_goal(g['goal'], float(g['duration']), excl)
        elif g['type'] == 'SpecificInGoal':
            vtw.attach_specific_in_goal(g['goal'], g['obj'], float(g['duration']))
        elif g['type'] == 'ManyInGoal':
            vtw.attach_many_in_goal(g['goal'], g['objlist'], float(g['duration']))
        elif g['type'] == 'AnyTouch':
            vtw.attach_any_touch(g['goal'], float(g['duration']))
        elif g['type'] == 'SpecificTouch':
            vtw.attach_specific_touch(g['goal'], g['obj'], float(g['duration']))
        else:
            raise Exception("In valid goal condition type given")

    return vtw

def reverse_world(w: VTWorld) -> VTWorld:
    """Flips a world around its x-axis

    Args:
        w (VTWorld): the world to flip

    Raises:
        Exception: if an object has an illegal type. In theory this should never happen

    Returns:
        VTWorld: the flipped world
    """    
    xdim = w.dims[0]
    def rev_pt(p):
        return (xdim - p[0], p[1])
    # Easier to do this as a dict than modify the objects themselves
    d = w.to_dict()
    for nm, o in d['objects'].items():
        if o['type'] == 'Poly' or o['type'] == 'Goal':
            o['vertices'] = [rev_pt(p) for p in o['vertices']]
            o['vertices'].reverse()
        elif o['type'] == 'Ball':
            o['position'] = rev_pt(o['position'])
        elif o['type'] == 'Segment':
            o['p1'] = rev_pt(o['p1'])
            o['p2'] = rev_pt(o['p2'])
        elif o['type'] == 'Container':
            o['points'] = [rev_pt(p) for p in o['points']]
            o['points'].reverse()
        elif o['type'] == 'Compound':
            for i, poly in enumerate(o['polys']):
                o['polys'][i] = [rev_pt(p) for p in poly]
                o['polys'][i].reverse()
        else:
            raise Exception("Invalid object type given: " + o['type'])
    
    for nm, b in d['blocks'].items():
        b['vertices'] = [rev_pt(p) for p in b['vertices']]
        b['vertices'].reverse()
    
    return load_vt_from_dict(d)
        