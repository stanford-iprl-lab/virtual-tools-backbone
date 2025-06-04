from .world import VTWorld
from .object import VTObject
from typing import Dict, List, Tuple
from .constants import *
from scipy.stats import norm, truncnorm
import numpy as np
import pymunk as pm
from copy import copy
import pickle

__all__ = ['noisify_world', 'trunc_norm', 'wrapped_norm']


def trunc_norm(mu: float, sig: float,
               lower: float=None, upper: float=None) -> float:
    """Provides a random draw from a truncated normal

    Args:
        mu (float): the mean of the (untruncated) distribution
        sig (float): the standard deviation of the (untruncated) distribution
        lower (float, optional): the truncation lower bound. Defaults to None, which coerces this to -20
        upper (float, optional): the truncation upper bound. Defaults to None, which coerces this to 20

    Returns:
        float: a random number drawn from this distribution
    """    
    if lower is None:
        a = -20
    else:
        a = (lower - mu) / sig
    if upper is None:
        b = 20
    else:
        b = (upper - mu) / sig
    return mu + sig * truncnorm.rvs(a, b, size=1)[0]


def wrapped_norm(mu: float, sig: float) -> float:
    """Provides a random draw from a wrapped normal (around 2pi radians)

    Args:
        mu (float): the mean of the (unwrapped) distribution
        sig (float): the standard deviation of the (unwrapped) distribution

    Returns:
        float: a random number drawn from this distribution; will be in the range of [0, 2*pi)
    """    
    return (mu + sig * norm.rvs(size=1))[0] % (2 * np.pi)

# Helper function to keep track of objects that are touching before the noisification


def _add_collisions(s1: VTObject,
                    s2: VTObject,
                    collision_list: List) -> List:
    o1n = s1.name
    o2n = s2.name
    # Ensure ordering
    if o1n < o2n:
        tmp = o1n
        o1n = o2n
        o2n = tmp
    matched = False
    for c in collision_list:
        if c[0] == o1n and c[1] == o2n:
            matched = True
    if not matched:
        collision_list.append([o1n, o2n])
    return collision_list

# Helper function to move static objects (differs by shape)


def _move_static(obj, pos_ch, space):
    # Seems like the best way to move static objects is to destroy and recreate them
    fric = obj.friction
    elast = obj.elasticity
    nm = obj.name

    if obj.type == 'Poly':
        space.remove(obj._cpShape)
        nverts = [v + pos_ch for v in obj.vertices]
        obj._cpShape = pm.Poly(space.static_body, nverts)
        obj._cpShape.friction = fric
        obj._cpShape.elasticity = elast
        obj._cpShape.collision_type = COLTYPE_SOLID
        obj._cpShape.name = nm
        space.add(obj._cpShape)
    elif obj.type == 'Ball':
        npos = obj._cpShape.offset + pos_ch
        rad = obj._cpShape.radius
        space.remove(obj._cpShape)
        obj._cpShape = pm.Circle(space.static_body, rad, npos)
        obj._cpShape.friction = fric
        obj._cpShape.elasticity = elast
        obj._cpShape.collision_type = COLTYPE_SOLID
        obj._cpShape.name = nm
        space.add(obj._cpShape)
    elif obj.type == 'Segment':
        a = obj._cpShape.a + pos_ch
        b = obj._cpShape.b + pos_ch
        rad = obj._cpShape.radius
        space.remove(obj._cpShape)
        obj._cpShape = pm.Segment(space.static_body, a, b, rad)
        obj._cpShape.friction = fric
        obj._cpShape.elasticity = elast
        obj._cpShape.collision_type = COLTYPE_SOLID
        obj._cpShape.name = nm
        space.add(obj._cpShape)
    elif obj.type == 'Container':
        space.remove(obj._cpPolyShapes)
        space.remove(obj._cpSensor)
        newshapes = []
        newpolys = []
        obj.seglist = [s + pos_ch for s in obj.seglist]
        for p in obj.getPolys():
            newverts = [v + pos_ch for v in p]
            newpolys.append(newverts)
            s = pm.Poly(space.static_body, newverts)
            s.friction = fric
            s.elasticity = elast
            s.collision_type = COLTYPE_SOLID
            s.name = nm
            newshapes.append(s)
        obj.polylist = newpolys
        obj._cpPolyShapes = newshapes
        space.add(obj._cpPolyShapes)
        sensvert = [v + pos_ch for v in obj._cpSensor.get_vertices()]
        obj._cpSensor = pm.Poly(space.static_body, sensvert)
        obj._cpSensor.sensor = True
        obj._cpSensor.collision_type = COLTYPE_SENSOR
        obj._cpSensor.name = nm
        space.add(obj._cpSensor)
    elif obj.type == 'Compound':
        space.remove(obj._cpShapes)
        newpolys = []
        newshapes = []
        for p in obj.getPolys():
            newverts = [v + pos_ch for v in p]
            newpolys.append(newverts)
            s = pm.Poly(space.static_body, newverts)
            s.friction = fric
            s.elasticity = elast
            s.collision_type = COLTYPE_SOLID
            s.name = nm
            newshapes.append(s)
        obj.polylist = newpolys
        obj._cpShapes = newshapes
        space.add(obj._cpShapes)


def noisify_world(gameworld: VTWorld,
                  noise_position_static: float=0.,
                  noise_position_moving: float=0.,
                  noise_collision_direction: float=0.,
                  noise_collision_elasticity: float=0.,
                  noise_gravity: float=0.,
                  noise_object_friction: float=0.,
                  noise_object_density: float=0.,
                  noise_object_elasticity: float=0.) -> VTWorld:
    """Creates a noisy version of a VTWorld, including: perceptual noise, collision noise, and property noise
    (NOTE: property noise currently is not implemented, except for gravity)
    
    perceptual noise:
        Changes the location of objects by perturbing each objects' positions by a gaussian around the original location. The standard deviation of this is set separately for static and moving objects
    
    collision noise:
        Noise in the dynamics of the system. This is added uncertainty whenever there is a collision. In pymunk, collisions are resolved by providing equal and opposite impulses at the point of collision; this noise perturbs both the direction and the magnitude of those impulses
        
    property noise:
        Currently only provides uncertainty about the gravity of the world; in theory should also add noise to friction, density, and elasticities of each object
    
    Args:
        gameworld (VTWorld): the world to make noisy
        noise_position_static (float, optional): the sd of an isotropic gaussian describing perceptual uncertainty for static objects. Defaults to 0..
        noise_position_moving (float, optional): the sd of an isotropic gaussian describing perceptual uncertainty for moving objects. Defaults to 0..
        noise_collision_direction (float, optional): the sd of a wrapped gaussian describing collision impulse perturbations. Defaults to 0..
        noise_collision_elasticity (float, optional): the sd of the elasticity of the colision, which directly impacts the magnitude. Defaults to 0..
        noise_gravity (float, optional): the sd of a truncated gaussian centered at 1 which is multiplied by the true gravity (but cannot bring gravity below 0). Defaults to 0..
        noise_object_friction (float, optional): NOT IMPLEMENTED. Defaults to 0..
        noise_object_density (float, optional): NOT IMPLEMENTED. Defaults to 0..
        noise_object_elasticity (float, optional): NOT IMPLEMENTED. Defaults to 0..

    Returns:
        VTWorld: a noisy version of the original world
    """    

    w = gameworld.copy()

    # Figure out the gravity (with adjustments)
    if noise_gravity > 0:
        grav = w.gravity * trunc_norm(1, noise_gravity, 0)
    else:
        grav = w.gravity

    # Turn things off (gravity & callbacks)
    w.gravity = 0
    w._cpSpace.add_collision_handler(COLTYPE_SOLID, COLTYPE_SOLID)
    w._cpSpace.add_collision_handler(COLTYPE_PLACED, COLTYPE_SOLID)
    w._cpSpace.add_collision_handler(COLTYPE_SOLID, COLTYPE_SENSOR)
    w._cpSpace.add_collision_handler(COLTYPE_PLACED, COLTYPE_SENSOR)

    # Adjust the object positions and attributes
    # First segment in to static vs not static & adjust properties
    wall_names = ["_LeftWall", "_BottomWall", "_RightWall", "_TopWall"]

    # With static noise, group all touching objects and move them together
    if noise_position_static > 0:
        # Make object groups (things that move together because they are touching)
        obj_groups = []
        objs = w.objects.values()
        for i in range(len(objs) - 1):
            o1 = objs[i]
            if o1.name not in wall_names:
                this_idx = -1
                for idx, og in enumerate(obj_groups):
                    if o1.name in [o.name for o in og]:
                        this_idx = idx
                if this_idx == -1:
                    this_idx = len(obj_groups)
                    obj_groups.append([o1])
                for j in range(i + 1, len(objs)):
                    o2 = objs[j]
                    if o1.checkContact(o2):
                        if o2.name not in [o.name for o in obj_groups[this_idx]] + wall_names:
                            obj_groups[this_idx].append(o2)

        # Now that the space is segmented, move all static items together
        for og in obj_groups:
            pos_change = noise_position_static * norm.rvs(size=2)
            for o in og:
                if o.is_static():
                    _move_static(o, pos_change, w._cpSpace)
                else:
                    o.position += pos_change

    # With moving noise, adjust objects individually but make sure they are still touching everything they already were
    if noise_position_moving > 0:
        # Find the things that need to be moved and cache their original positions and touching objects
        free_obj = []
        orig_pos = {}
        orig_vel = {}
        touch_dict = {}
        for onm, obj in w.objects.items():
            if not obj.is_static():
                free_obj.append(obj)
                orig_pos[onm] = obj.position
                orig_vel[onm] = obj.velocity

                obj.velocity = (0, 0)
                touch_dict[onm] = []
                for onm2, obj2 in w.objects.items():
                    if onm != onm2:
                        if obj.checkContact(obj2):
                            touch_dict[onm].append(obj2)

        # Catch to ensure moving static objects doesn't produce an impossible configuration
        noise_attempts = 0
        max_attempts = 500
        while len(free_obj) > 0 and noise_attempts < max_attempts:
            noise_attempts += 1

            # Randomly perturb everything
            for o in free_obj:
                o.position += noise_position_moving * norm.rvs(size=2)

            # Take tiny steps to resolve overlaps
            for i in range(10):
                w._cpSpace.step(.1)

            # Check that any contacts that existed already still remain - if not, reset
            checked_contacts = []
            for o in free_obj:
                stillgood = True
                touches = touch_dict[o.name]
                for o2 in w.objects.values():
                    if stillgood and o.name != o2.name:  # Make sure it's not the same thing; save compuatation
                        if o.checkContact(o2):
                            # If there's a contact, make sure that it's not a new one
                            if o2 not in touches:
                                stillgood = False
                        else:
                            # Otherwise, make sure if it's not touching it shouldn't be
                            if o2 in touches:
                                stillgood = False
                if stillgood:
                    checked_contacts.append(o.name)
                    o._cpBody.sleep()
                else:
                    o.position = orig_pos[o.name]

            # Now reduce free_obj

            # things are getting messed up when we reduce free_obj in the w._cpSpace.step() phase to not include all objects. I'm not sure why,
            # but the obvious fix for this for now is to just free up every object whenever we don't "make it" through to having no free objects
            curr_free_obj = [
                o for o in free_obj if o.name not in checked_contacts]
            if len(curr_free_obj) > 0:
                for o in free_obj:
                    o._cpBody.activate()  # wake things again if this isn't going to work
            else:
                free_obj = []

        # Wake things back up
        for onm, v in orig_vel.items():
            o = w.objects[onm]
            o._cpBody.activate()
            o.velocity = v

        # so as to prevent impossible configurations - just go back to original position
        if noise_attempts >= max_attempts:
            for onm, v in orig_vel.items():
                o = w.objects[onm]
                o.velocity = v
                o.position = orig_pos[onm]

    # Set the callbacks to add noise
    if noise_collision_direction > 0 or noise_collision_elasticity > 0:
        def noisify_arbiter(arb):
            # Make the restitution noisy
            if noise_collision_elasticity > 0:
                arb.restitution += trunc_norm(0,
                                              noise_collision_elasticity, -arb.restitution)
            # Make the contact normals noisy
            if noise_collision_direction > 0:
                newnorm = arb.contact_point_set.normal.rotated(
                    wrapped_norm(0, noise_collision_direction))
                setpoints = []
                for cp in arb.contact_point_set.points:
                    setpoints.append(pm.ContactPoint(
                        list(cp.point_a), list(cp.point_b), cp.distance))
                newcps = pm.ContactPointSet(list(newnorm), setpoints)
                arb.contact_point_set = newcps

        def do_solid_solid_pre(arb, space, data):
            if w.time > 0.001 and arb.is_first_contact:
                noisify_arbiter(arb)
            return w._solid_solid_pre(arb, space, data)

    else:
        def do_solid_solid_pre(arb, space, data):
            return w._solid_solid_pre(arb, space, data)

    # Reset the world
    w.gravity = grav

    def do_solid_solid_begin(arb, space, data):
        return w._solid_solid_begin(arb, space, data)

    def do_solid_solid_post(arb, space, data):
        return w._solid_solid_post(arb, space, data)

    def do_solid_solid_end(arb, space, data):
        return w._solid_solid_end(arb, space, data)

    def do_solid_goal_begin(arb, space, data):
        return w._solid_goal_begin(arb, space, data)

    def do_solid_goal_end(arb, space, data):
        return w._solid_goal_end(arb, space, data)

    ssch = w._cpSpace.add_collision_handler(COLTYPE_SOLID, COLTYPE_SOLID)
    ssch.begin = do_solid_solid_begin
    ssch.pre_solve = do_solid_solid_pre
    ssch.post_solve = do_solid_solid_post
    ssch.separate = do_solid_solid_end

    psch = w._cpSpace.add_collision_handler(COLTYPE_PLACED, COLTYPE_SOLID)
    psch.begin = do_solid_solid_begin
    psch.pre_solve = do_solid_solid_pre
    psch.post_solve = do_solid_solid_post
    psch.separate = do_solid_solid_end

    ssench = w._cpSpace.add_collision_handler(COLTYPE_SOLID, COLTYPE_SENSOR)
    ssench.begin = do_solid_goal_begin
    ssench.separate = do_solid_goal_end

    psench = w._cpSpace.add_collision_handler(COLTYPE_PLACED, COLTYPE_SENSOR)
    psench.begin = do_solid_goal_begin
    psench.separate = do_solid_goal_end

    w._cpSpace.step(.000001)
    return w
