from __future__ import division, print_function
import execjs
import json
import os
import pdb
from .js_contexts import modulepath, base_context, collision_context, context_phyre
from .helpers import filterCollisionEvents, stripGoal, updateObjects
from .world import loadFromDict
from .viewer import *
from .toolpicker_js import JSRunner, CollisionChecker
import pygame as pg
import numpy as np
import warnings

__all__ = ['PhyreActions', 'loadPhyreActions', 'JSRunner']


class PhyreActions(object):

    def __init__(self, worlddict, basicTimestep=0.1, worldTimestep=0.01,
                 maxTime=20., checkThruPy=True, tnm=None):
        self._worlddict = worlddict
        self._worlddict['bts'] = worldTimestep
        self._wdng = stripGoal(self._worlddict)
        self.bts = basicTimestep
        self.maxTime = maxTime
        self.wts = worldTimestep
        if tnm is not None:
            self._tnm = tnm
        self.t = 0
        ctxstr = (context_phyre.format(modulepath, json.dumps(self._worlddict)))
        self._ctx = execjs.compile(ctxstr)
        self._pycheck = checkThruPy
        if checkThruPy:
            self._pyworld = loadFromDict(self._worlddict)

    def _reset_pyworld(self):
        self._pyworld = loadFromDict(self._worlddict)

    def _get_image_array(self, worlddict, path, sample_ratio=1):
        if path is None:
            imgs = makeImageArrayNoPath(worlddict, self.maxTime/self.bts/sample_ratio)
        else:
            imgs = makeImageArray(worlddict, path, sample_ratio)
        imgdata = np.array([pg.surfarray.array3d(img).swapaxes(0,1) for img in imgs])
        return imgdata

    def drawPathSingleImage(self, wd, path):
        if path is None:
            world = loadFromDict(wd)
            sc = drawWorld(world, backgroundOnly=False)
            img = sc
        else:
            img = drawPathSingleImage(wd, path)

        imgdata = pg.surfarray.array3d(img).swapaxes(0,1)
        return imgdata

    def checkPlacementCollide(self, position, radius):
        #for tverts in self._tools[toolname]:
        #    if self._ctx.call('checkSinglePlaceCollide', tverts, position):
        #        return True
        #return False
        if any(np.array(position) <= 0.0) or any(np.array(position)>=600.0):
            return True
        if self._pycheck:
            if self._pyworld.checkCircleCollision(position, radius):
                return True
            return False
        else:
            return self._ctx.call('checkCircleCollide', position, radius)

    def runPlacement(self, position, radius, maxtime=20., returnDict=False,
                     stopOnGoal=True, objAdjust=None):
        # Make sure the tool can be placed
        if self.checkPlacementCollide(position, radius):
            return None, -1
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('runPhyrePlacement', wd, position, radius, maxtime, self.bts, {}, returnDict)

    def observePlacementPath(self, position, radius, maxtime=20., returnDict=False,
                             stopOnGoal=True, objAdjust=None):
        # Make sure the tool can be placed
        if self.checkPlacementCollide(position, radius):
            return None, None, -1
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyrePathPlacement', wd,
                              position, radius, maxtime,
                              self.bts, {}, returnDict)

    def observePath(self, maxtime=20., returnDict=False, stopOnGoal=True,
                    objAdjust=None):
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyreStatePath', wd, maxtime, self.bts,
                              {}, returnDict)

    def observeFullPlacementPath(self, position, radius, maxtime=20.,
                                 returnDict=False, stopOnGoal=True,
                                 objAdjust=None):
        # Make sure the tool can be placed
        if self.checkPlacementCollide(position, radius):
            return None, None, -1, None
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyrePathAndRotPlacement', wd,
                              position, radius, maxtime,
                              self.bts, {}, returnDict)

    def observeGeomPath(self, position, radius, maxtime=20.,
                        returnDict=False, stopOnGoal=True, objAdjust=None):
        # Make sure the tool can be placed
        if self.checkPlacementCollide(position, radius):
            return None, None, -1, None
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyreGeomPathPlacement', wd,
                              position, radius, maxtime,
                              self.bts, {}, returnDict)

    def observePlacementStatePath(self, position, radius, returnDict=False,
                                  stopOnGoal=True, objAdjust=None):
        # Make sure the tool can be placed
        if self.checkPlacementCollide(position, radius):
            if returnDict:
                return None, None, -1, None
            else:
                return None, None, -1
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyreStatePathPlacement', wd,
                              position, radius, self.maxTime,
                              self.bts, {}, returnDict)

    def observeCollisionEvents(self, position, radius, maxtime=20.,
                               collisionSlop=.2001, returnDict=False,
                               stopOnGoal=True, objAdjust=None):
        # Make sure the tool can be placed
        if self.checkPlacementCollide(position, radius):
            return None, None, -1, -1
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        if returnDict:
            path, col, end, t, w = self._ctx.call('getPhyreCollisionPathPlacement', wd,
                                               position, radius,
                                               maxtime, self.bts, {},True)
        else:
            path, col, end, t = self._ctx.call('getPhyreCollisionPathPlacement', wd,
                                               position, radius,
                                               maxtime, self.bts, {}, False)
        fcol = filterCollisionEvents(col, collisionSlop)
        r = [path, fcol, end, t]
        if returnDict:
            r.append(w)
        return r

    def observeFullCollisionEvents(self, position, radius, maxtime=20.,
                               collisionSlop=.2001, returnDict=False,
                               stopOnGoal=True, objAdjust=None):
        # Make sure the tool can be placed
        if self.checkPlacementCollide(position, radius):
            return None, None, -1, -1
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        if returnDict:
            path, col, end, t, w = self._ctx.call('getPhyreCollisionPathAndRotPlacement', wd,
                                               position, radius,
                                               maxtime, self.bts, {},True)
        else:
            path, col, end, t = self._ctx.call('getPhyreCollisionPathAndRotPlacement', wd,
                                               position, radius,
                                               maxtime, self.bts, {}, False)
        fcol = filterCollisionEvents(col, collisionSlop)
        r = [path, fcol, end, t]
        if returnDict:
            r.append(w)
        return r

    def placeObject(self, position, radius):
        raise NotImplementedError(
            'Direct placement not allowed in new PhyreAction')

    def noisifySelf(self, noise_position_static=5., noise_position_moving=5.,
                    noise_collision_direction=.2, noise_collision_elasticity=.2, noise_gravity=.1,
                    noise_object_friction=.1, noise_object_density=.1, noise_object_elasticity=.1):
        raise NotImplementedError(
            'Direct noisification not allowed in PhyreAction -- use runNoisyPlacement()')

    def runNoisyPlacement(self, position, radius, maxtime=20.,
                          noise_position_static=0, noise_position_moving=0,
                          noise_collision_direction=0, noise_collision_elasticity=0, noise_gravity=0,
                          noise_object_friction=0, noise_object_density=0, noise_object_elasticity=0,
                          returnDict=False, stopOnGoal=True, objAdjust=None):
        ndict = {
            'noise_position_static': noise_position_static,
            'noise_position_moving': noise_position_moving,
            'noise_collision_direction': noise_collision_direction,
            'noise_collision_elasticity': noise_collision_elasticity,
            'noise_gravity': noise_gravity,
            'noise_object_friction': noise_object_friction,
            'noise_object_density': noise_object_density,
            'noise_object_elasticity': noise_object_elasticity
        }
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('runPhyrePlacement', wd, position, radius,
                              maxtime, self.bts, ndict, returnDict)

    def observeNoisyPlacementStatePath(self, position, radius, ndict={},
                                       returnDict=False, stopOnGoal=True,
                                       objAdjust=None):
        # Make sure the tool can be placed
        if self.checkPlacementCollide(position, radius):
            if returnDict:
                return None, None, -1, None
            else:
                return None, None, -1
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyreStatePathPlacement', wd,
                              position, radius, self.maxTime,
                              self.bts, ndict, returnDict)

    def runNoisyPath(self, position, radius, maxtime=20.,
                     noise_position_static=0, noise_position_moving=0,
                     noise_collision_direction=0, noise_collision_elasticity=0, noise_gravity=0,
                     noise_object_friction=0, noise_object_density=0, noise_object_elasticity=0,
                     returnDict=False, stopOnGoal=True, objAdjust=None):
        ndict = {
            'noise_position_static': noise_position_static,
            'noise_position_moving': noise_position_moving,
            'noise_collision_direction': noise_collision_direction,
            'noise_collision_elasticity': noise_collision_elasticity,
            'noise_gravity': noise_gravity,
            'noise_object_friction': noise_object_friction,
            'noise_object_density': noise_object_density,
            'noise_object_elasticity': noise_object_elasticity
        }
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyrePathPlacement', wd, position, radius,
                              maxtime, self.bts, ndict, returnDict)

    def runFullNoisyPath(self, position, radius, maxtime=20.,
                     noise_position_static=0, noise_position_moving=0,
                     noise_collision_direction=0, noise_collision_elasticity=0, noise_gravity=0,
                     noise_object_friction=0, noise_object_density=0, noise_object_elasticity=0,
                     returnDict=False, stopOnGoal=True, objAdjust=None):
        ndict = {
            'noise_position_static': noise_position_static,
            'noise_position_moving': noise_position_moving,
            'noise_collision_direction': noise_collision_direction,
            'noise_collision_elasticity': noise_collision_elasticity,
            'noise_gravity': noise_gravity,
            'noise_object_friction': noise_object_friction,
            'noise_object_density': noise_object_density,
            'noise_object_elasticity': noise_object_elasticity
        }
        if all([v == 0 for v in ndict.values()]):
            ndict = {}
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyrePathAndRotPlacement', wd, position, radius,
                              maxtime, self.bts, ndict, returnDict)

    def runNoisyGeomPath(self, position, radius, maxtime=20.,
                     noise_position_static=0, noise_position_moving=0,
                     noise_collision_direction=0, noise_collision_elasticity=0, noise_gravity=0,
                     noise_object_friction=0, noise_object_density=0, noise_object_elasticity=0,
                     returnDict=False, stopOnGoal=True, objAdjust=None):
        ndict = {
            'noise_position_static': noise_position_static,
            'noise_position_moving': noise_position_moving,
            'noise_collision_direction': noise_collision_direction,
            'noise_collision_elasticity': noise_collision_elasticity,
            'noise_gravity': noise_gravity,
            'noise_object_friction': noise_object_friction,
            'noise_object_density': noise_object_density,
            'noise_object_elasticity': noise_object_elasticity
        }
        if all([v == 0 for v in ndict.values()]):
            ndict = {}
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyreGeomPathPlacement', wd, position, radius,
                              maxtime, self.bts, ndict, returnDict)

    def runFullNoisyPathDict(self, position, radius, maxtime=20.,ndict={},
                     returnDict=False, stopOnGoal=True, objAdjust=None):
        if ndict != {}:
            warnings.warn("Noise on objects not yet implemented -- will have no effect")
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        return self._ctx.call('getPhyrePathAndRotPlacement', wd, position, radius,
                              maxtime, self.bts, ndict, returnDict)

    def runNoisyBumpPath(self, position, radius, bumptime, bumpname, bumpimpulse, bumpLocation=None,
                         maxtime=20., noise_position_static=0,
                         noise_position_moving=0, noise_collision_direction=0,
                         noise_collision_elasticity=0, noise_gravity=0,
                         noise_object_friction=0, noise_object_density=0,
                         noise_object_elasticity=0, returnDict=False,
                         stopOnGoal=True, objAdjust=None):
        ndict = {
            'noise_position_static': 0,
            'noise_position_moving': 0,
            'noise_collision_direction': noise_collision_direction,
            'noise_collision_elasticity': noise_collision_elasticity,
            'noise_gravity': noise_gravity,
            'noise_object_friction': noise_object_friction,
            'noise_object_density': noise_object_density,
            'noise_object_elasticity': noise_object_elasticity
        }
        # Skip noisification if no noisy parameters added
        if all([v == 0 for v in ndict.values()]):
            ndict = {}

        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)

        if bumpLocation is not None:
            return self._ctx.call('getPhyrePathBumpAndNoiseLocationPlacement', wd,
                              position, radius,
                              bumptime, bumpname, bumpimpulse, bumpLocation,
                              maxtime, self.bts, ndict, returnDict)
        else:
            return self._ctx.call('getPhyrePathBumpAndNoisePlacement', wd,
                              position, radius,
                              bumptime, bumpname, bumpimpulse,
                              maxtime, self.bts, ndict, returnDict)

    def runNoisyBumpPathDict(self, position, radius, bumptime, bumpname, bumpimpulse, bumpLocation=None,
                         maxtime=20., ndict={}, returnDict=False,
                         stopOnGoal=True, objAdjust=None):

        # Skip noisification if no noisy parameters added
        if all([v == 0 for v in ndict.values()]) or ndict == {}:
            ndict = {}

        else:
            ndict['noise_position_static'] = 0
            ndict['noise_position_moving'] = 0

        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)

        if bumpLocation is not None:
            return self._ctx.call('getPhyrePathBumpAndNoiseLocationPlacement', wd,
                              position, radius,
                              bumptime, bumpname, bumpimpulse, bumpLocation,
                              maxtime, self.bts, ndict, returnDict)
        else:
            return self._ctx.call('getPhyrePathBumpAndNoisePlacement', wd,
                              position, radius,
                              bumptime, bumpname, bumpimpulse,
                              maxtime, self.bts, ndict, returnDict)

    def runNoisyStartBumpPathDict(self, position, radius, bumptime, bumpname, bumpimpulse, bumpLocation=None,
                         maxtime=20., ndict={}, returnDict=False, stopOnGoal=True,
                         objAdjust=None):

        # Skip noisification if no noisy parameters added
        if all([v == 0 for v in ndict.values()]):
            ndict = {}

        else:
            ndict['noise_position_static'] = 0
            ndict['noise_position_moving'] = 0

        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng

        if objAdjust:
            wd = updateObjects(wd, objAdjust)

        if bumpLocation is not None:
            return self._ctx.call('getPhyrePathWithBumpLocationPlacement', wd,
                              position, radius,
                              bumptime, bumpname, bumpimpulse, bumpLocation,
                              maxtime, self.bts, ndict, returnDict)
        else:
            return self._ctx.call('getPhyrePathWithBumpPlacement', wd,
                              position, radius,
                              bumptime, bumpname, bumpimpulse,
                              maxtime, self.bts, ndict, returnDict)


    def observeNoisyFullCollisionEvents(self, position, radius, maxtime=20.,
                                        noise_position_static=0,
                                        noise_position_moving=0, noise_collision_direction=0,
                                        noise_collision_elasticity=0, noise_gravity=0,
                                        noise_object_friction=0, noise_object_density=0,
                                        noise_object_elasticity=0,
                                        collisionSlop=.2001, returnDict=False,
                                        stopOnGoal=True, objAdjust=None):
        ndict = {
            'noise_position_static': 0,
            'noise_position_moving': 0,
            'noise_collision_direction': noise_collision_direction,
            'noise_collision_elasticity': noise_collision_elasticity,
            'noise_gravity': noise_gravity,
            'noise_object_friction': noise_object_friction,
            'noise_object_density': noise_object_density,
            'noise_object_elasticity': noise_object_elasticity
        }
        # Skip noisification if no noisy parameters added
        if all([v == 0 for v in ndict.values()]):
            ndict = {}
        if stopOnGoal:
            wd = self._worlddict
        else:
            wd = self._wdng
        if objAdjust:
            wd = updateObjects(wd, objAdjust)
        if returnDict:
            path, col, end, t, w = self._ctx.call('getPhyreCollisionPathAndRotPlacement', wd,
                                               position, radius,
                                               maxtime, self.bts, ndict,True)
        else:
            path, col, end, t = self._ctx.call('getPhyreCollisionPathAndRotPlacement', wd,
                                               position, radius,
                                               maxtime, self.bts, ndict, False)
        #fcol = filterCollisionEvents(col, collisionSlop)
        r = [path, end, t]
        if returnDict:
            r.append(w)
        return r

    def getWorldDims(self):
        return self._worlddict['dims']

    def getObjects(self):
        #this should only be used for getting initial positions of objects!
        world = loadFromDict(self._worlddict)
        return world.objects

    def exposeWorld(self):
        warnings.warn(
            "Exposing world returns a python object -- may differ from JS used in PhyreAction")
        return loadFromDict(self._worlddict)

def loadPhyreActions(jsonfile, basicTimestep=0.1):
    with open(jsonfile, 'rU') as jfl:
        return PhyreActions(json.load(jfl), basicTimestep)
