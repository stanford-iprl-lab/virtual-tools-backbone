from __future__ import division, print_function
import execjs
import json
import os
import pygame as pg
import copy
import numpy as np
import pdb
from .toolpicker_js import JSRunner, CollisionChecker
from .world import loadFromDict
from .viewer import drawWorld
import warnings

__all__ = ['SimpleDropper', 'make_default_simple_world']

"""

A class for running SimpleDropper games (ones with easy geometry, a single tool
to drop, and the goal of hitting a certain point on the ground with one object)

SimpleDropper game dictionaries should have the following entries:
    * 'world': a pyGameWorld world dict. This should *always* have an object
               named 'GoalBall' that is expected to hit the ground. Also not
               allowed to have any moveable compound objects
    * 'dropper': a list of [[x,y], ...] vertices for the tool to drop (must be
                 a convex polygon)
    * 'goal': a range [xmin, xmax] for where the object should hit
    * 'droprange': a range [xmin, xmax] where the tool *can* be dropped

"""
class SimpleDropper:

    """Initializes a SimpleDropper

    Args:
        gamedict [dict]: defined as above
        record_timestep [float]: how often to record positions
        world_timestep [float]: physics timestep for updating
    """
    def __init__(self, gamedict, record_timestep=0.1, world_timestep=0.01,
                 max_time=20.):

        # Check the cleanliness of the world
        world = gamedict['world']
        assert 'GoalBall' in world['objects'].keys(),\
            "Must have GoalBall object included"
        for onm, o in world['objects'].items():
            if o["density"] > 0:
                assert o['type'] not in ['Compound', 'Container'],\
                    "Not allowed to have moveable compounds"

        self._world = world
        self._drop = gamedict['dropper']
        self._goalrange = gamedict['goal']

        toolspan_min = min(v[0] for v in self._drop)
        toolspan_max = max(v[0] for v in self._drop)
        realrange = [gamedict['droprange'][0] + toolspan_min,
                     gamedict['droprange'][1] + toolspan_max]
        self._droprange = gamedict['droprange']
        self._block_droprange = realrange # for putting up blockers
        self._dim = self._world['dims']
        self._maxt = max_time

        self._record_ts = record_timestep
        self._world_ts = world_timestep

        # Add the goal along the floor
        dimx = self._world['dims'][0]
        self._world['objects']['FloorGoal'] = {
            "type": "Goal",
            "color": "none",
            "density": 0,
            "vertices": [
                [0,0], [0,5], [dimx, 5], [dimx, 0]
            ]
        }
        self._world['gcond'] = {
            "type": "SpecificInGoal",
            "goal": "FloorGoal",
            "obj": "GoalBall",
            "duration": 0.0
        }
        # Add blockers
        if self._block_droprange[0] > 0:
            self._world['blocks']['lblock'] = {
                "color": "grey",
                "vertices": [[0, 0], [0, self._dim[1]],
                             [self._block_droprange[0], self._dim[1]],
                             [self._block_droprange[0], 0]]
            }
        if self._block_droprange[1] < self._dim[0]:
            self._world['blocks']['rblock'] = {
                "color": "grey",
                "vertices": [[self._block_droprange[1], 0],
                             [self._block_droprange[1], self._dim[1]],
                             self._dim,
                             [self._dim[1], 0]]
            }

        self._runner = JSRunner()
        self._checker = CollisionChecker(self._world)

        self._raw_pygw = loadFromDict(self._world)
        # I'm gonna assume placing at y=550 in the midpoint of the range is
        # okay...
        self._place_pygw = loadFromDict(self.make_placed_world(
            [sum(self._droprange)/2,550]))


    """Runs the world and returns trajectories and outcomes from a single
    placement

    Args:
        position [x,y]: the position to place the dropper at

    Returns:
        A list of an outcome boolean if the goal has been hit, and a dictionary
        with all dynamic object trajectories & rotations. If the placement is
        illegal, just returns None, None
    """
    def run_placement(self, position):
        w = self.make_placed_world(position)
        if w is None:
            return None, None

        traj, ret, tm = self._runner.run_gw_path_and_rot(w, self._maxt,
                                                         self._record_ts)

        # Have to manually check for success... goal object as stands just
        # ends the trajectory
        gb_endpos = traj['GoalBall'][0][-1]
        success = self._goalrange[0] <= gb_endpos[0] <= self._goalrange[1]
        return success, traj

    """Makes a new world dictionary with the dropper placed in it

    Args:
        position [x,y]: the position to place the dropper at
        skipcheck [bool]: skip the check that you can place the tool? This
            is required because the check doesn't play nicely with Torch

    Returns:
        A dict object formatted to be run through JSRunner or translated into
        the lcp_physics engine
    """
    def make_placed_world(self, position):
        newworld = copy.deepcopy(self._world)
        # Add the dropped object
        drop = self._make_dropper(position)
        if drop is None:
            return None
        newworld['objects']['PLACED'] = drop
        return newworld

    """Checks whether this is a valid placement

    Args:
        position [x,y]: the dropper placement to check

    Returns:
        Boolean indicating if this is okay
    """
    def check_valid_placement(self, position):
        if position[0] < 0 or position[0] > self._world['dims'][0] or \
             position[1] < 0 or position[1] > self._world['dims'][1]:
           return False
        return not self._checker([self._drop], position)

    """Draws the initial state of the world

    Args:
        None

    Returns:
        A pygame.Surface object with the initial game image
    """
    def draw(self):
        # Have to reset the world...
        sc = drawWorld(self._raw_pygw)
        self._draw_goalbox(sc)
        return sc

    """Helper to draw the goalbox on the screen"""
    def _draw_goalbox(self, screen):
        gbox = [[self._goalrange[0], self._dim[1]],
                [self._goalrange[0], self._dim[1]-5],
                [self._goalrange[1], self._dim[1]-5],
                [self._goalrange[1], self._dim[1]]]
        pg.draw.polygon(screen, (0,255,0), gbox)

    """Draws the world with dynamic objects given a certain position

    Args:
        trajectory [dict]: a set of object position/rotations as given by the
            output of runPlacement[1]
        time [float]: the time in that trajectory to draw

    Returns:
        A pygame.Surface with the image of the world in that state
    """
    def draw_by_position(self, trajectory, time):
        idx = self._time_to_pos_idx(time)
        nsteps = len(list(trajectory.values())[0][0])
        assert 0 <= idx < nsteps, "Time input out of bounds"
        return self._draw_by_index(trajectory, idx)

    """As above, but uses indices privately"""
    def _draw_by_index(self, trajectory, index):
        for onm, traj in trajectory.items():
            assert onm in self._place_pygw.objects.keys(),\
                "Object name not found: " + onm
            obj = self._place_pygw.objects[onm]
            assert not obj.isStatic(), onm + " is not static!"
            pos = traj[0][index]
            rot = traj[1][index]
            obj.position = pos
            obj.rotation = rot
        sc = drawWorld(self._place_pygw)
        self._draw_goalbox(sc)
        return sc

    """Makes a movie of the outcome of a given placement

    Args:
        position [x,y]: the position to place the dropper at
        movie_screen [pygame.Surface]: the display screen to write to; if
            None, just writes to files
        output_dir [string]: a path to a directory where the movie images
            are written to individually (optional - can be None)
        return_screens [bool]: should this function return all of the
            pg.Surface objects it makes?
    """
    def make_placement_movie(self, position, movie_screen=None,
                             output_dir=None, return_screens=False):
        assert movie_screen is not None or output_dir is not None or return_screens,\
            "No display or place to write!"
        screens = []
        succ, traj = self.run_placement(position)
        assert traj is not None, "Illegal placement -- no trajectory"
        hz = 1 / self._record_ts
        nsteps = len(list(traj.values())[0][0])
        if movie_screen is not None:
            clk = pg.time.Clock()
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            odmask = os.path.join(output_dir, "img_{0:05d}.png")

        for idx in range(nsteps):
            draw_screen = self._draw_by_index(traj, idx)
            screens.append(draw_screen)
            if movie_screen is not None:
                pg.event.pump()
                movie_screen.blit(draw_screen, (0, 0))
                pg.display.flip()
                clk.tick(hz)
            if output_dir is not None:
                pg.image.save(draw_screen, odmask.format(idx))
        if return_screens:
            return succ, traj, screens


    """Expose the world dictionary"""
    def _get_world(self):
        return self._world
    world = property(_get_world)

    """Expose the dropper"""
    def _get_dropper(self):
        return self._drop
    dropper = property(_get_dropper)

    """Expose the droprange"""
    def _get_droprange(self):
        return self._droprange
    droprange = property(_get_droprange)

    """Expose the goal extent"""
    def _get_goal(self):
        return self._goalrange
    goal = property(_get_goal)

    """Helper so functions can take in time, return an index within a position
    vector"""
    def _time_to_pos_idx(self, time):
        rough_idx = time / self._record_ts
        idx = int(round(rough_idx,0))
        assert abs(idx-rough_idx) < (self._record_ts / 1000.),\
            "Time not found in positions index... ensure divisibility!"
        return idx

    """Helper to place the dropper"""
    def _make_dropper(self, position):
        # Make sure the position is allowed
        if not (self._droprange[0] <= position[0] and\
                    position[0] <= self._droprange[1]) or\
                self._checker(self._drop, position):
            return None
        newverts = [[position[0] + v[0], position[1] + v[1]]
                    for v in self._drop]
        return {
            "type": "Poly",
            "color": "blue",
            "density": 1,
            "vertices": newverts
        }

"""Creates a world dicitonary with defauts and containing walls for
ease of construction"""
def make_default_simple_world():
    return {
        "dims": [600, 600],
        "bts": 0.01,
        "gravity": 200,
        "defaults": {
            "density": 1,
            "friction": 0.5,
            "elasticity": 0.5,
            "color": "black",
            "bk_color": "white"
        },
        "objects": {
            "_LeftWall": {
              "type": "Poly",
              "color": "black",
              "density": 0,
              "vertices": [
                [-1, -1],
                [-1, 601],
                [1, 601],
                [1, -1]
              ]
            },
            "_BottomWall": {
              "type": "Poly",
              "color": "black",
              "density": 0,
              "vertices": [
                [-1, -1],
                [-1, 1],
                [601, 1],
                [601, -1]
              ]
            },
            "_RightWall": {
              "type": "Poly",
              "color": "black",
              "density": 0,
              "vertices": [
                [599, -1],
                [599, 601],
                [601, 601],
                [601, -1]
              ]
            },
            "_TopWall": {
              "type": "Poly",
              "color": "black",
              "density": 0,
              "vertices": [
                [-1, 599],
                [-1, 601],
                [601, 601],
                [601, 599]
              ]
            }
        },
        "blocks": {},
        "constraints": {},
        "gcond": {}
    }
