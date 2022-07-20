import numpy as np
from .abstracts import VTCond_Base
from .constants import *
from .object import *

__all__ = ["VTCond_AnyInGoal", "VTCond_SpecificInGoal", "VTCond_AnyTouch",
           "VTCond_SpecificTouch", "VTCond_ManyInGoal"]



class VTCond_AnyInGoal(VTCond_Base):

    def __init__(self, goalname, duration, parent, exclusions = []):
        self.type = "AnyInGoal"
        self.won = False
        self.goal = goalname
        self.excl = exclusions
        self.dur = duration
        self.ins = {}
        self.has_time = True
        self.parent = parent

    def _goes_in(self, obj, goal):
        if (goal.name == self.goal and \
                    (not obj.name in self.ins.keys()) and \
                    (not obj.name in self.excl)):
            self.ins[obj.name] = self.parent.time

    def _goes_out(self, obj, goal):
        if (goal.name == self.goal and \
            obj.name in self.ins.keys() and \
                    (not goal.pointIn(obj.position))):
            del self.ins[obj.name]

    def attach_hooks(self):
        self.parent.set_goal_collision_begin(self._goes_in)
        self.parent.set_goal_collision_end(self._goes_out)

    def _get_time_in(self):
        if len(self.ins) == 0:
            return -1
        mintime = min(min(self.ins.values()), self.parent.time)
        return mintime

class VTCond_ManyInGoal(VTCond_Base):

    def __init__(self, goalname, objlist, duration, parent):
        self.type = "ManyInGoal"
        self.won = False
        self.goal = goalname
        self.objlist = objlist
        self.objsin = []
        self.dur = duration
        self.tin = -1
        self.has_time = True
        self.parent = parent

    def _goes_in(self, obj, goal):
        if (goal.name == self.goal and
            obj.name in self.objlist and
            obj.name not in self.objsin):
            self.objsin.append(obj.name)
            if len(self.objsin) == 1:
                self.tin = self.parent.time

    def _goes_out(self, obj, goal):
        if (goal.name == self.goal and
            obj.name in self.objsin):
            self.objsin.remove(obj.name)
            if len(self.objsin) == 0:
                self.tin = -1

    def attach_hooks(self):
        self.parent.set_goal_collision_begin(self._goes_in)
        self.parent.set_goal_collision_end(self._goes_out)

    def _get_time_in(self):
        return self.tin


class VTCond_SpecificInGoal(VTCond_Base):

    def __init__(self, goalname, objname, duration, parent):
        self.type = "SpecificInGoal"
        self.won = False
        self.goal = goalname
        self.obj = objname
        self.dur = duration
        self.tin = -1
        self.has_time = True
        self.parent = parent

    def _goes_in(self, obj, goal):
        if goal.name == self.goal and obj.name == self.obj:
            self.tin = self.parent.time

    def _goes_out(self, obj, goal):
        if goal.name == self.goal and obj.name == self.obj and (not goal.point_in(obj.position)):
            self.tin = -1

    def attach_hooks(self):
        self.parent.set_goal_collision_begin(self._goes_in)
        self.parent.set_goal_collision_end(self._goes_out)

    def _get_time_in(self):
        return self.tin


class VTCond_AnyTouch(VTCond_Base):

    def __init__(self, objname, duration, parent):
        self.type = "AnyTouch"
        self.won = False
        self.goal = objname
        self.dur = duration
        self.tin = -1
        self.has_time = True
        self.parent = parent

    def _begin_touch(self, obj, goal):
        if obj.name == self.goal or goal.name == self.goal:
            self.tin = self.parent.time

    def _end_touch(self, obj, goal):
        if obj.name == self.goal or goal.name == self.goal:
            sefl.tin = -1

    def attach_hooks(self):
        self.parent.set_solid_collision_begin(self._begin_touch)
        self.parent.set_solid_collision_end(self._end_touch)

    def _get_time_in(self):
        return self.tin

class VTCond_SpecificTouch(VTCond_Base):

    def __init__(self, objname1, objname2, duration, parent):
        self.type = "SpecificTouch"
        self.won = False
        self.o1 = objname1
        self.o2 = objname2
        self.dur = duration
        self.tin = -1
        self.has_time = True
        self.parent = parent

    def _begin_touch(self, obj1, obj2):
        if (obj1.name == self.o1 and obj2.name == self.o2) or \
            (obj1.name == self.o2 and obj2.name == self.o1):
            self.tin = self.parent.time

    def _end_touch(self, obj1, obj2):
        if (obj1.name == self.o1 and obj2.name == self.o2) or \
            (obj1.name == self.o2 and obj2.name == self.o1):
            self.tin = -1

    def attach_hooks(self):
        self.parent.set_solid_collision_begin(self._begin_touch)
        self.parent.set_solid_collision_end(self._end_touch)

    def _get_time_in(self):
        return self.tin
