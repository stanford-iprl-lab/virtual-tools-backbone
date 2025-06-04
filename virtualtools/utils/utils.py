import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def limit_angle_range(angle):
    """
    Limits the angle to the range [-pi, pi]
    """
    angle = angle % (2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle

def min_distance_point_to_segment(segment, p):
    p = np.array(p)
    r = segment[1] - segment[0]
    a = segment[0] - p

    min_t = np.clip(-a.dot(r) / (r.dot(r)), 0, 1)
    d = a + min_t * r

    return np.sqrt(d.dot(d))
