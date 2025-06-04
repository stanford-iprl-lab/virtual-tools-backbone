import pandas as pd
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
import json
from pygame.surfarray import pixels3d
import os
import time
import torch
import scipy.spatial.distance as distance

from virtualtools.interfaces import ToolPicker
from virtualtools.world import load_vt_from_dict
from virtualtools.vtviewer import visualizePathSingleImageVT, draw_world
from virtualtools.vtviewer.visualization import makeImageArrayAsNumpy

from .utils import limit_angle_range

import sys
sys.path.append('..')

def get_list_of_objects(objects, obj_type="dynamic"):
    filtered_objects = {}
    for obj_name, obj in objects.items():
        if isinstance(obj, dict):
            if obj_type == "dynamic" and 'velocity' in obj:
                filtered_objects[obj_name] = obj
            else:
                continue
        else:
            if obj_type == "dynamic" and not obj.is_static():
                filtered_objects[obj_name] = obj
            elif obj_type == "static" and obj.is_static():
                filtered_objects[obj_name] = obj
            else:
                continue
    return filtered_objects

def get_collision_coordinates(tp, col, dynamic_objects):
    collisions = []
    first_collision_time = np.inf
    last_collision = (None, None, None)
    for c in col:
        obj1, obj2, time, _, positions = c
        
        if obj1 not in dynamic_objects or obj2 not in dynamic_objects:
            continue
        
        pos = positions[2][0][0]
        if pos[0] <= 1 or pos[1] <= 1:
            continue
        if last_collision[0] == obj1 and last_collision[1] == obj2 and np.linalg.norm(np.array(last_collision[2]) - np.array((pos[0], pos[1]))) < 25:
            continue
        last_collision = (obj1, obj2, (pos[0], pos[1]))

        collisions.append((obj1, obj2, (pos[0], pos[1]), time))
        if time < first_collision_time:
            first_collision_time = time
    return collisions, first_collision_time

def get_all_tools_from_json(set, name):
     # get human data
    path = os.path.dirname(os.path.abspath(__file__))
    path = '/'.join(path.split('/')[:-1])
    
    if set == 'Original':
        human_data = os.path.join(path, 'trials_data/original_levels/humans/FullGameSuccessData.csv')
    else:
        human_data = os.path.join(path, 'trials_data/cv_levels/humans/FullGameSuccessDataCV.csv')
    human_df = pd.read_csv(human_data)
    # get puzzle data
    human_df_puzzle = human_df[human_df['Trial'] == name]
    tools = human_df_puzzle.Tool.unique()
    # sort alphabetically
    tools = sorted(tools)
    return tools

def load_puzzle(set, name):
    # load environment
    path = os.path.dirname(os.path.abspath(__file__))
    path = '/'.join(path.split('/')[:-1])
    path = os.path.join(path, 'trials', set, name+'.json')
    with open(path) as f:
        btr = json.load(f)
    tp = ToolPicker(btr)

    # get dynamic objects list
    objects = tp.get_objects()
    dynamic_objects = get_list_of_objects(objects, obj_type="dynamic")

    # get tools
    tools = get_all_tools_from_json(set, name)

    return tp, tools, dynamic_objects

def get_puzzle_bounds(tp):
    # returns all the areas where the objects can be placed
    bounds = []
    margin = 25
    if 'white' in tp._worlddict:
        vertices = tp._worlddict['white']['vertices']
        for area in vertices:
            x_min, x_max = min(area, key=lambda x: x[0])[0], max(area, key=lambda x: x[0])[0]
            y_min, y_max = min(area, key=lambda x: x[1])[1], max(area, key=lambda x: x[1])[1]
            # add margin to min and max
            x_min, y_min, x_max, y_max = x_min + margin, y_min + margin, x_max - margin, y_max - margin
            bounds.append((x_min/600, y_min/600, x_max/600, y_max/600))
        return bounds
    else:
        return[((0 + margin)/600, (0 + margin)/600, (600 - margin)/600, (600 - margin)/600)]
    
def plot_puzzle_bounds(tp):
    bounds = get_puzzle_bounds(tp)
    fig, ax = plt.subplots()
    for bound in bounds:
        x_min, y_min, x_max, y_max = bound
        print(x_min, y_min, x_max, y_max)
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=None, edgecolor='red'))
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    plt.show()

def get_area_around_object(object, margin=20):
    pos = object.get_pos()
    pos = (pos[0], pos[1])
    if (object.type == "Poly"):
        vertices = np.array(object.get_vertices())
        min_x = min(vertices[:,0]) - margin
        max_x = max(vertices[:,0]) + margin
        min_y = min(vertices[:,1]) - margin
        max_y = max(vertices[:,1]) + margin
    elif (object.type == "Ball"):
        radius = object.get_radius()
        min_x = pos[0] - radius - margin
        max_x = pos[0] + radius + margin
        min_y = pos[1] - radius - margin
        max_y = pos[1] + radius + margin
    elif object.type == "Container" or object.type == "Compound":
        polys = np.array(object.get_polys()).reshape(-1,2)
        min_x = min(polys[:,0]) - margin
        max_x = max(polys[:,0]) + margin
        min_y = min(polys[:,1]) - margin
        max_y = max(polys[:,1]) + margin
    else:
        print("object type not recognized")
    return int(min_x), int(max_x), int(min_y), int(max_y)


KEY = None
def key_pressed(event):
    global KEY
    KEY = event.key

def gen_puzzle_variations_limits(set, name, tp):
    variations_limits = {}
    objects_bboxes = {}
    objects = tp.get_objects()
    for obj_name, obj in objects.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        vertices, points, position, polys = None, None, None, None
        if obj.type == 'Poly':
            vertices = np.array(obj.vertices)
            plt.scatter(vertices[:,0], vertices[:,1], c='magenta', s=100, marker='x')
            objects_bboxes[obj_name] = min(vertices[:,0]), max(vertices[:,0]), min(vertices[:,1]), max(vertices[:,1])
        elif obj.type == 'Container':
            points = np.array(obj.seglist)
            plt.scatter(points[:,0], points[:,1], c='magenta', s=100, marker='x')
            objects_bboxes[obj_name] = (min(points[:,0]), max(points[:,0]), min(points[:,1]), max(points[:,1]))
        elif obj.type == 'Ball':
            position = obj.position
            plt.scatter(position[0], position[1], c='magenta', s=100, marker='x')
            objects_bboxes[obj_name] = (position[0] - obj.radius, position[0] + obj.radius, position[1] - obj.radius, position[1] + obj.radius)
        elif obj.type == 'Compound':
            polys = np.array(obj.polys).reshape(-1,2)
            min_x, min_y, max_x, max_y = min(polys[:,0]), min(polys[:,1]), max(polys[:,0]), max(polys[:,1])
            corners = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
            plt.scatter(corners[:,0], corners[:,1], c='magenta', s=100, marker='x')
            objects_bboxes[obj_name] = (min_x, max_x, min_y, max_y)
        elif obj.type == 'Goal':
            vertices = np.array(obj.vertices)
            plt.scatter(vertices[:,0], vertices[:,1], c='magenta', s=100, marker='x')
            objects_bboxes[obj_name] = (min(vertices[:,0]), max(vertices[:,0]), min(vertices[:,1]), max(vertices[:,1]))
        else:
            print(Exception(f"Object type {obj.type} not recognized"))

        s = draw_world(load_vt_from_dict(tp._worlddict))
        pixels = pixels3d(s).transpose([1,0,2])
        plt.imshow(pixels, origin='lower')
        plt.grid()
        plt.gcf().canvas.mpl_connect('key_press_event', key_pressed)

        print(f"Do you want to add a range for this object ({obj_name})? (y/n)")
        while not plt.waitforbuttonpress(): pass
        global KEY
        if KEY != 'y':
            plt.close()
            continue

        print("Please click on the (min_x, min_y) corner")
        response = plt.ginput(1)
        min_x, min_y = response[0]
        min_x, min_y = int(min_x), int(min_y)
        print("Please click on the (max_x, max_y) corner")
        response = plt.ginput(1)
        max_x, max_y = response[0]
        max_x, max_y = int(max_x), int(max_y)

        # if min or max are close to previous bounding box, adjust them
        min_x = objects_bboxes[obj_name][0] if abs(min_x - objects_bboxes[obj_name][0]) < 7 else min_x
        min_y = objects_bboxes[obj_name][2] if abs(min_y - objects_bboxes[obj_name][2]) < 7 else min_y
        max_x = objects_bboxes[obj_name][1] if abs(max_x - objects_bboxes[obj_name][1]) < 7 else max_x
        max_y = objects_bboxes[obj_name][3] if abs(max_y - objects_bboxes[obj_name][3]) < 7 else max_y
        
        # if min or max are close to puzzle boundaries, adjust them
        min_x = 0 if min_x < 5 else min_x
        min_y = 0 if min_y < 5 else min_y
        max_x = 600 if max_x > 595 else max_x
        max_y = 600 if max_y > 595 else max_y

        # add to variations_limits
        variations_limits[obj_name] = {"prev": objects_bboxes[obj_name], "new": (min_x, max_x, min_y, max_y)}
        plt.close()

    # ask for object links
    print("Do you want to add object links? (y/n)")
    response = input()
    if response == 'y':
        fig, ax = plt.subplots(figsize=(10, 8))
        s = draw_world(load_vt_from_dict(tp._worlddict))
        pixels = pixels3d(s).transpose([1,0,2])
        plt.imshow(pixels, origin='lower')
        plt.grid()
        plt.show(block=False)
    while response == 'y':
        print("Please click on the first object")
        response = plt.ginput(1)
        obj1_x, obj1_y = response[0]
        obj1 = None
        closest = (None, 600)
        for obj_name, obj in objects.items():
            bbox = get_area_around_object(obj, margin=0)
            center = ((bbox[0] + bbox[1])/2, (bbox[2] + bbox[3])/2)
            distance = np.sqrt((center[0]-obj1_x)**2 + (center[1]-obj1_y)**2)
            if distance < closest[1]:
                closest = (obj_name, distance)
        obj1 = closest[0]
        print(f"Clicked on {obj1}")
        print("Please click on the second object")
        response = plt.ginput(1)
        obj2_x, obj2_y = response[0]
        obj2 = None
        closest = (None, 600)
        for obj_name, obj in objects.items():
            bbox = get_area_around_object(obj, margin=0)
            center = ((bbox[0] + bbox[1])/2, (bbox[2] + bbox[3])/2)
            distance = np.sqrt((center[0]-obj2_x)**2 + (center[1]-obj2_y)**2)
            if distance < closest[1]:
                closest = (obj_name, distance)
        obj2 = closest[0]
        print(f"Clicked on {obj2}")

        # make sure obj1 is below obj2 and to the left
        if objects_bboxes[obj1][2] > objects_bboxes[obj2][3] or objects_bboxes[obj1][0] > objects_bboxes[obj2][1]:
            obj1, obj2 = obj2, obj1
        # get x and y offset
        offset = (objects_bboxes[obj2][0] - objects_bboxes[obj1][1], objects_bboxes[obj2][2] - objects_bboxes[obj1][3])

        if 'links' not in variations_limits:
            variations_limits['links'] = {}
        # bottom up order and left to right order
        if obj1 not in variations_limits['links']:
            variations_limits['links'][obj1] = {}
        variations_limits['links'][obj1][obj2] = offset
        print("Do you want to add more object links? (y/n)")
        response = input()
    plt.close()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    path = os.path.dirname(os.path.abspath(__file__))
    path = '/'.join(path.split('/')[:-1])
    filename = os.path.join(path, 'trials', set, name, 'templates', name + '_'+timestamp+'.json')

    # check if folder exists
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    # save to json
    with open(filename, 'w') as f:
        json.dump(variations_limits, f, default=int)
    
    return variations_limits, filename

def gen_puzzle_from_limits(set, name, variations_limits, filename):
    path = os.path.dirname(os.path.abspath(__file__))
    path = '/'.join(path.split('/')[:-1])
    path = os.path.join(path, 'trials', set, name+'.json')
    template = open(path, 'r')
    variation = json.load(template)
    links = variations_limits['links'] if 'links' in variations_limits else {}
    tp = ToolPicker(variation)
    
    # shift and scale objects to new bounding box
    objects = tp.get_objects()
    for obj_name, obj in objects.items():
        if obj_name in variations_limits:
            if 'new' not in variations_limits[obj_name]:
                continue
            old_min_x, old_max_x, old_min_y, old_max_y = variations_limits[obj_name]['prev']
            new_min_x, new_max_x, new_min_y, new_max_y = variations_limits[obj_name]['new']
            
            if obj.type == 'Poly':
                vertices = np.array(obj.vertices)
                vertices = (vertices - np.array([old_min_x, old_min_y])) / np.array([old_max_x - old_min_x, old_max_y - old_min_y])
                vertices = vertices * np.array([new_max_x - new_min_x, new_max_y - new_min_y]) + np.array([new_min_x, new_min_y])
                variation['world']['objects'][obj_name]['vertices'] = vertices.tolist()

            elif obj.type == 'Container':
                points = np.array(obj.seglist)
                points = (points - np.array([old_min_x, old_min_y])) / np.array([old_max_x - old_min_x, old_max_y - old_min_y])
                points = points * np.array([new_max_x - new_min_x, new_max_y - new_min_y]) + np.array([new_min_x, new_min_y])
                variation['world']['objects'][obj_name]['points'] = points.tolist()

            elif obj.type == 'Compound':
                polys = np.array(obj.polys)
                polys = (polys - np.array([old_min_x, old_min_y])) / np.array([old_max_x - old_min_x, old_max_y - old_min_y])
                polys = polys * np.array([new_max_x - new_min_x, new_max_y - new_min_y]) + np.array([new_min_x, new_min_y])
                variation['world']['objects'][obj_name]['polys'] = polys.tolist()

            elif obj.type == 'Goal':
                vertices = np.array(obj.vertices)
                vertices = (vertices - np.array([old_min_x, old_min_y])) / np.array([old_max_x - old_min_x, old_max_y - old_min_y])
                vertices = vertices * np.array([new_max_x - new_min_x, new_max_y - new_min_y]) + np.array([new_min_x, new_min_y])
                variation['world']['objects'][obj_name]['vertices'] = vertices.tolist()

            elif obj.type == 'Ball':
                # new position is center of the new bbox
                position = (new_min_x + new_max_x) / 2, (new_min_y + new_max_y) / 2
                variation['world']['objects'][obj_name]['position'] = position
                # new radius is half the width of the new bbox
                radius = (new_max_x - new_min_x) / 2
                variation['world']['objects'][obj_name]['radius'] = radius

            else:
                print(Exception(f"Object type {obj.type} not recognized"))
    # reload puzzle items
    tp = ToolPicker(variation)
    objects = tp.get_objects()

    # sort links by obj1 position, left to right, bottom to top
    object_with_links = list(links.keys())
    bboxes = {}
    for obj_name in object_with_links:
        bboxes[obj_name] = get_area_around_object(objects[obj_name], margin=0)
    # sort by y position bottom to top
    object_with_links = sorted(object_with_links, key=lambda x: bboxes[x][3], reverse=True)
    # sort by x position left to right
    object_with_links = sorted(object_with_links, key=lambda x: bboxes[x][0], reverse=False)

    # shift objects to make them link compliant
    for obj1 in object_with_links:
        data = links[obj1]
        for obj2, offset in data.items():
            # given the new position of obj1, compute the new position of obj2
            # so that the offset is satisfied
            obj1_bbox = get_area_around_object(objects[obj1], margin=0)
            obj2_bbox = get_area_around_object(objects[obj2], margin=0)

            current_y_offset = obj2_bbox[2] - obj1_bbox[3]
            current_x_offset = obj2_bbox[0] - obj1_bbox[1]

            new_offset = [0,0]
            if current_y_offset > offset[1]:
                new_offset[1] = -(current_y_offset-offset[1])
            else:
                new_offset[1] = offset[1] - current_y_offset
            
            if current_x_offset > offset[0]:
                new_offset[0] = -(current_x_offset-offset[0])
            else:
                new_offset[0] = offset[0] - current_x_offset

            offset = np.array(new_offset)

            # shift obj2 to new position
            obj2_aux = objects[obj2]
            if obj2_aux.type == 'Poly':
                vertices = np.array(obj2_aux.vertices)
                vertices = vertices + np.array(offset)
                variation['world']['objects'][obj2]['vertices'] = vertices.tolist()

            elif obj2_aux.type == 'Container':
                points = np.array(obj2_aux.seglist)
                points = points + np.array(offset)
                variation['world']['objects'][obj2]['points'] = points.tolist()

            elif obj2_aux.type == 'Compound':
                polys = np.array(obj2_aux.polys)
                polys = polys + np.array(offset)
                variation['world']['objects'][obj2]['polys'] = polys.tolist()

            elif obj2_aux.type == 'Goal':
                vertices = np.array(obj2_aux.vertices)
                vertices = vertices + np.array(offset)
                variation['world']['objects'][obj2]['vertices'] = vertices.tolist()
            
            elif obj2_aux.type == 'Ball':
                position = obj2_aux.position
                position = (position[0] + offset[0], position[1] + offset[1])
                variation['world']['objects'][obj2]['position'] = position

            else:
                print(Exception(f"Object type {obj2_aux.type} not recognized"))

            # update puzzle at every iteration
            tp = ToolPicker(variation)
            objects = tp.get_objects()

    # display new puzzle
    tp = ToolPicker(variation)
    s = draw_world(load_vt_from_dict(tp._worlddict))
    pixels = pixels3d(s).transpose([1,0,2])
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.imshow(pixels)
    plt.grid()
    plt.title("New puzzle")
    plt.show()

    # save the puzzle
    filename = filename.split('template')
    filename = filename[0] + filename[1][1:]
    # check if folder exists
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    # save to json
    with open(filename, 'w') as f:
        json.dump(variation, f, default=int)


def get_collision_areas(tp, obj, tool):
    wd = load_vt_from_dict(tp._worlddict)
    obj_bb = obj.get_bounding_box()
    tool_bb = tp.tool_bbox(tool)

    tool_width = tool_bb[1][0] - tool_bb[0][0]
    tool_height = tool_bb[1][1] - tool_bb[0][1]

    # area on top of the object
    x_min, x_max = obj_bb[0] - min(tool_width//2,10) + 2, obj_bb[2] + min(tool_width//2,10) - 2
    y_min, y_max = obj_bb[3] + min(tool_height//2,10) + 2, 600 - min(tool_height//2,10) - 2
    if x_min < x_max and y_min < y_max:
        areas = [(x_min, y_min, x_max, y_max)]
    else:
        areas = []

    # area at the bottom of the object
    x_min, x_max = obj_bb[0] - tool_width//2 + 2, obj_bb[2] + tool_width//2 - 2
    y_min, y_max = 0 + tool_height//2 + 2, obj_bb[1] - tool_height//2 - 2
    if x_min < x_max and y_min < y_max:
        areas.append((x_min, y_min, x_max, y_max))

    final_areas = []

    for ii, area in enumerate(areas):
        x_min, y_min, x_max, y_max = area

        left_margin = x_min - tool_width//2
        if left_margin < 0:
            x_min += abs(left_margin) + 5

        right_margin = x_max + tool_width//2
        if right_margin > 600:
            x_max -= right_margin - 600 + 5

        top_margin = y_max + tool_height//2
        if top_margin > 600:
            y_max -= top_margin - 600 + 5

        bottom_margin = y_min - tool_height//2
        if bottom_margin < 0:
            y_min += abs(bottom_margin) + 5

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        points_to_check = [(x_min, y_min), (x_mid, y_min), (x_max, y_min), 
                            (x_min, y_mid), (x_mid, y_mid), (x_max, y_mid), 
                            (x_min, y_max), (x_mid, y_max), (x_max, y_max)]
        
        collision = False
        for point in points_to_check:
            try:
                tp.place({'tool': tool, 'position': point})
            except Exception as e:
                collision = True
                break

        if not collision and x_min < x_max and y_min < y_max:
            final_areas.append((x_min, y_min, x_max, y_max))
            continue
        
        # check for collisions 
        # 1) in case there is a block on the left
        collision = True
        while collision and x_min < x_max:
            for point in [(x_min, y_min), (x_min, (y_max-y_min)//3 + y_min), (x_min, y_max - (y_max-y_min)//3), (x_min, y_max)]:
                try:
                    tp.place({'tool': tool, 'position': point})
                    collision = False
                    break
                except Exception as e:
                    continue
            x_min += 1

        # 2) in case there is a block on the right
        collision = True
        while collision and x_min < x_max:
            for point in [(x_max, y_min), (x_max, (y_max-y_min)//3 + y_min), (x_max, y_max - (y_max-y_min)//3), (x_max, y_max)]:
                try:
                    tp.place({'tool': tool, 'position': point})
                    collision = False
                    break
                except Exception as e:
                    continue
            x_max -= 1

        x_mid = (x_min + x_max) / 2
        
        # 3) in case there is a block on the top
        collision = True
        while collision and y_min < y_max:
            for point in [(x_min, y_max), ((x_max-x_min)//3 +x_min, y_max), (x_max - (x_max-x_min)//3, y_max), (x_max, y_max)]:
                try:
                    tp.place({'tool': tool, 'position': point})
                    collision = False
                    break
                except Exception as e:
                    continue
            y_max -= 1

        # 4) in case there is a block on the bottom
        collision = True
        while collision and y_min < y_max:
            for point in [(x_min, y_min), ((x_max-x_min)//3 +x_min, y_min), (x_max - (x_max-x_min)//3, y_min), (x_max, y_min)]:
                try:
                    tp.place({'tool': tool, 'position': point})
                    collision = False
                    break
                except Exception as e:
                    continue
            y_min += 1

        if x_min >= x_max or y_min >= y_max:
            continue

        final_areas.append((x_min, y_min, x_max, y_max))

    return final_areas


def get_puzzle_bboxes(dynamic_objects):
    bboxes = {}
    for obj in dynamic_objects:
        bbox = obj.get_bounding_box()
        # make sure bbox is minx, miny, maxx, maxy
        bbox = np.array([min(bbox[0],bbox[2])-1, 
                            min(600-bbox[1],600-bbox[3])-1, 
                            max(bbox[0],bbox[2])+1,
                            max(600-bbox[1],600-bbox[3])+1]).astype(int)
        # make sure bbox is inside image
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(600, bbox[2])
        bbox[3] = min(600, bbox[3])
        bboxes[obj.name] = bbox
    return bboxes

def plot_inference_result(tp, action, bboxes, objects, ground_truth=False):
    try:
        path_dict, _, _, wd = tp.observe_placement_path_bounding_boxes(action=action,maxtime=20.,return_world=True, stop_on_goal=False)
        img_array = makeImageArrayAsNumpy(wd, path_dict, sample_ratio=1)
        pixels = img_array[2]
    except Exception as e:
        path_dict, _, _, wd = tp.observe_placement_path_bounding_boxes(action=None,maxtime=20.,return_world=True, stop_on_goal=False)
        img_array = makeImageArrayAsNumpy(wd, path_dict, sample_ratio=1)
        pixels = img_array[2]

    # define cmap of colors as long as the number of objects
    # cmap = plt.cm.get_cmap('tab20', len(objects))

    _, ax = plt.subplots(figsize=(10, 8))
    plt.imshow(pixels)

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()

    if not ground_truth:
        for jj, step in enumerate(bboxes[:-3]):
            for ii, bbox in enumerate(step):
                min_x, min_y, max_x, max_y = bbox
                if jj < 2:
                    continue
                else:
                    color = 'red' if 'Ball' in objects[ii] else 'blue'
                    center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                    ax.add_patch(plt.Circle(center, 3, fill=True, edgecolor=color, facecolor=color))
                if ii == len(objects)-1:
                    break
        # draw lines connecting points
        for jj, step in enumerate(bboxes[:-3]):
            for ii, bbox in enumerate(step):
                if jj < 2:
                    continue
                min_x, min_y, max_x, max_y = bbox
                center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                prev_bbox = bboxes[jj-1][ii]
                prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
                color = 'red' if 'Ball' in objects[ii] else 'blue'
                ax.plot([prev_center[0], center[0]], [prev_center[1], center[1]], color=color, linewidth=1)
                if ii == len(objects)-1:
                    break
        # add legend that indicates that the red dots and blue dots are the predicted bounding boxes
        # put the two patches in the same legend
        from matplotlib import patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Ball')
        blue_patch = mpatches.Patch(color='blue', label='Poly')
        ax.legend(handles=[red_patch, blue_patch], labels=["","Prediction"], loc='upper right', ncol=3, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5, fontsize=16)
    else:
        # plot ground truth bounding boxes
        for obj in objects:
            if obj not in path_dict:
                bbox = path_dict['PLACED']
            else:
                bbox = path_dict[obj]
            for step in range(2, 42, 2):
                min_x, min_y, max_x, max_y = bbox[step][3]
                # invert y axis
                min_y = 600 - min_y
                max_y = 600 - max_y
                if step < 6:
                    continue
                center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                ax.add_patch(plt.Circle(center, 2, fill=True, edgecolor='black', facecolor='black'))
        # draw lines connecting points
        for obj in objects:
            if obj not in path_dict:
                bbox = path_dict['PLACED']
            else:
                bbox = path_dict[obj]
            for step in range(2, 42, 2):
                min_x, min_y, max_x, max_y = bbox[step][3]
                # invert y axis
                min_y = 600 - min_y
                max_y = 600 - max_y
                if step < 6:
                    continue
                center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                prev_bbox = bbox[step-2][3]
                prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (600-prev_bbox[1] + 600-prev_bbox[3]) / 2)
                ax.plot([prev_center[0], center[0]], [prev_center[1], center[1]], color='black', linewidth=1)
        # add legend that indicates that the black dots are the ground truth
        ax.legend(['Ground truth'], loc='upper right', fontsize=16)
    filename = f'inference_result_{action["position"][0]}-{action["position"][1]}.svg'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='svg')
    plt.savefig(filename.replace('svg', 'png'), dpi=300, bbox_inches='tight', format='png')
    plt.show()

def get_velocity_vector_from_bboxes_vt(tp, args, bboxes, objects):
    velocity_vectors = []
    taken_actions = tp.get_runs()
    # bboxes shape is (num_actions, num_steps, num_objects, 4)
    for jj, action in enumerate(bboxes):
        if (int(args[jj]['position'][0].item()), int(args[jj]['position'][1].item()), args[jj]['tool']) in taken_actions:
            velocity_vectors.append(taken_actions[(int(args[jj]['position'][0].item()), int(args[jj]['position'][1].item()), args[jj]['tool'])]['v_vec'])
            continue
        objs = objects[jj]
        # replace 'objX' with 'PLACED'
        objs = ['PLACED' if 'obj' in obj else obj for obj in objs]

        # get object trajectories
        obj_traj = {}
        for ii, obj in enumerate(objs):
            bbox_seq = action[:,ii]
            center_seq = torch.tensor([((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2) for bbox in bbox_seq])
            obj_traj[obj] = center_seq

        v_vec = get_velocity_vector_from_trajectories_vt(obj_traj)
        velocity_vectors.append(v_vec)
            
    return velocity_vectors

def get_velocity_vector_from_trajectories_vt(obj_traj):
    STEPS_AHEAD = 10
    MAX_STEPS = 20
    v_vec = {}
    objs = list(obj_traj.keys())

    # np array to torch tensor
    for k, v in obj_traj.items():
        if not isinstance(v, torch.Tensor):
            obj_traj[k] = torch.tensor(v, dtype=torch.float32)

    # check if tool was placed
    if 'PLACED' not in obj_traj:
        for ii in range(len(objs)):
            initial_center = obj_traj[objs[ii]][0]
            final_center = obj_traj[objs[ii]][STEPS_AHEAD-1]
            # if the movement is too small, ignore it (< 10 pixels)
            if torch.linalg.norm(final_center - initial_center) < 10:
                v_vec[objs[ii]] = torch.tensor([torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32)])
                continue
            v_vec[objs[ii]] = (final_center[0] - initial_center[0], final_center[1] - initial_center[1])
        return v_vec

    # get initial and final step if there is collision
    # otherwise, get first and STEPS_AHEAD steps ahead

    # get trajectories for each object
    tool_traj = obj_traj['PLACED']
    dynamic_objects_traj = {obj: obj_traj[obj] for obj in objs if obj != 'PLACED'}
    dynamic_objects = list(dynamic_objects_traj.keys())
    dynamic_objects_traj = [dynamic_objects_traj[obj] for obj in dynamic_objects]
    dynamic_objects_traj = torch.stack(dynamic_objects_traj, dim=0)

    # dynamic_objects_traj shape is (num_objects, num_steps, 2)
    # tool_traj shape is (num_steps, 2)
    # compute distances between tool and objects at each step (num_objects, num_steps)
    dist = torch.linalg.norm(tool_traj - dynamic_objects_traj, dim=2)
    closest_point_per_obj = torch.argmin(dist, dim=1)
    closest_point_per_obj_dist = torch.min(dist, dim=1).values

    # get the point where any dynamic object collides with the tool (distance < 100 pixels)
    # then take the smallest point index (collision that happens first)

    colliding_points = torch.where(closest_point_per_obj_dist < 100)[0]
    if len(colliding_points) > 0:
        closest_point = torch.min(closest_point_per_obj[colliding_points])
        initial_step = min(closest_point, min(MAX_STEPS-1, dynamic_objects_traj.shape[1]))
        final_step = min(closest_point + STEPS_AHEAD, min(MAX_STEPS, dynamic_objects_traj.shape[1]))
    else:
        initial_step = 0
        final_step = STEPS_AHEAD

    if final_step - initial_step < 2:
        initial_step = 0
        final_step = STEPS_AHEAD

    for ii in range(len(dynamic_objects)):
        initial_center = dynamic_objects_traj[ii, initial_step]
        final_center = dynamic_objects_traj[ii, final_step-1]

        # if the movement is too small, ignore it (< 10 pixels)
        if torch.linalg.norm(final_center - initial_center) < 10:
            v_vec[dynamic_objects[ii]] = torch.tensor([torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32)])
            continue
            
        v_vec[dynamic_objects[ii]] = (final_center[0] - initial_center[0], final_center[1] - initial_center[1])

    return v_vec

