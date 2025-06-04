#imports
import json
import numpy as np
from scipy.stats import qmc
import cv2
from tqdm import tqdm
from virtualtools.interfaces import ToolPicker
from virtualtools.world import load_vt_from_dict
from virtualtools.vtviewer.visualization import makeImageArrayAsNumpy, visualizePathSingleImageVT
from .utils import get_collision_areas
import os
import matplotlib.pyplot as plt

def get_feasible_actions(set, name, variation, size, tool, balance=0.5, with_variations=True):
    print(f"Generating {size} feasible actions for {set} {name} {variation}")
    feasible_actions = []
    num_actions_with_collision = int(size * balance)

    # load environment
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "trials", set)

    if with_variations:
        path = os.path.join(path, name, variation)
        with open(path,'r') as f:
            btr = json.load(f)
    else:
        path = os.path.join(path, name + '.json')
        with open(path,'r') as f:
            btr = json.load(f)

    tp = ToolPicker(btr)
    wd = load_vt_from_dict(tp._worlddict)

    areas = []
    dynamic_objects = wd.get_dynamic_objects()

    for obj in dynamic_objects:
        if obj.name == 'PLACED':
            continue
        col_areas = get_collision_areas(tp, obj, tool)
        if len(col_areas) == 0:
            continue
        areas.extend(col_areas)

    dynamic_objects = [obj.name for obj in dynamic_objects]

    actions_with_collision = 0
    attempts = 0
    max_attempts = 10*num_actions_with_collision # max attempts to generate a feasible action (some tools can't be placed in some areas)
    actions_without_collision = []

    while actions_with_collision < num_actions_with_collision and max_attempts > attempts:
        attempts += 1
        area = areas[np.random.randint(0, len(areas))]
        x = np.random.randint(area[0], area[2])
        y = np.random.randint(area[1], area[3])

        try:
            tp.place({"tool": tool, "position": (x, y)})
            path_dict, fcol, success, _, aux_wd = tp.observe_collision_events(action={'tool': tool, 'position': (x,y)}, maxtime=20., stop_on_goal=True, return_world=True)
            if not path_dict or not fcol:
                continue
            
            has_collision = False
            for c in fcol:
                obj1, obj2, _, _, _ = c
                if obj1 == 'PLACED' and obj2 in dynamic_objects or obj2 == 'PLACED' and obj1 in dynamic_objects:
                    actions_with_collision += 1
                    # visualizePathSingleImageVT(wd, path_dict)
                    feasible_actions.append((x,y))
                    has_collision = True
                    break
            if not has_collision:
                actions_without_collision.append((x,y))
        except:
            continue
            
    print(f"Generated {len(feasible_actions)} actions with collision")
    print(f"Generated {len(actions_without_collision)} actions without collision")
    actions_without_collision = actions_without_collision[:size-len(feasible_actions)]
    feasible_actions.extend(actions_without_collision)

    if len(feasible_actions) == size:
        return feasible_actions

    sampler = qmc.Sobol(2, scramble=True)
    n_actions_base_2 = int(np.log2((size-len(feasible_actions))*1.2))
    print(f"Generating {2**n_actions_base_2} extra actions without collision")
    actions = sampler.random_base2(max(n_actions_base_2, 14))

    # rescale actions to world dims
    actions[:,0] = actions[:,0]*600
    actions[:,1] = actions[:,1]*600

    i = 0
    while len(feasible_actions) < size:
        action = actions[i]
        try:
            tp.place({"tool": tool, "position": (action[0], action[1])})
            feasible_actions.append(action)
        except:
            pass
        i += 1

    print(f"Total generated actions: {len(feasible_actions)}")
    
    return feasible_actions

class NoPathException(Exception):
    pass


def simulate_single_puzzle(set, name, variation, actions, size, tool, output_size=128, mask_size=21, with_variations=True):
    # load environment
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "trials", set)

    if with_variations:
        path = os.path.join(path, name, variation)
        with open(path,'r') as f:
            btr = json.load(f)
    else:
        path = os.path.join(path, name + '.json')
        with open(path,'r') as f:
            btr = json.load(f)
    tp = ToolPicker(btr)
    wd = load_vt_from_dict(tp._worlddict)

    boxes = []
    masks = []
    imgs = []
    labels = []
    final_actions = []
    final_actions_no_tool = 0

    for act in tqdm(actions):
        try:
            # get path
            if act is None:
                path_dict, _, _, wd = tp.observe_placement_path_bounding_boxes(action=act,return_world=True, stop_on_goal=False)
                success = False
                final_actions_no_tool += 1
            else:
                path_dict, success, _, wd = tp.observe_placement_path_bounding_boxes(action={'tool':tool, 'position':(act[0], act[1])},return_world=True, stop_on_goal=False)
            if not path_dict:
                raise NoPathException
        except (NoPathException):
            path_dict, _, _, wd = tp.observe_placement_path_bounding_boxes(action=None,return_world=True, stop_on_goal=False)
            success = False
            final_actions_no_tool += 1


        # get dynamic objects list
        dynamic_objects = wd.get_dynamic_objects()
        
        final_actions.append(act)
        img_array = makeImageArrayAsNumpy(wd, path_dict, sample_ratio=1)
        # resize images
        original_size = img_array[0].shape
        img_array = [cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_NEAREST) for img in img_array]
        img_array = np.array(img_array)

        imgs.append(img_array)
        
        act_boxes = []
        act_masks = []

        error = False
        for ii in range(len(path_dict[list(path_dict.keys())[0]])):
            if error:
                break
            step_boxes = []
            step_masks = []
            for obj in dynamic_objects:
                if error:
                    break
                if 'FAKE' in obj.name:
                    continue
                bbox = path_dict[obj.name][ii][3]
                obj_geom = path_dict[obj.name][ii][4]
                # x, y as center of the object
                x = (bbox[0] + bbox[2]) / 2
                y = (bbox[1] + bbox[3]) / 2
                    
                # make sure bbox is minx, miny, maxx, maxy
                bbox = np.array([min(bbox[0],bbox[2]), 
                                min(original_size[1]-bbox[1],original_size[1]-bbox[3]), 
                                max(bbox[0],bbox[2]),
                                max(original_size[1]-bbox[1],original_size[1]-bbox[3])])
                # make sure bbox is inside image
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(original_size[0], bbox[2])
                bbox[3] = min(original_size[1], bbox[3])

                # protect against random errors
                if bbox[2] - bbox[0] < 1 or bbox[3] - bbox[1] < 1:
                    error = True
                    break

                y = original_size[1] - y
                mask = np.zeros((round(bbox[3]-bbox[1]), round(bbox[2]-bbox[0])))
                if obj.type == 'Ball':
                    _,_,radius = obj_geom[0][0], obj_geom[0][1], obj_geom[1]
                    mask = cv2.circle(mask, (int((bbox[2]-bbox[0])/2),int((bbox[3]-bbox[1])/2)), int((bbox[2]-bbox[0])//2), 1, -1)
                elif obj.type == 'Poly':
                    vertices = np.array(obj_geom).reshape(-1,2)
                    vertices[:,1] = original_size[1] - vertices[:,1]
                    vertices = np.rint(vertices - np.array([bbox[0], bbox[1]])).astype(int)
                    mask = cv2.fillPoly(mask, [vertices], 1)
                elif obj.type == 'Container' or obj.type == 'Compound':
                    polys = obj_geom
                    for poly in polys:
                        vertices = np.array(poly).flatten().reshape(-1,2)
                        vertices[:,1] = original_size[1] - vertices[:,1]
                        vertices = np.rint(vertices - np.array([bbox[0], bbox[1]])).astype(int)
                        mask = cv2.fillPoly(mask, [vertices], 1)
                else:
                    raise Exception('Shape type "' + obj.type + '" not found')

                # resize mask to mask size
                mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)

                # resize bbox to output size
                bbox = np.array(bbox)
                bbox[0] = bbox[0]*output_size/original_size[0]
                bbox[1] = bbox[1]*output_size/original_size[1]
                bbox[2] = bbox[2]*output_size/original_size[0]
                bbox[3] = bbox[3]*output_size/original_size[1]

                step_boxes.append(bbox)
                step_masks.append(mask)

            if error:
                break

            act_boxes.append(step_boxes)
            act_masks.append(step_masks)

        if error:
            continue  
        boxes.append(act_boxes)
        masks.append(act_masks)
        labels.append(1 if success else 0)

    # return all actions without tool (placed at the end of the list) plus
    # size - number of actions without tool actions from the beginning of the list

    imgs = imgs[:size - final_actions_no_tool] + imgs[-final_actions_no_tool:]
    labels = labels[:size - final_actions_no_tool] + labels[-final_actions_no_tool:]
    boxes = boxes[:size - final_actions_no_tool] + boxes[-final_actions_no_tool:]
    masks = masks[:size - final_actions_no_tool] + masks[-final_actions_no_tool:]
    final_actions = final_actions[:size - final_actions_no_tool] + final_actions[-final_actions_no_tool:]
    
    return imgs, labels, boxes, masks, final_actions