import h5py
import sys
import os
import csv
import pdb
import pygame as pg
import numpy as np

DEF_EMP_DAT = os.path.join(
    os.path.dirname(__file__), '..', '..', 'JSGaming',
    'Analysis', 'SingleDrop_Data.csv')

DEF_EMP_DAT = os.path.join(os.path.dirname(__file__), '..', '..','Analysis','EmpiricalData', 'FullGameSuccessData.csv')
COLORDICT = {'obj1': (255,0,255), 'obj2': (255,255,0), 'obj3': (0,255,255)}

def visualize_bayesopt_likelihoods(tp, world, domain, kernel, obj_func_creator, bo_data, obj_data, data, X_step, Y_step, likelihood_fig_folder):
    pg.init()
    pg.display.set_mode((10,10)) # Needed for alpha jazz


    if trnm not in ['IntroPlay2', 'Introduction1', 'Introduction2_Lower']:
        fnc = make_bayesopt_distribution(tp, world, domain, kernel, obj_func_creator, bo_data, obj_data, data, X_step, Y_step)
        o1, o2, o3 = make_trial_llh_img(tp, fnc)

        o1.blit(o2, (0, 0))
        o1.blit(o3, (0, 0))
        #i.blit(o3, (0, 0))
        fbase = os.path.join(likelihood_fig_folder, trnm + '_last.png')
        pg.image.save(o1, fbase)

        #pg.image.save(o2, fbase + "2.png")
        #pg.image.save(o3, fbase + "3.png")
    pg.quit()


def make_bayesopt_distribution(tp, world, domain, kernel, obj_func_creator, bo_data, obj_data, data, X_step, Y_step):
    bo_step = {}
    obj_func = {}
    for i,tool in enumerate(['obj1','obj2','obj3']):
        new_act = TPJSAction(tp, tp._toolNames[tool], [0,0])
        obj_func[tool] = obj_func_creator(new_act, world, None, data, obj_data)

        bo_step[tool] = makeBayesOpt(obj_func[tool], domain.copy(), X_step[X_step[:,-1] == i, 0:-1], Y_step[X_step[:,-1] == i],
                    kernel.copy(), tp, bo_data['stddev_y'], bo_data['stddev_x'], normal=False, acq_type='EI')


    def comb_fnc(x):
        tool, pos = x
        acq_val = bo_step[tool].model.predict(pos)[0]
        return acq_val
    return comb_fnc

def find_bayesopt_functions(tp, dat):
    """
    Produces likelihood functions for each trial

    Parameters
    ----------
    model_data : dict
        Output of extract_data
    data_group : hdf5.group
        Group where data is located

    Returns
    -------
    A dict with each trial name with elements that are likelihood functions
    """
    fncs = make_bayesopt_distribution(tp, dat['world'], dat['domain'], dat['kernel'], dat['obj_func_creator'], 
            dat['bo_data'], dat['obj_data'], dat['data'], dat['X_step'], dat['Y_step'])
    return fncs


def p2purp(p):
    g = 255 - (255*p)
    return (255, int(g), 255)

def llh2purp(llh):
    # clamp to -16 to -8
    llh = min(max(llh, -16), -8)
    g = int((llh+8) * (-255 / 8))
    return (255, g, 255)

def llh2col(llh, tool, maxAlpha=196):
    if tool == 'obj1':
        # clamp to -16 to -8
        llh = min(max(llh, -16), -8)
        g = int((llh+8) * (-255 / 8))
        return (255, 0, 255, 255-g)
    elif tool == 'obj2':
        llh = min(max(llh, -16), -8)
        b = int((llh+8) * (-255 / 8))
        return (255, 255, 0, 255-b)
    else:
        llh = min(max(llh, -16), -8)
        r= int((llh+8) * (-255 / 8))
        return (0, 255, 255, 255-r)


def make_single_llh_field(llh_fnc, tool, dim_size):
    """
    Makes a small pygame surface with likelihood for one tool

    Parameters
    ----------
    llh_fnc : function
        A likelihood function
    tool : string
        The tool name
    dim_size : int
        Size of surface dimensions (smaller for interpolation)

    Returns
    -------
    A pygame.Surface of size (dim_size, dim_size) with just the likelihood
    field (no level information)
    """
    px_per = 600 / dim_size / 2
    pts_raw = np.linspace(0, 600, dim_size, False) + px_per
    pts = [int(p) for p in pts_raw]

    s = pg.Surface((dim_size, dim_size)).convert_alpha()
    s.fill((0,0,0,0))
    for i, x in enumerate(pts):
        for j, y in enumerate(pts):
            llh = llh_fnc((tool, (x,y)))
            #lh = np.exp(llh)
            s.set_at((i, dim_size-j), llh2col(llh,tool))
    return s

def make_trial_llh_img(tp, llh_fnc, tool_separate=False, dim_size=60):
    """
    Makes a pygame surface

    Parameters
    ----------
    tp : ToolPicker
        The toolpicker object
    llh_fnc : function
        The likelihood function
    empdat : dict
        Empirical data from load_singledrop_empirical
    dim_size : int
        Size of surface dimensions for creating the llh fields

    Returns
    -------
    Three pygame.Surface objects (one for each tool)
    """
    r = []
    world = tp.world
    def makept(p):
        return [int(i) for i in world._invert(p)]

    for tool in tp.toolNames:
        s_sm = make_single_llh_field(llh_fnc, tool, dim_size)
        s = pg.transform.smoothscale(s_sm, (600, 600))
        baseC = COLORDICT[tool]
        if tool_separate:
            for b in world.blockers.values():
                drawpts = [makept(p) for p in b.vertices]
                pg.draw.polygon(s, b.color, drawpts)

            for o in world.objects.values():
                _draw_obj(o, s, makept)

        r.append(s)
    if not tool_separate:
        for b in world.blockers.values():
            drawpts = [makept(p) for p in b.vertices]
            pg.draw.polygon(s, b.color, drawpts)

        for o in world.objects.values():
            _draw_obj(o, s, makept)

        for etool, epos in empdat:
            baseC = COLORDICT[etool]
            pg.draw.circle(r[-1],(0,0,0,255),makept(epos),3+2)
            pg.draw.circle(r[-1], (baseC[0], baseC[1], baseC[2]), makept(epos), 3) 
    return r


def show_screen(s):
    pg.init()
    sc = pg.display.set_mode(s.get_size())
    sc.blit(s, (0, 0))
    pg.display.flip()
    while True:
        for e in pg.event.get():
            if e.type == pg.constants.QUIT:
                return
