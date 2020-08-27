# colorplot

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import pdb

def calc_metrics(pts):
    true = np.array(pts[0]).reshape(len(pts[0]),1)
    pred = np.array(pts[1]).reshape(len(pts[1]),1)
    ER = np.sqrt(np.sum(np.square(true-pred), axis=1))
    MAE = np.mean(ER)
    norm_ER = ER/np.max(ER)
    sort_ER = np.sort(ER)
    CEP_ind = int(sort_ER.shape[0]*0.5)
    CEP95_ind = int(sort_ER.shape[0]*0.95)
    if 0: 
        print('MAE = ', MAE)
        print('CEP = ', sort_ER[CEP_ind])
        print('CEP95 = ', sort_ER[CEP95_ind])
    return ER, MAE, sort_ER[CEP_ind], sort_ER[CEP95_ind]

def bubbles_plot(pts, str_name, savepath=None):
    # rng_pos = true pos
    # cur_pos = predicted pos

    ER, MAE, CEP, CE95 = calc_metrics(pts)
    rng_pos = np.array(pts[0]).reshape(len(pts[0]),1)
    cur_pos = np.array(pts[1]).reshape(len(pts[1]),1)
    
#     pdb.set_trace()

    fig1 = plt.figure()
    fig1.suptitle('Scenario: '+ str_name)
    ax0 = fig1.add_subplot(111)
    ax0.plot(rng_pos, 'bo', zorder = 1, mfc='none', label='target')
    im0 = ax0.plot(cur_pos, 'ro', mfc='none', label='pred', zorder = 3)
    if 0:
        im0.set_clim(0.0, 200.0)
    title = 'MAE= %.3f' % MAE, 'CEP= %.3f' % CEP, 'CE95= %.3f' % CE95
    ax0.set_title(title)
    fig1.gca().invert_yaxis()
    ax0.grid()


